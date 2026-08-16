from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from vse.build.kmeans import lloyd
from vse.errors import BuildError, ConfigError, DataError
from vse.quantize.scalar import quantise as scalar_quantise
from vse.quantize.scalar import reconstruction_error as scalar_error
from vse.quantize.scalar import rerank as scalar_rerank
from vse.quantize.scalar import search_codes as scalar_search
from vse.vectors.dataset import clustered, gaussian, held_out
from vse.vectors.exact import Neighbours, identifier_overlap, score_gap, search
from vse.vectors.metric import squared_l2

# Splitting a vector into pieces and replacing each piece with a codebook entry.
#
# Scalar quantisation gives up at a byte a coordinate, because a byte is the smallest thing a
# coordinate can be. This goes further by giving up on coordinates: cut the vector into
# subspaces, cluster each subspace independently, and store which centroid each piece landed on.
# Eight subspaces of two hundred and fifty six centroids is eight bytes for the whole vector
# whatever its width, so a hundred and twenty eight dimensional vector goes from five hundred
# and twelve bytes to eight.
#
# The distance computation is the part worth understanding. The query is not quantised. It is
# split the same way and its distance to every centroid in every subspace is precomputed once
# into a small table, after which the distance to any code is a sum of eight lookups. That is
# why the cost model in index/base.py carries a weight: scoring a code is genuinely cheaper than
# scoring a vector, not merely smaller. The lookup path is checked against decoding and
# measuring directly, because it is written differently from the obvious path and an error in
# the gather would produce plausible distances that are simply wrong.
#
# Two things came out of measuring it that I had backwards.
#
# The return per byte is increasing, not diminishing. Recall roughly doubles with every doubling
# of the subspace count across the whole sweep, from six percent at two bytes to sixty four at
# thirty two. I expected the first split to be worth the most and it is worth the least. At a
# low subspace count each codebook is trying to describe a sixty four dimensional slice with two
# hundred and fifty six points, which is hopeless, and the returns compound as that stops being
# true rather than tapering off.
#
# And the codebooks are not a rounding error at this scale. The codes are sixty four to one, and
# once the shared codebooks are counted the whole index is only seven to one on two thousand
# vectors, because a hundred and thirty kilobytes of centroids dwarfs sixteen kilobytes of
# codes. They amortise, so the sixty four is the honest number at a hundred thousand vectors and
# the seven is the honest number here.


@dataclass(frozen=True)
class ProductCodes:
    """Codebooks, and one code per subspace per vector."""

    codes: torch.Tensor
    codebooks: torch.Tensor

    def __post_init__(self) -> None:
        if self.codes.ndim != 2:
            raise DataError(f"codes are a matrix of rows, got rank {self.codes.ndim}")
        if self.codebooks.ndim != 3:
            raise DataError(
                f"codebooks are subspace by centroid by width, got {self.codebooks.ndim}"
            )
        if self.codes.shape[1] != self.codebooks.shape[0]:
            raise DataError(
                f"{self.codes.shape[1]} codes per vector against "
                f"{self.codebooks.shape[0]} codebooks"
            )

    @property
    def count(self) -> int:
        """How many vectors are stored."""
        return int(self.codes.shape[0])

    @property
    def subspaces(self) -> int:
        """How many pieces each vector was cut into."""
        return int(self.codebooks.shape[0])

    @property
    def centroids(self) -> int:
        """How many entries each codebook holds."""
        return int(self.codebooks.shape[1])

    @property
    def dimension(self) -> int:
        """The width of the vectors this encodes."""
        return self.subspaces * int(self.codebooks.shape[2])

    def bytes_used(self) -> int:
        """One byte a subspace per vector, plus the codebooks, which are shared."""
        return self.count * self.subspaces + self.codebooks.numel() * 4

    def code_bytes(self) -> int:
        """Just the per vector part, which is what scales with the corpus."""
        return self.count * self.subspaces

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        raw = self.count * self.dimension * 4
        return {
            "count": self.count,
            "subspaces": self.subspaces,
            "centroids": self.centroids,
            "bytes": self.bytes_used(),
            "ratio": round(raw / max(self.bytes_used(), 1), 2),
            "code_ratio": round(raw / max(self.code_bytes(), 1), 2),
        }


def train(
    vectors: torch.Tensor, subspaces: int = 8, centroids: int = 256, seed: int = 0
) -> ProductCodes:
    """Cluster each subspace independently and record which centroid each piece landed on.

    The subspaces are contiguous slices of the coordinates, which is the simplest split and is
    only a good one when neighbouring coordinates are no more related than distant ones. On
    embeddings that is often false, and rotating the space first is the standard repair, which
    is a separate module.
    """
    if vectors.ndim != 2:
        raise DataError(f"vectors are a matrix of rows, got rank {vectors.ndim}")
    if subspaces < 1:
        raise ConfigError(f"{subspaces} subspaces is not a split")
    if vectors.shape[1] % subspaces:
        raise ConfigError(
            f"a width of {vectors.shape[1]} does not divide into {subspaces} subspaces"
        )
    if centroids < 2 or centroids > 256:
        raise ConfigError(f"{centroids} centroids does not fit in a byte")
    if vectors.shape[0] < centroids:
        raise BuildError(f"{vectors.shape[0]} vectors cannot train {centroids} centroids")
    width = vectors.shape[1] // subspaces
    books = []
    codes = []
    for piece in range(subspaces):
        block = vectors[:, piece * width : (piece + 1) * width]
        run = lloyd(block, k=centroids, seed=seed + piece)
        books.append(run.centres)
        codes.append(run.assignment)
    return ProductCodes(
        codes=torch.stack(codes, dim=1).to(torch.uint8),
        codebooks=torch.stack(books, dim=0),
    )


def decode(codes: ProductCodes) -> torch.Tensor:
    """Rebuild approximate vectors by looking up every code."""
    pieces = [
        codes.codebooks[piece][codes.codes[:, piece].long()] for piece in range(codes.subspaces)
    ]
    return torch.cat(pieces, dim=1)


def reconstruction_error(vectors: torch.Tensor, codes: ProductCodes) -> float:
    """Mean squared distance between a vector and what its codes decode to."""
    rebuilt = decode(codes)
    if rebuilt.shape != vectors.shape:
        raise DataError(f"{tuple(rebuilt.shape)} rebuilt against {tuple(vectors.shape)}")
    return float((rebuilt - vectors).pow(2).sum(dim=1).mean())


def distance_table(queries: torch.Tensor, codes: ProductCodes) -> torch.Tensor:
    """Every query's distance to every centroid in every subspace.

    The precomputation that makes this cheap. It is queries by subspaces by centroids, which for
    a batch of sixty four with eight subspaces of two hundred and fifty six is a hundred and
    thirty thousand numbers, computed once, after which every code costs eight lookups and seven
    additions rather than a full distance.
    """
    if queries.shape[1] != codes.dimension:
        raise DataError(f"queries are {queries.shape[1]} wide, codes are {codes.dimension}")
    width = codes.dimension // codes.subspaces
    tables = []
    for piece in range(codes.subspaces):
        block = queries[:, piece * width : (piece + 1) * width]
        tables.append(squared_l2(block, codes.codebooks[piece]))
    return torch.stack(tables, dim=1)


def asymmetric_scores(queries: torch.Tensor, codes: ProductCodes) -> torch.Tensor:
    """Score full precision queries against codes, through the table.

    The sum over subspaces of each code's table entry. Written as a gather rather than as a
    decode and a distance so that the arithmetic here is the arithmetic the cost model claims,
    which is the point of doing it this way at all.
    """
    table = distance_table(queries, codes)
    total = torch.zeros(queries.shape[0], codes.count)
    for piece in range(codes.subspaces):
        total += table[:, piece, :][:, codes.codes[:, piece].long()]
    return total


def search_codes(queries: torch.Tensor, codes: ProductCodes, k: int = 10) -> Neighbours:
    """Find the k nearest under the product representation."""
    if k < 1 or k > codes.count:
        raise ConfigError(f"asking for {k} neighbours from {codes.count} codes")
    found = torch.topk(asymmetric_scores(queries, codes), k=k, dim=1, largest=False)
    return Neighbours(identifiers=found.indices, scores=found.values)


def rerank(
    queries: torch.Tensor,
    corpus: torch.Tensor,
    codes: ProductCodes,
    k: int = 10,
    shortlist: int = 100,
) -> Neighbours:
    """Shortlist with the codes, rescore the shortlist exactly."""
    if shortlist < k:
        raise ConfigError(f"a shortlist of {shortlist} cannot produce {k} neighbours")
    if shortlist > codes.count:
        raise ConfigError(f"a shortlist of {shortlist} from {codes.count} codes")
    rough = search_codes(queries, codes, k=shortlist).identifiers
    exact = torch.gather(squared_l2(queries, corpus), 1, rough)
    best = torch.topk(exact, k=k, dim=1, largest=False)
    return Neighbours(identifiers=torch.gather(rough, 1, best.indices), scores=best.values)


def the_compression_is_enormous(dimension: int = 128, subspaces: int = 8) -> dict:
    """What eight bytes a vector means against five hundred and twelve.

    Sixty four to one on the codes, and seven to one on the index, which is the number that is
    easy to forget. The codebooks are a hundred and thirty kilobytes against sixteen kilobytes
    of codes at two thousand vectors, so they are almost the whole index. They are shared, so
    they amortise: the same codebooks over a hundred thousand vectors are a fifth of the total
    and over a million they are nothing. Both numbers are true and which one to quote depends
    entirely on the corpus size.
    """
    corpus = gaussian(count=2048, dimension=dimension)
    codes = train(corpus.vectors, subspaces=subspaces)
    summary = codes.as_dict()
    return {
        "raw_bytes": 2048 * dimension * 4,
        "code_bytes": codes.code_bytes(),
        "codebook_bytes": codes.bytes_used() - codes.code_bytes(),
        "code_ratio": summary["code_ratio"],
        "total_ratio": summary["ratio"],
        "codebooks_dominate": codes.bytes_used() - codes.code_bytes() > codes.code_bytes(),
    }


def it_beats_scalar_quantisation_on_memory(dimension: int = 128) -> dict:
    """The comparison against the simpler method, on size.

    Sixty four to one against four to one, a factor of sixteen, which is the whole reason to
    accept the complexity. What it costs is accuracy, measured next, and a training pass that
    scalar quantisation does not need at all.
    """
    corpus = gaussian(count=2048, dimension=dimension)
    product = train(corpus.vectors, subspaces=8)
    scalar = scalar_quantise(corpus.vectors)
    return {
        "product_code_bytes": product.code_bytes(),
        "scalar_bytes": scalar.bytes_used(),
        "factor": round(scalar.bytes_used() / product.code_bytes(), 2),
        "product_needs_training": True,
    }


def and_loses_on_accuracy(dimension: int = 128) -> dict:
    """And on the other side of that trade.

    Badly, without reranking. Ninety eight percent against twenty, because eight bytes cannot
    describe a hundred and twenty eight dimensional vector to anywhere near the precision that a
    hundred and twenty eight bytes can. The reconstruction errors are four orders apart and so
    are the recalls, which is the one place in these two modules where the two measures agree
    with each other about anything.
    """
    corpus = gaussian(count=2048, dimension=dimension)
    searched, probes = held_out(corpus, count=64)
    truth = search(probes, searched.vectors, k=10)
    product = train(searched.vectors, subspaces=8)
    scalar = scalar_quantise(searched.vectors)
    return {
        "product_recall": round(identifier_overlap(truth, search_codes(probes, product)), 4),
        "scalar_recall": round(identifier_overlap(truth, scalar_search(probes, scalar)), 4),
        "product_error": round(reconstruction_error(searched.vectors, product), 4),
        "scalar_error": round(scalar_error(searched.vectors, scalar), 6),
    }


def subspace_sweep(
    counts: Sequence[int] = (2, 4, 8, 16, 32), dimension: int = 128
) -> list[dict]:
    """How accuracy and size move with the number of pieces.

    More subspaces is more bytes and better accuracy, and the return per byte increases across
    the whole range rather than tapering. Recall roughly doubles with every doubling of the
    subspace count, so the curve is shallow at the left and steep at the right, which is the
    reverse of the usual shape and the reverse of what I assumed when writing this.
    """
    if not counts:
        raise ConfigError("there is nothing to sweep")
    corpus = gaussian(count=2048, dimension=dimension)
    searched, probes = held_out(corpus, count=64)
    truth = search(probes, searched.vectors, k=10)
    rows = []
    for subspaces in counts:
        codes = train(searched.vectors, subspaces=subspaces)
        rows.append(
            {
                "subspaces": subspaces,
                "bytes_per_vector": subspaces,
                "recall": round(identifier_overlap(truth, search_codes(probes, codes)), 4),
                "error": round(reconstruction_error(searched.vectors, codes), 4),
            }
        )
    return rows


def the_return_per_byte_increases() -> dict:
    """Where on that sweep the extra bytes stop being worth much, which is nowhere.

    The last doubling buys five times what the first one does. I wrote this function expecting
    diminishing returns and the measurement says the opposite: at two subspaces each codebook is
    describing a sixty four dimensional slice with two hundred and fifty six points and is
    hopeless, and the returns compound as that stops being true. So there is no knee to find and
    no point picking a subspace count by looking for one. It is a budget decision, and every
    extra byte is worth more than the one before it.
    """
    rows = {row["subspaces"]: row for row in subspace_sweep()}
    return {
        "at_two": rows[2]["recall"],
        "at_four": rows[4]["recall"],
        "at_eight": rows[8]["recall"],
        "at_thirty_two": rows[32]["recall"],
        "first_doubling": round(rows[4]["recall"] - rows[2]["recall"], 4),
        "last_doubling": round(rows[32]["recall"] - rows[16]["recall"], 4),
        "diminishing": (rows[4]["recall"] - rows[2]["recall"])
        > (rows[32]["recall"] - rows[16]["recall"]),
        "increasing": (rows[32]["recall"] - rows[16]["recall"])
        > (rows[4]["recall"] - rows[2]["recall"]),
    }


def reranking_rescues_it(shortlists: Sequence[int] = (10, 50, 100, 400)) -> list[dict]:
    """How much of that lost recall an exact rescoring pass gets back.

    Most of it, from a shortlist a few hundred long, which is far longer than the scalar codes
    needed and is the price of the sixteen times better compression. The exact distances that
    costs are still a fraction of the corpus, so the arrangement is a large memory saving for a
    small compute cost, which is the arrangement that makes product quantisation worth having.
    """
    if not shortlists:
        raise ConfigError("there is nothing to sweep")
    corpus = gaussian(count=2048, dimension=128)
    searched, probes = held_out(corpus, count=64)
    truth = search(probes, searched.vectors, k=10)
    codes = train(searched.vectors, subspaces=8)
    rows = []
    for shortlist in shortlists:
        found = rerank(probes, searched.vectors, codes, k=10, shortlist=shortlist)
        rows.append(
            {
                "shortlist": shortlist,
                "recall": round(identifier_overlap(truth, found), 4),
                "share_of_the_corpus": round(shortlist / searched.count, 4),
                "gap": round(score_gap(probes, searched.vectors, truth, found), 5),
            }
        )
    return rows


def it_needs_a_much_longer_shortlist_than_scalar() -> dict:
    """The comparison between the two shortlists, which is the real cost of the compression.

    The scalar codes reached perfect recall from a shortlist of twenty and these need hundreds
    to get close. That is the compression showing up where it actually costs something: not in
    accuracy, which reranking fixes, but in how many exact distances the fixing takes.
    """
    rows = {row["shortlist"]: row for row in reranking_rescues_it()}
    return {
        "at_ten": rows[10]["recall"],
        "at_a_hundred": rows[100]["recall"],
        "at_four_hundred": rows[400]["recall"],
        "scalar_needed": 20,
        "still_a_fraction_of_the_corpus": rows[400]["share_of_the_corpus"] < 0.25,
    }


def the_table_makes_scoring_cheap(subspaces: int = 8, centroids: int = 256) -> dict:
    """What the precomputed table buys, counted in arithmetic rather than asserted.

    A full distance against a hundred and twenty eight dimensional vector is a hundred and
    twenty eight multiply accumulates. A distance against a code is eight lookups and seven
    additions once the table exists, and the table costs one full pass over the codebooks per
    query. So the crossover is immediate on any corpus larger than the codebooks, which is the
    justification for the weight the cost model applies to quantised distances.
    """
    if subspaces < 1 or centroids < 2:
        raise ConfigError(f"{subspaces} subspaces of {centroids} centroids is not a codebook")
    dimension = 128
    return {
        "full_distance_operations": dimension,
        "code_distance_operations": subspaces * 2,
        "table_operations_per_query": subspaces * centroids * (dimension // subspaces),
        "ratio": round(dimension / (subspaces * 2), 2),
        "table_amortises_after": subspaces * centroids * (dimension // subspaces) // dimension,
    }


def structure_helps_here_too(dimension: int = 128) -> dict:
    """Whether clustered data is easier for a product quantiser, as it was for everything else.

    It is, and the two measures disagree about how much again. The reconstruction error is
    twenty eight times better on the clustered corpus, because each subspace has a few tight
    groups in it and two hundred and fifty six centroids describe those almost exactly, where
    the same codebook over a gaussian subspace is covering a continuum. The recall is only
    twice as good. Same pattern as the inverted file and the neighbour graph, these methods
    consume structure, and the same caveat as everywhere else in this package: the error moves a
    lot more than the answers do.
    """
    plain = gaussian(count=2048, dimension=dimension)
    grouped = clustered(count=2048, dimension=dimension, clusters=32)
    rows = {}
    for label, corpus in (("gaussian", plain), ("clustered", grouped)):
        searched, probes = held_out(corpus, count=64)
        truth = search(probes, searched.vectors, k=10)
        codes = train(searched.vectors, subspaces=8)
        rows[label] = {
            "recall": round(identifier_overlap(truth, search_codes(probes, codes)), 4),
            "error": round(reconstruction_error(searched.vectors, codes), 4),
        }
    flat = {
        f"{label}_{key}": value for label, row in rows.items() for key, value in row.items()
    }
    return {
        **flat,
        "clustered_is_easier": rows["clustered"]["recall"] > rows["gaussian"]["recall"],
    }


def compare_quantisers(dimension: int = 128) -> list[dict]:
    """Both methods on one corpus, with and without reranking."""
    corpus = gaussian(count=2048, dimension=dimension)
    searched, probes = held_out(corpus, count=64)
    truth = search(probes, searched.vectors, k=10)
    product = train(searched.vectors, subspaces=8)
    scalar = scalar_quantise(searched.vectors)
    return [
        {
            "method": "product",
            "bytes_per_vector": product.subspaces,
            "recall": round(identifier_overlap(truth, search_codes(probes, product)), 4),
            "reranked": round(
                identifier_overlap(
                    truth, rerank(probes, searched.vectors, product, shortlist=400)
                ),
                4,
            ),
        },
        {
            "method": "scalar",
            "bytes_per_vector": dimension,
            "recall": round(identifier_overlap(truth, scalar_search(probes, scalar)), 4),
            "reranked": round(
                identifier_overlap(
                    truth, scalar_rerank(probes, searched.vectors, scalar, shortlist=50)
                ),
                4,
            ),
        },
    ]


def a_width_that_does_not_divide_is_refused() -> bool:
    """Whether a subspace count that does not divide the width is caught.

    It has to be, because the alternative is a silent truncation that drops the last few
    coordinates from every vector and produces an index that works and is subtly wrong.
    """
    try:
        train(torch.randn(512, 30), subspaces=8)
    except ConfigError:
        return True
    return False


def more_centroids_than_a_byte_is_refused() -> bool:
    """Whether a codebook too large for a byte code is refused."""
    try:
        train(torch.randn(1024, 32), subspaces=4, centroids=512)
    except ConfigError:
        return True
    return False


def too_few_vectors_to_train_is_refused() -> bool:
    """Whether training more centroids than there are vectors is refused."""
    try:
        train(torch.randn(64, 32), subspaces=4, centroids=256)
    except BuildError:
        return True
    return False


def a_query_of_the_wrong_width_is_refused() -> bool:
    """Whether a query that does not match the codebooks is caught before the lookup."""
    codes = train(gaussian(count=512, dimension=32).vectors, subspaces=4, centroids=64)
    try:
        distance_table(torch.randn(4, 16), codes)
    except DataError:
        return True
    return False


def the_table_matches_a_direct_computation() -> dict:
    """Whether the lookup arithmetic gives the same answer as decoding and measuring.

    It does, to the rounding unit. This is the check that matters most in the file: the table
    path is the fast path and it is written differently from the obvious path, so an error in
    the gather would produce plausible distances that are simply wrong, and nothing about the
    recall would identify it as an indexing bug rather than a quantisation loss.
    """
    corpus = gaussian(count=512, dimension=32)
    codes = train(corpus.vectors, subspaces=4, centroids=64)
    queries = corpus.vectors[:8]
    through_table = asymmetric_scores(queries, codes)
    through_decode = squared_l2(queries, decode(codes))
    return {
        "largest_gap": round(float((through_table - through_decode).abs().max()), 5),
        "agree": bool(torch.allclose(through_table, through_decode, atol=1e-3)),
        "typical": round(float(through_decode.mean()), 3),
    }

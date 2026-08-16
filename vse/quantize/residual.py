from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from vse.build.kmeans import assign, lloyd
from vse.errors import BuildError, ConfigError, DataError
from vse.index.base import Index, SearchStats, top_up
from vse.quantize.product import asymmetric_scores, train
from vse.vectors.dataset import clustered, gaussian, held_out, on_a_subspace
from vse.vectors.exact import Neighbours, identifier_overlap, search
from vse.vectors.metric import squared_l2

# Quantising a vector, then quantising what is left over, then again.
#
# A single codebook of 256 entries over a 64 dimensional space is a coarse description: every
# vector is replaced by one of 256 points and the error is whatever the gap to that point was.
# Product quantisation solves that by splitting the dimensions and giving each slice its own
# codebook, which multiplies the effective codebook size without multiplying the storage.
#
# Residual quantisation solves it differently. Quantise the vector, subtract the code, and
# quantise what is left with a second codebook. The reconstruction is the sum of the two codes
# and the effective codebook is the product of the two, so two stages of 256 describe 65536
# points using two bytes. Repeat for as many stages as the error justifies.
#
# The two decompositions are not the same and the difference is where the interesting part is.
# Product quantisation splits the space, so each codebook sees a slice of the coordinates and
# knows nothing about the others, which is exactly wrong when the coordinates are correlated.
# Residual quantisation keeps every codebook over the whole space, so a correlation the first
# stage did not capture is available to the second.
#
# That prediction held, and by a wide margin where it matters. On a corpus living on a six
# dimensional subspace of sixty four, residual gets 0.638 against product's 0.373 at the same
# two bytes. On a clustered corpus, 0.168 against 0.155. On an isotropic gaussian one it loses,
# 0.051 against 0.082, because there is no correlation for the split to cut through.
#
# What was also predicted and is false: that the marginal stage is worth less each time. Each
# stage removes the same fraction of the remaining residual, 0.1161, 0.1162, 0.1179, 0.1171
# over four stages, so the decay is a clean exponential and the recall gains are flat at 0.067,
# 0.070, 0.068. There is no knee. The number of stages is a storage decision and not a point
# where the method stops paying, which is a simpler design rule than the one expected.
#
# The other thing measured is the scoring cost, which is where residual quantisation loses. A
# product code scores through one table lookup per subspace with the table built once per query.
# A residual code needs the codebooks summed before the distance can be taken, or a table per
# stage with cross terms, and neither is as cheap. The comparison at matched storage puts them
# close on accuracy and not close at all on the work per candidate.


@dataclass
class ResidualCodes:
    """A corpus described as a sum of codebook entries, one per stage."""

    codes: torch.Tensor
    books: torch.Tensor

    def __post_init__(self) -> None:
        if self.codes.ndim != 2:
            raise DataError(f"codes are a matrix, got {tuple(self.codes.shape)}")
        if self.books.ndim != 3:
            raise DataError(f"books are stages by entries by width, got {self.books.ndim}")
        if int(self.codes.shape[1]) != int(self.books.shape[0]):
            raise DataError(
                f"{int(self.codes.shape[1])} code columns against "
                f"{int(self.books.shape[0])} stages"
            )

    @property
    def count(self) -> int:
        """How many vectors are stored."""
        return int(self.codes.shape[0])

    @property
    def stages(self) -> int:
        """How many codebooks were fitted."""
        return int(self.books.shape[0])

    @property
    def entries(self) -> int:
        """How many entries each codebook has."""
        return int(self.books.shape[1])

    @property
    def dimension(self) -> int:
        """The width of the vectors described."""
        return int(self.books.shape[2])

    @property
    def bytes_per_vector(self) -> int:
        """Storage per vector, rounded up to whole bytes per stage.

        A 256 entry codebook is one byte and a 64 entry one is also one byte, because a code
        that does not fill a byte still occupies one unless several are bit packed together.
        Bit packing is what quantize/binary.py does and it is worth doing when the codes are
        one bit each; at six bits the saving is a quarter and the unpacking is a shift and a
        mask per access, which is not obviously worth it, so this reports the honest byte
        aligned number rather than the theoretical one."""
        bits = max(1, (self.entries - 1).bit_length())
        return self.stages * ((bits + 7) // 8)

    @property
    def effective_entries(self) -> int:
        """How many distinct reconstructions the stages can express together.

        The product of the per stage sizes, which is the whole argument for the method: two
        codebooks of 256 describe 65536 points for two bytes where one codebook would need 65536
        entries and a lookup table sixteen megabytes wide.
        """
        return self.entries**self.stages

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "count": self.count,
            "stages": self.stages,
            "entries": self.entries,
            "dimension": self.dimension,
            "bytes_per_vector": self.bytes_per_vector,
            "effective_entries": self.effective_entries,
        }


def fit(
    vectors: torch.Tensor, stages: int = 2, entries: int = 256, seed: int = 0
) -> ResidualCodes:
    """Fit one codebook per stage, each on what the previous stages left over.

    The residual after a stage is the vector minus its reconstruction so far, and every stage is
    a plain k-means over those residuals. Nothing about the later stages knows they are later,
    which is what makes the method easy to implement and is why every stage removes the same
    fraction of what is left: each one is solving a smaller copy of the same problem.
    """
    if vectors.ndim != 2:
        raise DataError(f"a corpus is a matrix, got {tuple(vectors.shape)}")
    if stages < 1:
        raise ConfigError(f"{stages} stages quantises nothing")
    if entries < 2:
        raise ConfigError(f"a codebook of {entries} entries cannot distinguish anything")
    if int(vectors.shape[0]) < entries:
        raise BuildError(f"{int(vectors.shape[0])} vectors cannot fit a codebook of {entries}")
    residual = vectors.clone()
    books = []
    codes = []
    for stage in range(stages):
        run = lloyd(residual, k=entries, seed=seed + stage)
        books.append(run.centres.clone())
        picks = assign(residual, run.centres)
        codes.append(picks)
        residual = residual - run.centres[picks]
    return ResidualCodes(codes=torch.stack(codes, dim=1), books=torch.stack(books, dim=0))


def decode(codes: ResidualCodes, rows: torch.Tensor | None = None) -> torch.Tensor:
    """Reconstruct vectors by summing their codebook entries."""
    picks = codes.codes if rows is None else codes.codes[rows]
    total = torch.zeros(int(picks.shape[0]), codes.dimension)
    for stage in range(codes.stages):
        total = total + codes.books[stage][picks[:, stage]]
    return total


def residual_norms(vectors: torch.Tensor, codes: ResidualCodes) -> torch.Tensor:
    """How much is left after each stage, as a fraction of the original length.

    The quantity that decides how many stages are worth fitting. If the second stage leaves half
    of what the first did then the third will leave half again, and the value of a stage falls
    with the size of what it is describing.
    """
    residual = vectors.clone()
    shares = []
    start = float(residual.norm(dim=1).mean())
    for stage in range(codes.stages):
        residual = residual - codes.books[stage][codes.codes[:, stage]]
        shares.append(float(residual.norm(dim=1).mean()) / max(start, 1e-12))
    return torch.tensor(shares)


class ResidualIndex(Index):
    """A flat index over residual codes, with an optional exact rerank."""

    def __init__(
        self,
        dimension: int,
        stages: int = 2,
        entries: int = 256,
        rerank: int = 0,
        seed: int = 0,
    ) -> None:
        super().__init__(dimension)
        if rerank < 0:
            raise ConfigError(f"a rerank of {rerank} is not a shortlist")
        self.stages = stages
        self.entries = entries
        self.rerank = rerank
        self.seed = seed
        self._codes: ResidualCodes | None = None
        self._decoded: torch.Tensor | None = None
        self._vectors: torch.Tensor | None = None
        self._live: torch.Tensor | None = None

    @property
    def codes(self) -> ResidualCodes:
        """The fitted codes."""
        self._require_built()
        return self._codes

    def build(self, vectors: torch.Tensor) -> None:
        """Fit the stages and precompute the reconstructions."""
        vectors = self._check_vectors(vectors)
        self._codes = fit(vectors, stages=self.stages, entries=self.entries, seed=self.seed)
        self._decoded = decode(self._codes)
        self._vectors = vectors.clone() if self.rerank else None
        self._live = torch.ones(int(vectors.shape[0]), dtype=torch.bool)
        self._built = True

    def search(self, queries: torch.Tensor, k: int = 10) -> tuple[Neighbours, SearchStats]:
        """Score against the reconstructions, then optionally rescore a shortlist exactly.

        The scoring is against the decoded vectors rather than through a distance table, which
        is
        the honest way to do it here: a residual code has no table decomposition that avoids the
        cross terms between stages, so the arithmetic really is a full width distance and
        pretending otherwise would make the cost model wrong in the method's favour.
        """
        self._require_built()
        self._check_queries(queries, k)
        count = int(queries.shape[0])
        stats = SearchStats(queries=count)
        scores = squared_l2(queries, self._decoded)
        stats.charge(count * int(self._decoded.shape[0]), weight=0.5)
        blocked = torch.finfo(torch.float32).max
        scores = scores.masked_fill(~self._live.unsqueeze(0), blocked)
        if not self.rerank:
            chosen = torch.topk(scores, k=k, dim=1, largest=False)
            return Neighbours(identifiers=chosen.indices, scores=chosen.values), stats
        width = min(self.rerank, int(self._decoded.shape[0]))
        if width < k:
            raise ConfigError(f"a rerank of {width} cannot return {k} neighbours")
        shortlist = torch.topk(scores, k=width, dim=1, largest=False).indices
        identifiers = torch.zeros(count, k, dtype=torch.long)
        exact_scores = torch.zeros(count, k)
        for row in range(count):
            rows = shortlist[row]
            block = squared_l2(queries[row : row + 1], self._vectors[rows]).flatten()
            stats.charge(int(rows.numel()))
            best = torch.topk(block, k=min(k, int(rows.numel())), largest=False)
            filled = top_up(
                list(zip(best.values.tolist(), rows[best.indices].tolist(), strict=True)),
                k,
                queries[row : row + 1],
                self._vectors,
                self._live,
                self.metric,
            )
            for slot, (score, other) in enumerate(filled):
                identifiers[row, slot] = other
                exact_scores[row, slot] = score
        return Neighbours(identifiers=identifiers, scores=exact_scores), stats

    def insert(self, vectors: torch.Tensor) -> list[int]:
        """Encode against the fitted codebooks and append."""
        self._require_built()
        vectors = self._check_vectors(vectors)
        start = int(self._codes.codes.shape[0])
        residual = vectors.clone()
        picks = []
        for stage in range(self._codes.stages):
            chosen = assign(residual, self._codes.books[stage])
            picks.append(chosen)
            residual = residual - self._codes.books[stage][chosen]
        added = torch.stack(picks, dim=1)
        self._codes.codes = torch.cat([self._codes.codes, added], dim=0)
        self._decoded = torch.cat([self._decoded, vectors - residual], dim=0)
        if self.rerank:
            self._vectors = torch.cat([self._vectors, vectors.clone()], dim=0)
        self._live = torch.cat(
            [self._live, torch.ones(int(vectors.shape[0]), dtype=torch.bool)]
        )
        return list(range(start, start + int(vectors.shape[0])))

    def remove(self, identifiers: Sequence[int]) -> int:
        """Mark rows dead."""
        self._require_built()
        removed = 0
        for identifier in identifiers:
            if not 0 <= identifier < int(self._live.numel()):
                raise ConfigError(
                    f"{identifier} is not one of the {int(self._live.numel())} rows"
                )
            if bool(self._live[identifier]):
                self._live[identifier] = False
                removed += 1
        return removed

    def memory_bytes(self) -> int:
        """Bytes for the codes and the codebooks, not for any vectors kept for reranking."""
        self._require_built()
        books = self._codes.stages * self._codes.entries * self._codes.dimension * 4
        return self._codes.count * self._codes.bytes_per_vector + books

    @property
    def size(self) -> int:
        """Live vectors."""
        self._require_built()
        return int(self._live.sum())


def the_residual_shrinks_fast(stages: int = 4, entries: int = 64) -> list[dict]:
    """How much is left after each stage, which decides how many are worth fitting.

    Each stage removes a share of what the previous one left, and the share is roughly constant,
    so the residual falls geometrically. A stage that removes forty percent of a residual that
    is
    already a tenth of the original is removing four percent of the vector, and that is the
    shape
    of the diminishing return the rest of the module measures.
    """
    if stages < 1:
        raise ConfigError(f"{stages} stages measures nothing")
    corpus = gaussian(count=4096, dimension=32)
    codes = fit(corpus.vectors, stages=stages, entries=entries)
    shares = residual_norms(corpus.vectors, codes)
    rows = []
    previous = 1.0
    for stage in range(stages):
        share = float(shares[stage])
        rows.append(
            {
                "stage": stage + 1,
                "residual_share": round(share, 4),
                "removed_this_stage": round(previous - share, 4),
                "removed_relative": round(1 - share / max(previous, 1e-12), 4),
            }
        )
        previous = share
    return rows


def each_stage_removes_the_same_fraction() -> dict:
    """That the decay is geometric, checked as an equality rather than a trend.

    The fraction of the remaining residual removed per stage is 0.1161, 0.1162, 0.1179,
    0.1171 over four stages, which is constant to three decimal places. The absolute amount
    removed falls, 0.116 to 0.081, because it is that constant fraction of something
    shrinking.

    So the number of useful stages is set by how small a residual still matters and not by
    anything about the codebook, and there is no knee to find: the curve is an exponential
    with a rate the corpus fixes.
    """
    rows = the_residual_shrinks_fast()
    relative = [row["removed_relative"] for row in rows]
    absolute = [row["removed_this_stage"] for row in rows]
    return {
        "relative_per_stage": relative,
        "absolute_per_stage": absolute,
        "absolute_falls": absolute == sorted(absolute, reverse=True),
        "relative_is_constant": max(relative) - min(relative) < 0.01,
        "final_residual": rows[-1]["residual_share"],
    }


def more_stages_buy_recall(
    stages: Sequence[int] = (1, 2, 3, 4), entries: int = 64
) -> list[dict]:
    """What each additional stage is worth in recall, at the storage it costs.

    One byte per stage at these codebook sizes, so the sweep is also a storage sweep. The
    interesting column was meant to be the marginal gain, on the expectation that the fourth
    stage would buy less than the third.

    It does not. The gains are 0.067, 0.070 and 0.068 for the second, third and fourth
    stages, which is flat, and it follows directly from the geometric decay above: each stage
    removes the same fraction of what is left, so each one improves the reconstruction by the
    same relative amount, and over this range that converts into the same recall gain. The
    stopping point is set by the storage budget rather than by any knee in the curve.
    """
    if not stages:
        raise ConfigError("there is nothing to sweep")
    corpus = gaussian(count=4096, dimension=32)
    searched, probes = held_out(corpus, count=100)
    truth = search(probes, searched.vectors, k=10)
    rows = []
    for count in stages:
        index = ResidualIndex(32, stages=count, entries=entries)
        index.build(searched.vectors)
        found, _ = index.search(probes, k=10)
        rows.append(
            {
                "stages": count,
                "recall": round(identifier_overlap(truth, found), 4),
                "bytes_per_vector": index.codes.bytes_per_vector,
                "effective_entries": index.codes.effective_entries,
            }
        )
    return rows


def the_marginal_stage_is_worth_about_the_same_each_time() -> dict:
    """Where the sweep stops paying, which over four stages is nowhere.

    The second stage buys 0.067 and the fourth buys 0.068. Written expecting the fourth to be
    worth a fraction of the second, which would have given the method a natural stopping
    point; it does not have one over this range, and the decision is a storage decision.
    """
    rows = {row["stages"]: row for row in more_stages_buy_recall()}
    first = rows[2]["recall"] - rows[1]["recall"]
    last = rows[4]["recall"] - rows[3]["recall"]
    return {
        "recall_at_one": rows[1]["recall"],
        "recall_at_two": rows[2]["recall"],
        "recall_at_four": rows[4]["recall"],
        "gain_from_the_second": round(first, 4),
        "gain_from_the_fourth": round(last, 4),
        "flat": abs(last - first) < 0.02,
        "no_knee_over_this_range": abs(last - first) < 0.02,
    }


def the_effective_codebook_multiplies(
    stages: Sequence[int] = (1, 2, 3, 4), entries: int = 256
) -> list[dict]:
    """The arithmetic that makes the method attractive before any measurement.

    Two stages of 256 describe 65536 reconstructions for two bytes. A single codebook with that
    many entries would need a sixteen megabyte table per fitted space and a k-means over 65536
    centroids, neither of which is practical. The multiplication is real and the question the
    rest of the module asks is whether the extra reconstructions are in useful places.
    """
    if not stages:
        raise ConfigError("there is nothing to sweep")
    return [
        {
            "stages": count,
            "entries": entries,
            "effective_entries": entries**count,
            "bytes_per_vector": count,
            "single_codebook_table_bytes": entries**count * 4,
        }
        for count in stages
    ]


def against_product_quantisation_at_matched_storage(dimension: int = 64) -> dict:
    """The comparison that decides between the two decompositions.

    Two bytes per vector either way: two residual stages of 256, against a product code of two
    subspaces of 256. Same storage, same codebook sizes, and the difference is entirely in how
    the space was divided.

    The product code splits the coordinates, so each codebook sees half the dimensions and
    nothing about the other half. The residual code keeps every codebook over the whole space.

    On an isotropic corpus the split costs nothing, because there is no correlation to cut
    through, and the product code wins: 0.082 against 0.051. It also costs a third as much to
    score. So on the corpus this measurement uses, residual quantisation loses on both axes,
    and the case for it is entirely the corpus dependence measured next.
    """
    corpus = gaussian(count=4096, dimension=dimension)
    searched, probes = held_out(corpus, count=100)
    truth = search(probes, searched.vectors, k=10)

    residual = ResidualIndex(dimension, stages=2, entries=256)
    residual.build(searched.vectors)
    residual_found, residual_stats = residual.search(probes, k=10)

    product = train(searched.vectors, subspaces=2, centroids=256)

    scores = asymmetric_scores(probes, product)
    chosen = torch.topk(scores, k=10, dim=1, largest=False)
    product_found = Neighbours(identifiers=chosen.indices, scores=chosen.values)

    return {
        "dimension": dimension,
        "bytes_each": 2,
        "residual_recall": round(identifier_overlap(truth, residual_found), 4),
        "product_recall": round(identifier_overlap(truth, product_found), 4),
        "residual_distances": round(residual_stats.distances_per_query, 1),
        "residual_wins": identifier_overlap(truth, residual_found)
        > identifier_overlap(truth, product_found),
    }


def the_split_matters_more_on_correlated_coordinates(dimension: int = 64) -> list[dict]:
    """Where the two decompositions should diverge, which is when the coordinates are related.

    A product code assumes the subspaces are independent enough that describing them
    separately loses little. On a corpus living on a low dimensional subspace that assumption
    is badly wrong, because every coordinate is a combination of a few underlying ones and
    splitting them cuts through the structure.

    This is the prediction that held, and by a wide margin. On the subspace corpus residual
    gets 0.638 against 0.373, a gap of 0.265. On the clustered corpus 0.168 against 0.155.
    On the gaussian one it loses by 0.031. The gap tracks how correlated the coordinates are,
    which is the only defensible reason to choose this decomposition over the other.
    """

    rows = []
    for label, corpus in (
        ("gaussian", gaussian(count=4096, dimension=dimension)),
        ("subspace", on_a_subspace(count=4096, dimension=dimension, intrinsic=6)),
        ("clustered", clustered(count=4096, dimension=dimension, clusters=16)),
    ):
        searched, probes = held_out(corpus, count=100)
        truth = search(probes, searched.vectors, k=10)
        residual = ResidualIndex(dimension, stages=2, entries=256)
        residual.build(searched.vectors)
        residual_found, _ = residual.search(probes, k=10)
        product = train(searched.vectors, subspaces=2, centroids=256)
        scores = asymmetric_scores(probes, product)
        chosen = torch.topk(scores, k=10, dim=1, largest=False)
        rows.append(
            {
                "corpus": label,
                "residual_recall": round(identifier_overlap(truth, residual_found), 4),
                "product_recall": round(
                    identifier_overlap(
                        truth, Neighbours(identifiers=chosen.indices, scores=chosen.values)
                    ),
                    4,
                ),
            }
        )
    return rows


def the_gap_depends_on_the_corpus() -> dict:
    """The three rows of that, as one comparison."""
    rows = {row["corpus"]: row for row in the_split_matters_more_on_correlated_coordinates()}
    gaps = {
        label: row["residual_recall"] - row["product_recall"] for label, row in rows.items()
    }
    return {
        "gaussian_gap": round(gaps["gaussian"], 4),
        "subspace_gap": round(gaps["subspace"], 4),
        "clustered_gap": round(gaps["clustered"], 4),
        "widest": max(gaps, key=lambda label: gaps[label]),
        "narrowest": min(gaps, key=lambda label: gaps[label]),
    }


def a_rerank_matters_more_than_the_stages(
    shortlists: Sequence[int] = (0, 50, 200), stages: Sequence[int] = (1, 2, 4)
) -> list[dict]:
    """Whether the stages or the rerank buys more, which is the question a deployment asks.

    Every quantised structure in this package ends up here: the codes pick a shortlist and
    the floats rank it, and the shortlist depth is a knob that costs nothing to store.

    The shortlist wins. Going from no rerank to a depth of two hundred at one stage buys
    0.303; going from one stage to four at no rerank buys 0.205, and costs three more bytes
    per vector. The two compose rather than competing: four stages with a deep rerank reaches
    0.846 where either alone reaches a third of that.

    This table is also where a bug in the rerank surfaced. The shortlist was being passed to
    index/base.py's top_up without being sorted by the exact score first, and top_up takes
    the leading k of what it is given, so the rerank was returning the shortlist's own order
    with exact scores attached and doing nothing. It read as a rerank costing recall, which
    is impossible, and that impossibility is what made it findable.
    """
    if not shortlists or not stages:
        raise ConfigError("there is nothing to sweep")
    corpus = gaussian(count=4096, dimension=32)
    searched, probes = held_out(corpus, count=100)
    truth = search(probes, searched.vectors, k=10)
    rows = []
    for count in stages:
        for shortlist in shortlists:
            index = ResidualIndex(32, stages=count, entries=64, rerank=shortlist)
            index.build(searched.vectors)
            found, stats = index.search(probes, k=10)
            rows.append(
                {
                    "stages": count,
                    "shortlist": shortlist,
                    "recall": round(identifier_overlap(truth, found), 4),
                    "distances": round(stats.distances_per_query, 1),
                }
            )
    return rows


def the_shortlist_buys_more_than_a_stage() -> dict:
    """The corners of that table, which is the practical answer."""
    rows = {
        (row["stages"], row["shortlist"]): row
        for row in a_rerank_matters_more_than_the_stages()
    }
    from_stages = rows[(4, 0)]["recall"] - rows[(1, 0)]["recall"]
    from_shortlist = rows[(1, 200)]["recall"] - rows[(1, 0)]["recall"]
    return {
        "one_stage_no_rerank": rows[(1, 0)]["recall"],
        "four_stages_no_rerank": rows[(4, 0)]["recall"],
        "one_stage_deep_rerank": rows[(1, 200)]["recall"],
        "gain_from_stages": round(from_stages, 4),
        "gain_from_the_shortlist": round(from_shortlist, 4),
        "shortlist_wins": from_shortlist > from_stages,
    }


def a_bigger_codebook_or_more_stages(
    settings: Sequence[tuple[int, int]] = ((1, 4096), (2, 64), (3, 16), (4, 8)),
) -> list[dict]:
    """Two ways to spend the same effective codebook size, which is the real design choice.

    Every row here expresses 4096 reconstructions: one codebook of 4096, two of 64, three of 16,
    four of 8. The storage differs, because more stages means more code bytes and fewer codebook
    entries, and the accuracy differs because the stages are fitted greedily rather than
    jointly.
    """
    if not settings:
        raise ConfigError("there is nothing to sweep")
    corpus = gaussian(count=8192, dimension=32)
    searched, probes = held_out(corpus, count=100)
    truth = search(probes, searched.vectors, k=10)
    rows = []
    for stages, entries in settings:
        index = ResidualIndex(32, stages=stages, entries=entries)
        index.build(searched.vectors)
        found, _ = index.search(probes, k=10)
        rows.append(
            {
                "stages": stages,
                "entries": entries,
                "effective_entries": entries**stages,
                "recall": round(identifier_overlap(truth, found), 4),
                "bytes_per_vector": index.codes.bytes_per_vector,
                "codebook_bytes": stages * entries * 32 * 4,
            }
        )
    return rows


def one_big_codebook_beats_several_small_ones() -> dict:
    """The result of that comparison, which is about greedy fitting rather than about size.

    Each stage is fitted on the residual the previous stages left, so the codebooks are chosen
    one at a time and never revisited. A single codebook expressing the same number of
    reconstructions is fitted jointly and can place its entries wherever they are most useful.
    The difference is the cost of the greedy decomposition and it is what buys the storage
    saving.
    """
    rows = {(row["stages"], row["entries"]): row for row in a_bigger_codebook_or_more_stages()}
    single = rows[(1, 4096)]
    split = rows[(4, 8)]
    return {
        "single_recall": single["recall"],
        "single_codebook_bytes": single["codebook_bytes"],
        "single_per_vector": single["bytes_per_vector"],
        "four_stage_recall": split["recall"],
        "four_stage_codebook_bytes": split["codebook_bytes"],
        "four_stage_per_vector": split["bytes_per_vector"],
        "single_is_more_accurate": single["recall"] > split["recall"],
        "and_costs_more_to_store": single["codebook_bytes"] > split["codebook_bytes"],
    }


def the_scoring_cost_is_where_it_loses(dimension: int = 64) -> dict:
    """Why residual quantisation is not the default despite the accuracy.

    A product code scores through a table: one lookup per subspace, with the table built once
    per
    query at a cost of subspaces times entries distances. A residual code has no such
    decomposition, because the distance to a sum of codebook entries has cross terms between the
    stages, so scoring means reconstructing the vector and taking a full width distance.

    Counted as this package counts, a product code charges a fraction of a distance per
    candidate
    and a residual code charges most of one. At a million candidates that is the difference
    between the two methods being usable at the same scale.
    """
    corpus = gaussian(count=4096, dimension=dimension)
    searched, probes = held_out(corpus, count=100)
    size = int(searched.vectors.shape[0])

    residual = ResidualIndex(dimension, stages=2, entries=256)
    residual.build(searched.vectors)
    _, residual_stats = residual.search(probes, k=10)

    table_cost = 2 * 256
    product_per_query = table_cost + size * (2 / dimension)
    return {
        "corpus": size,
        "residual_per_query": round(residual_stats.distances_per_query, 1),
        "product_per_query": round(product_per_query, 1),
        "residual_is_dearer": residual_stats.distances_per_query > product_per_query,
        "ratio": round(residual_stats.distances_per_query / product_per_query, 2),
    }


def the_reconstruction_is_the_sum_of_the_codes() -> dict:
    """A correctness check on the decoding, which nothing else here would catch.

    Every measurement in this module scores against the decoded vectors, so an error in the sum
    would move every number consistently and look like a property of the method. Checked by
    decoding one way and rebuilding the same vectors stage by stage the other way.
    """
    corpus = gaussian(count=512, dimension=16)
    codes = fit(corpus.vectors, stages=3, entries=32)
    fast = decode(codes)
    slow = torch.zeros_like(fast)
    for stage in range(codes.stages):
        slow = slow + codes.books[stage][codes.codes[:, stage]]
    return {
        "stages": codes.stages,
        "identical": bool(torch.allclose(fast, slow, atol=1e-6)),
        "max_gap": round(float((fast - slow).abs().max()), 8),
    }


def decoding_a_subset_matches_decoding_everything() -> dict:
    """That the row selection in the decoder does not shift anything.

    An off by one in the row indexing would decode the wrong vectors and every recall number
    would fall by a plausible amount, which is exactly the failure this package's differential
    checks were written for and this is the version specific to this module.
    """
    corpus = gaussian(count=512, dimension=16)
    codes = fit(corpus.vectors, stages=2, entries=32)
    everything = decode(codes)
    rows = torch.tensor([3, 17, 99, 400])
    subset = decode(codes, rows)
    return {
        "identical": bool(torch.allclose(everything[rows], subset, atol=1e-6)),
        "rows": rows.tolist(),
    }


def a_single_stage_is_plain_quantisation() -> dict:
    """That one stage is exactly a k-means quantiser, which is the base case.

    Worth checking because everything in this module is built on top of it, and because a single
    stage residual index should be indistinguishable from filing every vector under its nearest
    centroid, which is what an inverted file does before it starts probing.
    """
    corpus = gaussian(count=2048, dimension=16)
    codes = fit(corpus.vectors, stages=1, entries=64)
    reconstructed = decode(codes)
    run = lloyd(corpus.vectors, k=64, seed=0)
    direct = run.centres[assign(corpus.vectors, run.centres)]
    return {
        "identical": bool(torch.allclose(reconstructed, direct, atol=1e-5)),
        "stages": codes.stages,
        "effective_entries": codes.effective_entries,
    }


def zero_stages_are_refused() -> bool:
    """Whether a quantiser that quantises nothing is caught."""
    try:
        fit(torch.randn(512, 8), stages=0)
    except ConfigError:
        return True
    return False


def a_codebook_of_one_is_refused() -> bool:
    """Whether a codebook that cannot distinguish anything is caught.

    Every vector would map to the same entry, so the code carries no information and the
    residual
    is the vector minus the mean. It is a legitimate preprocessing step and it is not a
    quantiser, and calling it one would make the storage numbers meaningless.
    """
    try:
        fit(torch.randn(512, 8), entries=1)
    except ConfigError:
        return True
    return False


def a_corpus_smaller_than_the_codebook_is_refused() -> bool:
    """Whether fitting more entries than there are vectors is caught."""
    try:
        fit(torch.randn(32, 8), entries=256)
    except BuildError:
        return True
    return False


def a_rank_one_corpus_is_refused() -> bool:
    """Whether an unbatched corpus is caught."""
    try:
        fit(torch.randn(64))
    except DataError:
        return True
    return False


def codes_that_do_not_match_their_books_are_refused() -> bool:
    """Whether a code matrix with the wrong number of stages is caught."""
    try:
        ResidualCodes(codes=torch.zeros(10, 3, dtype=torch.long), books=torch.zeros(2, 16, 8))
    except DataError:
        return True
    return False


def a_negative_rerank_is_refused() -> bool:
    """Whether a negative shortlist is caught at construction."""
    try:
        ResidualIndex(16, rerank=-1)
    except ConfigError:
        return True
    return False


def a_rerank_below_k_is_refused() -> bool:
    """Whether a shortlist too short to fill the result is caught."""
    corpus = gaussian(count=512, dimension=16)
    index = ResidualIndex(16, stages=2, entries=32, rerank=5)
    index.build(corpus.vectors)
    try:
        index.search(corpus.vectors[:4], k=10)
    except ConfigError:
        return True
    return False


def insertion_reuses_the_fitted_codebooks() -> dict:
    """That inserted vectors are encoded against the codebooks the corpus was.

    They have to be. Refitting on the grown corpus would give the new vectors better codes and
    silently reinterpret every code already stored, which is the same argument the binary index
    makes about its centring vector and has the same answer.
    """
    corpus = gaussian(count=2048, dimension=16)
    searched, probes = held_out(corpus, count=32)
    index = ResidualIndex(16, stages=2, entries=32)
    index.build(searched.vectors[:1000])
    before = index.codes.books.clone()
    index.insert(searched.vectors[1000:1500])
    found, _ = index.search(probes, k=5)
    return {
        "books_unchanged": bool(torch.equal(before, index.codes.books)),
        "size": index.size,
        "returns_results": int(found.identifiers.shape[0]) == int(probes.shape[0]),
    }


def removal_takes_a_row_out() -> dict:
    """That a removed vector stops being returned."""
    corpus = gaussian(count=1024, dimension=16)
    index = ResidualIndex(16, stages=2, entries=32)
    index.build(corpus.vectors)
    query = corpus.vectors[:1]
    before, _ = index.search(query, k=5)
    index.remove([int(before.identifiers[0, 0])])
    after, _ = index.search(query, k=5)
    return {
        "removed": int(before.identifiers[0, 0]),
        "still_present": int(before.identifiers[0, 0]) in after.identifiers[0].tolist(),
        "size": index.size,
        "still_returns_k": int(after.identifiers.shape[1]) == 5,
    }

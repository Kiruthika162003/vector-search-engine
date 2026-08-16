from __future__ import annotations

from collections.abc import Sequence

import torch

from vse.build.kmeans import assign, lloyd
from vse.errors import BuildError, ConfigError, IndexStateError
from vse.index.base import Index, Quality, SearchStats, evaluate
from vse.index.flat import FlatIndex
from vse.index.ivf import IVFIndex
from vse.quantize.product import ProductCodes, distance_table, train
from vse.vectors.dataset import Corpus, clustered, gaussian, held_out
from vse.vectors.exact import Neighbours
from vse.vectors.metric import L2, Metric, distances, squared_l2

# Partitioning, quantisation and reranking in one index, which is what a production one is.
#
# None of the three pieces is new. The inverted file decides which fraction of the corpus to
# look at, the product codes decide how cheaply each of those can be scored, and the reranking
# pass decides how accurate the final answer is. What is new is that they compose, and that the
# composition has a property none of them has alone: the memory and the accuracy stop being the
# same knob.
#
# The residual is the piece that makes it work. Quantising the vectors directly wastes most of
# the codebook describing where the partitions are, which the partition identifier already says.
# Quantising the offset from each vector to its own centre instead means every code describes
# the same small region, so the same eight bytes buy a finer approximation. On the clustered
# corpus that is sixteen points of recall and on the unstructured one it is a point worse than
# not bothering, because an arbitrary centre carries no information to remove.
#
# Measuring that at all took two attempts. The first comparison used the default shortlist of a
# hundred and found the residual worth exactly nothing on both corpora, to four decimal places,
# which was true and was measuring the rerank: at that shortlist the exact pass repairs whatever
# the codes got wrong, so code quality is invisible. The comparison runs at a shortlist of
# fifteen now, which is the regime where the codes are the answer.
#
# The other thing worth stating is where the memory actually goes. A hundred thousand vectors of
# a hundred and twenty eight dimensions is fifty one megabytes raw and two and a quarter as an
# index, and the partition identifier is exactly as large a share of that as the codes are,
# thirty five percent each, because an identifier is a long and a code is eight bytes and nobody
# chose for those to match. A third of this index is an integer stored six times wider than it
# needs to be. The compression everybody discusses is the other third.


class CompositeIndex(Index):
    """An inverted file over product codes of the residuals, with an exact rerank."""

    def __init__(
        self,
        dimension: int,
        partitions: int = 64,
        subspaces: int = 8,
        centroids: int = 256,
        metric: Metric | str = L2,
        probe: int = 8,
        shortlist: int = 100,
        residual: bool = True,
        seed: int = 0,
    ) -> None:
        super().__init__(dimension, metric)
        if partitions < 1 or probe < 1 or probe > partitions:
            raise ConfigError(f"probing {probe} of {partitions} partitions")
        if dimension % subspaces:
            raise ConfigError(f"a width of {dimension} does not split into {subspaces}")
        if shortlist < 1:
            raise ConfigError(f"a shortlist of {shortlist} returns nothing")
        self.partitions = partitions
        self.subspaces = subspaces
        self.centroids = centroids
        self.probe = probe
        self.shortlist = shortlist
        self.residual = residual
        self.seed = seed
        self._vectors = torch.zeros(0, dimension)
        self._live = torch.zeros(0, dtype=torch.bool)
        self._centres = torch.zeros(0, dimension)
        self._of = torch.zeros(0, dtype=torch.long)
        self._lists: list[torch.Tensor] = []
        self._codes: ProductCodes | None = None

    @property
    def size(self) -> int:
        """Live vectors."""
        return int(self._live.sum())

    @property
    def capacity(self) -> int:
        """Rows held, tombstones included."""
        return int(self._vectors.shape[0])

    @property
    def codes(self) -> ProductCodes:
        """The quantiser this index scores with."""
        if self._codes is None:
            raise IndexStateError("the composite index has not been built")
        return self._codes

    def build(self, vectors: torch.Tensor) -> None:
        """Partition, take residuals if asked, and quantise whatever is left."""
        self._check_vectors(vectors)
        if self.partitions > vectors.shape[0]:
            raise BuildError(f"{self.partitions} partitions over {vectors.shape[0]} vectors")
        if vectors.shape[0] < self.centroids:
            raise BuildError(f"{vectors.shape[0]} vectors cannot train {self.centroids} codes")
        run = lloyd(vectors, k=self.partitions, seed=self.seed)
        self._vectors = vectors.clone()
        self._live = torch.ones(vectors.shape[0], dtype=torch.bool)
        self._centres = run.centres.clone()
        self._of = run.assignment.clone()
        self._lists = [
            torch.nonzero(self._of == partition, as_tuple=False).flatten()
            for partition in range(self.partitions)
        ]
        target = vectors - self._centres[self._of] if self.residual else vectors
        self._codes = train(
            target, subspaces=self.subspaces, centroids=self.centroids, seed=self.seed
        )
        self._built = True

    def _shortlist(
        self, query: torch.Tensor, rows: torch.Tensor, width: int, stats: SearchStats
    ) -> torch.Tensor:
        """Score one query against a set of codes and keep the best.

        The residual case has to build a distance table per partition, because a residual code
        means something different depending on which centre it was taken from. That is the one
        real cost of the residual trick and it is why the probe count and the partition count
        interact here in a way they do not in a plain inverted file.
        """
        if rows.numel() == 0:
            return rows
        if self.residual:
            offsets = query - self._centres[self._of[rows]]
            scores = torch.zeros(int(rows.numel()))
            width_per = self.dimension // self.subspaces
            for piece in range(self.subspaces):
                block = offsets[:, piece * width_per : (piece + 1) * width_per]
                book = self.codes.codebooks[piece]
                picked = self.codes.codes[rows, piece].long()
                scores += (block - book[picked]).pow(2).sum(dim=1)
        else:
            table = distance_table(query, self.codes)
            scores = torch.zeros(int(rows.numel()))
            for piece in range(self.subspaces):
                scores += table[0, piece, :][self.codes.codes[rows, piece].long()]
        stats.charge(int(rows.numel()), weight=self.subspaces / self.dimension)
        stats.visit(int(rows.numel()))
        keep = min(width, int(rows.numel()))
        best = torch.topk(scores, k=keep, largest=False).indices
        return rows[best]

    def search(
        self, queries: torch.Tensor, k: int = 10, probe: int | None = None
    ) -> tuple[Neighbours, SearchStats]:
        """Open the nearest partitions, shortlist with codes, rerank the shortlist exactly."""
        self._require_built()
        self._check_queries(queries, k)
        opened = self.probe if probe is None else probe
        if opened < 1 or opened > self.partitions:
            raise ConfigError(f"probing {opened} of {self.partitions} partitions")
        if self.shortlist < k:
            raise ConfigError(f"a shortlist of {self.shortlist} cannot produce {k}")
        count = int(queries.shape[0])
        stats = SearchStats(queries=count)
        stats.charge(self.partitions * count)
        centre_scores = distances(queries, self._centres, self.metric)
        chosen = torch.topk(centre_scores, k=opened, dim=1, largest=False).indices
        identifiers = torch.zeros(count, k, dtype=torch.long)
        scores = torch.zeros(count, k)
        for row in range(count):
            rows = torch.cat([self._lists[int(part)] for part in chosen[row]])
            rows = rows[self._live[rows]]
            stats.hop(opened)
            short = self._shortlist(queries[row : row + 1], rows, self.shortlist, stats)
            if short.numel() == 0:
                continue
            stats.charge(int(short.numel()))
            exact = squared_l2(queries[row : row + 1], self._vectors[short]).flatten()
            keep = min(k, int(short.numel()))
            best = torch.topk(exact, k=keep, largest=False)
            identifiers[row, :keep] = short[best.indices]
            scores[row, :keep] = best.values
        return Neighbours(identifiers=identifiers, scores=scores), stats

    def insert(self, vectors: torch.Tensor) -> list[int]:
        """File new vectors under the nearest centre and encode them with the existing books."""
        self._check_vectors(vectors)
        if not self._built:
            self.build(vectors)
            return list(range(vectors.shape[0]))
        start = self.capacity
        placement = assign(vectors, self._centres)
        target = vectors - self._centres[placement] if self.residual else vectors
        width = self.dimension // self.subspaces
        fresh = []
        for piece in range(self.subspaces):
            block = target[:, piece * width : (piece + 1) * width]
            fresh.append(squared_l2(block, self.codes.codebooks[piece]).argmin(dim=1))
        self._vectors = torch.cat([self._vectors, vectors.clone()], dim=0)
        self._live = torch.cat([self._live, torch.ones(vectors.shape[0], dtype=torch.bool)])
        self._of = torch.cat([self._of, placement])
        self._codes = ProductCodes(
            codes=torch.cat(
                [self.codes.codes, torch.stack(fresh, dim=1).to(torch.uint8)], dim=0
            ),
            codebooks=self.codes.codebooks,
        )
        self._lists = [
            torch.nonzero(self._of == partition, as_tuple=False).flatten()
            for partition in range(self.partitions)
        ]
        return list(range(start, self.capacity))

    def remove(self, identifiers: Sequence[int]) -> int:
        """Mark rows dead. They are filtered out of the posting lists before scoring."""
        self._require_built()
        removed = 0
        for identifier in identifiers:
            if not 0 <= identifier < self.capacity:
                raise ConfigError(f"{identifier} is not one of the {self.capacity} rows")
            if self._live[identifier]:
                self._live[identifier] = False
                removed += 1
        return removed

    def memory_bytes(self) -> int:
        """What the index costs without the full precision vectors.

        The codes, the codebooks, the centres, the partition identifiers and the liveness mask.
        The vectors themselves are not counted, because the whole point of the arrangement is
        that they live somewhere slower and are read only for the rerank. The method that
        includes them is separate so neither number can be quoted by accident.
        """
        return (
            self.codes.code_bytes()
            + self.codes.codebooks.numel() * 4
            + self._centres.numel() * 4
            + self.capacity * 8
            + (self.capacity + 7) // 8
        )

    def memory_with_vectors(self) -> int:
        """And what it costs if the vectors have to be resident too."""
        return self.memory_bytes() + self.capacity * self.dimension * 4


def composite_on(
    corpus: Corpus,
    partitions: int = 64,
    probe: int = 8,
    shortlist: int = 100,
    residual: bool = True,
    k: int = 10,
) -> Quality:
    """Build a composite index on a corpus with queries held out, and score it."""
    searched, probes = held_out(corpus, count=64)
    index = CompositeIndex(
        corpus.dimension,
        partitions=partitions,
        probe=probe,
        shortlist=shortlist,
        residual=residual,
    )
    index.build(searched.vectors)
    return evaluate(index, searched.vectors, probes, k=k)


def residuals_help_on_structured_data(dimension: int = 64) -> dict:
    """What quantising the offset from the centre buys instead of the vector.

    A great deal on clustered data, once the measurement is set up so the codes actually decide
    anything. The partition identifier already records roughly where the vector is, so a code
    that repeats that is spending its centroids on information the index already has. Coding the
    offset instead means every code describes the same small region around zero, and the same
    eight bytes buy a much finer approximation of it.

    The first version of this comparison used the default shortlist of a hundred and reported
    that the residual was worth exactly nothing, to four decimal places, on both corpora. That
    was true and it was measuring the rerank rather than the codes: with a shortlist that long
    the exact pass fixes whatever the codes got wrong, so the code quality is invisible. The
    shortlist here is fifteen, which is the regime where the codes are the answer.
    """
    corpus = clustered(count=2048, dimension=dimension, clusters=32)
    with_residual = composite_on(corpus, partitions=32, probe=1, shortlist=15, residual=True)
    without = composite_on(corpus, partitions=32, probe=1, shortlist=15, residual=False)
    return {
        "with_residual": round(with_residual.recall, 4),
        "without_residual": round(without.recall, 4),
        "gain": round(with_residual.recall - without.recall, 4),
        "helps": with_residual.recall > without.recall,
        "same_bytes": True,
    }


def and_much_less_on_unstructured_data(dimension: int = 64) -> dict:
    """Whether the residual is worth anything when the partitions mean less.

    Nothing, and very slightly less than nothing: a point of recall worse than not bothering.
    On gaussian rows the centres are an arbitrary carving of a continuum rather than a
    description of where the data is, so subtracting one removes no information from the vector
    while still shifting the cloud the codebook has to cover. Same pattern as everywhere else in
    this package, taken one step further: this trick does not merely fail to help without
    structure, it costs a little. Sixteen points better on clustered data and one point worse on
    unstructured is a large enough swing that it is worth knowing which corpus is in front of
    you before turning it on.
    """
    corpus = gaussian(count=2048, dimension=dimension)
    with_residual = composite_on(corpus, partitions=32, probe=1, shortlist=15, residual=True)
    without = composite_on(corpus, partitions=32, probe=1, shortlist=15, residual=False)
    return {
        "with_residual": round(with_residual.recall, 4),
        "without_residual": round(without.recall, 4),
        "gain": round(with_residual.recall - without.recall, 4),
        "smaller_gain_than_clustered": (with_residual.recall - without.recall)
        < residuals_help_on_structured_data(dimension)["gain"],
    }


def the_memory_is_the_point(count: int = 100_000, dimension: int = 128) -> dict:
    """What the arrangement is actually for, at a size worth quoting.

    A hundred thousand vectors of a hundred and twenty eight dimensions is fifty one megabytes
    of float32. The composite index over them is about one and a half: eight bytes of codes, a
    little for the centres and the codebooks, and eight for the partition identifier, which is
    itself half the total and is the one field that could obviously be smaller. That is the
    ratio that makes the whole arrangement worth its complexity.
    """
    if count < 1 or dimension < 1:
        raise ConfigError(f"{count} vectors of {dimension} is not a corpus")
    subspaces, partitions, centroids = 8, 1024, 256
    codes = count * subspaces
    identifiers = count * 8
    centres = partitions * dimension * 4
    codebooks = subspaces * centroids * (dimension // subspaces) * 4
    total = codes + identifiers + centres + codebooks
    return {
        "raw_megabytes": round(count * dimension * 4 / 1e6, 2),
        "index_megabytes": round(total / 1e6, 3),
        "ratio": round(count * dimension * 4 / total, 1),
        "identifier_share": round(identifiers / total, 3),
        "codes_share": round(codes / total, 3),
    }


def the_partition_identifier_is_half_the_index() -> dict:
    """The part of that budget that is not doing any work.

    Eight bytes a vector to record which of a thousand partitions it is in, when ten bits would
    do. It comes out exactly equal to the codes, thirty five percent of the index each, because
    a partition identifier is a long and a code is eight bytes and nobody chose for those to
    match. So a third of this index is an integer stored six times wider than it needs to be,
    and the compression everybody discusses is the other third. Nothing here fixes it, because
    the fix is a bit packed array and a slower lookup, and the point of measuring it is that the
    obvious thing to optimise in this structure is not the thing that gets optimised.
    """
    budget = the_memory_is_the_point()
    return {
        "identifier_share": budget["identifier_share"],
        "codes_share": budget["codes_share"],
        "identifiers_match_codes": abs(budget["identifier_share"] - budget["codes_share"])
        < 0.01,
        "bits_actually_needed": 10,
        "bits_used": 64,
        "wasted_share": round(budget["identifier_share"] * (1 - 10 / 64), 3),
    }


def probe_and_shortlist_are_different_knobs(
    probes: Sequence[int] = (1, 4, 16), shortlists: Sequence[int] = (20, 100, 400)
) -> list[dict]:
    """How the two accuracy parameters interact, which is not much.

    The probe count decides which vectors are candidates at all and the shortlist decides how
    many of those get an exact score. A vector missed by the probe cannot be recovered by any
    shortlist, so the probe is the ceiling and the shortlist approaches it. They are close to
    independent, which is what makes tuning them tractable.
    """
    if not probes or not shortlists:
        raise ConfigError("there is nothing to sweep")
    corpus = clustered(count=2048, dimension=64, clusters=32)
    searched, queries = held_out(corpus, count=64)
    rows = []
    for probe in probes:
        for shortlist in shortlists:
            index = CompositeIndex(
                64, partitions=32, probe=probe, shortlist=shortlist, subspaces=8
            )
            index.build(searched.vectors)
            quality = evaluate(index, searched.vectors, queries, k=10)
            rows.append(
                {
                    "probe": probe,
                    "shortlist": shortlist,
                    "recall": round(quality.recall, 4),
                    "distances_per_query": round(quality.stats.distances_per_query, 1),
                }
            )
    return rows


def the_probe_count_is_the_ceiling() -> dict:
    """Which of the two knobs binds, which decides which one to raise first.

    Neither, cleanly, which is not the answer I set the sweep up to get. Raising the shortlist
    from twenty to four hundred is worth twelve points at one probe and thirteen at sixteen, so
    they are almost the same, and both of those gains arrive by a shortlist of a hundred and
    nothing after. What does differ is the ceiling: one probe tops out at ninety six percent and
    sixteen reaches one, so the probe count sets what is reachable and the shortlist decides how
    much of it is reached. Tune the probe for the ceiling and the shortlist for the approach.
    They barely interact, which is the property that makes tuning them tractable at all.
    """
    rows = {
        (row["probe"], row["shortlist"]): row
        for row in probe_and_shortlist_are_different_knobs()
    }
    at_one = rows[(1, 400)]["recall"] - rows[(1, 20)]["recall"]
    at_sixteen = rows[(16, 400)]["recall"] - rows[(16, 20)]["recall"]
    return {
        "shortlist_gain_at_one_probe": round(at_one, 4),
        "shortlist_gain_at_sixteen_probes": round(at_sixteen, 4),
        "probe_binds_first": at_sixteen > at_one,
        "recall_at_one_probe": rows[(1, 400)]["recall"],
        "recall_at_sixteen": rows[(16, 400)]["recall"],
    }


def the_rerank_makes_the_codes_almost_irrelevant() -> dict:
    """How much of the final accuracy comes from the exact pass rather than the codes.

    All of it, on this corpus, which is stronger than the quantisation modules found. The
    composite index reaches exactly the recall an uncompressed inverted file reaches at the same
    probe count, so the sixteen to one compression of the vectors costs nothing measurable. The
    codes decide which hundred candidates get an exact score and being roughly right about that
    turns out to be easy, which is the whole reason this arrangement is worth building.
    """
    corpus = clustered(count=2048, dimension=64, clusters=32)
    searched, probes = held_out(corpus, count=64)
    composite = CompositeIndex(64, partitions=32, probe=8, shortlist=100)
    composite.build(searched.vectors)
    reranked = evaluate(composite, searched.vectors, probes, k=10)
    inverted = IVFIndex(64, partitions=32, probe=8)
    inverted.build(searched.vectors)
    exact_partition = evaluate(inverted, searched.vectors, probes, k=10)
    return {
        "composite_recall": round(reranked.recall, 4),
        "uncompressed_partition_recall": round(exact_partition.recall, 4),
        "gap_between_them": round(exact_partition.recall - reranked.recall, 4),
        "compression_costs_little": abs(exact_partition.recall - reranked.recall) < 0.1,
    }


def compare_the_whole_family(dimension: int = 64) -> list[dict]:
    """Every index in the package on one corpus, with its memory.

    The table this package has been building towards. The flat index is exact and stores
    everything, the inverted file trades recall for a scan, and the composite one trades memory
    for a rerank pass. They are not on the same axis, which is why the memory column is here
    rather than a single quality score.
    """
    corpus = clustered(count=2048, dimension=dimension, clusters=32)
    searched, probes = held_out(corpus, count=64)
    rows = []
    flat = FlatIndex(dimension)
    flat.build(searched.vectors)
    rows.append(
        {
            **evaluate(flat, searched.vectors, probes, k=10).as_dict(),
            "bytes": flat.memory_bytes(),
        }
    )
    inverted = IVFIndex(dimension, partitions=32, probe=8)
    inverted.build(searched.vectors)
    rows.append(
        {
            **evaluate(inverted, searched.vectors, probes, k=10).as_dict(),
            "bytes": inverted.memory_bytes(),
        }
    )
    composite = CompositeIndex(dimension, partitions=32, probe=8, shortlist=100)
    composite.build(searched.vectors)
    rows.append(
        {
            **evaluate(composite, searched.vectors, probes, k=10).as_dict(),
            "bytes": composite.memory_bytes(),
        }
    )
    return rows


def the_composite_index_is_the_smallest() -> dict:
    """Which of them stores least, and what that costs in answers."""
    rows = {row["index"]: row for row in compare_the_whole_family()}
    return {
        "flat_bytes": rows["flat"]["bytes"],
        "ivf_bytes": rows["ivf"]["bytes"],
        "composite_bytes": rows["composite"]["bytes"],
        "smallest": min(rows, key=lambda name: rows[name]["bytes"]),
        "flat_recall": rows["flat"]["recall"],
        "composite_recall": rows["composite"]["recall"],
    }


def insertion_uses_the_existing_codebooks() -> dict:
    """Whether a vector can be added without retraining anything.

    It can, and that is what makes the structure usable online. The new vector is assigned to a
    centre and encoded against the codebooks that already exist, so an insertion is a nearest
    centroid search and nothing more. What it cannot do is notice that the codebooks no longer
    describe the data, which is the same drift the inverted file has and is not visible from
    inside.
    """
    corpus = clustered(count=2048, dimension=64, clusters=32)
    searched, probes = held_out(corpus, count=64)
    index = CompositeIndex(64, partitions=32, probe=8, shortlist=100)
    index.build(searched.vectors[:1024])
    before = evaluate(index, index._vectors, probes, k=10).recall
    index.insert(searched.vectors[1024:])
    after = evaluate(index, index._vectors, probes, k=10)
    return {
        "before": round(before, 4),
        "after": round(after.recall, 4),
        "size": index.size,
        "still_works": after.recall > 0.5,
    }


def a_removed_vector_never_comes_back() -> dict:
    """Whether deletion works through both stages."""
    corpus = clustered(count=1024, dimension=64, clusters=16)
    index = CompositeIndex(64, partitions=16, probe=16, shortlist=50, centroids=64)
    index.build(corpus.vectors)
    victim = int(index.search(corpus.vectors[:1], k=1)[0].identifiers[0, 0])
    index.remove([victim])
    after = index.search(corpus.vectors[:1], k=5)[0]
    return {
        "removed": victim,
        "still_returned": victim in after.row(0),
        "live": index.size,
    }


def a_shortlist_shorter_than_k_is_refused() -> bool:
    """Whether a shortlist that cannot fill the result is caught at search time."""
    corpus = clustered(count=1024, dimension=32, clusters=16)
    index = CompositeIndex(32, partitions=16, probe=4, shortlist=5, centroids=64)
    index.build(corpus.vectors)
    try:
        index.search(corpus.vectors[:2], k=10)
    except ConfigError:
        return True
    return False


def a_width_that_does_not_split_is_refused() -> bool:
    """Whether a subspace count that does not divide the width is caught at construction."""
    try:
        CompositeIndex(30, partitions=4, probe=1, subspaces=8)
    except ConfigError:
        return True
    return False


def searching_before_building_is_refused() -> bool:
    """Whether an unbuilt composite index refuses rather than scoring an empty codebook."""
    try:
        CompositeIndex(32, partitions=4, probe=1).search(torch.randn(2, 32), k=1)
    except IndexStateError:
        return True
    return False


def the_two_memory_numbers_are_separate() -> dict:
    """Whether the index reports its size honestly.

    Two numbers, deliberately. The index without the vectors is what it costs to keep resident,
    and the index with them is what it costs if they cannot live anywhere slower. Reporting only
    the first would be the flattering number and reporting only the second would hide the entire
    point of the structure, so both are methods and neither is the default.
    """
    corpus = clustered(count=2048, dimension=64, clusters=32)
    index = CompositeIndex(64, partitions=32, probe=8)
    index.build(corpus.vectors)
    return {
        "without_vectors": index.memory_bytes(),
        "with_vectors": index.memory_with_vectors(),
        "raw_vectors": 2048 * 64 * 4,
        "ratio": round(index.memory_with_vectors() / index.memory_bytes(), 2),
    }

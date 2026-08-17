from __future__ import annotations

from collections.abc import Sequence

import torch

from vse.errors import ConfigError, DataError, IndexStateError
from vse.index.base import Index, Quality, SearchStats, evaluate, evaluate_on
from vse.vectors.dataset import Corpus, gaussian, held_out
from vse.vectors.exact import Neighbours
from vse.vectors.metric import L2, Metric, distances

# The exact index, which is the baseline and is also a real answer for a great many corpora.
#
# It scores every query against every vector and takes the best. Recall is one by construction
# and the gap is zero, so the only interesting column is the cost, and the cost is the entire
# corpus per query. Everything else in this package exists to reduce that number and gives up
# some of the first two columns to do it.
#
# It is worth being clear about how far this goes before anything else is needed, because the
# answer surprised me. Four thousand vectors of thirty two dimensions is half a megabyte and a
# hundred and thirty thousand multiply accumulates per query, which is nothing. The crossover
# where a structure earns its complexity is not at a thousand vectors and it is not at ten
# thousand: a batched exact search over a hundred thousand vectors is a single matrix product
# that a laptop does in milliseconds. What actually forces an index is the product of corpus
# size and query rate, and a good deal of published work on this compares against an exact
# search that was written badly rather than against one written like this.
#
# Deletion here is a tombstone rather than a compaction, which is the same choice every other
# index in this package makes and is measured properly in the module on dynamic updates. The
# reason it is a tombstone even in the trivial case is that identifiers have to stay stable:
# compacting the array would renumber every vector after the hole, and an application holding
# an identifier from a previous query would silently get a different vector back.


class FlatIndex(Index):
    """Every vector, every query, no structure."""

    def __init__(self, dimension: int, metric: Metric | str = L2) -> None:
        super().__init__(dimension, metric)
        self._vectors = torch.zeros(0, dimension)
        self._live = torch.zeros(0, dtype=torch.bool)

    @property
    def size(self) -> int:
        """Live vectors, tombstones excluded."""
        return int(self._live.sum())

    @property
    def capacity(self) -> int:
        """Rows held, tombstones included. What it actually costs to scan."""
        return int(self._vectors.shape[0])

    @property
    def tombstones(self) -> int:
        """Rows that are still scanned and can never be returned."""
        return self.capacity - self.size

    def build(self, vectors: torch.Tensor) -> None:
        """Keep the vectors. There is nothing else to do."""
        self._check_vectors(vectors)
        self._vectors = vectors.clone()
        self._live = torch.ones(vectors.shape[0], dtype=torch.bool)
        self._built = True

    def search(self, queries: torch.Tensor, k: int = 10) -> tuple[Neighbours, SearchStats]:
        """Score everything and take the best k that are still live."""
        self._require_built()
        self._check_queries(queries, k)
        stats = SearchStats(queries=int(queries.shape[0]))
        stats.charge(self.capacity * int(queries.shape[0]))
        stats.visit(self.capacity * int(queries.shape[0]))
        scores = distances(queries, self._vectors, self.metric)
        if self.tombstones:
            limit = torch.finfo(scores.dtype).max
            blocked = limit if self.metric.smaller_is_closer else -limit
            scores = scores.masked_fill(~self._live.unsqueeze(0), blocked)
        found = torch.topk(scores, k=k, dim=1, largest=not self.metric.smaller_is_closer)
        return Neighbours(identifiers=found.indices, scores=found.values), stats

    def insert(self, vectors: torch.Tensor) -> list[int]:
        """Append rows. Identifiers are positions and never move."""
        self._check_vectors(vectors)
        if not self._built:
            self.build(vectors)
            return list(range(vectors.shape[0]))
        start = self.capacity
        self._vectors = torch.cat([self._vectors, vectors.clone()], dim=0)
        self._live = torch.cat([self._live, torch.ones(vectors.shape[0], dtype=torch.bool)])
        return list(range(start, self.capacity))

    def remove(self, identifiers: Sequence[int]) -> int:
        """Mark rows dead. They keep costing until a rebuild."""
        self._require_built()
        removed = 0
        for identifier in identifiers:
            if not 0 <= identifier < self.capacity:
                raise ConfigError(f"{identifier} is not one of the {self.capacity} rows")
            if self._live[identifier]:
                self._live[identifier] = False
                removed += 1
        return removed

    def compact(self) -> dict:
        """Drop the tombstones, renumbering everything after each hole.

        Not called automatically, and the reason is in the return value: it changes the
        identifier of every vector that follows a deleted one. Anything holding an identifier
        from an earlier query is now holding a reference to a different vector, which is a bug
        that produces plausible wrong answers rather than an error.
        """
        self._require_built()
        moved = int((~self._live).cumsum(dim=0)[self._live].sum())
        self._vectors = self._vectors[self._live].clone()
        reclaimed = int((~self._live).sum())
        self._live = torch.ones(self._vectors.shape[0], dtype=torch.bool)
        return {"reclaimed": reclaimed, "identifiers_changed": moved > 0, "shifted_by": moved}

    def memory_bytes(self) -> int:
        """Four bytes a coordinate plus one bit a row for the liveness mask."""
        return self.capacity * self.dimension * 4 + (self.capacity + 7) // 8

    def vector(self, identifier: int) -> torch.Tensor:
        """One stored vector, by identifier."""
        self._require_built()
        if not 0 <= identifier < self.capacity:
            raise ConfigError(f"{identifier} is not one of the {self.capacity} rows")
        if not bool(self._live[identifier]):
            raise IndexStateError(f"vector {identifier} was removed")
        return self._vectors[identifier]


def flat_on(corpus: Corpus, k: int = 10, queries: int = 64) -> Quality:
    """Build a flat index on a corpus and score it. The baseline row of every table."""
    return evaluate_on(FlatIndex(corpus.dimension), corpus, k=k, queries=queries)


def the_baseline_is_exact() -> dict:
    """Whether the flat index agrees with exact search, which it had better.

    It does, on both counts, and the check is not a formality. It is the only thing standing
    between a sign error in the metric dispatch and every recall number in this package being
    measured against the wrong answer.
    """
    quality = flat_on(gaussian(count=2048, dimension=32))
    return {
        "recall": quality.recall,
        "gap": quality.gap,
        "exact": quality.recall == 1.0 and quality.gap == 0.0,
    }


def it_costs_the_whole_corpus() -> dict:
    """And what that costs, which is the number everything else is trying to beat."""
    corpus = gaussian(count=2048, dimension=32)
    quality = flat_on(corpus)
    return {
        "corpus": quality.corpus_size,
        "distances_per_query": quality.stats.distances_per_query,
        "scanned": quality.scanned,
        "speedup": quality.speedup,
    }


def the_cost_does_not_depend_on_k(values: Sequence[int] = (1, 10, 100)) -> dict:
    """Whether asking for more neighbours costs more.

    Not measurably. The distances are the whole cost and there are the same number of them
    whatever k is, so a caller that wants a hundred candidates to filter down to ten should ask
    for a hundred rather than issue a second query.
    """
    if not values:
        raise ConfigError("there is nothing to sweep")
    corpus = gaussian(count=2048, dimension=32)
    counts = {k: flat_on(corpus, k=k).stats.distances_per_query for k in values}
    return {
        "counts": {str(k): value for k, value in counts.items()},
        "identical": len(set(counts.values())) == 1,
    }


def the_cost_grows_with_the_corpus(
    sizes: Sequence[int] = (1024, 2048, 4096, 8192),
) -> list[dict]:
    """How the exact search scales, which is linearly and exactly.

    Doubling the corpus doubles the distances, with no constant factor and no threshold. That is
    the shape every structure in this package is trying to bend, and it is worth having the
    straight line drawn before looking at anything that claims to beat it.
    """
    if not sizes:
        raise ConfigError("there is nothing to sweep")
    rows = []
    for size in sizes:
        quality = flat_on(gaussian(count=size, dimension=32))
        rows.append(
            {
                "corpus": size,
                "searched": quality.corpus_size,
                "distances_per_query": quality.stats.distances_per_query,
                "scanned": quality.scanned,
                "bytes": size * 32 * 4,
            }
        )
    return rows


def it_is_exactly_linear() -> dict:
    """The straight line, checked.

    The right statement is that the search scans the corpus exactly once, which holds to the
    last digit at every size. The ratio between consecutive rows comes out at about two and a
    fiftieth rather than two, and that is not a nonlinearity: sixty four queries are held out of
    each corpus, so the searched sizes are the nominal ones less a constant, and a constant
    offset shows up in a ratio and not in a slope.
    """
    rows = {row["corpus"]: row for row in the_cost_grows_with_the_corpus()}
    ratios = [
        rows[8192]["distances_per_query"] / rows[4096]["distances_per_query"],
        rows[4096]["distances_per_query"] / rows[2048]["distances_per_query"],
    ]
    return {
        "ratios": [round(ratio, 4) for ratio in ratios],
        "scans_the_corpus_once": all(row["scanned"] == 1.0 for row in rows.values()),
        "held_out_explains_the_offset": all(
            row["corpus"] - row["searched"] == 64 for row in rows.values()
        ),
        "close_to_two": all(abs(ratio - 2.0) < 0.05 for ratio in ratios),
    }


def a_hundred_thousand_vectors_is_not_a_problem(
    count: int = 100_000, dimension: int = 128, queries: int = 1
) -> dict:
    """What an exact search over a corpus people call large actually costs.

    Thirteen million multiply accumulates for one query, which is a fraction of what a single
    layer of the model that produced the embeddings already did. Stated in bytes it is fifty
    megabytes read once. The reason to build an index is the query rate multiplied by this, not
    this, and a comparison against a badly written exact search flatters every structure that
    follows.
    """
    if count < 1 or dimension < 1:
        raise ConfigError(f"{count} vectors of {dimension} is not a corpus")
    return {
        "vectors": count,
        "dimension": dimension,
        "multiply_accumulates": count * dimension * queries,
        "bytes_read": count * dimension * 4,
        "megabytes": round(count * dimension * 4 / 1e6, 1),
    }


def tombstones_still_cost() -> dict:
    """What a removal does to the cost of the searches that follow.

    Nothing good. The row stays in the array and stays in the matrix product, so removing half
    the corpus leaves the search exactly as expensive as it was and returns half as many
    candidates. Every structure here behaves this way and the module on dynamic updates measures
    what it does to recall.
    """
    corpus = gaussian(count=2048, dimension=32)
    searched, probes = held_out(corpus, count=64)
    index = FlatIndex(corpus.dimension)
    index.build(searched.vectors)
    before = evaluate(index, searched.vectors, probes, k=10)
    index.remove(range(0, index.capacity, 2))
    after_cost = index.search(probes, k=10)[1].distances_per_query
    return {
        "before": before.stats.distances_per_query,
        "after_removing_half": after_cost,
        "unchanged": abs(before.stats.distances_per_query - after_cost) < 1e-9,
        "live": index.size,
        "capacity": index.capacity,
    }


def a_removed_vector_never_comes_back() -> dict:
    """Whether the mask actually works.

    It does, and the way it works is worth noting. The scores of dead rows are set to the
    largest representable value rather than dropped, so the shape of the score matrix does not
    change and the selection does not have to know about deletion at all. The cost is that a
    corpus which is mostly tombstones is mostly computing distances to values that will be
    thrown away.
    """
    corpus = gaussian(count=512, dimension=16)
    index = FlatIndex(16)
    index.build(corpus.vectors)
    victim = int(index.search(corpus.vectors[:1], k=1)[0].identifiers[0, 0])
    index.remove([victim])
    after = index.search(corpus.vectors[:1], k=5)[0]
    return {
        "removed": victim,
        "still_returned": victim in after.row(0),
        "live": index.size,
        "tombstones": index.tombstones,
    }


def compaction_renumbers_everything() -> dict:
    """Why compaction is not automatic.

    Because it moves identifiers. Removing early rows and compacting shifts every later vector
    down, so an identifier held by a caller from a previous query now points at a different
    vector. There is no error and no warning, just a different answer, which is the reason the
    tombstone stays until somebody asks for it to go.
    """
    corpus = gaussian(count=256, dimension=8)
    index = FlatIndex(8)
    index.build(corpus.vectors)
    kept = corpus.vectors[100].clone()
    index.remove([0, 1, 2])
    result = index.compact()
    return {
        **result,
        "vector_at_the_old_position": bool(torch.equal(index.vector(100), kept)),
        "vector_at_the_new_position": bool(torch.equal(index.vector(97), kept)),
        "size": index.size,
    }


def compaction_reclaims_the_memory() -> dict:
    """And what it buys, which is the cost of the tombstones back.

    Both the memory and the distances. After compacting, the array holds only live rows, so the
    matrix product shrinks to match. That is the whole reason to ever do it, and it has to be
    weighed against renumbering.
    """
    corpus = gaussian(count=2048, dimension=32)
    index = FlatIndex(32)
    index.build(corpus.vectors)
    before_bytes = index.memory_bytes()
    index.remove(range(0, 2048, 2))
    during_bytes = index.memory_bytes()
    index.compact()
    return {
        "before": before_bytes,
        "after_removing_half": during_bytes,
        "after_compacting": index.memory_bytes(),
        "removal_freed_nothing": before_bytes == during_bytes,
        "compaction_halved_it": index.memory_bytes() < before_bytes * 0.55,
    }


def insertion_is_an_append() -> dict:
    """What it costs a flat index to accept new vectors.

    A concatenation and nothing else. This is the one thing a flat index is unambiguously best
    at: there is no structure to maintain, so an insertion cannot degrade the quality of
    anything, which is exactly the property the graph indexes lose.
    """
    corpus = gaussian(count=1024, dimension=16)
    index = FlatIndex(16)
    index.build(corpus.vectors[:512])
    before = index.size
    identifiers = index.insert(corpus.vectors[512:])
    quality = evaluate(index, corpus.vectors, corpus.vectors[:32], k=10)
    return {
        "before": before,
        "after": index.size,
        "first_new_identifier": identifiers[0],
        "recall_after_inserting": quality.recall,
        "still_exact": quality.recall == 1.0,
    }


def building_on_an_empty_index_by_inserting() -> dict:
    """Whether the first insertion into an unbuilt index works.

    It does, by building. An index that required a build before an insert would force every
    caller to special case the first batch, and there is nothing a build does here that the
    first insert cannot.
    """
    index = FlatIndex(8)
    identifiers = index.insert(gaussian(count=64, dimension=8).vectors)
    return {"built": index.built, "size": index.size, "identifiers": len(identifiers)}


def memory_is_the_vectors_plus_a_bit() -> dict:
    """What the structure costs over the raw data.

    One bit per vector, for the liveness mask. That is the honest floor: any index that stores
    less than the vectors cannot answer exactly, and every structure in this package that beats
    this number has given up exactness to do it.
    """
    index = FlatIndex(32)
    index.build(gaussian(count=4096, dimension=32).vectors)
    raw = 4096 * 32 * 4
    return {
        "total": index.memory_bytes(),
        "raw_vectors": raw,
        "overhead": index.memory_bytes() - raw,
        "overhead_share": round((index.memory_bytes() - raw) / raw, 6),
    }


def compare_dimensions(dimensions: Sequence[int] = (8, 32, 128, 512)) -> list[dict]:
    """The baseline across widths.

    The distance count does not change with dimension and the work per distance does, which is
    the one respect in which counting distances is a poor cost model. A five hundred dimensional
    distance is sixteen times the arithmetic of a thirty two dimensional one and this counter
    calls them the same. The bytes column is there so that is visible rather than hidden.
    """
    if not dimensions:
        raise ConfigError("there is nothing to sweep")
    rows = []
    for dimension in dimensions:
        corpus = gaussian(count=2048, dimension=dimension)
        quality = flat_on(corpus)
        rows.append(
            {
                "dimension": dimension,
                "recall": quality.recall,
                "distances_per_query": quality.stats.distances_per_query,
                "multiply_accumulates": int(quality.stats.distances_per_query) * dimension,
            }
        )
    return rows


def the_distance_count_hides_the_dimension() -> dict:
    """The one place the cost model is misleading, stated plainly."""
    rows = {row["dimension"]: row for row in compare_dimensions()}
    return {
        "distances_at_eight": rows[8]["distances_per_query"],
        "distances_at_five_hundred": rows[512]["distances_per_query"],
        "identical": rows[8]["distances_per_query"] == rows[512]["distances_per_query"],
        "arithmetic_ratio": (
            rows[512]["multiply_accumulates"] // rows[8]["multiply_accumulates"]
        ),
    }


def searching_before_building_is_refused() -> bool:
    """Whether an unbuilt index refuses rather than returning an empty result."""
    try:
        FlatIndex(8).search(torch.randn(2, 8), k=1)
    except IndexStateError:
        return True
    return False


def removing_a_row_that_does_not_exist_is_refused() -> bool:
    """Whether an identifier outside the index is caught."""
    index = FlatIndex(8)
    index.build(gaussian(count=32, dimension=8).vectors)
    try:
        index.remove([99])
    except ConfigError:
        return True
    return False


def removing_the_same_row_twice_counts_once() -> dict:
    """What a repeated removal reports.

    One removal, not two. The count is how many rows changed state, so a caller deleting a batch
    with duplicates in it gets a number it can trust rather than one that overstates.
    """
    index = FlatIndex(8)
    index.build(gaussian(count=32, dimension=8).vectors)
    return {
        "first": index.remove([3, 4]),
        "again": index.remove([3, 4]),
        "size": index.size,
    }


def a_query_of_the_wrong_width_is_refused() -> bool:
    """Whether a query that does not match the index width is caught."""
    index = FlatIndex(16)
    index.build(gaussian(count=64, dimension=16).vectors)
    try:
        index.search(torch.randn(2, 8), k=1)
    except DataError:
        return True
    return False

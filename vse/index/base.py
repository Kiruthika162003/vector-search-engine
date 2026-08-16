from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

from vse.errors import ConfigError, DataError, IndexStateError
from vse.vectors.dataset import Corpus, gaussian, held_out
from vse.vectors.exact import Neighbours, distances, identifier_overlap, score_gap, search
from vse.vectors.metric import L2, Metric, metric_named

# What every index in this package has to be, and what it is measured in.
#
# The interface is small on purpose: build from a corpus, search a batch of queries, insert,
# remove, and say how much memory it is using. Everything else is a property of a particular
# structure and lives with it.
#
# The unit of cost is one distance computation, counted rather than timed. Wall clock time on
# one laptop for one Python implementation would say more about torch dispatch overhead than
# about the structures, and it would not be reproducible between runs let alone between
# machines. Distance computations are exactly reproducible, they are what every published
# comparison of these structures is really about, and an exact search costs one per corpus
# vector per query, which makes the ratio against it the natural way to state a speedup.
#
# The counting is not free of judgement, and the place it bends is worth stating up front.
# Counting a distance against a full precision vector and a distance against a quantised code as
# one each is wrong: a code is several times cheaper to score. So the counter records both a
# count and a weight, and the quantised structures charge themselves a fraction. That is a model
# and the fraction is a choice, so it is stated where it is set rather than buried.
#
# Every index is measured on the same three numbers: recall against exact search, the score gap
# from vectors/exact.py because recall alone overstates a near miss, and distances per query.
# Anything that reports two of the three is hiding the third.


@dataclass
class SearchStats:
    """What a search cost, counted rather than timed."""

    distances: float = 0.0
    candidates: int = 0
    hops: int = 0
    queries: int = 0

    def charge(self, count: int, weight: float = 1.0) -> None:
        """Record distance computations, optionally at a discount."""
        if count < 0:
            raise ConfigError(f"cannot charge {count} distances")
        if weight < 0:
            raise ConfigError(f"a weight of {weight} is not a weight")
        self.distances += count * weight

    def visit(self, count: int = 1) -> None:
        """Record candidates pulled out of a list and looked at."""
        if count < 0:
            raise ConfigError(f"cannot visit {count} candidates")
        self.candidates += count

    def hop(self, count: int = 1) -> None:
        """Record steps taken through a graph or partitions opened."""
        if count < 0:
            raise ConfigError(f"cannot hop {count} times")
        self.hops += count

    @property
    def distances_per_query(self) -> float:
        """The number that matters, normalised by the batch size."""
        if self.queries == 0:
            return 0.0
        return self.distances / self.queries

    def merge(self, other: SearchStats) -> None:
        """Fold another search's counts into this one."""
        self.distances += other.distances
        self.candidates += other.candidates
        self.hops += other.hops
        self.queries += other.queries

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "queries": self.queries,
            "distances_per_query": round(self.distances_per_query, 2),
            "candidates": self.candidates,
            "hops": self.hops,
        }


class Index(ABC):
    """The interface every structure here implements.

    An index owns its vectors. It is handed a corpus once, it answers batches of queries, and it
    accepts insertions and removals. Nothing in the interface promises that the answers are
    exact, and nothing promises that insertion leaves the structure as good as a rebuild would,
    which is the subject of a later module rather than a gap here.
    """

    def __init__(self, dimension: int, metric: Metric | str = L2) -> None:
        if dimension < 1:
            raise ConfigError(f"a dimension of {dimension} is not a width")
        self.dimension = dimension
        self.metric = metric if isinstance(metric, Metric) else metric_named(metric)
        self._built = False

    @property
    def built(self) -> bool:
        """Whether it has been given anything to search."""
        return self._built

    @property
    @abstractmethod
    def size(self) -> int:
        """How many live vectors it holds."""

    @abstractmethod
    def build(self, vectors: torch.Tensor) -> None:
        """Take a corpus and prepare to answer queries."""

    @abstractmethod
    def search(self, queries: torch.Tensor, k: int = 10) -> tuple[Neighbours, SearchStats]:
        """Answer a batch, and say what it cost."""

    @abstractmethod
    def insert(self, vectors: torch.Tensor) -> list[int]:
        """Add vectors, returning their identifiers."""

    @abstractmethod
    def remove(self, identifiers: Sequence[int]) -> int:
        """Delete vectors, returning how many were actually removed."""

    @abstractmethod
    def memory_bytes(self) -> int:
        """What it costs to hold, vectors and structure together."""

    @property
    def name(self) -> str:
        """A short label for tables."""
        return type(self).__name__.replace("Index", "").lower()

    def _require_built(self) -> None:
        """Refuse to answer before there is anything to answer with."""
        if not self._built:
            raise IndexStateError(f"the {self.name} index has not been built")

    def _check_queries(self, queries: torch.Tensor, k: int) -> None:
        """Reject a batch this index cannot answer."""
        if queries.ndim != 2:
            raise DataError(f"queries are a matrix of rows, got rank {queries.ndim}")
        if queries.shape[1] != self.dimension:
            raise DataError(
                f"queries are {queries.shape[1]} wide and the index is {self.dimension}"
            )
        if k < 1:
            raise ConfigError(f"asking for {k} neighbours is not a query")
        if k > self.size:
            raise ConfigError(f"asking for {k} neighbours from {self.size} vectors")

    def _check_vectors(self, vectors: torch.Tensor) -> torch.Tensor:
        """Reject vectors this index cannot hold."""
        if vectors.ndim != 2:
            raise DataError(f"vectors are a matrix of rows, got rank {vectors.ndim}")
        if vectors.shape[0] == 0:
            raise DataError("there are no vectors here")
        if vectors.shape[1] != self.dimension:
            raise DataError(
                f"vectors are {vectors.shape[1]} wide and the index is {self.dimension}"
            )
        if not vectors.dtype.is_floating_point:
            raise DataError(f"vectors have to be floating point, got {vectors.dtype}")
        return vectors

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "index": self.name,
            "size": self.size if self._built else 0,
            "dimension": self.dimension,
            "metric": self.metric.name,
            "bytes": self.memory_bytes() if self._built else 0,
        }


@dataclass
class Quality:
    """How well one index answered one batch, and what it spent doing it."""

    index: str
    recall: float
    gap: float
    stats: SearchStats = field(default_factory=SearchStats)
    corpus_size: int = 0

    @property
    def speedup(self) -> float:
        """Distances an exact search would have done, over what this one did."""
        if self.stats.distances_per_query <= 0:
            return 0.0
        return self.corpus_size / self.stats.distances_per_query

    @property
    def scanned(self) -> float:
        """The share of the corpus it looked at."""
        if self.corpus_size == 0:
            return 0.0
        return self.stats.distances_per_query / self.corpus_size

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "index": self.index,
            "recall": round(self.recall, 4),
            "gap": round(self.gap, 6),
            "distances_per_query": round(self.stats.distances_per_query, 1),
            "scanned": round(self.scanned, 5),
            "speedup": round(self.speedup, 2),
        }


def top_up(
    found: Sequence[tuple[float, int]],
    k: int,
    query: torch.Tensor,
    corpus: torch.Tensor,
    live: torch.Tensor,
    metric: Metric | str = L2,
) -> list[tuple[float, int]]:
    """Fill a short result from rows the structure never reached.

    An approximate structure can come back with fewer than k candidates, which happens when a
    hash bucket is nearly empty or a walk is trapped in a small component. The contract says a
    search returns exactly k, so the shortfall has to be filled with something, and the obvious
    fillers are all wrong: repeating the last result breaks distinctness, leaving zeros claims
    identifier zero is a neighbour at distance zero, and returning fewer rows breaks the shape
    every caller is written against.

    So the fill comes from live rows the result does not already contain, scored honestly and
    sorted in with everything else. Those are real vectors with real distances, and if one of
    them happens to be a true neighbour then the structure got lucky rather than being credited
    with something it did not find. On a corpus of any size the chance of that is negligible and
    the alternative is a result that lies about its own contents.

    The differential sweep in verify/differential.py found three structures doing the zero fill
    before this existed, and it was invisible to every recall measurement in the package.
    """
    if k < 1:
        raise ConfigError(f"{k} is not a result width")
    kept = list(found)[:k]
    if len(kept) >= k:
        return kept
    taken = {identifier for _, identifier in kept}
    available = torch.nonzero(live, as_tuple=False).flatten()
    for candidate in available.tolist():
        if len(kept) >= k:
            break
        if candidate in taken:
            continue
        score = float(distances(query, corpus[candidate : candidate + 1], metric))
        kept.append((score, candidate))
        taken.add(candidate)
    if len(kept) < k:
        raise ConfigError(f"a corpus of {int(available.numel())} cannot supply {k} neighbours")
    kept.sort()
    return kept


def evaluate(
    index: Index,
    corpus: torch.Tensor,
    queries: torch.Tensor,
    k: int = 10,
    truth: Neighbours | None = None,
) -> Quality:
    """Run a batch through an index and score it against exact search.

    Three numbers and not one. Recall says how many of the right identifiers came back, the gap
    says how much worse the wrong ones were, and the distance count says what it cost. An index
    that improves any two of them at the expense of the third has not improved.
    """
    exact = truth if truth is not None else search(queries, corpus, k=k, metric=index.metric)
    found, stats = index.search(queries, k=k)
    return Quality(
        index=index.name,
        recall=identifier_overlap(exact, found),
        gap=score_gap(queries, corpus, exact, found, index.metric),
        stats=stats,
        corpus_size=int(corpus.shape[0]),
    )


def evaluate_result(
    index: Index,
    corpus: torch.Tensor,
    queries: torch.Tensor,
    found: Neighbours,
    stats: SearchStats,
    truth: Neighbours | None = None,
) -> Quality:
    """Score a result that has already been produced.

    The same three numbers as evaluate, for callers that ran the search themselves. A sweep over
    a search time parameter has one index and several results, and making it go back through
    evaluate would rebuild the index once per row for nothing.
    """
    exact = (
        truth if truth is not None else search(queries, corpus, k=found.k, metric=index.metric)
    )
    return Quality(
        index=index.name,
        recall=identifier_overlap(exact, found),
        gap=score_gap(queries, corpus, exact, found, index.metric),
        stats=stats,
        corpus_size=int(corpus.shape[0]),
    )


def evaluate_on(index: Index, corpus: Corpus, k: int = 10, queries: int = 64) -> Quality:
    """Build an index on a corpus with queries held out of it, and score it."""
    searched, probes = held_out(corpus, count=queries)
    index.build(searched.vectors)
    return evaluate(index, searched.vectors, probes, k=k)


def default_corpus(count: int = 4096, dimension: int = 32) -> Corpus:
    """The corpus the index comparisons run on unless they say otherwise."""
    return gaussian(count=count, dimension=dimension)


def an_unbuilt_index_refuses_to_search(index: Index) -> bool:
    """Whether searching before building raises rather than returning nothing.

    An empty result from an unbuilt index looks exactly like an index that found nothing, and
    the two have very different fixes.
    """
    try:
        index.search(torch.randn(2, index.dimension), k=1)
    except IndexStateError:
        return True
    return False


def stats_add_up() -> dict:
    """Whether merging two searches gives the same totals as one batch of both.

    It does, which is what makes it legitimate to accumulate counts across batches and report
    one number. The check exists because the per query normalisation divides by a count that
    also has to be merged, and forgetting that is the obvious way to get a plausible wrong
    answer.
    """
    first = SearchStats(distances=300.0, candidates=12, hops=3, queries=3)
    second = SearchStats(distances=700.0, candidates=8, hops=2, queries=7)
    merged = SearchStats()
    merged.merge(first)
    merged.merge(second)
    return {
        "distances": merged.distances,
        "queries": merged.queries,
        "per_query": merged.distances_per_query,
        "matches_the_mean": abs(merged.distances_per_query - 100.0) < 1e-9,
    }


def an_empty_stat_reports_nothing() -> dict:
    """What the per query number does before anything has been counted.

    Zero rather than an error from dividing by nothing. A stats object with no queries in it is
    a normal state, not a mistake, and it appears every time a table is assembled before the
    searches run.
    """
    empty = SearchStats()
    return {
        "per_query": empty.distances_per_query,
        "distances": empty.distances,
        "no_division_error": True,
    }


def a_discount_lowers_the_charge(weight: float = 0.25) -> dict:
    """What the weight on the counter is for.

    Scoring against a quantised code is cheaper than scoring against a full vector, so a
    structure that does the former charges itself a fraction. The fraction is a modelling
    choice: it is the ratio of the code width to the vector width, which ignores that the two
    have different memory access patterns, and it is stated here rather than hidden inside the
    quantised index.
    """
    if not 0 < weight <= 1:
        raise ConfigError(f"a discount of {weight} is not a discount")
    full = SearchStats(queries=1)
    full.charge(1000)
    cheap = SearchStats(queries=1)
    cheap.charge(1000, weight=weight)
    return {
        "full_price": full.distances_per_query,
        "discounted": cheap.distances_per_query,
        "ratio": round(full.distances_per_query / cheap.distances_per_query, 3),
    }


def a_negative_charge_is_refused() -> bool:
    """Whether a negative count is refused rather than reducing the total."""
    try:
        SearchStats().charge(-5)
    except ConfigError:
        return True
    return False


def the_speedup_is_the_corpus_over_the_count() -> dict:
    """What the speedup column means, written out.

    An exact search does one distance per corpus vector, so an index that does a hundredth of
    them is a hundred times faster by this measure and probably not a hundred times faster in
    seconds. The measure ignores everything that is not a distance: the graph walk, the
    candidate heap, the cache misses. It is a lower bound on the work and an upper bound on the
    benefit.
    """
    stats = SearchStats(queries=10)
    stats.charge(4096)
    quality = Quality(index="example", recall=0.9, gap=0.01, stats=stats, corpus_size=4096)
    return {
        "distances_per_query": quality.stats.distances_per_query,
        "corpus": quality.corpus_size,
        "speedup": quality.speedup,
        "scanned": round(quality.scanned, 4),
    }


def a_quality_with_no_cost_reports_no_speedup() -> dict:
    """What the ratio does when nothing was counted.

    Zero, not infinity. An index that reports no distances has almost certainly failed to count
    them rather than achieved an infinite speedup, and a zero in a table is easier to notice
    than a very large number.
    """
    quality = Quality(index="uncounted", recall=1.0, gap=0.0, corpus_size=4096)
    return {"speedup": quality.speedup, "scanned": quality.scanned}


def compare(qualities: Sequence[Quality]) -> list[dict]:
    """A table of results, best recall first."""
    if not qualities:
        raise ConfigError("there is nothing to compare")
    return [quality.as_dict() for quality in sorted(qualities, key=lambda row: -row.recall)]


def the_three_numbers_are_independent() -> dict:
    """Why all three are reported rather than one summary.

    Because any two can be made to look good. An index that scans the whole corpus has perfect
    recall and no speedup. An index that returns the first ten vectors it sees has a large
    speedup and no recall. An index that returns near misses has poor recall and a tiny gap.
    Only the three together describe a structure.
    """
    exhaustive = SearchStats(queries=1)
    exhaustive.charge(4096)
    lazy = SearchStats(queries=1)
    lazy.charge(10)
    return {
        "exhaustive": Quality(
            index="exhaustive", recall=1.0, gap=0.0, stats=exhaustive, corpus_size=4096
        ).as_dict(),
        "lazy": Quality(
            index="lazy", recall=0.02, gap=4.1, stats=lazy, corpus_size=4096
        ).as_dict(),
    }

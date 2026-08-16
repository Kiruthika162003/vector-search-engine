from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from vse.errors import ConfigError, DataError
from vse.index.base import SearchStats, top_up
from vse.index.graph import GraphIndex
from vse.index.ivf import IVFIndex
from vse.vectors.dataset import clustered, gaussian, held_out
from vse.vectors.exact import Neighbours, identifier_overlap, search
from vse.vectors.metric import squared_l2

# Stopping a search before it is finished, which every production system does and no benchmark
# measures.
#
# A service has a deadline. When the deadline arrives the search has to return whatever it has,
# and the interesting question is not whether that costs recall, which it obviously does, but
# whether the cost is predictable and whether it falls on the queries you would expect.
#
# The budget here is distance computations rather than milliseconds, following the rest of the
# package, and that is not a simplification for convenience: a budget in distances is
# reproducible and a budget in milliseconds is a property of the machine. A real service
# converts one to the other with a measured rate, which is a calibration problem rather
# than a search problem.
#
# Three things are measured.
#
# What a budget costs, which is the obvious sweep and is the baseline for the rest.
#
# Whether the cost is fair. A global budget spent in query order gives the whole budget to the
# first queries in a batch and nothing to the last, which is the naive implementation and is
# catastrophic. Splitting the budget per query is the fix and it is not free: a query that would
# have finished early cannot donate its unused budget to a query that needs it.
#
# And which queries get cut off, which came out backwards from what was written here. The
# expectation was that a deadline would concentrate its damage on the queries already served
# worst, since those are the ones whose neighbourhoods are spread across many partitions.
# Measured, the loss is 0.163 among the worse served half and 0.220 among the better served
# half. The queries with the most to lose lose the most, which is arithmetic rather than
# insight, and it means a deadline levels the batch down rather than compounding an existing
# unfairness. That is a better outcome than the one predicted and it is worth having
# measured, because the predicted one would have been an argument for an adaptive budget.
#
# One design note. The budget supersedes the probe count rather than capping it: a search
# with a generous budget opens partitions past what probe would have allowed, so at 1600
# distances it reaches 0.881 where the probe eight search reaches 0.553. The budget is the
# knob, and probe is only used to decide whether a query counts as truncated.


@dataclass
class Budget:
    """A cap on distance computations, and what was spent against it."""

    limit: float
    spent: float = 0.0

    def __post_init__(self) -> None:
        if self.limit <= 0:
            raise ConfigError(f"a budget of {self.limit} allows no search at all")

    @property
    def remaining(self) -> float:
        """What is left."""
        return max(0.0, self.limit - self.spent)

    @property
    def exhausted(self) -> bool:
        """Whether there is anything left."""
        return self.spent >= self.limit

    def charge(self, count: float) -> float:
        """Spend against the budget, returning what was actually allowed.

        Returning the allowed amount rather than raising, because a search that hits its limit
        mid partition should score the part of the partition it can afford rather than throwing
        away work already done. The difference is a few percent of recall at tight budgets
        and it is free.
        """
        if count < 0:
            raise ConfigError(f"cannot charge {count} distances")
        allowed = min(count, self.remaining)
        self.spent += allowed
        return allowed

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "limit": self.limit,
            "spent": round(self.spent, 1),
            "remaining": round(self.remaining, 1),
            "exhausted": self.exhausted,
        }


@dataclass
class Served:
    """What a bounded search returned and how completely."""

    found: Neighbours
    stats: SearchStats
    completed: int
    truncated: int

    @property
    def queries(self) -> int:
        """How many were answered at all."""
        return self.completed + self.truncated

    @property
    def truncated_share(self) -> float:
        """The share that hit the limit."""
        if self.queries == 0:
            return 0.0
        return self.truncated / self.queries

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "queries": self.queries,
            "completed": self.completed,
            "truncated": self.truncated,
            "truncated_share": round(self.truncated_share, 4),
            "distances_per_query": round(self.stats.distances_per_query, 1),
        }


def bounded_partition_search(
    index: IVFIndex,
    queries: torch.Tensor,
    k: int,
    per_query: float,
) -> Served:
    """Probe partitions until the per query budget runs out.

    Partitions are opened in order of centroid distance, so a truncated search has looked at the
    most promising ones and skipped the rest. That is the best possible truncation for this
    structure and it means the recall loss is much smaller than the fraction of work skipped,
    which is the measurement below.
    """
    if per_query <= 0:
        raise ConfigError(f"a per query budget of {per_query} allows no search")
    count = int(queries.shape[0])
    stats = SearchStats(queries=count)
    identifiers = torch.zeros(count, k, dtype=torch.long)
    scores = torch.zeros(count, k)
    completed = 0
    truncated = 0
    centre_scores = squared_l2(queries, index._centres)
    order = torch.argsort(centre_scores, dim=1)
    for row in range(count):
        budget = Budget(limit=per_query)
        budget.charge(float(index.partitions))
        stats.charge(index.partitions)
        reached: list[tuple[float, int]] = []
        opened = 0
        for position in range(index.partitions):
            if budget.exhausted:
                break
            rows = index._lists[int(order[row, position])]
            rows = rows[index._live[rows]]
            if int(rows.numel()) == 0:
                continue
            allowed = int(budget.charge(float(rows.numel())))
            if allowed == 0:
                break
            rows = rows[:allowed]
            stats.charge(allowed)
            stats.visit(allowed)
            block = squared_l2(queries[row : row + 1], index._vectors[rows]).flatten()
            reached.extend(zip(block.tolist(), rows.tolist(), strict=True))
            reached.sort()
            del reached[k:]
            opened += 1
        stats.hop(opened)
        if opened >= index.probe:
            completed += 1
        else:
            truncated += 1
        filled = top_up(
            reached, k, queries[row : row + 1], index._vectors, index._live, index.metric
        )
        for slot, (score, other) in enumerate(filled):
            identifiers[row, slot] = other
            scores[row, slot] = score
    return Served(
        found=Neighbours(identifiers=identifiers, scores=scores),
        stats=stats,
        completed=completed,
        truncated=truncated,
    )


def shared_budget_search(
    index: IVFIndex, queries: torch.Tensor, k: int, total: float
) -> Served:
    """Spend one budget across the whole batch in query order, which is the naive design.

    The first queries get everything they want and the last get nothing. It is the
    implementation somebody writes when the deadline is a wall clock check inside a loop, and it
    is catastrophic in exactly the way the measurement below shows: the mean recall looks
    tolerable and the distribution is two populations, one served perfectly and one not at all.
    """
    if total <= 0:
        raise ConfigError(f"a total budget of {total} allows no search")
    count = int(queries.shape[0])
    budget = Budget(limit=total)
    stats = SearchStats(queries=count)
    identifiers = torch.zeros(count, k, dtype=torch.long)
    scores = torch.zeros(count, k)
    completed = 0
    truncated = 0
    centre_scores = squared_l2(queries, index._centres)
    order = torch.argsort(centre_scores, dim=1)
    for row in range(count):
        reached: list[tuple[float, int]] = []
        opened = 0
        for position in range(index.probe):
            if budget.exhausted:
                break
            rows = index._lists[int(order[row, position])]
            rows = rows[index._live[rows]]
            if int(rows.numel()) == 0:
                continue
            allowed = int(budget.charge(float(rows.numel())))
            if allowed == 0:
                break
            rows = rows[:allowed]
            stats.charge(allowed)
            stats.visit(allowed)
            block = squared_l2(queries[row : row + 1], index._vectors[rows]).flatten()
            reached.extend(zip(block.tolist(), rows.tolist(), strict=True))
            reached.sort()
            del reached[k:]
            opened += 1
        stats.hop(opened)
        if opened >= index.probe:
            completed += 1
        else:
            truncated += 1
        filled = top_up(
            reached, k, queries[row : row + 1], index._vectors, index._live, index.metric
        )
        for slot, (score, other) in enumerate(filled):
            identifiers[row, slot] = other
            scores[row, slot] = score
    return Served(
        found=Neighbours(identifiers=identifiers, scores=scores),
        stats=stats,
        completed=completed,
        truncated=truncated,
    )


def _setup(count: int = 4096, dimension: int = 32, queries: int = 100, probe: int = 8):
    """A built inverted file with its queries and their true answers."""
    corpus = gaussian(count=count, dimension=dimension)
    searched, probes = held_out(corpus, count=queries)
    truth = search(probes, searched.vectors, k=10)
    index = IVFIndex(dimension, partitions=64, probe=probe)
    index.build(searched.vectors)
    return index, searched.vectors, probes, truth


def a_budget_costs_recall(budgets: Sequence[float] = (100, 200, 400, 800, 1600)) -> list[dict]:
    """The baseline sweep, which is what a deadline buys and costs.

    Recall against the per query distance budget. It rises and saturates, and the saturation
    point is where the budget stops binding, which is the number a deployment needs: past it the
    deadline is free and before it every millisecond of headroom is worth something.
    """
    if not budgets:
        raise ConfigError("there is nothing to sweep")
    index, _, queries, truth = _setup()
    rows = []
    for limit in budgets:
        served = bounded_partition_search(index, queries, 10, limit)
        rows.append(
            {
                "budget": limit,
                "recall": round(identifier_overlap(truth, served.found), 4),
                "spent": round(served.stats.distances_per_query, 1),
                "truncated_share": round(served.truncated_share, 4),
            }
        )
    return rows


def the_recall_saturates_where_the_budget_stops_binding() -> dict:
    """Where that curve flattens, which is the useful number out of it."""
    rows = {row["budget"]: row for row in a_budget_costs_recall()}
    return {
        "recall_at_a_hundred": rows[100]["recall"],
        "recall_at_sixteen_hundred": rows[1600]["recall"],
        "truncated_at_a_hundred": rows[100]["truncated_share"],
        "truncated_at_sixteen_hundred": rows[1600]["truncated_share"],
        "rises": rows[1600]["recall"] > rows[100]["recall"],
        "truncation_falls": rows[1600]["truncated_share"] < rows[100]["truncated_share"],
    }


def truncating_costs_less_recall_than_work(budget: float = 300.0) -> dict:
    """Whether a search cut off halfway has lost half its recall, which it has not.

    Far less, because the partitions are opened in order of promise. A budget that allows half
    the work skips the half that was least likely to contain anything, so the recall loss is
    much
    smaller than the work loss. That is the whole reason a deadline is survivable and it is a
    property of the ordering rather than of the budget.
    """
    index, _, queries, truth = _setup()
    full, full_stats = index.search(queries, k=10)
    served = bounded_partition_search(index, queries, 10, budget)
    work_share = served.stats.distances_per_query / full_stats.distances_per_query
    recall_share = identifier_overlap(truth, served.found) / max(
        identifier_overlap(truth, full), 1e-9
    )
    return {
        "budget": budget,
        "full_recall": round(identifier_overlap(truth, full), 4),
        "bounded_recall": round(identifier_overlap(truth, served.found), 4),
        "work_share": round(work_share, 4),
        "recall_share": round(recall_share, 4),
        "recall_survives_better_than_work": recall_share > work_share,
    }


def a_shared_budget_is_unfair(total_per_query: float = 300.0) -> dict:
    """The naive implementation, which serves the first queries and starves the rest.

    One budget spent in query order. The mean recall is comparable to the per query version
    because the total work is the same, and the distribution is not: the early queries are
    answered completely and the late ones get nothing at all. Reporting the mean makes the two
    designs look interchangeable and they are not.
    """
    index, _, queries, truth = _setup()
    count = int(queries.shape[0])
    fair = bounded_partition_search(index, queries, 10, total_per_query)
    shared = shared_budget_search(index, queries, 10, total_per_query * count)
    return {
        "fair_recall": round(identifier_overlap(truth, fair.found), 4),
        "shared_recall": round(identifier_overlap(truth, shared.found), 4),
        "fair_truncated": fair.truncated,
        "shared_truncated": shared.truncated,
        "fair_spent": round(fair.stats.distances_per_query, 1),
        "shared_spent": round(shared.stats.distances_per_query, 1),
        "means_are_close": abs(
            identifier_overlap(truth, fair.found) - identifier_overlap(truth, shared.found)
        )
        < 0.15,
    }


def the_shared_budget_splits_the_batch_in_two(total_per_query: float = 300.0) -> dict:
    """The distribution behind that, which is the point.

    Under a shared budget the batch divides into queries answered as if there were no deadline
    and queries answered with nothing. Under a per query budget every query gets the same
    treatment. Same total work, same mean, and one of them has a tail of complete failures that
    a mean cannot show.
    """
    index, _, queries, truth = _setup()
    count = int(queries.shape[0])
    shared = shared_budget_search(index, queries, 10, total_per_query * count)
    fair = bounded_partition_search(index, queries, 10, total_per_query)
    shared_rows = _per_query_recall(truth, shared.found)
    fair_rows = _per_query_recall(truth, fair.found)
    return {
        "shared_zero_share": round(float((shared_rows == 0.0).float().mean()), 4),
        "fair_zero_share": round(float((fair_rows == 0.0).float().mean()), 4),
        "shared_spread": round(float(shared_rows.std(unbiased=True)), 4),
        "fair_spread": round(float(fair_rows.std(unbiased=True)), 4),
        "shared_is_more_spread": float(shared_rows.std(unbiased=True))
        > float(fair_rows.std(unbiased=True)),
    }


def _per_query_recall(truth: Neighbours, found: Neighbours) -> torch.Tensor:
    """Recall for each query separately."""
    rows = int(truth.identifiers.shape[0])
    hits = torch.zeros(rows)
    for row in range(rows):
        wanted = set(truth.identifiers[row].tolist())
        hits[row] = len(wanted & set(found.identifiers[row].tolist())) / float(len(wanted))
    return hits


def a_deadline_levels_the_batch_down(budget: float = 300.0) -> dict:
    """Which queries a deadline hits, which is not the ones it was expected to.

    The expectation was that truncation would fall on queries already served badly, since a
    query whose neighbourhood is spread across many partitions is both hard to serve and
    likely to be cut off. Measured: the loss is 0.163 among the worse served half and 0.220
    among the better served half.

    The reason is arithmetic. A query already at zero recall cannot lose any, and a query at
    one has everything to lose, so the damage is concentrated where there was something to
    damage. A deadline levels the batch down. That is a milder failure than the one predicted
    and it removes the argument for an adaptive per query budget, which would have been
    expensive and is now unmotivated.
    """
    index, _, queries, truth = _setup()
    full, _ = index.search(queries, k=10)
    served = bounded_partition_search(index, queries, 10, budget)
    unbounded = _per_query_recall(truth, full)
    bounded = _per_query_recall(truth, served.found)
    hurt = unbounded - bounded
    badly_served = unbounded <= float(unbounded.median())
    return {
        "mean_loss": round(float(hurt.mean()), 4),
        "loss_among_the_worst_half": round(float(hurt[badly_served].mean()), 4),
        "loss_among_the_best_half": round(float(hurt[~badly_served].mean()), 4),
        "the_best_lose_more": float(hurt[~badly_served].mean())
        > float(hurt[badly_served].mean()),
    }


def partial_work_inside_a_partition_is_kept(budget: float = 250.0) -> dict:
    """Whether a search that runs out mid partition throws away what it did.

    It does not. The budget returns what it could afford rather than refusing, so a partition
    that is half payable gets half scored and those candidates count. Compared against a version
    that discards a partition it cannot pay for in full, the difference is a few points at tight
    budgets and nothing at loose ones, which is the shape a small optimisation should have.
    """
    index, _, queries, truth = _setup()
    kept = bounded_partition_search(index, queries, 10, budget)
    whole = _whole_partitions_only(index, queries, 10, budget)
    return {
        "budget": budget,
        "partial_recall": round(identifier_overlap(truth, kept.found), 4),
        "whole_only_recall": round(identifier_overlap(truth, whole.found), 4),
        "partial_spent": round(kept.stats.distances_per_query, 1),
        "whole_only_spent": round(whole.stats.distances_per_query, 1),
        "partial_is_better": identifier_overlap(truth, kept.found)
        >= identifier_overlap(truth, whole.found),
    }


def _whole_partitions_only(
    index: IVFIndex, queries: torch.Tensor, k: int, per_query: float
) -> Served:
    """The same search, refusing any partition it cannot pay for in full."""
    count = int(queries.shape[0])
    stats = SearchStats(queries=count)
    identifiers = torch.zeros(count, k, dtype=torch.long)
    scores = torch.zeros(count, k)
    completed = 0
    truncated = 0
    order = torch.argsort(squared_l2(queries, index._centres), dim=1)
    for row in range(count):
        budget = Budget(limit=per_query)
        budget.charge(float(index.partitions))
        stats.charge(index.partitions)
        reached: list[tuple[float, int]] = []
        opened = 0
        for position in range(index.partitions):
            rows = index._lists[int(order[row, position])]
            rows = rows[index._live[rows]]
            if int(rows.numel()) == 0:
                continue
            if float(rows.numel()) > budget.remaining:
                break
            budget.charge(float(rows.numel()))
            stats.charge(int(rows.numel()))
            stats.visit(int(rows.numel()))
            block = squared_l2(queries[row : row + 1], index._vectors[rows]).flatten()
            reached.extend(zip(block.tolist(), rows.tolist(), strict=True))
            reached.sort()
            del reached[k:]
            opened += 1
        stats.hop(opened)
        if opened >= index.probe:
            completed += 1
        else:
            truncated += 1
        filled = top_up(
            reached, k, queries[row : row + 1], index._vectors, index._live, index.metric
        )
        for slot, (score, other) in enumerate(filled):
            identifiers[row, slot] = other
            scores[row, slot] = score
    return Served(
        found=Neighbours(identifiers=identifiers, scores=scores),
        stats=stats,
        completed=completed,
        truncated=truncated,
    )


def a_clustered_corpus_survives_a_deadline_better(budget: float = 300.0) -> dict:
    """Whether the corpus shape changes how much a deadline hurts.

    It should, and the direction is worth checking rather than assuming. A clustered corpus puts
    a query's neighbours in fewer partitions, so a truncated search that opened the first few
    partitions has more of them, and the same budget buys more recall.
    """
    rows = {}
    for label, corpus in (
        ("gaussian", gaussian(count=4096, dimension=32)),
        ("clustered", clustered(count=4096, dimension=32, clusters=16)),
    ):
        searched, queries = held_out(corpus, count=100)
        truth = search(queries, searched.vectors, k=10)
        index = IVFIndex(32, partitions=64, probe=8)
        index.build(searched.vectors)
        full, _ = index.search(queries, k=10)
        served = bounded_partition_search(index, queries, 10, budget)
        rows[label] = {
            "full": identifier_overlap(truth, full),
            "bounded": identifier_overlap(truth, served.found),
        }
    return {
        "gaussian_full": round(rows["gaussian"]["full"], 4),
        "gaussian_bounded": round(rows["gaussian"]["bounded"], 4),
        "clustered_full": round(rows["clustered"]["full"], 4),
        "clustered_bounded": round(rows["clustered"]["bounded"], 4),
        "gaussian_loss": round(rows["gaussian"]["full"] - rows["gaussian"]["bounded"], 4),
        "clustered_loss": round(rows["clustered"]["full"] - rows["clustered"]["bounded"], 4),
        "clustered_loses_less": (rows["clustered"]["full"] - rows["clustered"]["bounded"])
        < (rows["gaussian"]["full"] - rows["gaussian"]["bounded"]),
    }


def a_graph_handles_a_budget_better(budget: float = 300.0) -> dict:
    """Whether every structure degrades the same way under a budget, which they do not.

    Written expecting the partitioned index to win. It opens partitions in order of promise,
    so stopping early leaves out the least useful work, where a graph walk has no such
    ordering and holds whatever it had reached.

    At a budget of 300 distances the graph gets 0.676 for 261 and the partitioned search gets
    0.362 for 300. Nearly twice the recall for less work. The ordering argument is sound and
    it is swamped by the structures simply not being equal at this cost point, which is the
    same conclusion index/forest.py reached about the fitted partitioning from a different
    direction.

    A budget is approximated here by capping the beam rather than by instrumenting the walk,
    which is the
    honest way to do it without rewriting the graph's search: a smaller beam is what a budget
    buys, and the comparison is between the two structures at matched distance counts.
    """
    corpus = gaussian(count=4096, dimension=32)
    searched, queries = held_out(corpus, count=100)
    truth = search(queries, searched.vectors, k=10)

    partitioned = IVFIndex(32, partitions=64, probe=8)
    partitioned.build(searched.vectors)
    served = bounded_partition_search(partitioned, queries, 10, budget)

    graph = GraphIndex(32, degree=16, ef=10)
    graph.build(searched.vectors)
    best = None
    for width in (10, 16, 24, 32, 48, 64):
        graph.ef = width
        found, stats = graph.search(queries, k=10)
        if stats.distances_per_query > budget:
            break
        best = (width, found, stats)
    if best is None:
        raise ConfigError(f"a budget of {budget} does not allow even the smallest beam")
    width, found, stats = best
    return {
        "budget": budget,
        "partitioned_recall": round(identifier_overlap(truth, served.found), 4),
        "partitioned_spent": round(served.stats.distances_per_query, 1),
        "graph_beam": width,
        "graph_recall": round(identifier_overlap(truth, found), 4),
        "graph_spent": round(stats.distances_per_query, 1),
    }


def the_budget_is_respected(budgets: Sequence[float] = (100, 300, 1000)) -> list[dict]:
    """That the search never spends more than it was allowed.

    The one thing a budget has to do. A deadline that is exceeded by a few percent is a deadline
    that is exceeded, and in a service the consequence is a timeout upstream rather than a
    slightly slow response, so this is checked as an inequality rather than an approximation.
    """
    if not budgets:
        raise ConfigError("there is nothing to sweep")
    index, _, queries, _ = _setup()
    rows = []
    for limit in budgets:
        served = bounded_partition_search(index, queries, 10, limit)
        rows.append(
            {
                "budget": limit,
                "spent": round(served.stats.distances_per_query, 1),
                "within": served.stats.distances_per_query <= limit + 1e-6,
            }
        )
    return rows


def no_budget_is_ever_exceeded() -> dict:
    """The conclusion of that table, which is the only acceptable one."""
    rows = the_budget_is_respected()
    return {
        "checked": len(rows),
        "all_within": all(row["within"] for row in rows),
        "worst_overrun": round(max(row["spent"] - row["budget"] for row in rows), 4),
    }


def a_bounded_search_still_returns_k(budget: float = 80.0) -> dict:
    """That a truncated search returns a well formed result.

    Exactly k identifiers, distinct, sorted and correctly scored, even at a budget so tight that
    almost nothing was searched. The shortfall is filled the same way verify/differential.py
    found the unbounded structures should fill it, which is from live rows the search never
    reached, so a truncated result is honest about being poor rather than malformed.
    """
    index, _, queries, truth = _setup()
    served = bounded_partition_search(index, queries, 10, budget)
    identifiers = served.found.identifiers
    distinct = all(
        int(torch.unique(identifiers[row]).numel()) == 10
        for row in range(int(identifiers.shape[0]))
    )
    sorted_rows = bool(
        torch.all(served.found.scores[:, 1:] >= served.found.scores[:, :-1] - 1e-5)
    )
    return {
        "budget": budget,
        "shape": tuple(identifiers.shape),
        "distinct": distinct,
        "sorted": sorted_rows,
        "recall": round(identifier_overlap(truth, served.found), 4),
        "truncated_share": round(served.truncated_share, 4),
    }


def a_budget_of_nothing_is_refused() -> bool:
    """Whether a budget that allows no work is caught at construction."""
    try:
        Budget(limit=0.0)
    except ConfigError:
        return True
    return False


def a_negative_budget_is_refused() -> bool:
    """Whether a negative limit is caught."""
    try:
        Budget(limit=-100.0)
    except ConfigError:
        return True
    return False


def a_negative_charge_is_refused() -> bool:
    """Whether spending a negative amount is caught.

    It would refund the budget, which turns a cap into something a caller can grow by charging
    itself backwards, and the only way that happens by accident is a subtraction in the wrong
    order somewhere upstream.
    """
    try:
        Budget(limit=100.0).charge(-10.0)
    except ConfigError:
        return True
    return False


def a_search_with_no_budget_is_refused() -> bool:
    """Whether a per query budget of zero is caught before the search runs."""
    index, _, queries, _ = _setup(count=512, queries=8, probe=2)
    try:
        bounded_partition_search(index, queries, 5, 0.0)
    except ConfigError:
        return True
    return False


def a_shared_search_with_no_budget_is_refused() -> bool:
    """The same for the shared version."""
    index, _, queries, _ = _setup(count=512, queries=8, probe=2)
    try:
        shared_budget_search(index, queries, 5, 0.0)
    except ConfigError:
        return True
    return False


def a_budget_tracks_what_it_allowed() -> dict:
    """That a budget reports its own state correctly.

    Charged more than it holds, it allows what is left and says it is exhausted. That partial
    allowance is what makes the mid partition case work, and getting it wrong in the other
    direction would let a search overspend by a whole partition on every query.
    """
    budget = Budget(limit=100.0)
    first = budget.charge(60.0)
    second = budget.charge(60.0)
    third = budget.charge(60.0)
    return {
        "first": first,
        "second": second,
        "third": third,
        "spent": budget.spent,
        "exhausted": budget.exhausted,
        "never_overspent": budget.spent <= 100.0,
    }


def an_empty_served_result_divides_safely() -> dict:
    """That the reporting handles a batch of nothing without dividing by it."""
    served = Served(
        found=Neighbours(torch.zeros(0, 10, dtype=torch.long), torch.zeros(0, 10)),
        stats=SearchStats(),
        completed=0,
        truncated=0,
    )
    return {
        "queries": served.queries,
        "truncated_share": served.truncated_share,
        "safe": served.truncated_share == 0.0,
    }


def a_served_result_serialises() -> dict:
    """That the summary a service would log has the fields it needs."""
    stats = SearchStats(queries=4)
    stats.charge(400)
    served = Served(
        found=Neighbours(torch.zeros(4, 10, dtype=torch.long), torch.zeros(4, 10)),
        stats=stats,
        completed=3,
        truncated=1,
    )
    row = served.as_dict()
    return {
        "queries": row["queries"],
        "truncated_share": row["truncated_share"],
        "distances_per_query": row["distances_per_query"],
        "has_everything": set(row)
        == {
            "queries",
            "completed",
            "truncated",
            "truncated_share",
            "distances_per_query",
        },
    }


def a_result_of_the_wrong_shape_is_refused() -> bool:
    """Whether a served result whose identifiers and scores disagree is caught.

    Not by this module, by Neighbours, which is where it belongs. Checked here because every
    function in this module builds a Neighbours by hand rather than through a search, so the
    usual guarantee that the shapes match does not apply.
    """
    try:
        Neighbours(torch.zeros(4, 10, dtype=torch.long), torch.zeros(4, 5))
    except DataError:
        return True
    return False

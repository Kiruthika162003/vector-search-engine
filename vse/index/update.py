from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

from vse.build.neighbours import Graph, components
from vse.errors import BuildError, ConfigError
from vse.index.base import Index, evaluate_result
from vse.index.graph import GraphIndex
from vse.index.ivf import IVFIndex
from vse.vectors.dataset import Corpus, clustered, gaussian, held_out
from vse.vectors.exact import Neighbours, search

# What happens to an index after it stops being the thing that was built.
#
# Every index in this package is built once from a corpus and then searched. That is the case
# the literature measures and it is not the case anyone runs: a real corpus grows, shrinks, and
# drifts. This module was written to measure how much recall that costs, and the answer for
# three of the four cases is none, which is not what it was written expecting.
#
# Inserts from the same distribution cost nothing. That is measured in ivf.py rather than here,
# and this module does not repeat it, but it is the baseline everything below sits against: a
# partitioned index absorbing stationary writes is not degrading, it is just bigger.
#
# Drift costs nothing either, in recall, and this was the surprise. Queries drawn from the
# drifted region score 0.814 against 0.582 for queries in the original region. Higher, not
# lower. What drift actually does is unbalance: the arriving blob lands on a handful of
# centroids, the spread goes from 0.40 to 0.75, and a query into that region opens partitions
# holding most of the new corpus. It pays 879 distances against 493. So drift is a cost problem
# wearing a recall problem's clothes, and the recall it does not cost is why it goes unnoticed.
#
# What a rebuild fixes is therefore not damage. It is that the centroids were fitted on a
# sample: refitting on the full corpus halves the spread, from 0.30 to 0.15, and buys two points
# of recall that were never lost in the first place. That is still worth doing and it is a
# different argument from the one usually made for it.
#
# The one real degradation is deletion on a graph, and it is severe. A tombstoned vertex keeps
# its edges, because unlinking it cuts every path through it, so deleted vectors are traversed
# and paid for while contributing nothing to the result. The component count says the graph is
# still connected; recounting with the dead vertices removed from the edge lists says the live
# ones alone are not. The structure holding the graph together is made of vectors that no longer
# exist.


@dataclass
class Churn:
    """A record of what has been written to an index since it was built."""

    inserted: int = 0
    removed: int = 0
    built_size: int = 0

    @property
    def size(self) -> int:
        """How many live vectors the index should hold."""
        return self.built_size + self.inserted - self.removed

    @property
    def insert_share(self) -> float:
        """Inserts as a fraction of the built corpus."""
        if self.built_size == 0:
            return 0.0
        return self.inserted / self.built_size

    @property
    def remove_share(self) -> float:
        """Removals as a fraction of the built corpus."""
        if self.built_size == 0:
            return 0.0
        return self.removed / self.built_size

    @property
    def churn(self) -> float:
        """Total writes as a fraction of the built corpus.

        The single number a rebuild policy can be written against. Half a corpus inserted and
        half removed is a churn of one even though the size did not move, which is the point:
        the damage tracks writes, not the net.
        """
        if self.built_size == 0:
            return 0.0
        return (self.inserted + self.removed) / self.built_size

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "built_size": self.built_size,
            "inserted": self.inserted,
            "removed": self.removed,
            "size": self.size,
            "churn": round(self.churn, 4),
        }


@dataclass
class Degradation:
    """Recall and cost at one point along a churn schedule."""

    churn: float
    recall: float
    distances: float
    size: int
    detail: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        row = {
            "churn": round(self.churn, 4),
            "recall": round(self.recall, 4),
            "distances": round(self.distances, 1),
            "size": self.size,
        }
        row.update(self.detail)
        return row


def split_for_churn(
    corpus: Corpus, built: int = 2048, queries: int = 100
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Divide a corpus into what the index is built on, what arrives later, and the queries.

    The arriving vectors come from the same distribution as the built ones, which is the kind
    case. A corpus that drifts gives up recall faster than anything measured here and the drift
    case is separated out below so the two effects do not get confused.
    """
    total = int(corpus.vectors.shape[0])
    if built + queries >= total:
        raise ConfigError(f"{total} vectors cannot supply {built} built and {queries} queries")
    searched, probes = held_out(corpus, count=queries)
    pool = searched.vectors
    return pool[:built], pool[built:], probes


def measure(
    index: Index,
    corpus: torch.Tensor,
    queries: torch.Tensor,
    churn: Churn,
    k: int = 10,
    alive: torch.Tensor | None = None,
    detail: dict | None = None,
) -> Degradation:
    """Score an index against exact search over whatever it currently holds.

    The ground truth is recomputed from the live corpus at every point, not carried forward from
    build time. Carrying it forward would measure the index against a corpus that no longer
    exists and would make every degradation look worse than it is.

    The live corpus is given as a mask over the original rows rather than as a smaller tensor,
    because an index that has had rows removed still returns the identifiers those rows had at
    build time. Compacting the corpus and searching it directly produces truth in a different
    identifier space, and comparing the two silently scores every result against the wrong
    vector, which is a bug that reads as a plausible recall number.
    """
    if alive is None:
        truth = search(queries, corpus, k=k, metric=index.metric)
        size = int(corpus.shape[0])
    else:
        rows = torch.nonzero(alive, as_tuple=False).flatten()
        local = search(queries, corpus[rows], k=k, metric=index.metric)
        truth = Neighbours(identifiers=rows[local.identifiers], scores=local.scores)
        size = int(rows.numel())
    found, stats = index.search(queries, k=k)
    quality = evaluate_result(index, corpus, queries, found, stats, truth=truth)
    return Degradation(
        churn=churn.churn,
        recall=quality.recall,
        distances=quality.stats.distances_per_query,
        size=size,
        detail=detail or {},
    )


def insert_in_batches(
    index: Index,
    built: torch.Tensor,
    arriving: torch.Tensor,
    queries: torch.Tensor,
    batches: int = 8,
    k: int = 10,
) -> list[Degradation]:
    """Insert arriving vectors in equal batches, measuring after each one."""
    if batches < 1:
        raise ConfigError(f"{batches} batches inserts nothing")
    per_batch = int(arriving.shape[0]) // batches
    if per_batch < 1:
        raise ConfigError(f"{int(arriving.shape[0])} vectors do not divide into {batches}")
    churn = Churn(built_size=int(built.shape[0]))
    live = built
    rows = [measure(index, live, queries, churn, k=k)]
    for batch in range(batches):
        block = arriving[batch * per_batch : (batch + 1) * per_batch]
        index.insert(block)
        live = torch.cat([live, block], dim=0)
        churn.inserted += int(block.shape[0])
        rows.append(measure(index, live, queries, churn, k=k))
    return rows


def remove_in_batches(
    index: Index,
    built: torch.Tensor,
    queries: torch.Tensor,
    batches: int = 8,
    share: float = 0.4,
    k: int = 10,
    seed: int = 0,
) -> list[Degradation]:
    """Remove a share of the corpus in equal batches, measuring after each one.

    Removals are drawn uniformly at random. A real deletion pattern is usually clustered, since
    what gets deleted is a tenant or a date range or a document, and clustered deletion is
    measured separately because it does something different to a partitioned index.
    """
    if not 0.0 < share < 1.0:
        raise ConfigError(f"removing a share of {share} is not a removal")
    if batches < 1:
        raise ConfigError(f"{batches} batches removes nothing")
    count = int(built.shape[0])
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(count, generator=generator)
    doomed = order[: int(count * share)]
    per_batch = int(doomed.numel()) // batches
    if per_batch < 1:
        raise ConfigError(f"{int(doomed.numel())} removals do not divide into {batches}")
    churn = Churn(built_size=count)
    alive = torch.ones(count, dtype=torch.bool)
    rows = [measure(index, built, queries, churn, k=k, alive=alive)]
    for batch in range(batches):
        block = doomed[batch * per_batch : (batch + 1) * per_batch]
        index.remove([int(row) for row in block])
        alive[block] = False
        churn.removed += int(block.numel())
        rows.append(
            measure(
                index,
                built,
                queries,
                churn,
                k=k,
                alive=alive,
                detail={"live_share": round(float(alive.float().mean()), 4)},
            )
        )
    return rows


def the_partitions_go_uneven(batches: int = 8) -> dict:
    """What is actually wrong with the index after a lot of inserts.

    Nothing, on a stationary stream. The spread was expected to grow, since arriving vectors go
    to whichever centroid is nearest and nothing rebalances. It falls, from 0.3155 to 0.3033.

    The reason is that a partition is large because its cell covers a dense region, and vectors
    from the same distribution arrive into cells in proportion to how large those cells already
    are. So the systematic part of the imbalance is preserved and the sampling part, which is
    the fluctuation of a finite count, shrinks as the counts grow. Relative spread goes down.

    Which locates the imbalance where it actually is: at build time, not after it. The centroids
    were fitted on 2048 vectors and the spread they produce is a property of that fit.
    """
    corpus = gaussian(count=4096, dimension=32)
    built, arriving, _ = split_for_churn(corpus, built=2048)
    index = IVFIndex(32, partitions=45, probe=6)
    index.build(built)
    before = _partition_spread(index)
    per_batch = int(arriving.shape[0]) // batches
    for batch in range(batches):
        index.insert(arriving[batch * per_batch : (batch + 1) * per_batch])
    after = _partition_spread(index)
    return {
        "spread_at_build": round(before, 4),
        "spread_after": round(after, 4),
        "grew": after > before,
        "ratio": round(after / before, 3) if before > 0 else None,
    }


def _partition_spread(index: IVFIndex) -> float:
    """The coefficient of variation of the posting list lengths."""
    sizes = torch.tensor([float(rows.numel()) for rows in index._lists])
    if float(sizes.mean()) == 0.0:
        return 0.0
    return float(sizes.std(unbiased=False) / sizes.mean())


def a_rebuild_fixes_something_that_was_never_broken() -> dict:
    """What rebuilding buys, given that nothing degraded.

    Quite a lot, for a reason that has nothing to do with the writes. Refitting the centroids on
    the corpus that is actually there halves the spread, 0.3036 to 0.1492, drops the distance
    count from 623 to 579 and lifts recall from 0.543 to 0.561. None of that is recovery of
    something lost. It is the difference between centroids fitted to half the corpus and
    centroids fitted to all of it, and it was there from the first insert.

    So the case for rebuilding is real and the usual argument for it is not. Rebuild because the
    fit improves with the data available, not because writes damaged anything.
    """
    corpus = gaussian(count=4096, dimension=32)
    built, arriving, queries = split_for_churn(corpus, built=2048)
    index = IVFIndex(32, partitions=45, probe=6)
    index.build(built)
    live = torch.cat([built, arriving], dim=0)
    index.insert(arriving)
    churn = Churn(built_size=int(built.shape[0]), inserted=int(arriving.shape[0]))
    before = measure(index, live, queries, churn)
    spread_before = _partition_spread(index)
    index.rebuild()
    after = measure(index, live, queries, churn)
    return {
        "recall_before": round(before.recall, 4),
        "recall_after": round(after.recall, 4),
        "distances_before": round(before.distances, 1),
        "distances_after": round(after.distances, 1),
        "spread_before": round(spread_before, 4),
        "spread_after": round(_partition_spread(index), 4),
        "cost_recovered": after.distances < before.distances,
    }


def a_graph_degrades_on_deletion(batches: int = 8, share: float = 0.4) -> list[dict]:
    """How a neighbour graph holds up as vectors are removed, which is badly.

    A removed vertex is tombstoned rather than unlinked, because unlinking it would cut every
    path that went through it and the graph relies on those paths for connectivity. So the
    vertex is still traversed, still costs a distance computation to evaluate, and is then
    discarded from the result. At forty percent removed, the walk pays close to the original
    cost for a corpus that is sixty percent of its original size.
    """
    corpus = gaussian(count=3072, dimension=32)
    built, _, queries = split_for_churn(corpus, built=2048)
    index = GraphIndex(32, degree=16, ef=32)
    index.build(built)
    rows = remove_in_batches(index, built, queries, batches=batches, share=share)
    return [row.as_dict() for row in rows]


def the_tombstones_are_still_traversed() -> dict:
    """The number that makes that concrete.

    Distances per query does not fall at all. Not approximately: 372.7 at build and 372.7 after
    forty percent of the corpus is gone, identical to the last digit at every one of the eight
    measurement points in between. The walk visits exactly the same vertices it always did and
    evaluates every one of them, including the dead. A fresh graph over the survivors costs
    339.9, so the price of not rebuilding is the whole difference plus the sixty six percent
    rise in cost per live vector.
    """
    rows = a_graph_degrades_on_deletion()
    first, last = rows[0], rows[-1]
    corpus = gaussian(count=3072, dimension=32)
    built, _, queries = split_for_churn(corpus, built=2048)
    generator = torch.Generator().manual_seed(0)
    order = torch.randperm(int(built.shape[0]), generator=generator)
    doomed = order[: int(int(built.shape[0]) * 0.4)]
    alive = torch.ones(int(built.shape[0]), dtype=torch.bool)
    alive[doomed[: (int(doomed.numel()) // 8) * 8]] = False
    fresh = GraphIndex(32, degree=16, ef=32)
    fresh.build(built[alive])
    _, fresh_stats = fresh.search(queries, k=10)
    return {
        "distances_at_build": first["distances"],
        "distances_after_removals": last["distances"],
        "distances_if_rebuilt": round(fresh_stats.distances_per_query, 1),
        "size_at_build": first["size"],
        "size_after": last["size"],
        "cost_barely_fell": last["distances"] > first["distances"] * 0.8,
        "rebuilt_is_cheaper": fresh_stats.distances_per_query < last["distances"],
    }


def deletion_fragments_the_graph(shares: Sequence[float] = (0.0, 0.2, 0.4, 0.6)) -> list[dict]:
    """Whether removing vertices breaks the graph into pieces, which is the real risk.

    Counted the usual way, no: one component at every deletion level up to sixty percent, which
    is the number a health check would report and it is reassuring and useless. Counted with the
    dead vertices taken out of the edge lists, the live vectors alone fall into two pieces at
    twenty percent removed, three at forty, and sixteen at sixty.

    So the graph is not surviving deletion, it is deferring it. Every path that makes it look
    connected runs through vertices that are not in the corpus any more, and the compaction that
    would make the cost honest is exactly the operation that breaks it.
    """
    if not shares:
        raise ConfigError("there is nothing to sweep")
    corpus = gaussian(count=2048, dimension=32)
    built, _, _ = split_for_churn(corpus, built=1536, queries=64)
    rows = []
    for share in shares:
        index = GraphIndex(32, degree=16, ef=32)
        index.build(built)
        count = int(built.shape[0])
        generator = torch.Generator().manual_seed(1)
        order = torch.randperm(count, generator=generator)
        doomed = order[: int(count * share)]
        alive = torch.ones(count, dtype=torch.bool)
        if int(doomed.numel()) > 0:
            index.remove([int(row) for row in doomed])
            alive[doomed] = False
        whole = components(index.graph)
        live_only = _components_over_live(index.graph, alive)
        rows.append(
            {
                "removed_share": share,
                "components_with_tombstones": whole,
                "components_over_live_only": live_only,
                "live": int(alive.sum()),
            }
        )
    return rows


def _components_over_live(graph: Graph, alive: torch.Tensor) -> int:
    """Connected components counted with the dead vertices taken out of the edge lists."""
    seen = torch.zeros(int(alive.numel()), dtype=torch.bool)
    seen[~alive] = True
    found = 0
    for start in range(int(alive.numel())):
        if bool(seen[start]):
            continue
        found += 1
        stack = [start]
        seen[start] = True
        while stack:
            here = stack.pop()
            for other in graph.edges[here]:
                if not bool(seen[other]):
                    seen[other] = True
                    stack.append(int(other))
    return found


def the_tombstones_are_holding_the_graph_together() -> dict:
    """The two component counts side by side, which is the finding.

    With tombstones in the edge lists the graph is one component at every deletion level. Take
    the dead vertices out and the live vectors alone fall into many. So the graph is not
    surviving deletion, it is deferring it: the structure that keeps it searchable is made of
    vertices that no longer exist, and a compaction that actually removed them would break it.
    """
    rows = {row["removed_share"]: row for row in deletion_fragments_the_graph()}
    heavy = rows[0.6]
    return {
        "components_with_tombstones": heavy["components_with_tombstones"],
        "components_over_live_only": heavy["components_over_live_only"],
        "removed_share": 0.6,
        "connected_only_through_the_dead": heavy["components_over_live_only"]
        > heavy["components_with_tombstones"],
    }


def a_partitioned_index_survives_deletion_better(share: float = 0.4) -> dict:
    """The same deletion schedule on both structures.

    The inverted file wins clearly. A posting list with dead entries filtered out is smaller and
    cheaper, so removals make it faster, where the graph gets no cheaper at all. Neither loses
    much recall, so this is entirely a cost story, and it is the reverse of the ranking these
    two have on every other measurement in the package.
    """
    corpus = gaussian(count=3072, dimension=32)
    built, _, queries = split_for_churn(corpus, built=2048)

    partitioned = IVFIndex(32, partitions=45, probe=6)
    partitioned.build(built)
    ivf_rows = remove_in_batches(partitioned, built, queries, batches=4, share=share)

    graph = GraphIndex(32, degree=16, ef=32)
    graph.build(built)
    graph_rows = remove_in_batches(graph, built, queries, batches=4, share=share)

    return {
        "ivf_cost_ratio": round(ivf_rows[-1].distances / ivf_rows[0].distances, 3),
        "graph_cost_ratio": round(graph_rows[-1].distances / graph_rows[0].distances, 3),
        "ivf_recall_change": round(ivf_rows[-1].recall - ivf_rows[0].recall, 4),
        "graph_recall_change": round(graph_rows[-1].recall - graph_rows[0].recall, 4),
        "ivf_got_cheaper": ivf_rows[-1].distances < ivf_rows[0].distances,
        "graph_did_not": graph_rows[-1].distances > graph_rows[0].distances * 0.8,
    }


def clustered_deletion_is_worse_than_uniform(share: float = 0.4) -> dict:
    """Whether it matters which vectors get deleted, which it does for a partitioned index.

    Deleting a tenant or a date range empties whole partitions rather than thinning all of them,
    so a probe that opens six partitions can find several of them empty and come back with
    almost nothing. Uniform deletion leaves every partition usable. The measurement compares
    both at the same deletion volume and the recall gap is the answer.
    """
    corpus = clustered(count=3072, dimension=32, clusters=16)
    built, _, queries = split_for_churn(corpus, built=2048)
    count = int(built.shape[0])

    generator = torch.Generator().manual_seed(2)
    uniform = torch.randperm(count, generator=generator)[: int(count * share)]

    centre = built.mean(dim=0, keepdim=True)
    order = torch.argsort(((built - centre) ** 2).sum(dim=1))
    grouped = order[: int(count * share)]

    rows = {}
    for label, doomed in (("uniform", uniform), ("clustered", grouped)):
        index = IVFIndex(32, partitions=45, probe=6)
        index.build(built)
        index.remove([int(row) for row in doomed])
        alive = torch.ones(count, dtype=torch.bool)
        alive[doomed] = False
        churn = Churn(built_size=count, removed=int(doomed.numel()))
        rows[label] = measure(index, built, queries, churn, alive=alive)
    return {
        "uniform_recall": round(rows["uniform"].recall, 4),
        "clustered_recall": round(rows["clustered"].recall, 4),
        "uniform_distances": round(rows["uniform"].distances, 1),
        "clustered_distances": round(rows["clustered"].distances, 1),
        "clustered_is_worse": rows["clustered"].recall < rows["uniform"].recall,
    }


def drift_lands_the_new_corpus_on_a_few_centroids(
    shifts: Sequence[float] = (0.0, 2.0, 4.0, 8.0),
) -> list[dict]:
    """What a moving distribution does to an index, which is not what it does to recall.

    It unbalances it. The arriving vectors sit somewhere no centroid was fitted, so they all
    take whichever few centroids happen to lie in that direction, and those posting lists grow
    without limit while the rest are untouched. The spread goes from 0.40 at build to 0.75 at a
    shift of eight, and the largest partition doubles.

    This is the quantity to watch. It is a pass over the posting list lengths, it moves early,
    and it moves on the workload where recall does not move at all.
    """
    if not shifts:
        raise ConfigError("there is nothing to sweep")
    corpus = gaussian(count=3072, dimension=32)
    built, arriving, _ = split_for_churn(corpus, built=2048)
    direction = torch.randn(1, 32, generator=torch.Generator().manual_seed(7))
    direction = direction / direction.norm()
    rows = []
    for shift in shifts:
        index = IVFIndex(32, partitions=45, probe=6)
        index.build(built)
        before = _partition_spread(index)
        index.insert(arriving + direction * shift)
        sizes = sorted(int(rows_in.numel()) for rows_in in index._lists)
        rows.append(
            {
                "shift": shift,
                "spread_at_build": round(before, 4),
                "spread_after": round(_partition_spread(index), 4),
                "largest_partition": sizes[-1],
            }
        )
    return rows


def the_spread_is_where_drift_shows_up() -> dict:
    """The two ends of that sweep."""
    rows = {row["shift"]: row for row in drift_lands_the_new_corpus_on_a_few_centroids()}
    return {
        "spread_without_drift": rows[0.0]["spread_after"],
        "spread_at_eight": rows[8.0]["spread_after"],
        "largest_without_drift": rows[0.0]["largest_partition"],
        "largest_at_eight": rows[8.0]["largest_partition"],
        "grows": rows[8.0]["spread_after"] > rows[0.0]["spread_after"] * 1.5,
    }


def queries_in_the_drifted_region_recall_more_and_cost_more(
    shifts: Sequence[float] = (0.0, 2.0, 4.0, 8.0),
) -> list[dict]:
    """Whether the drifted vectors are hard to find, which they are not.

    This module was written expecting the arriving vectors to be the ones the index fails on,
    since no centroid was fitted where they live. The opposite happens. A query from the drifted
    region gets 0.814 recall against 0.582 for a query from the original one, because the whole
    drifted blob is concentrated in a few partitions and probing them scans most of it.

    The index is accidentally exact over the new corpus and pays for it: 879 distances per query
    against 493. Which is a nicer way to be wrong than the alternative, and it is still wrong,
    because the cost is unbounded in the size of the drifted region.
    """
    if not shifts:
        raise ConfigError("there is nothing to sweep")
    corpus = gaussian(count=3072, dimension=32)
    built, arriving, probes = split_for_churn(corpus, built=2048)
    direction = torch.randn(1, 32, generator=torch.Generator().manual_seed(7))
    direction = direction / direction.norm()
    rows = []
    for shift in shifts:
        index = IVFIndex(32, partitions=45, probe=6)
        index.build(built)
        moved = arriving + direction * shift
        index.insert(moved)
        live = torch.cat([built, moved], dim=0)
        churn = Churn(built_size=int(built.shape[0]), inserted=int(moved.shape[0]))
        original = measure(index, live, probes[:50], churn)
        drifted = measure(index, live, probes[:50] + direction * shift, churn)
        rows.append(
            {
                "shift": shift,
                "original_region_recall": round(original.recall, 4),
                "drifted_region_recall": round(drifted.recall, 4),
                "original_region_distances": round(original.distances, 1),
                "drifted_region_distances": round(drifted.distances, 1),
            }
        )
    return rows


def the_drifted_region_is_the_expensive_one() -> dict:
    """The two ends of that sweep, which is the finding this module was rewritten around."""
    rows = {
        row["shift"]: row for row in queries_in_the_drifted_region_recall_more_and_cost_more()
    }
    heavy = rows[8.0]
    return {
        "original_region_recall": heavy["original_region_recall"],
        "drifted_region_recall": heavy["drifted_region_recall"],
        "original_region_distances": heavy["original_region_distances"],
        "drifted_region_distances": heavy["drifted_region_distances"],
        "recall_is_higher": heavy["drifted_region_recall"] > heavy["original_region_recall"],
        "cost_is_higher": heavy["drifted_region_distances"]
        > heavy["original_region_distances"] * 1.5,
    }


def an_insert_into_an_unbuilt_index_builds_it() -> dict:
    """What writing to an index that was never built does, which is not what it should.

    The inverted file builds itself from whatever arrived first. Four vectors in and it has a
    four vector fit and a partitioning nobody chose, and every insert after that is filed
    against
    centroids computed from a sample of four. Nothing raises and nothing warns.

    The graph refuses, because its build needs more vectors than a degree, and the accident of
    that check is the only reason the two behave differently. Recorded here rather than fixed,
    because changing it would change IVFIndex.insert, and the honest note is that the two
    indexes
    disagree about whether an unbuilt write is an error.
    """
    partitioned = IVFIndex(8, partitions=4)
    identifiers = partitioned.insert(torch.randn(4, 8))
    graph_refused = False
    try:
        GraphIndex(8, degree=32).insert(torch.randn(4, 8))
    except BuildError:
        graph_refused = True
    return {
        "ivf_accepted": len(identifiers) == 4,
        "ivf_size_after": partitioned.size,
        "ivf_partitions_fitted_on": 4,
        "graph_refused": graph_refused,
        "they_disagree": graph_refused,
    }


def a_rebuild_after_drift_recovers_it() -> dict:
    """Whether the same repair works on the harder case.

    It does, which is the reassuring half. Refitting the centroids on the corpus that is
    actually there puts partitions where the vectors are, drifted or not, and the recall
    returns. Drift is not a different failure from imbalance, it is imbalance arriving faster.
    """
    corpus = gaussian(count=3072, dimension=32)
    built, arriving, queries = split_for_churn(corpus, built=2048)
    index = IVFIndex(32, partitions=45, probe=6)
    index.build(built)
    direction = torch.randn(1, 32, generator=torch.Generator().manual_seed(7))
    direction = direction / direction.norm()
    moved = arriving + direction * 2.0
    index.insert(moved)
    live = torch.cat([built, moved], dim=0)
    churn = Churn(built_size=int(built.shape[0]), inserted=int(moved.shape[0]))
    before = measure(index, live, queries, churn)
    index.rebuild()
    after = measure(index, live, queries, churn)
    return {
        "recall_before": round(before.recall, 4),
        "recall_after": round(after.recall, 4),
        "recovered": after.recall > before.recall,
        "gain": round(after.recall - before.recall, 4),
    }


def removing_something_that_is_not_there_is_not_an_error() -> dict:
    """What a removal of an unknown identifier does, which is nothing quietly.

    Returns a count of what it actually removed rather than raising. A delete that arrives twice
    is ordinary in any system with retries, and making the second one an error pushes the
    problem onto every caller.
    """
    corpus = gaussian(count=256, dimension=8)
    index = IVFIndex(8, partitions=8, probe=2)
    index.build(corpus.vectors)
    first = index.remove([1, 2, 3])
    second = index.remove([1, 2, 3])
    return {
        "first_removal": first,
        "second_removal": second,
        "idempotent": second == 0,
        "size": index.size,
    }


def a_zero_batch_insert_is_refused() -> bool:
    """Whether an insert schedule with no batches is caught."""
    corpus = gaussian(count=512, dimension=8)
    built, arriving, queries = split_for_churn(corpus, built=256, queries=32)
    index = IVFIndex(8, partitions=16, probe=2)
    index.build(built)
    try:
        insert_in_batches(index, built, arriving, queries, batches=0)
    except ConfigError:
        return True
    return False


def a_removal_share_of_one_is_refused() -> bool:
    """Whether removing the entire corpus as a share is caught.

    An index with nothing in it is not a degraded index, it is a different object, and every
    measurement downstream would divide by zero or return an empty result that scores as perfect
    recall against an empty truth.
    """
    corpus = gaussian(count=512, dimension=8)
    built, _, queries = split_for_churn(corpus, built=256, queries=32)
    index = IVFIndex(8, partitions=16, probe=2)
    index.build(built)
    try:
        remove_in_batches(index, built, queries, share=1.0)
    except ConfigError:
        return True
    return False


def a_split_that_leaves_nothing_to_insert_is_refused() -> bool:
    """Whether asking for more built vectors than the corpus holds is caught."""
    try:
        split_for_churn(gaussian(count=256, dimension=8), built=512, queries=32)
    except ConfigError:
        return True
    return False


def the_churn_counter_tracks_writes_not_size() -> dict:
    """That a corpus which grows and shrinks back has still been churned.

    Half inserted and half removed leaves the size where it started and the index in the state
    of one that absorbed a corpus worth of writes. A rebuild policy keyed on size would never
    fire on that workload, which is why the counter reports writes.
    """
    churn = Churn(built_size=1000, inserted=500, removed=500)
    return {
        "built": 1000,
        "size": churn.size,
        "churn": churn.churn,
        "size_unchanged": churn.size == 1000,
        "churn_is_one": churn.churn == 1.0,
    }


def compare_structures_under_churn() -> list[dict]:
    """Both structures under both kinds of write, as one table.

    Four rows. The inverted file absorbs inserts with a cost rise and removals with a cost fall.
    The graph absorbs inserts about as well and removals not at all. Which is a clean split: the
    structure that wins on every search measurement is the one that cannot be maintained.
    """
    corpus = gaussian(count=3072, dimension=32)
    built, arriving, queries = split_for_churn(corpus, built=2048)
    rows = []
    for label, make in (
        ("ivf", lambda: IVFIndex(32, partitions=45, probe=6)),
        ("graph", lambda: GraphIndex(32, degree=16, ef=32)),
    ):
        index = make()
        index.build(built)
        grown = insert_in_batches(index, built, arriving, queries, batches=4)
        rows.append(
            {
                "index": label,
                "write": "insert",
                "recall_change": round(grown[-1].recall - grown[0].recall, 4),
                "cost_ratio": round(grown[-1].distances / grown[0].distances, 3),
            }
        )
        index = make()
        index.build(built)
        shrunk = remove_in_batches(index, built, queries, batches=4, share=0.4)
        rows.append(
            {
                "index": label,
                "write": "remove",
                "recall_change": round(shrunk[-1].recall - shrunk[0].recall, 4),
                "cost_ratio": round(shrunk[-1].distances / shrunk[0].distances, 3),
            }
        )
    return rows

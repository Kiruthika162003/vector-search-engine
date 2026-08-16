from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from vse.errors import ConfigError, DataError
from vse.index.base import Index
from vse.index.forest import ForestIndex
from vse.index.graph import GraphIndex
from vse.index.hnsw import HNSWIndex
from vse.index.ivf import IVFIndex
from vse.quantize.binary import BinaryIndex
from vse.vectors.dataset import clustered, gaussian, held_out
from vse.vectors.exact import identifier_overlap, search
from vse.vectors.metric import squared_l2

# Queries chosen to be hard rather than drawn at random, which is the difference between a
# benchmark and a guarantee.
#
# Every recall number in this package is a mean over queries sampled from the same distribution
# as the corpus. That is the right measurement for a system whose traffic looks like its data,
# and it says nothing about the worst case. A service does not average over its users, and the
# user whose query happens to sit on a partition boundary sees the tail rather than the mean.
#
# Four ways to construct a hard query, in increasing order of how much they need to know:
#
#   drawn from the sparse regions of the corpus, which needs only the corpus
#   placed exactly between two partition centroids, which needs the fitted index
#   placed where the graph's entry point is furthest away, which needs the built graph
#   found by searching for the queries the index already answers worst, which needs the truth
#
# The last one is not an attack, it is a selection, and it is the strongest of the four by a
# wide margin because it optimises directly against the thing being measured. It is the one a
# real adversary cannot use, since it needs the ground truth. The first three are what somebody
# with the corpus and the code could actually do.
#
# The question the module answers is how far below the mean the worst case sits, and whether the
# structures differ in how exposed they are. A structure whose adversarial recall is close to
# its average recall fails evenly; one with a wide gap has a small set of queries it reliably
# fails, and those queries can be found.


@dataclass
class Attack:
    """A set of queries built to be hard, and how they were built."""

    queries: torch.Tensor
    name: str
    needs: str

    def __post_init__(self) -> None:
        if self.queries.ndim != 2:
            raise DataError(f"queries are a matrix, got {tuple(self.queries.shape)}")
        if int(self.queries.shape[0]) == 0:
            raise DataError("an attack with no queries measures nothing")

    @property
    def count(self) -> int:
        """How many queries it holds."""
        return int(self.queries.shape[0])

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {"name": self.name, "queries": self.count, "needs": self.needs}


def from_sparse_regions(
    corpus: torch.Tensor, count: int = 100, sample: int = 512, seed: int = 0
) -> Attack:
    """Queries drawn where the corpus is thinnest.

    Needs nothing but the corpus. Each candidate is scored by its distance to its own tenth
    nearest neighbour, which is a local density estimate, and the sparsest are kept. A query in
    a sparse region has neighbours far away and spread out, which is exactly the case
    every structure here handles worst.
    """
    if count < 1:
        raise ConfigError(f"{count} queries is not an attack")
    total = int(corpus.shape[0])
    if sample > total:
        raise ConfigError(f"cannot sample {sample} from {total}")
    generator = torch.Generator().manual_seed(seed)
    picks = torch.randperm(total, generator=generator)[:sample]
    scores = squared_l2(corpus[picks], corpus)
    tenth = torch.topk(scores, k=11, dim=1, largest=False).values[:, -1]
    sparsest = picks[torch.argsort(tenth, descending=True)[:count]]
    return Attack(queries=corpus[sparsest].clone(), name="sparse regions", needs="the corpus")


def between_partitions(index: IVFIndex, count: int = 100, seed: int = 0) -> Attack:
    """Queries placed exactly halfway between two centroids.

    Needs the fitted index. A query on the midpoint of two centroids is equidistant from both,
    so the probe order between them is arbitrary and its true neighbours split across the two
    partitions. A probe of one gets at most half of them by construction.
    """
    if count < 1:
        raise ConfigError(f"{count} queries is not an attack")
    centres = index._centres
    partitions = int(centres.shape[0])
    if partitions < 2:
        raise ConfigError("a midpoint needs at least two partitions")
    generator = torch.Generator().manual_seed(seed)
    left = torch.randint(0, partitions, (count,), generator=generator)
    right = (left + 1 + torch.randint(0, partitions - 1, (count,), generator=generator)) % (
        partitions
    )
    return Attack(
        queries=(centres[left] + centres[right]) / 2,
        name="between partitions",
        needs="the fitted index",
    )


def far_from_the_entry_point(
    index: GraphIndex, corpus: torch.Tensor, count: int = 100
) -> Attack:
    """Queries at the corpus vectors furthest from where the graph's walk begins.

    Needs the built graph. Every walk starts at one vertex, so a query far from it needs many
    hops to reach its neighbourhood and a bounded beam may not get there. Which vertex is the
    entry point is a property of the build rather than of the data, so this is an attack on the
    implementation as much as on the structure.
    """
    if count < 1:
        raise ConfigError(f"{count} queries is not an attack")
    start = index._entry_point
    scores = squared_l2(corpus[start : start + 1], corpus).flatten()
    furthest = torch.argsort(scores, descending=True)[:count]
    return Attack(
        queries=corpus[furthest].clone(),
        name="far from the entry point",
        needs="the built graph",
    )


def the_worst_answered(
    index: Index, corpus: torch.Tensor, pool: torch.Tensor, count: int = 100, k: int = 10
) -> Attack:
    """The queries this index already answers worst, selected against the ground truth.

    Needs the truth, so no real adversary has it. Included because it is the upper bound on how
    bad a selected set can be, and because comparing the constructible attacks against it says
    how much of the available damage they actually find.
    """
    if count < 1:
        raise ConfigError(f"{count} queries is not an attack")
    if count > int(pool.shape[0]):
        raise ConfigError(f"cannot select {count} from a pool of {int(pool.shape[0])}")
    truth = search(pool, corpus, k=k)
    found, _ = index.search(pool, k=k)
    recalls = torch.zeros(int(pool.shape[0]))
    for row in range(int(pool.shape[0])):
        wanted = set(truth.identifiers[row].tolist())
        recalls[row] = len(wanted & set(found.identifiers[row].tolist())) / float(len(wanted))
    worst = torch.argsort(recalls)[:count]
    return Attack(
        queries=pool[worst].clone(), name="the worst answered", needs="the ground truth"
    )


def score(index: Index, corpus: torch.Tensor, attack: Attack, k: int = 10) -> dict:
    """Run one attack against one index and report what it did."""
    truth = search(attack.queries, corpus, k=k)
    found, stats = index.search(attack.queries, k=k)
    return {
        "attack": attack.name,
        "queries": attack.count,
        "recall": round(identifier_overlap(truth, found), 4),
        "distances": round(stats.distances_per_query, 1),
    }


def _setup(count: int = 4096, dimension: int = 32, queries: int = 200):
    """A corpus, a held out pool of ordinary queries, and their true answers."""
    corpus = gaussian(count=count, dimension=dimension)
    searched, probes = held_out(corpus, count=queries)
    return searched.vectors, probes, search(probes, searched.vectors, k=10)


def the_average_case_is_the_baseline() -> dict:
    """What an ordinary query sample gets, which every attack is measured against.

    Stated first because an attack that lowers the recall by two points has found nothing and
    one
    that lowers it by forty has found something, and neither statement means anything without
    the baseline next to it.
    """
    corpus, probes, truth = _setup()
    index = IVFIndex(32, partitions=64, probe=8)
    index.build(corpus)
    found, stats = index.search(probes, k=10)
    return {
        "queries": int(probes.shape[0]),
        "recall": round(identifier_overlap(truth, found), 4),
        "distances": round(stats.distances_per_query, 1),
    }


def sparse_regions_are_easier() -> dict:
    """What the cheapest attack costs, which turns out to be nothing.

    Queries from the sparsest tenth of the corpus score 0.631 against 0.552 for ordinary ones.
    Sparsity makes a query easier, not harder, and the construction is not at fault: the picked
    queries really do sit 1.23 times further from their tenth neighbour than an ordinary corpus
    vector does.

    The reason is the concentration effect the rest of the package keeps running into. In a
    sparse region the ten true neighbours are far apart from each other, so the ordering between
    them is unambiguous and an approximate index that reaches the neighbourhood at all gets them
    in the right order. In a dense region the tenth and the eleventh are nearly equidistant and
    any approximation swaps them.

    Which inverts the intuition an attacker would start from. The hard queries are in the
    crowded places, not the empty ones.
    """
    corpus, probes, _ = _setup()
    index = IVFIndex(32, partitions=64, probe=8)
    index.build(corpus)
    ordinary = score(index, corpus, Attack(probes, "ordinary", "nothing"))
    sparse = score(index, corpus, from_sparse_regions(corpus))
    return {
        "ordinary_recall": ordinary["recall"],
        "sparse_recall": sparse["recall"],
        "gap": round(ordinary["recall"] - sparse["recall"], 4),
        "easier_not_harder": sparse["recall"] > ordinary["recall"],
        "distances_are_similar": abs(sparse["distances"] - ordinary["distances"])
        < ordinary["distances"] * 0.5,
    }


def midpoints_between_partitions_are_harder() -> dict:
    """What knowing the fitted centroids buys an attacker.

    A query on the midpoint of two centroids has its neighbourhood split across both partitions
    by construction, so a probe count of p covers at most what p partitions hold and the split
    guarantees the miss. This is the most mechanically motivated attack here and it needs the
    index rather than only the corpus.
    """
    corpus, probes, _ = _setup()
    index = IVFIndex(32, partitions=64, probe=8)
    index.build(corpus)
    ordinary = score(index, corpus, Attack(probes, "ordinary", "nothing"))
    midpoints = score(index, corpus, between_partitions(index))
    return {
        "ordinary_recall": ordinary["recall"],
        "midpoint_recall": midpoints["recall"],
        "gap": round(ordinary["recall"] - midpoints["recall"], 4),
        "harder": midpoints["recall"] < ordinary["recall"],
    }


def the_selected_worst_is_the_upper_bound() -> dict:
    """How bad a query set can be made when the truth is available.

    Selecting the two hundred queries this index already answers worst, out of a pool of two
    hundred, is a degenerate case: it selects everything. Drawing from a larger pool is the real
    version and it is what this measures, so the number is the recall of the worst tenth rather
    than of the whole pool.
    """
    corpus, probes, _ = _setup(queries=1000)
    index = IVFIndex(32, partitions=64, probe=8)
    index.build(corpus)
    ordinary = score(index, corpus, Attack(probes, "ordinary", "nothing"))
    worst = score(index, corpus, the_worst_answered(index, corpus, probes, count=100))
    return {
        "pool": int(probes.shape[0]),
        "selected": 100,
        "ordinary_recall": ordinary["recall"],
        "worst_recall": worst["recall"],
        "gap": round(ordinary["recall"] - worst["recall"], 4),
    }


def the_constructible_attacks_find_some_of_it() -> dict:
    """How much of the available damage an attacker without the truth can reach.

    The gap the selected worst achieves is the ceiling, at 0.304. The midpoint attack reaches
    0.087 of it, which is 28.5 percent, and it is the only constructible attack here that
    reaches anything: the sparse region attack has a negative share because it makes queries
    easier.

    So an attacker holding the fitted centroids can find about a quarter of the damage that
    holding the ground truth would allow, and an attacker holding only the corpus can find none
    of it by this route. Which is a reasonable summary of the exposure: the index is worth
    protecting and the corpus alone is not enough to attack it.
    """
    sparse = sparse_regions_are_easier()
    midpoints = midpoints_between_partitions_are_harder()
    ceiling = the_selected_worst_is_the_upper_bound()
    return {
        "ceiling_gap": ceiling["gap"],
        "sparse_gap": sparse["gap"],
        "midpoint_gap": midpoints["gap"],
        "sparse_share": round(sparse["gap"] / max(ceiling["gap"], 1e-9), 3),
        "midpoint_share": round(midpoints["gap"] / max(ceiling["gap"], 1e-9), 3),
        "constructible_reaches_less": max(sparse["gap"], midpoints["gap"]) < ceiling["gap"],
    }


def every_structure_under_every_attack(k: int = 10) -> list[dict]:
    """The whole table, which is what says whether the exposure is structural.

    Five structures against the three attacks that do not need the truth, plus the ordinary
    baseline. If one structure's gap is much wider than the others then the exposure is a
    property of that structure rather than of the problem, which is the only thing that would
    make this actionable.
    """
    corpus, probes, _ = _setup()
    partitioned = IVFIndex(32, partitions=64, probe=8)
    partitioned.build(corpus)
    graph = GraphIndex(32, degree=16, ef=32)
    graph.build(corpus)
    attacks = [
        Attack(probes, "ordinary", "nothing"),
        from_sparse_regions(corpus),
        between_partitions(partitioned),
        far_from_the_entry_point(graph, corpus),
    ]
    rows = []
    for label, index in (
        ("ivf", partitioned),
        ("graph", graph),
        ("hnsw", _built(HNSWIndex(32, degree=16, ef=32), corpus)),
        ("forest", _built(ForestIndex(32, trees=8, leaf_size=64), corpus)),
        ("binary", _built(BinaryIndex(32, rerank=100), corpus)),
    ):
        for attack in attacks:
            row = score(index, corpus, attack, k=k)
            row["index"] = label
            rows.append(row)
    return rows


def _built(index: Index, corpus: torch.Tensor) -> Index:
    """Build an index and hand it back, so the table above reads as a list."""
    index.build(corpus)
    return index


def the_exposure_is_not_the_same_for_every_structure() -> dict:
    """Which structures have the widest gap between ordinary and adversarial queries.

    The number a deployment would act on. A structure whose adversarial recall is near its
    ordinary recall fails evenly and there is nothing to target; one with a wide gap has a
    findable set of queries it reliably fails.
    """
    rows = {}
    for row in every_structure_under_every_attack():
        rows.setdefault(row["index"], {})[row["attack"]] = row["recall"]
    gaps = {
        name: max(
            values["ordinary"] - values[attack] for attack in values if attack != "ordinary"
        )
        for name, values in rows.items()
    }
    return {
        "worst_gaps": {name: round(value, 4) for name, value in gaps.items()},
        "most_exposed": max(gaps, key=lambda name: gaps[name]),
        "least_exposed": min(gaps, key=lambda name: gaps[name]),
        "spread": round(max(gaps.values()) - min(gaps.values()), 4),
        "they_differ": max(gaps.values()) - min(gaps.values()) > 0.05,
    }


def each_attack_hits_its_own_structure_hardest() -> dict:
    """Whether an attack built against one structure transfers to the others.

    The midpoint attack is built from an inverted file's centroids and the entry point attack
    from a graph's start vertex, so each should hurt the structure it was built against most.
    Whether it also hurts the others says whether an attacker needs to know which index is
    running.
    """
    rows = {}
    for row in every_structure_under_every_attack():
        rows.setdefault(row["attack"], {})[row["index"]] = row["recall"]
    midpoint = rows["between partitions"]
    entry = rows["far from the entry point"]
    ordinary = rows["ordinary"]
    return {
        "midpoint_hurts_ivf": round(ordinary["ivf"] - midpoint["ivf"], 4),
        "midpoint_hurts_graph": round(ordinary["graph"] - midpoint["graph"], 4),
        "entry_hurts_graph": round(ordinary["graph"] - entry["graph"], 4),
        "entry_hurts_ivf": round(ordinary["ivf"] - entry["ivf"], 4),
        "midpoint_is_targeted": (ordinary["ivf"] - midpoint["ivf"])
        > (ordinary["graph"] - midpoint["graph"]),
        "entry_is_targeted": (ordinary["graph"] - entry["graph"])
        > (ordinary["ivf"] - entry["ivf"]),
    }


def a_higher_setting_closes_the_gap(
    probes: Sequence[int] = (2, 8, 32, 64),
) -> list[dict]:
    """Whether the obvious defence works, which is spending more per query.

    Turning the probe count up covers more partitions, so a query straddling two of them is
    eventually caught. The question is whether the gap closes faster than the ordinary recall
    saturates, because if it does not then the defence is just running the expensive index for
    everybody.
    """
    if not probes:
        raise ConfigError("there is nothing to sweep")
    corpus, ordinary_queries, _ = _setup()
    index = IVFIndex(32, partitions=64, probe=probes[0])
    index.build(corpus)
    attack = between_partitions(index)
    rows = []
    for probe in probes:
        index.probe = probe
        ordinary = score(index, corpus, Attack(ordinary_queries, "ordinary", "nothing"))
        hard = score(index, corpus, attack)
        rows.append(
            {
                "probe": probe,
                "ordinary_recall": ordinary["recall"],
                "adversarial_recall": hard["recall"],
                "gap": round(ordinary["recall"] - hard["recall"], 4),
                "distances": ordinary["distances"],
            }
        )
    return rows


def the_gap_closes_more_slowly_than_the_mean_rises() -> dict:
    """The shape of that defence, which decides whether it is one.

    If the gap shrinks as fast as the ordinary recall rises then spending more helps everybody
    equally and the adversarial case is not special. If the gap persists then the worst queries
    stay worst at every setting and the only fix is a different structure.
    """
    rows = {row["probe"]: row for row in a_higher_setting_closes_the_gap()}
    low, high = rows[2], rows[64]
    return {
        "gap_at_two": low["gap"],
        "gap_at_sixty_four": high["gap"],
        "ordinary_at_two": low["ordinary_recall"],
        "ordinary_at_sixty_four": high["ordinary_recall"],
        "gap_closes": high["gap"] < low["gap"],
        "closes_completely": high["gap"] < 0.05,
    }


def a_clustered_corpus_is_more_exposed() -> dict:
    """Whether the corpus shape changes how much an attacker can do.

    A clustered corpus has real gaps between clusters, so a query in one of those gaps is far
    from everything and its true neighbours are spread across several clusters. An isotropic
    corpus has no such places. The sparse region attack is the one that should notice.
    """
    rows = {}
    for label, corpus in (
        ("gaussian", gaussian(count=4096, dimension=32)),
        ("clustered", clustered(count=4096, dimension=32, clusters=16)),
    ):
        searched, probes = held_out(corpus, count=200)
        index = IVFIndex(32, partitions=64, probe=8)
        index.build(searched.vectors)
        ordinary = score(index, searched.vectors, Attack(probes, "ordinary", "nothing"))
        sparse = score(index, searched.vectors, from_sparse_regions(searched.vectors))
        rows[label] = {
            "ordinary": ordinary["recall"],
            "sparse": sparse["recall"],
            "gap": ordinary["recall"] - sparse["recall"],
        }
    return {
        "gaussian_ordinary": round(rows["gaussian"]["ordinary"], 4),
        "gaussian_sparse": round(rows["gaussian"]["sparse"], 4),
        "clustered_ordinary": round(rows["clustered"]["ordinary"], 4),
        "clustered_sparse": round(rows["clustered"]["sparse"], 4),
        "gaussian_gap": round(rows["gaussian"]["gap"], 4),
        "clustered_gap": round(rows["clustered"]["gap"], 4),
        "clustered_is_more_exposed": rows["clustered"]["gap"] > rows["gaussian"]["gap"],
    }


def the_attacks_do_not_change_the_cost() -> dict:
    """That a hard query is answered wrongly rather than slowly.

    Worth separating, because a structure that spent ten times as long on adversarial queries
    would be exposed to a different attack entirely, one on the service's latency rather than on
    its accuracy. Measured on the same index at the same settings across all four query sets.
    """
    rows = {}
    for row in every_structure_under_every_attack():
        rows.setdefault(row["index"], {})[row["attack"]] = row["distances"]
    spreads = {
        name: max(values.values()) / max(min(values.values()), 1e-9)
        for name, values in rows.items()
    }
    return {
        "cost_ratios": {name: round(value, 3) for name, value in spreads.items()},
        "widest": max(spreads, key=lambda name: spreads[name]),
        "widest_ratio": round(max(spreads.values()), 3),
        "cost_is_stable": max(spreads.values()) < 2.0,
    }


def an_attack_with_no_queries_is_refused() -> bool:
    """Whether an empty query set is caught at construction.

    It would score a recall of nothing over nothing, which several of the measurements here
    would report as a perfect or an undefined number depending on which one ran first.
    """
    try:
        Attack(torch.zeros(0, 8), "empty", "nothing")
    except DataError:
        return True
    return False


def a_rank_one_attack_is_refused() -> bool:
    """Whether an unbatched query set is caught."""
    try:
        Attack(torch.randn(8), "flat", "nothing")
    except DataError:
        return True
    return False


def a_zero_count_attack_is_refused() -> bool:
    """Whether asking a constructor for no queries is caught."""
    corpus = gaussian(count=512, dimension=8).vectors
    try:
        from_sparse_regions(corpus, count=0)
    except ConfigError:
        return True
    return False


def sampling_more_than_the_corpus_is_refused() -> bool:
    """Whether the sparse region search is asked to look at more than exists."""
    corpus = gaussian(count=128, dimension=8).vectors
    try:
        from_sparse_regions(corpus, count=10, sample=512)
    except ConfigError:
        return True
    return False


def a_midpoint_needs_two_partitions() -> bool:
    """Whether building a midpoint attack against a single partition is caught."""
    corpus = gaussian(count=512, dimension=8).vectors
    index = IVFIndex(8, partitions=1, probe=1)
    index.build(corpus)
    try:
        between_partitions(index)
    except ConfigError:
        return True
    return False


def selecting_more_than_the_pool_is_refused() -> bool:
    """Whether the worst answered selector is asked for more queries than it was given."""
    corpus, probes, _ = _setup(count=1024, queries=32)
    index = IVFIndex(32, partitions=16, probe=4)
    index.build(corpus)
    try:
        the_worst_answered(index, corpus, probes, count=200)
    except ConfigError:
        return True
    return False


def an_attack_reports_what_it_needs() -> dict:
    """That each attack says what an adversary would have to hold to build it.

    The difference between a threat and a curiosity. Sparse regions need the corpus, which is
    often public. Midpoints need the fitted index, which is usually not. The worst answered set
    needs the ground truth, which nobody has, and it is here as a bound rather than a threat.
    """
    corpus, probes, _ = _setup(count=2048, queries=64)
    index = IVFIndex(32, partitions=32, probe=4)
    index.build(corpus)
    graph = GraphIndex(32, degree=16, ef=32)
    graph.build(corpus)
    attacks = [
        from_sparse_regions(corpus, count=20),
        between_partitions(index, count=20),
        far_from_the_entry_point(graph, corpus, count=20),
        the_worst_answered(index, corpus, probes, count=20),
    ]
    return {
        "attacks": [attack.as_dict() for attack in attacks],
        "needs": sorted({attack.needs for attack in attacks}),
        "all_the_same_size": len({attack.count for attack in attacks}) == 1,
        "four_distinct_requirements": len({attack.needs for attack in attacks}) == 4,
    }


def the_attacks_are_deterministic() -> dict:
    """That an attack built twice from the same inputs is the same attack.

    Every measurement here compares an attack's recall against the baseline, so an attack that
    differed between runs would make the comparison unrepeatable and the numbers in these
    docstrings unverifiable.
    """
    corpus = gaussian(count=2048, dimension=16).vectors
    index = IVFIndex(16, partitions=32, probe=4)
    index.build(corpus)
    return {
        "sparse_identical": bool(
            torch.equal(
                from_sparse_regions(corpus, count=20).queries,
                from_sparse_regions(corpus, count=20).queries,
            )
        ),
        "midpoint_identical": bool(
            torch.equal(
                between_partitions(index, count=20).queries,
                between_partitions(index, count=20).queries,
            )
        ),
        "seeds_differ": not bool(
            torch.equal(
                from_sparse_regions(corpus, count=20, seed=0).queries,
                from_sparse_regions(corpus, count=20, seed=1).queries,
            )
        ),
    }


def the_sparse_queries_really_are_sparse() -> dict:
    """A check on the construction, which the recall numbers would not distinguish from noise.

    The selected queries should have a larger distance to their tenth neighbour than ordinary
    corpus vectors do. If they do not then the density estimate is wrong and everything measured
    on top of it is measuring something else.
    """
    corpus = gaussian(count=4096, dimension=32).vectors
    attack = from_sparse_regions(corpus, count=100)
    generator = torch.Generator().manual_seed(3)
    ordinary = corpus[torch.randperm(int(corpus.shape[0]), generator=generator)[:100]]
    sparse_tenth = torch.topk(
        squared_l2(attack.queries, corpus), k=11, dim=1, largest=False
    ).values[:, -1]
    ordinary_tenth = torch.topk(
        squared_l2(ordinary, corpus), k=11, dim=1, largest=False
    ).values[:, -1]
    return {
        "sparse_mean": round(float(sparse_tenth.mean()), 4),
        "ordinary_mean": round(float(ordinary_tenth.mean()), 4),
        "sparser": float(sparse_tenth.mean()) > float(ordinary_tenth.mean()),
        "ratio": round(float(sparse_tenth.mean()) / float(ordinary_tenth.mean()), 3),
    }


def the_midpoints_really_are_equidistant() -> dict:
    """The same check on the other construction.

    A midpoint between two centroids should be the same distance from both, so the ratio of the
    two centroid distances should be one. An error in the pairing would produce queries near one
    centroid, which is an easy query rather than a hard one, and the recall would come out high
    for a reason nothing else here would explain.
    """
    corpus = gaussian(count=4096, dimension=32).vectors
    index = IVFIndex(32, partitions=64, probe=8)
    index.build(corpus)
    attack = between_partitions(index, count=100)
    scores = squared_l2(attack.queries, index._centres).clamp_min(0.0).sqrt()
    nearest = torch.topk(scores, k=2, dim=1, largest=False).values
    ratio = nearest[:, 1] / nearest[:, 0].clamp_min(1e-9)
    return {
        "mean_ratio": round(float(ratio.mean()), 4),
        "worst_ratio": round(float(ratio.max()), 4),
        "near_one": float(ratio.mean()) < 1.2,
        "queries": attack.count,
    }


def compare_the_attacks_on_one_index() -> list[dict]:
    """All four attacks on one index, as the summary table.

    Ordered by how much an attacker has to know, so the table reads as a progression: the more
    the attacker holds, the worse they can make it, and the constructible attacks sit somewhere
    between the ordinary baseline and the bound the truth allows.
    """
    corpus, probes, _ = _setup(queries=1000)
    index = IVFIndex(32, partitions=64, probe=8)
    index.build(corpus)
    graph = GraphIndex(32, degree=16, ef=32)
    graph.build(corpus)
    attacks = [
        Attack(probes[:100], "ordinary", "nothing"),
        from_sparse_regions(corpus, count=100),
        between_partitions(index, count=100),
        far_from_the_entry_point(graph, corpus, count=100),
        the_worst_answered(index, corpus, probes, count=100),
    ]
    return [{**score(index, corpus, attack), "needs": attack.needs} for attack in attacks]

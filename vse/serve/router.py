from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

from vse.errors import ConfigError, DataError
from vse.index.base import Index
from vse.index.graph import GraphIndex
from vse.index.ivf import IVFIndex
from vse.quantize.binary import BinaryIndex
from vse.vectors.dataset import clustered, gaussian, held_out
from vse.vectors.exact import Neighbours, identifier_overlap, search
from vse.vectors.metric import squared_l2

# Sending a query to the cheapest index that can answer it, which is the last optimisation
# available once the individual structures have stopped improving.
#
# Every module before this one asks how to make one index better. This one takes several
# indexes as given and asks which query goes to which. Queries differ in difficulty: one
# sitting in the middle of a dense cluster is answered by almost anything, and
# a query on a boundary needs the expensive structure. If difficulty can be predicted before the
# search, the cheap index can take most of the traffic and the expensive one can take the rest.
#
# The whole thing turns on whether difficulty is predictable, and predictable cheaply. A
# predictor costing as much as the expensive index has saved nothing, so the candidates are
# computable from the query alone or from the cheap index's own output:
#
#   the distance to the nearest centroid, which says how central the query is
#   the gap between the first and second centroid, which says how near a boundary it is
#   the spread of the cheap index's own top ten scores, which says how confident it was
#   the cheap index's best score, which says whether it found anything close at all
#
# Only the last two use the cheap search, and it has to run anyway under any routing scheme that
# escalates rather than choosing up front, so they are free.
#
# Two designs. Choosing up front means running one index per query and needing a predictor
# working on the query alone. Escalating means running the cheap one always and the expensive
# one when the cheap result looks bad, which costs the cheap search on every query and gets a
# much better signal.
#
# The result spread signal had its sign backwards in the first version of this module. A wide
# gap between the first and tenth score was supposed to mean the index had run out of good
# candidates. It correlates with the cheap tier doing well, at minus 0.254: a wide spread
# means a genuinely close first result and progressively worse ones, which is ranking working,
# and a narrow one means all ten are equally mediocre. Escalating on the wrong sign sent the
# easy queries to the expensive tier and scored below escalating at random, which is the only
# thing that caught it.
#
# With the sign right the signal is real and small: 0.387 against 0.372 for a random split at
# the same cost. And it is not enough. Interpolating between always cheap and always expensive
# gives 0.5416 at the router's cost and the router reaches 0.5335, so on two settings of one
# structure, routing buys nothing that turning the probe count up would not.
#
# It was expected to have something to offer where the tiers are different structures, since
# the line between two structures is not reachable by any single setting of either. It does
# not: with a binary index in front of a graph the signal reaches 0.446 against 0.452 for a
# random split, because the spread of a binary index's reranked scores means something
# different from the spread of a partitioned index's and the signal was calibrated on the
# wrong tier.
#
# So the module ends with nothing to recommend, which is the result. Routing is the obvious
# next optimisation once the structures stop improving, and on these tiers it beats a random
# split by 0.015 on one corpus, ties on another, loses to turning the probe count up, and does
# not transfer between structures. Worth measuring precisely because it is obvious.


@dataclass
class Tier:
    """One index in a routing scheme, with what it costs to consult."""

    name: str
    index: Index
    cost: float

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {"name": self.name, "cost": round(self.cost, 1)}


@dataclass
class Routed:
    """What a routing scheme returned and where it sent things."""

    found: Neighbours
    sent: dict = field(default_factory=dict)
    distances: float = 0.0
    queries: int = 0

    @property
    def escalated(self) -> int:
        """How many queries reached the expensive tier."""
        return self.sent.get("expensive", 0)

    @property
    def escalation_rate(self) -> float:
        """What share of the traffic escalated."""
        if self.queries == 0:
            return 0.0
        return self.escalated / self.queries

    @property
    def distances_per_query(self) -> float:
        """What the scheme cost on average."""
        if self.queries == 0:
            return 0.0
        return self.distances / self.queries

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "queries": self.queries,
            "sent": dict(self.sent),
            "escalation_rate": round(self.escalation_rate, 4),
            "distances_per_query": round(self.distances_per_query, 1),
        }


def centrality(queries: torch.Tensor, centres: torch.Tensor) -> torch.Tensor:
    """How far each query is from its nearest centroid.

    Computable before any search, at the cost of one pass over the centroids, which every
    partitioned search does anyway. A query far from every centroid is in a sparse region and
    plausibly hard, which is the hypothesis this measures rather than assumes.
    """
    if queries.ndim != 2 or centres.ndim != 2:
        raise DataError("centrality takes a batch of queries and a set of centres")
    return squared_l2(queries, centres).min(dim=1).values.clamp_min(0.0).sqrt()


def boundary_closeness(queries: torch.Tensor, centres: torch.Tensor) -> torch.Tensor:
    """How close each query is to the boundary between its two nearest partitions.

    The gap between the first and second centroid distances, small when the query sits between
    two partitions and large when it is firmly inside one. A query on a boundary needs both
    partitions opened, so this is the most mechanically motivated of the four signals.
    """
    if int(centres.shape[0]) < 2:
        raise ConfigError("a boundary needs at least two partitions")
    scores = squared_l2(queries, centres).clamp_min(0.0).sqrt()
    nearest = torch.topk(scores, k=2, dim=1, largest=False).values
    return nearest[:, 1] - nearest[:, 0]


def result_spread(found: Neighbours) -> torch.Tensor:
    """How spread out the cheap index's own scores were.

    Free, because the search has already run. A tight cluster of scores means the index found a
    dense neighbourhood and the ordering inside it is what it is; a wide spread means the tenth
    result is much worse than the first, which usually means the index ran out of good
    candidates.
    """
    return found.scores[:, -1] - found.scores[:, 0]


def best_score(found: Neighbours) -> torch.Tensor:
    """How close the cheap index's best answer was.

    Also free. A query whose nearest returned vector is far away either lives in a sparse region
    or was answered badly, and the two are not distinguishable from this number alone, which is
    the limitation the measurements have to account for.
    """
    return found.scores[:, 0]


def always(tier: Tier, queries: torch.Tensor, k: int) -> Routed:
    """Send everything to one tier, which is the baseline any router has to beat."""
    found, stats = tier.index.search(queries, k=k)
    count = int(queries.shape[0])
    return Routed(
        found=found,
        sent={tier.name: count},
        distances=stats.distances_per_query * count,
        queries=count,
    )


def escalate(
    cheap: Tier,
    expensive: Tier,
    queries: torch.Tensor,
    k: int,
    signal,
    share: float = 0.2,
) -> Routed:
    """Run the cheap tier on everything and the expensive one on the worst looking share.

    The cheap search is not wasted when a query escalates: its result is discarded, but it
    had to run to produce the signal, and that cost is charged. So escalation is never
    cheaper than
    always cheap and the question is only whether the accuracy it buys is worth the difference.
    """
    if not 0.0 <= share <= 1.0:
        raise ConfigError(f"a share of {share} is not a share")
    count = int(queries.shape[0])
    cheap_found, cheap_stats = cheap.index.search(queries, k=k)
    scores = signal(queries, cheap_found)
    escalating = round(count * share)
    order = torch.argsort(scores, descending=True)
    chosen = order[:escalating]
    identifiers = cheap_found.identifiers.clone()
    result_scores = cheap_found.scores.clone()
    total = cheap_stats.distances_per_query * count
    if escalating:
        better, better_stats = expensive.index.search(queries[chosen], k=k)
        identifiers[chosen] = better.identifiers
        result_scores[chosen] = better.scores
        total += better_stats.distances_per_query * escalating
    return Routed(
        found=Neighbours(identifiers=identifiers, scores=result_scores),
        sent={cheap.name: count - escalating, "expensive": escalating},
        distances=total,
        queries=count,
    )


def choose_up_front(
    cheap: Tier,
    expensive: Tier,
    queries: torch.Tensor,
    k: int,
    signal,
    share: float = 0.2,
) -> Routed:
    """Decide from the query alone, so only one index runs per query.

    Cheaper than escalating by the cost of the cheap search on the escalated queries, and it
    needs a signal that does not depend on any search having run. Only the two centroid based
    signals qualify, which is the constraint that decides how good this design can be.
    """
    if not 0.0 <= share <= 1.0:
        raise ConfigError(f"a share of {share} is not a share")
    count = int(queries.shape[0])
    scores = signal(queries, None)
    escalating = round(count * share)
    order = torch.argsort(scores, descending=True)
    chosen = order[:escalating]
    mask = torch.zeros(count, dtype=torch.bool)
    mask[chosen] = True
    identifiers = torch.zeros(count, k, dtype=torch.long)
    result_scores = torch.zeros(count, k)
    total = 0.0
    if int((~mask).sum()):
        found, stats = cheap.index.search(queries[~mask], k=k)
        identifiers[~mask] = found.identifiers
        result_scores[~mask] = found.scores
        total += stats.distances_per_query * int((~mask).sum())
    if int(mask.sum()):
        found, stats = expensive.index.search(queries[mask], k=k)
        identifiers[mask] = found.identifiers
        result_scores[mask] = found.scores
        total += stats.distances_per_query * int(mask.sum())
    return Routed(
        found=Neighbours(identifiers=identifiers, scores=result_scores),
        sent={cheap.name: count - escalating, "expensive": escalating},
        distances=total,
        queries=count,
    )


def _setup(count: int = 4096, dimension: int = 32, queries: int = 200):
    """Two tiers over one corpus, with the truth and the centroids the signals need."""
    corpus = gaussian(count=count, dimension=dimension)
    searched, probes = held_out(corpus, count=queries)
    truth = search(probes, searched.vectors, k=10)
    cheap_index = IVFIndex(dimension, partitions=64, probe=2)
    cheap_index.build(searched.vectors)
    expensive_index = IVFIndex(dimension, partitions=64, probe=32)
    expensive_index.build(searched.vectors)
    return (
        Tier("cheap", cheap_index, 0.0),
        Tier("expensive", expensive_index, 0.0),
        probes,
        truth,
        cheap_index._centres,
    )


def _per_query_recall(truth: Neighbours, found: Neighbours) -> torch.Tensor:
    """Recall for each query separately, which is what difficulty means here."""
    rows = int(truth.identifiers.shape[0])
    hits = torch.zeros(rows)
    for row in range(rows):
        wanted = set(truth.identifiers[row].tolist())
        hits[row] = len(wanted & set(found.identifiers[row].tolist())) / float(len(wanted))
    return hits


def the_signals_are_measured_against_difficulty() -> list[dict]:
    """How well each candidate signal predicts which queries the cheap tier answers badly.

    Correlation between the signal and the cheap tier's per query recall, which is the only
    question that matters before any routing is built. A signal uncorrelated with difficulty
    cannot route, however cheap it is, and one strongly correlated can route even if it costs
    something.

    The two centroid signals are computable before any search. The two result signals need the
    cheap search to have run, which is free under escalation and impossible under choosing up
    front, so the table is also a statement about which design each signal can serve.
    """
    cheap, _, probes, truth, centres = _setup()
    found, _ = cheap.index.search(probes, k=10)
    difficulty = 1.0 - _per_query_recall(truth, found)
    rows = []
    for label, values, needs_search in (
        ("centrality", centrality(probes, centres), False),
        ("boundary closeness", -boundary_closeness(probes, centres), False),
        ("result spread", result_spread(found), True),
        ("best score", best_score(found), True),
    ):
        pair = torch.stack([values, difficulty])
        rows.append(
            {
                "signal": label,
                "correlation": round(float(torch.corrcoef(pair)[0, 1]), 4),
                "needs_the_cheap_search": needs_search,
            }
        )
    return rows


def the_best_signal_is_the_one_that_costs_a_search() -> dict:
    """Which signal predicts best, and whether it is one a cheap design can use."""
    rows = {row["signal"]: row for row in the_signals_are_measured_against_difficulty()}
    free = max(
        (row for row in rows.values() if not row["needs_the_cheap_search"]),
        key=lambda row: abs(row["correlation"]),
    )
    paid = max(
        (row for row in rows.values() if row["needs_the_cheap_search"]),
        key=lambda row: abs(row["correlation"]),
    )
    return {
        "best_free_signal": free["signal"],
        "best_free_correlation": free["correlation"],
        "best_paid_signal": paid["signal"],
        "best_paid_correlation": paid["correlation"],
        "paid_is_stronger": abs(paid["correlation"]) > abs(free["correlation"]),
    }


def _spread_signal(queries: torch.Tensor, found: Neighbours) -> torch.Tensor:  # noqa: ARG001
    """The result spread, negated, wrapped so escalate can call it uniformly.

    Negated because the correlation came out the other way from what was expected. A wide spread
    between the first and tenth score was supposed to mean the index had run out of good
    candidates; measured, it correlates with the cheap tier doing well, at minus 0.254. A wide
    spread means the index found a genuinely close first result and progressively worse ones,
    which is what ranking properly looks like. A narrow spread means all ten are equally
    mediocre and it never found the neighbourhood at all.

    The first version escalated on high spread and therefore sent the easy queries to the
    expensive tier, scoring below escalating at random. Getting a signal backwards is not a
    small error and the only thing that caught it was the random baseline.
    """
    return -result_spread(found)


def _best_signal(queries: torch.Tensor, found: Neighbours) -> torch.Tensor:  # noqa: ARG001
    """The best score, wrapped the same way."""
    return best_score(found)


def _random_signal(queries: torch.Tensor, found: Neighbours) -> torch.Tensor:  # noqa: ARG001
    """A signal with no information, which is what every router has to beat.

    Escalating a random fifth of the traffic still helps, because a fifth of the queries get the
    better index, so the comparison that matters is against this rather than against always
    cheap. A router that beats always cheap and not random has learned nothing.
    """
    generator = torch.Generator().manual_seed(17)
    return torch.rand(int(queries.shape[0]), generator=generator)


def the_two_fixed_policies_bracket_everything() -> dict:
    """Always cheap and always expensive, which any router sits between.

    No routing scheme can be more accurate than always expensive or cheaper than always cheap,
    so the useful question is where in that interval a scheme lands and at what cost. Reporting
    the brackets first makes the rest of the module readable.
    """
    cheap, expensive, probes, truth, _ = _setup()
    low = always(cheap, probes, 10)
    high = always(expensive, probes, 10)
    return {
        "cheap_recall": round(identifier_overlap(truth, low.found), 4),
        "cheap_distances": round(low.distances_per_query, 1),
        "expensive_recall": round(identifier_overlap(truth, high.found), 4),
        "expensive_distances": round(high.distances_per_query, 1),
        "recall_gap": round(
            identifier_overlap(truth, high.found) - identifier_overlap(truth, low.found), 4
        ),
        "cost_ratio": round(high.distances_per_query / low.distances_per_query, 2),
    }


def escalating_beats_a_random_fifth(share: float = 0.2) -> dict:
    """Whether the signal is doing anything at all.

    The only test that separates a router
    from a coin.

    Escalating a fifth of the traffic on the result spread against escalating a random fifth.
    Both cost the same, both send the same volume to the expensive tier, and the only difference
    is which queries. Any gap is what the signal is worth.
    """
    cheap, expensive, probes, truth, _ = _setup()
    informed = escalate(cheap, expensive, probes, 10, _spread_signal, share=share)
    arbitrary = escalate(cheap, expensive, probes, 10, _random_signal, share=share)
    return {
        "share": share,
        "informed_recall": round(identifier_overlap(truth, informed.found), 4),
        "random_recall": round(identifier_overlap(truth, arbitrary.found), 4),
        "informed_distances": round(informed.distances_per_query, 1),
        "random_distances": round(arbitrary.distances_per_query, 1),
        "same_cost": abs(informed.distances_per_query - arbitrary.distances_per_query) < 1.0,
        "signal_is_worth_something": identifier_overlap(truth, informed.found)
        > identifier_overlap(truth, arbitrary.found),
    }


def the_escalation_share_trades_cost_for_recall(
    shares: Sequence[float] = (0.0, 0.1, 0.2, 0.4, 0.8, 1.0),
) -> list[dict]:
    """The router's knob, which is what fraction of traffic to escalate.

    At zero it is always cheap plus a wasted signal computation, at one it is always expensive
    plus the cheap search on every query. Both ends are worse than the corresponding fixed
    policy by exactly the overhead, which is the honest way to show what routing costs before
    showing what it buys.
    """
    if not shares:
        raise ConfigError("there is nothing to sweep")
    cheap, expensive, probes, truth, _ = _setup()
    rows = []
    for share in shares:
        routed = escalate(cheap, expensive, probes, 10, _spread_signal, share=share)
        rows.append(
            {
                "share": share,
                "recall": round(identifier_overlap(truth, routed.found), 4),
                "distances": round(routed.distances_per_query, 1),
                "escalated": routed.escalated,
            }
        )
    return rows


def routing_does_not_beat_tuning_the_cheap_index() -> dict:
    """Whether a router beats simply spending the same budget on a larger probe count.

    The test that decides whether routing is worth building, and it comes out negative.
    Interpolating between always cheap and always expensive gives 0.5416 at the router's
    cost of 1024 distances. The router reaches 0.5335, which is 0.008 below the line and well
    inside the standard error.

    So on this pair of tiers, escalating a share of the traffic on the best available signal
    buys nothing that turning the probe count up would not. The signal is real, it beats a
    random split by 0.015, and it is not strong enough to beat the alternative of not routing
    at all.

    That is a useful thing to have measured rather than assumed, because the two tiers here
    are settings of one structure and the interpolation between them is therefore something a
    single index can actually achieve. Where the tiers are different structures the line is
    not reachable by any single setting, and routing_across_different_structures is the case
    where the same scheme has something to offer.
    """
    brackets = the_two_fixed_policies_bracket_everything()
    rows = {row["share"]: row for row in the_escalation_share_trades_cost_for_recall()}
    middle = rows[0.4]
    span = brackets["expensive_distances"] - brackets["cheap_distances"]
    position = (middle["distances"] - brackets["cheap_distances"]) / max(span, 1e-9)
    line = brackets["cheap_recall"] + position * brackets["recall_gap"]
    return {
        "router_cost": middle["distances"],
        "router_recall": middle["recall"],
        "the_line_at_that_cost": round(line, 4),
        "above_the_line": middle["recall"] > line,
        "inside_the_noise": abs(middle["recall"] - line) < 0.035,
        "margin": round(middle["recall"] - line, 4),
    }


def choosing_up_front_is_cheaper_and_worse(share: float = 0.2) -> dict:
    """The other design, which trades signal quality for the cheap search it does not run.

    Choosing up front runs one index per query, so it saves the cheap search on every escalated
    query. What it gives up is the two signals that need a search to have happened, leaving only
    the centroid based ones. The comparison is at the same escalation share so the volumes
    match.
    """
    cheap, expensive, probes, truth, centres = _setup()

    def up_front(queries: torch.Tensor, found) -> torch.Tensor:  # noqa: ARG001
        return centrality(queries, centres)

    front = choose_up_front(cheap, expensive, probes, 10, up_front, share=share)
    after = escalate(cheap, expensive, probes, 10, _spread_signal, share=share)
    return {
        "share": share,
        "up_front_recall": round(identifier_overlap(truth, front.found), 4),
        "up_front_distances": round(front.distances_per_query, 1),
        "escalating_recall": round(identifier_overlap(truth, after.found), 4),
        "escalating_distances": round(after.distances_per_query, 1),
        "up_front_is_cheaper": front.distances_per_query < after.distances_per_query,
        "escalating_is_more_accurate": identifier_overlap(truth, after.found)
        > identifier_overlap(truth, front.found),
    }


def structure_does_not_make_difficulty_more_predictable(share: float = 0.2) -> dict:
    """Whether difficulty is easier to predict when the corpus has structure.

    The expectation was yes: a clustered corpus has queries plainly inside a cluster and
    queries plainly between them, so a signal has something to see, where an isotropic corpus
    offers no such distinction.

    Measured, the signal is worth 0.015 on the gaussian corpus and minus 0.013 on the
    clustered one, so it helps slightly on the corpus with no structure and hurts slightly on
    the one with plenty. Both numbers are inside the standard error and the honest reading is
    that the signal is worth nothing on either.
    """
    rows = {}
    for label, corpus in (
        ("gaussian", gaussian(count=4096, dimension=32)),
        ("clustered", clustered(count=4096, dimension=32, clusters=16)),
    ):
        searched, probes = held_out(corpus, count=200)
        truth = search(probes, searched.vectors, k=10)
        cheap_index = IVFIndex(32, partitions=64, probe=2)
        cheap_index.build(searched.vectors)
        expensive_index = IVFIndex(32, partitions=64, probe=32)
        expensive_index.build(searched.vectors)
        cheap = Tier("cheap", cheap_index, 0.0)
        expensive = Tier("expensive", expensive_index, 0.0)
        informed = escalate(cheap, expensive, probes, 10, _spread_signal, share=share)
        arbitrary = escalate(cheap, expensive, probes, 10, _random_signal, share=share)
        rows[label] = {
            "informed": identifier_overlap(truth, informed.found),
            "random": identifier_overlap(truth, arbitrary.found),
        }
    return {
        "gaussian_gain": round(rows["gaussian"]["informed"] - rows["gaussian"]["random"], 4),
        "clustered_gain": round(rows["clustered"]["informed"] - rows["clustered"]["random"], 4),
        "gaussian_informed": round(rows["gaussian"]["informed"], 4),
        "clustered_informed": round(rows["clustered"]["informed"], 4),
        "both_inside_the_noise": abs(rows["gaussian"]["informed"] - rows["gaussian"]["random"])
        < 0.035
        and abs(rows["clustered"]["informed"] - rows["clustered"]["random"]) < 0.035,
    }


def routing_across_different_structures(share: float = 0.2) -> dict:
    """Two tiers that are different structures rather than two settings of one.

    A binary index as the cheap tier and a graph as the expensive one, which is what a real
    deployment looks like: a compressed scan in front of a traversal. This was the case the
    module expected to recommend, since the line between two different structures is not
    reachable by any single setting of either.

    The signal does not survive it. Informed escalation reaches 0.446 against 0.452 for a
    random split, so it is behind by six thousandths, which is nothing and is on the wrong
    side of nothing. The scores the two tiers produce are not comparable, and the spread of a
    binary index's reranked scores says something different from the spread of a partitioned
    index's, so the signal was calibrated on a tier it is no longer reading.

    Which leaves the module with no case to recommend. Escalation on the best signal measured
    beats a random split by 0.015 on one corpus, ties on another, does not beat tuning the
    cheap index, and does not transfer between structures. That is a negative result and it
    is worth the module: routing is the obvious next optimisation and on these tiers it is
    not one.
    """
    corpus = gaussian(count=4096, dimension=32)
    searched, probes = held_out(corpus, count=200)
    truth = search(probes, searched.vectors, k=10)
    cheap_index = BinaryIndex(32, rerank=50)
    cheap_index.build(searched.vectors)
    expensive_index = GraphIndex(32, degree=16, ef=64)
    expensive_index.build(searched.vectors)
    cheap = Tier("cheap", cheap_index, 0.0)
    expensive = Tier("expensive", expensive_index, 0.0)
    informed = escalate(cheap, expensive, probes, 10, _spread_signal, share=share)
    arbitrary = escalate(cheap, expensive, probes, 10, _random_signal, share=share)
    low = always(cheap, probes, 10)
    high = always(expensive, probes, 10)
    return {
        "cheap_recall": round(identifier_overlap(truth, low.found), 4),
        "expensive_recall": round(identifier_overlap(truth, high.found), 4),
        "informed_recall": round(identifier_overlap(truth, informed.found), 4),
        "random_recall": round(identifier_overlap(truth, arbitrary.found), 4),
        "informed_distances": round(informed.distances_per_query, 1),
        "signal_transfers": identifier_overlap(truth, informed.found)
        > identifier_overlap(truth, arbitrary.found),
        "gap": round(
            identifier_overlap(truth, informed.found)
            - identifier_overlap(truth, arbitrary.found),
            4,
        ),
    }


def the_escalated_queries_are_the_hard_ones(share: float = 0.2) -> dict:
    """That the router is picking the queries it was meant to pick.

    Comparing the cheap tier's recall on the escalated fifth against its recall on the rest. If
    the escalated group is not worse under the cheap tier then the signal selected arbitrarily,
    and every accuracy gain downstream is the gain from giving a random fifth a better index.
    """
    cheap, _, probes, truth, _ = _setup()
    found, _ = cheap.index.search(probes, k=10)
    recalls = _per_query_recall(truth, found)
    scores = _spread_signal(probes, found)
    count = int(probes.shape[0])
    escalating = round(count * share)
    order = torch.argsort(scores, descending=True)
    chosen = order[:escalating]
    rest = order[escalating:]
    return {
        "share": share,
        "escalated_recall_under_the_cheap_tier": round(float(recalls[chosen].mean()), 4),
        "kept_recall_under_the_cheap_tier": round(float(recalls[rest].mean()), 4),
        "the_escalated_are_worse": float(recalls[chosen].mean()) < float(recalls[rest].mean()),
        "gap": round(float(recalls[rest].mean()) - float(recalls[chosen].mean()), 4),
    }


def a_share_of_zero_is_always_cheap_plus_overhead() -> dict:
    """That the router degenerates correctly at its endpoints.

    At a share of zero every query is answered by the cheap tier and the result should be
    identical to always cheap, because the signal is computed and then used to escalate nothing.
    At a share of one every query is answered by the expensive tier and the recall should match
    always expensive while the cost is higher by the cheap search.
    """
    cheap, expensive, probes, _, _ = _setup()
    low = always(cheap, probes, 10)
    high = always(expensive, probes, 10)
    none = escalate(cheap, expensive, probes, 10, _spread_signal, share=0.0)
    everything = escalate(cheap, expensive, probes, 10, _spread_signal, share=1.0)
    return {
        "zero_matches_cheap": bool(torch.equal(none.found.identifiers, low.found.identifiers)),
        "one_matches_expensive": bool(
            torch.equal(everything.found.identifiers, high.found.identifiers)
        ),
        "zero_costs_the_same": abs(none.distances_per_query - low.distances_per_query) < 1e-6,
        "one_costs_more": everything.distances_per_query > high.distances_per_query,
        "overhead": round(everything.distances_per_query - high.distances_per_query, 1),
    }


def the_routed_result_is_well_formed(share: float = 0.3) -> dict:
    """That mixing two indexes' answers produces a result nothing downstream can trip over.

    The escalated rows come from one index and the rest from another, so the shapes have to line
    up and the scores have to stay sorted. A router that returned the expensive index's scores
    for the cheap index's identifiers would be the exact failure verify/differential.py's score
    agreement rule exists to catch, and mixing two sources is where it would happen.
    """
    cheap, expensive, probes, _, _ = _setup()
    routed = escalate(cheap, expensive, probes, 10, _spread_signal, share=share)
    identifiers = routed.found.identifiers
    distinct = all(
        int(torch.unique(identifiers[row]).numel()) == 10
        for row in range(int(identifiers.shape[0]))
    )
    sorted_rows = bool(
        torch.all(routed.found.scores[:, 1:] >= routed.found.scores[:, :-1] - 1e-5)
    )
    return {
        "shape": tuple(identifiers.shape),
        "distinct": distinct,
        "sorted": sorted_rows,
        "escalated": routed.escalated,
        "accounted_for": sum(routed.sent.values()) == routed.queries,
    }


def a_share_outside_the_unit_interval_is_refused() -> bool:
    """Whether escalating more than everything is caught."""
    cheap, expensive, probes, _, _ = _setup(count=512, queries=16)
    try:
        escalate(cheap, expensive, probes, 5, _spread_signal, share=1.5)
    except ConfigError:
        return True
    return False


def a_negative_share_is_refused() -> bool:
    """The same at the other end."""
    cheap, expensive, probes, _, _ = _setup(count=512, queries=16)
    try:
        escalate(cheap, expensive, probes, 5, _spread_signal, share=-0.1)
    except ConfigError:
        return True
    return False


def an_up_front_share_outside_the_interval_is_refused() -> bool:
    """And on the other design."""
    cheap, expensive, probes, _, centres = _setup(count=512, queries=16)
    try:
        choose_up_front(
            cheap,
            expensive,
            probes,
            5,
            lambda queries, found: centrality(queries, centres),  # noqa: ARG005
            share=2.0,
        )
    except ConfigError:
        return True
    return False


def a_boundary_signal_needs_two_partitions() -> bool:
    """Whether measuring a boundary with one partition is caught.

    There is no boundary with one partition, and the topk would fail with a message about
    dimensions rather than about the thing that is wrong.
    """
    try:
        boundary_closeness(torch.randn(4, 8), torch.randn(1, 8))
    except ConfigError:
        return True
    return False


def a_rank_one_query_batch_is_refused() -> bool:
    """Whether an unbatched query reaches the centrality signal."""
    try:
        centrality(torch.randn(8), torch.randn(4, 8))
    except DataError:
        return True
    return False


def an_empty_routed_result_divides_safely() -> dict:
    """That the reporting handles a batch of nothing."""
    routed = Routed(found=Neighbours(torch.zeros(0, 10, dtype=torch.long), torch.zeros(0, 10)))
    return {
        "queries": routed.queries,
        "escalation_rate": routed.escalation_rate,
        "distances_per_query": routed.distances_per_query,
        "safe": routed.escalation_rate == 0.0 and routed.distances_per_query == 0.0,
    }


def a_routed_result_serialises() -> dict:
    """That the summary a service would log has what it needs."""
    routed = Routed(
        found=Neighbours(torch.zeros(10, 5, dtype=torch.long), torch.zeros(10, 5)),
        sent={"cheap": 8, "expensive": 2},
        distances=1000.0,
        queries=10,
    )
    row = routed.as_dict()
    return {
        "escalation_rate": row["escalation_rate"],
        "distances_per_query": row["distances_per_query"],
        "sent": row["sent"],
        "has_everything": set(row)
        == {"queries", "sent", "escalation_rate", "distances_per_query"},
    }


def a_tier_serialises() -> dict:
    """That a tier reports its name and cost."""
    corpus = gaussian(count=256, dimension=8)
    index = IVFIndex(8, partitions=8, probe=2)
    index.build(corpus.vectors)
    return Tier("cheap", index, 42.0).as_dict()


def compare_every_routing_scheme(share: float = 0.2) -> list[dict]:
    """The four schemes and the two brackets, as one table.

    Always cheap, always expensive, escalating on a signal, escalating at random, and choosing
    up
    front. Six rows, and the useful reading is the two comparisons inside it: informed against
    random says whether the signal works, and escalating against choosing up front says whether
    the extra search is worth what it buys.
    """
    cheap, expensive, probes, truth, centres = _setup()
    rows = [
        ("always cheap", always(cheap, probes, 10)),
        ("always expensive", always(expensive, probes, 10)),
        (
            "escalate on spread",
            escalate(cheap, expensive, probes, 10, _spread_signal, share=share),
        ),
        (
            "escalate on best score",
            escalate(cheap, expensive, probes, 10, _best_signal, share=share),
        ),
        (
            "escalate at random",
            escalate(cheap, expensive, probes, 10, _random_signal, share=share),
        ),
        (
            "choose up front",
            choose_up_front(
                cheap,
                expensive,
                probes,
                10,
                lambda queries, found: centrality(queries, centres),  # noqa: ARG005
                share=share,
            ),
        ),
    ]
    return [
        {
            "scheme": name,
            "recall": round(identifier_overlap(truth, routed.found), 4),
            "distances": round(routed.distances_per_query, 1),
            "escalated": routed.escalated,
        }
        for name, routed in rows
    ]

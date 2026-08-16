from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from vse.errors import ConfigError, DataError
from vse.index.base import SearchStats
from vse.index.graph import GraphIndex
from vse.index.ivf import IVFIndex
from vse.vectors.dataset import Corpus, clustered, gaussian, held_out
from vse.vectors.exact import Neighbours, identifier_overlap
from vse.vectors.metric import squared_l2

# Searching with a condition attached, which is the request every real system gets and almost
# no index answers well.
#
# The query is not find the nearest ten, it is find the nearest ten that are in stock, or
# owned by this tenant, or published after some date. There are three ways to do it and which
# one is right depends on how many vectors pass the condition, which is the selectivity.
#
# Search then filter runs the ordinary index, throws away whatever fails, and hopes enough is
# left. Exact at half selectivity, eighty six percent at a tenth, twenty percent at two percent
# and five at half of one. That is a cliff rather than a slope, and the way it fails is worse
# than the number suggests: the result comes back short rather than wrong, so a caller that does
# not count what it got sees a shorter list and concludes there was less to find.
#
# Filter then search restricts the corpus first and scans what is left. It is exact at every
# selectivity and it costs the size of the matching set, so it is cheap when the condition is
# tight and is a full corpus scan when it is loose.
#
# Filtering inside the traversal is the one people want: walk the index and skip non matching
# candidates as they come up. It is exact and cheaper than either extreme in the middle, and it
# has a failure the other two do not: on a graph index the matching vectors may not be connected
# to each other through matching vectors, so the walk gets stranded and the recall collapses
# without any indication that it has.
#
# There is no crossover between the two simple strategies as measured here, which is worth
# stating because I set out expecting to find one. Over a flat scan the cheap strategy always
# costs the whole corpus and restricting first never costs more than that, so restricting wins
# at every selectivity on both accuracy and cost. A crossover needs the unfiltered path to be an
# approximate index examining far fewer than the corpus, and then it sits where the matching set
# equals whatever that index would have examined.
#
# The correlated condition is the other thing I had backwards. A filter passing one region of
# the space is much harder than one passing a random sample at the same selectivity, fifteen
# percent recall against fifty, because a query outside the matching region finds none of its
# neighbours matching and those queries dominate the average. Real filters are usually
# correlated, so the random measurement here is the optimistic one, not the pessimistic one.


@dataclass(frozen=True)
class Predicate:
    """A mask over the corpus and the share of it that passes."""

    mask: torch.Tensor

    def __post_init__(self) -> None:
        if self.mask.ndim != 1:
            raise DataError(f"a predicate is a vector of flags, got rank {self.mask.ndim}")
        if self.mask.dtype != torch.bool:
            raise DataError(f"a predicate is boolean, got {self.mask.dtype}")
        if int(self.mask.sum()) == 0:
            raise ConfigError("a predicate that matches nothing has no answer")

    @property
    def size(self) -> int:
        """How many vectors it was built over."""
        return int(self.mask.shape[0])

    @property
    def matching(self) -> int:
        """How many pass."""
        return int(self.mask.sum())

    @property
    def selectivity(self) -> float:
        """The share that pass. Small means a tight condition."""
        return self.matching / self.size

    def rows(self) -> torch.Tensor:
        """The indices that pass."""
        return torch.nonzero(self.mask, as_tuple=False).flatten()

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "size": self.size,
            "matching": self.matching,
            "selectivity": round(self.selectivity, 5),
        }


def random_predicate(count: int, selectivity: float = 0.1, seed: int = 0) -> Predicate:
    """A condition that passes a given share of the corpus, chosen at random.

    Random membership is the easy case and it is worth being explicit that it is easy. A real
    condition correlates with the vectors, because whatever attribute it tests was probably
    related to whatever the embedding encodes, and a correlated condition behaves quite
    differently. The correlated fixture below is the one that matters.
    """
    if not 0 < selectivity <= 1:
        raise ConfigError(f"a selectivity of {selectivity} is not a share")
    if count < 1:
        raise ConfigError(f"a corpus of {count} has nothing to filter")
    generator = torch.Generator().manual_seed(seed)
    keep = max(1, round(count * selectivity))
    chosen = torch.randperm(count, generator=generator)[:keep]
    mask = torch.zeros(count, dtype=torch.bool)
    mask[chosen] = True
    return Predicate(mask=mask)


def clustered_predicate(corpus: Corpus, selectivity: float = 0.1, seed: int = 0) -> Predicate:
    """A condition that passes vectors from one region of the space.

    What a real attribute filter usually looks like. Documents from one customer, or one
    language, or one time window, are not scattered uniformly through an embedding space, they
    sit together. This picks a random centre and passes its nearest neighbours, which makes the
    matching set a connected region rather than a random sample.
    """
    if not 0 < selectivity <= 1:
        raise ConfigError(f"a selectivity of {selectivity} is not a share")
    generator = torch.Generator().manual_seed(seed)
    pivot = int(torch.randint(0, corpus.count, (1,), generator=generator))
    scores = squared_l2(corpus.vectors[pivot : pivot + 1], corpus.vectors).flatten()
    keep = max(1, round(corpus.count * selectivity))
    chosen = torch.topk(scores, k=keep, largest=False).indices
    mask = torch.zeros(corpus.count, dtype=torch.bool)
    mask[chosen] = True
    return Predicate(mask=mask)


def exact_filtered(
    queries: torch.Tensor, corpus: torch.Tensor, predicate: Predicate, k: int = 10
) -> Neighbours:
    """The right answer: the k nearest among the matching vectors.

    Every strategy below is measured against this. It is a scan of the matching set, which is
    also the filter then search strategy, so that one is exact by construction and the only
    thing worth measuring about it is what it costs.
    """
    rows = predicate.rows()
    if k > int(rows.numel()):
        raise ConfigError(f"asking for {k} neighbours from {int(rows.numel())} matches")
    scores = squared_l2(queries, corpus[rows])
    found = torch.topk(scores, k=k, dim=1, largest=False)
    return Neighbours(identifiers=rows[found.indices], scores=found.values)


def search_then_filter(
    queries: torch.Tensor,
    corpus: torch.Tensor,
    predicate: Predicate,
    k: int = 10,
    over_fetch: int = 10,
) -> tuple[Neighbours, SearchStats]:
    """Retrieve more than needed, then drop what fails the condition.

    The strategy that costs nothing to build and fails silently. Padding the result with the
    over fetch factor buys some headroom and cannot fix the problem: at a selectivity of one
    percent a top hundred contains one matching vector on average, so no constant factor is
    enough and the shortfall shows up as short results rather than as an error.
    """
    if over_fetch < 1:
        raise ConfigError(f"an over fetch of {over_fetch} fetches less than asked")
    width = min(k * over_fetch, corpus.shape[0])
    stats = SearchStats(queries=int(queries.shape[0]))
    stats.charge(corpus.shape[0] * int(queries.shape[0]))
    scores = squared_l2(queries, corpus)
    rough = torch.topk(scores, k=width, dim=1, largest=False).indices
    identifiers = torch.zeros(queries.shape[0], k, dtype=torch.long)
    kept = torch.zeros(queries.shape[0], k)
    for row in range(int(queries.shape[0])):
        passing = [int(other) for other in rough[row] if bool(predicate.mask[int(other)])]
        for slot, other in enumerate(passing[:k]):
            identifiers[row, slot] = other
            kept[row, slot] = scores[row, other]
    return Neighbours(identifiers=identifiers, scores=kept), stats


def filter_then_search(
    queries: torch.Tensor, corpus: torch.Tensor, predicate: Predicate, k: int = 10
) -> tuple[Neighbours, SearchStats]:
    """Restrict the corpus, then scan it. Exact at every selectivity."""
    rows = predicate.rows()
    stats = SearchStats(queries=int(queries.shape[0]))
    stats.charge(int(rows.numel()) * int(queries.shape[0]))
    stats.visit(int(rows.numel()) * int(queries.shape[0]))
    return exact_filtered(queries, corpus, predicate, k=k), stats


def filter_during_partitions(
    queries: torch.Tensor,
    corpus: torch.Tensor,
    predicate: Predicate,
    index: IVFIndex,
    k: int = 10,
    probe: int = 8,
) -> tuple[Neighbours, SearchStats]:
    """Open partitions and score only the matching vectors inside them.

    The version that works. A partitioned index has no connectivity to lose: the posting lists
    are just lists, so skipping non matching entries costs nothing and breaks nothing. Whether
    the answer is right still depends on whether the matching vectors are in the partitions that
    were opened, which is the ordinary recall question rather than a new one.
    """
    stats = SearchStats(queries=int(queries.shape[0]))
    stats.charge(index.partitions * int(queries.shape[0]))
    centre_scores = squared_l2(queries, index._centres)
    chosen = torch.topk(centre_scores, k=probe, dim=1, largest=False).indices
    identifiers = torch.zeros(queries.shape[0], k, dtype=torch.long)
    kept = torch.zeros(queries.shape[0], k)
    for row in range(int(queries.shape[0])):
        rows = torch.cat([index._lists[int(part)] for part in chosen[row]])
        rows = rows[predicate.mask[rows]]
        stats.hop(probe)
        stats.charge(int(rows.numel()))
        stats.visit(int(rows.numel()))
        if rows.numel() == 0:
            continue
        scores = squared_l2(queries[row : row + 1], corpus[rows]).flatten()
        keep = min(k, int(rows.numel()))
        best = torch.topk(scores, k=keep, largest=False)
        identifiers[row, :keep] = rows[best.indices]
        kept[row, :keep] = best.values
    return Neighbours(identifiers=identifiers, scores=kept), stats


def filter_during_graph(
    queries: torch.Tensor,
    corpus: torch.Tensor,
    predicate: Predicate,
    index: GraphIndex,
    k: int = 10,
    ef: int = 64,
) -> tuple[Neighbours, SearchStats]:
    """Walk the graph and keep only matching vertices in the result.

    The version that has a failure mode. The walk still traverses through non matching vertices,
    which is necessary, and the result keeps only the matching ones, so at a tight selectivity
    the beam fills with vertices that will all be discarded and the search terminates having
    found almost nothing. Nothing errors. The result is just short and wrong.
    """
    found, stats = index.search(queries, k=min(ef, index.size), ef=ef)
    identifiers = torch.zeros(queries.shape[0], k, dtype=torch.long)
    kept = torch.zeros(queries.shape[0], k)
    for row in range(int(queries.shape[0])):
        passing = [
            int(other) for other in found.identifiers[row] if bool(predicate.mask[int(other)])
        ]
        for slot, other in enumerate(passing[:k]):
            identifiers[row, slot] = other
            kept[row, slot] = float(
                squared_l2(queries[row : row + 1], corpus[other : other + 1])
            )
    return Neighbours(identifiers=identifiers, scores=kept), stats


def search_then_filter_returns_nothing(
    selectivities: Sequence[float] = (0.5, 0.1, 0.02, 0.005),
) -> list[dict]:
    """How the cheap strategy degrades as the condition tightens.

    Not gracefully. At half selectivity a top hundred contains fifty matches and the answer is
    exact. At half a percent it contains half a match on average, so most queries come back with
    nothing at all and the ones that do not come back with whatever happened to be there. The
    result is short rather than wrong, which is worse: a caller that does not count what it got
    sees an empty list and concludes there was nothing to find.
    """
    if not selectivities:
        raise ConfigError("there is nothing to sweep")
    corpus = gaussian(count=2048, dimension=32)
    searched, probes = held_out(corpus, count=64)
    rows = []
    for share in selectivities:
        predicate = random_predicate(searched.count, selectivity=share)
        truth = exact_filtered(probes, searched.vectors, predicate, k=10)
        found, stats = search_then_filter(probes, searched.vectors, predicate, k=10)
        filled = float((found.identifiers != 0).any(dim=0).float().mean())
        rows.append(
            {
                "selectivity": share,
                "matching": predicate.matching,
                "recall": round(identifier_overlap(truth, found), 4),
                "slots_filled": round(filled, 3),
                "distances_per_query": round(stats.distances_per_query, 1),
            }
        )
    return rows


def the_cheap_strategy_collapses() -> dict:
    """The two ends of that sweep, and where the collapse starts.

    Between ten percent and two percent. Above it the over fetch is enough and the answer is
    exact; below it there are not a hundred matching vectors anywhere near the query and no
    fixed multiple of k can find them. The failure is a cliff rather than a slope, which is why
    a system that works in testing on a loose filter breaks the first time somebody applies a
    tight one.
    """
    rows = {row["selectivity"]: row for row in search_then_filter_returns_nothing()}
    return {
        "at_half": rows[0.5]["recall"],
        "at_a_tenth": rows[0.1]["recall"],
        "at_two_percent": rows[0.02]["recall"],
        "at_half_a_percent": rows[0.005]["recall"],
        "collapsed": rows[0.005]["recall"] < rows[0.5]["recall"] / 2,
        "cost_unchanged": rows[0.005]["distances_per_query"]
        == rows[0.5]["distances_per_query"],
    }


def filter_then_search_is_exact_and_scales_the_other_way(
    selectivities: Sequence[float] = (0.5, 0.1, 0.02, 0.005),
) -> list[dict]:
    """The strategy that never fails, and what it costs.

    Exact at every selectivity by construction, since it scans exactly the matching set. Its
    cost is the size of that set, so it gets cheaper as the condition tightens, which is the
    opposite of the other strategy in every respect. Between them they cover the whole range and
    neither covers the middle well.
    """
    if not selectivities:
        raise ConfigError("there is nothing to sweep")
    corpus = gaussian(count=2048, dimension=32)
    searched, probes = held_out(corpus, count=64)
    rows = []
    for share in selectivities:
        predicate = random_predicate(searched.count, selectivity=share)
        truth = exact_filtered(probes, searched.vectors, predicate, k=10)
        found, stats = filter_then_search(probes, searched.vectors, predicate, k=10)
        rows.append(
            {
                "selectivity": share,
                "recall": round(identifier_overlap(truth, found), 4),
                "distances_per_query": round(stats.distances_per_query, 1),
                "share_of_the_corpus": round(stats.distances_per_query / searched.count, 4),
            }
        )
    return rows


def the_cheap_strategy_is_never_cheaper_here(selectivity: float = 0.05) -> dict:
    """Where the cheap one stops being cheaper, which turns out to be nowhere.

    I wrote this expecting a crossover and there is not one. Over a flat scan the cheap strategy
    costs the whole corpus at every selectivity, because it has to score everything before it
    can take a top hundred, and restricting first costs the matching set, which is never larger.
    So restricting wins on cost at every selectivity and on accuracy at every selectivity, and
    there is no regime in this measurement where the other choice is defensible.

    A crossover does exist and it needs something this comparison does not have: an approximate
    index on the unfiltered path examining far fewer vectors than the corpus. Then the cheap
    strategy costs whatever that index examines rather than everything, and the crossover sits
    where the matching set is about that size. That is a different measurement and it is the one
    to make before choosing, rather than the one written here.
    """
    corpus = gaussian(count=2048, dimension=32)
    searched, probes = held_out(corpus, count=64)
    predicate = random_predicate(searched.count, selectivity=selectivity)
    truth = exact_filtered(probes, searched.vectors, predicate, k=10)
    cheap, cheap_stats = search_then_filter(probes, searched.vectors, predicate, k=10)
    restricted, restricted_stats = filter_then_search(probes, searched.vectors, predicate, k=10)
    return {
        "selectivity": selectivity,
        "search_then_filter_recall": round(identifier_overlap(truth, cheap), 4),
        "filter_then_search_recall": round(identifier_overlap(truth, restricted), 4),
        "search_then_filter_cost": round(cheap_stats.distances_per_query, 1),
        "filter_then_search_cost": round(restricted_stats.distances_per_query, 1),
        "restricting_is_cheaper": restricted_stats.distances_per_query
        < cheap_stats.distances_per_query,
        "restricting_is_also_more_accurate": identifier_overlap(truth, restricted)
        > identifier_overlap(truth, cheap),
    }


def filtering_inside_a_partitioned_index_works(
    selectivities: Sequence[float] = (0.5, 0.1, 0.02),
) -> list[dict]:
    """Whether skipping non matching entries during the traversal works on posting lists.

    It does, without qualification. A posting list is a list, so skipping entries costs nothing
    and breaks nothing, and the result is exact whenever the matching vectors are in the opened
    partitions. That last condition is the ordinary recall question and not a new failure, which
    is why this is the arrangement to reach for.
    """
    if not selectivities:
        raise ConfigError("there is nothing to sweep")
    corpus = clustered(count=2048, dimension=32, clusters=32)
    searched, probes = held_out(corpus, count=64)
    index = IVFIndex(32, partitions=32, probe=8)
    index.build(searched.vectors)
    rows = []
    for share in selectivities:
        predicate = random_predicate(searched.count, selectivity=share)
        truth = exact_filtered(probes, searched.vectors, predicate, k=10)
        found, stats = filter_during_partitions(
            probes, searched.vectors, predicate, index, k=10, probe=8
        )
        rows.append(
            {
                "selectivity": share,
                "recall": round(identifier_overlap(truth, found), 4),
                "distances_per_query": round(stats.distances_per_query, 1),
            }
        )
    return rows


def filtering_inside_a_graph_strands_the_walk(
    selectivities: Sequence[float] = (0.5, 0.1, 0.02),
) -> list[dict]:
    """And whether the same trick works on a graph, which it does not.

    The walk has to traverse through non matching vertices to get anywhere, so it cannot skip
    them, and the result can only keep the matching ones it happened to pass. At a tight
    selectivity the beam fills entirely with vertices that will be discarded and the search
    finishes having found almost nothing. Nothing errors and nothing warns. The recall simply
    goes to nearly zero while the cost stays exactly where it was.
    """
    if not selectivities:
        raise ConfigError("there is nothing to sweep")
    corpus = gaussian(count=2048, dimension=32)
    searched, probes = held_out(corpus, count=64)
    index = GraphIndex(32, degree=16)
    index.build(searched.vectors)
    rows = []
    for share in selectivities:
        predicate = random_predicate(searched.count, selectivity=share)
        truth = exact_filtered(probes, searched.vectors, predicate, k=10)
        found, stats = filter_during_graph(
            probes, searched.vectors, predicate, index, k=10, ef=64
        )
        rows.append(
            {
                "selectivity": share,
                "recall": round(identifier_overlap(truth, found), 4),
                "distances_per_query": round(stats.distances_per_query, 1),
            }
        )
    return rows


def the_graph_fails_where_the_partitions_do_not() -> dict:
    """The comparison between the two, which is the practical conclusion of this module.

    A partitioned index filters cleanly and a graph index does not. Both cost the same as they
    did unfiltered, and only one of them is still answering. That is the reason systems that
    need attribute filtering reach for inverted files even when a graph would be faster
    unfiltered, and the reason the ones that do use graphs build a separate index per tenant.
    """
    partitioned = {
        row["selectivity"]: row for row in filtering_inside_a_partitioned_index_works()
    }
    graph = {row["selectivity"]: row for row in filtering_inside_a_graph_strands_the_walk()}
    return {
        "partitioned_at_two_percent": partitioned[0.02]["recall"],
        "graph_at_two_percent": graph[0.02]["recall"],
        "partitioned_at_half": partitioned[0.5]["recall"],
        "graph_at_half": graph[0.5]["recall"],
        "graph_collapses": graph[0.02]["recall"] < graph[0.5]["recall"] / 2,
        "partitioned_holds": partitioned[0.02]["recall"] > partitioned[0.5]["recall"] / 2,
    }


def a_correlated_condition_behaves_differently(selectivity: float = 0.05) -> dict:
    """Whether a condition that picks a region behaves like one that picks at random.

    Not at all, and it is the harder case, which is the reverse of what I wrote before running
    it. At five percent selectivity a random condition gives fifty percent recall and a
    condition passing one region gives fifteen. The reason is asymmetry: a query inside the
    matching region finds all of its neighbours matching and does fine, and a query outside it
    finds none of them matching and returns nothing useful, and the second kind is ninety five
    percent of the queries. Real filters correlate with the embedding, because whatever
    attribute they test is usually related to whatever the model encoded, so the random
    measurement above is the optimistic one rather than the pessimistic one.
    """
    corpus = clustered(count=2048, dimension=32, clusters=32)
    searched, probes = held_out(corpus, count=64)
    rows = {}
    for label, predicate in (
        ("random", random_predicate(searched.count, selectivity=selectivity)),
        ("clustered", clustered_predicate(searched, selectivity=selectivity)),
    ):
        truth = exact_filtered(probes, searched.vectors, predicate, k=10)
        found, _ = search_then_filter(probes, searched.vectors, predicate, k=10)
        rows[label] = round(identifier_overlap(truth, found), 4)
    return {
        **rows,
        "correlated_is_easier": rows["clustered"] > rows["random"],
        "selectivity": selectivity,
    }


def compare_strategies(selectivity: float = 0.05) -> list[dict]:
    """Every strategy at one selectivity, as one table."""
    corpus = clustered(count=2048, dimension=32, clusters=32)
    searched, probes = held_out(corpus, count=64)
    predicate = random_predicate(searched.count, selectivity=selectivity)
    truth = exact_filtered(probes, searched.vectors, predicate, k=10)
    index = IVFIndex(32, partitions=32, probe=8)
    index.build(searched.vectors)
    rows = []
    for label, result in (
        ("search then filter", search_then_filter(probes, searched.vectors, predicate, k=10)),
        ("filter then search", filter_then_search(probes, searched.vectors, predicate, k=10)),
        (
            "filter during",
            filter_during_partitions(probes, searched.vectors, predicate, index, k=10),
        ),
    ):
        found, stats = result
        rows.append(
            {
                "strategy": label,
                "recall": round(identifier_overlap(truth, found), 4),
                "distances_per_query": round(stats.distances_per_query, 1),
            }
        )
    return rows


def a_predicate_that_matches_nothing_is_refused() -> bool:
    """Whether an empty condition is refused rather than returning an empty result.

    An empty result is indistinguishable from a search that found nothing, and the two have
    completely different fixes, so this is caught where the predicate is built.
    """
    try:
        Predicate(mask=torch.zeros(64, dtype=torch.bool))
    except ConfigError:
        return True
    return False


def a_selectivity_above_one_is_refused() -> bool:
    """Whether a share larger than the corpus is caught."""
    try:
        random_predicate(1024, selectivity=1.5)
    except ConfigError:
        return True
    return False


def asking_for_more_than_the_matching_set_is_refused() -> bool:
    """Whether a k larger than the matching set is caught rather than silently shortened."""
    corpus = gaussian(count=512, dimension=16)
    predicate = random_predicate(512, selectivity=0.01)
    try:
        exact_filtered(corpus.vectors[:4], corpus.vectors, predicate, k=50)
    except ConfigError:
        return True
    return False


def a_float_predicate_is_refused() -> bool:
    """Whether a mask that is not boolean is refused at construction."""
    try:
        Predicate(mask=torch.ones(64))
    except DataError:
        return True
    return False


def the_exact_answer_only_contains_matching_vectors() -> dict:
    """The property every strategy here is checked against.

    A filtered search that returns a vector failing the condition is not slightly wrong, it is
    a correctness bug that a recall number would report as a small loss. This checks the ground
    truth itself has the property, since everything else is measured against it.
    """
    corpus = gaussian(count=1024, dimension=32)
    predicate = random_predicate(1024, selectivity=0.1)
    found = exact_filtered(corpus.vectors[:32], corpus.vectors, predicate, k=10)
    passing = predicate.mask[found.identifiers]
    return {
        "all_match": bool(passing.all()),
        "returned": int(found.identifiers.numel()),
        "matching_in_corpus": predicate.matching,
    }


def selectivity_sweep(shares: Sequence[float] = (0.5, 0.2, 0.05, 0.01)) -> list[dict]:
    """Both simple strategies across the range, with their costs, as one table.

    The table an implementation would consult to pick a strategy. The cheap one is exact and
    cheap at the top, exact and expensive in the middle, and broken at the bottom. The
    restricting one is exact everywhere and its cost falls as the other one's usefulness does.
    """
    if not shares:
        raise ConfigError("there is nothing to sweep")
    corpus = gaussian(count=2048, dimension=32)
    searched, probes = held_out(corpus, count=64)
    rows = []
    for share in shares:
        predicate = random_predicate(searched.count, selectivity=share)
        truth = exact_filtered(probes, searched.vectors, predicate, k=10)
        cheap, cheap_stats = search_then_filter(probes, searched.vectors, predicate, k=10)
        restricted, restricted_stats = filter_then_search(
            probes, searched.vectors, predicate, k=10
        )
        rows.append(
            {
                "selectivity": share,
                "cheap_recall": round(identifier_overlap(truth, cheap), 4),
                "restricted_recall": round(identifier_overlap(truth, restricted), 4),
                "cheap_cost": round(cheap_stats.distances_per_query, 1),
                "restricted_cost": round(restricted_stats.distances_per_query, 1),
            }
        )
    return rows

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache

import torch

from vse.errors import ConfigError
from vse.index.base import Index, evaluate_result
from vse.index.graph import GraphIndex
from vse.index.hnsw import HNSWIndex
from vse.index.ivf import IVFIndex
from vse.vectors.dataset import clustered, gaussian, held_out
from vse.vectors.exact import Neighbours, distances, search

# Choosing the search parameter, which is the only decision a user of this package makes.
#
# Every index here has exactly one knob that trades accuracy for work: probe count on the
# inverted file, beam width on the graph and the hierarchy, shortlist on the composite. Building
# the index is a decision made once by someone who read the code. Setting the knob is a decision
# made on every deployment by someone who did not, and the honest answer to what it should be is
# that it depends on the corpus, so the only defensible way to pick it is to measure.
#
# That is what this does: sweep the knob, record recall and distance count at each setting, and
# return the cheapest setting that clears a recall target. Three questions decide whether it is
# trustworthy, and all three are measured rather than assumed.
#
# Is the recall curve monotone in the knob? Bisection over the sweep is the obvious
# implementation and it is only correct if it is, and the docstrings here first said it mostly
# is and not always. That was wrong. Twelve thousand per query observations across an inverted
# file and a graph found not one drop, because both knobs grow the candidate set by inclusion:
# the nearest p plus one centroids contain the nearest p, and a wider beam explores a superset
# of a narrower one's frontier, so the top k is taken over a superset and cannot get worse.
#
# The scan stays anyway, for a better reason than the one it was written for. A bisection gives
# back a setting. A scan gives back the whole recall against cost curve, and that curve is what
# tells a deployment whether the target it asked for was the right target.
#
# Does a setting tuned on one query sample hold on another? If it does not, tuning is fitting
# noise and the number it produces is worthless. It holds, with a margin that has to be built
# in, since tuning to exactly the target leaves some of the fresh samples below it.
#
# And does the answer transfer? Not between index structures, where the graph meets the same
# target at a beam of forty eight for a third of the distance count the inverted file needs at
# probe thirty two, and not between corpora either: a clustered corpus reaches the target at
# probe four where a gaussian one of the same size needs thirty two. Six and a half times the
# work, from the shape of the data alone. That last number is the argument for the module, since
# a probe count shipped as a library default cannot see the data it will run on.


@dataclass
class Setting:
    """One knob value and what it bought."""

    value: int
    recall: float
    distances: float

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "value": self.value,
            "recall": round(self.recall, 4),
            "distances": round(self.distances, 1),
        }


@dataclass
class Tuning:
    """The result of a sweep, and the cheapest setting that met the target."""

    target: float
    sweep: list[Setting]
    chosen: Setting | None

    @property
    def met(self) -> bool:
        """Whether any setting in the sweep reached the target."""
        return self.chosen is not None

    @property
    def best_available(self) -> Setting:
        """The highest recall the sweep found, met or not."""
        if not self.sweep:
            raise ConfigError("an empty sweep has no best setting")
        return max(self.sweep, key=lambda setting: setting.recall)

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "target": self.target,
            "met": self.met,
            "chosen": None if self.chosen is None else self.chosen.as_dict(),
            "settings_tried": len(self.sweep),
        }


def sweep_setting(
    index: Index,
    corpus: torch.Tensor,
    queries: torch.Tensor,
    truth: Neighbours,
    values: Sequence[int],
    apply: Callable[[Index, int], None],
    k: int = 10,
) -> list[Setting]:
    """Measure recall and cost at each value of one search parameter.

    The parameter is applied to a built index rather than rebuilding it, which is what makes
    this cheap enough to run at deployment time. An index whose knob is a build parameter cannot
    be tuned this way and none of the ones here have that problem.
    """
    if not values:
        raise ConfigError("there is nothing to sweep")
    settings = []
    for value in values:
        apply(index, value)
        found, stats = index.search(queries, k=k)
        quality = evaluate_result(index, corpus, queries, found, stats, truth=truth)
        settings.append(
            Setting(
                value=value, recall=quality.recall, distances=quality.stats.distances_per_query
            )
        )
    return settings


def cheapest_that_clears(settings: Sequence[Setting], target: float) -> Setting | None:
    """The lowest cost setting whose recall reaches the target.

    Picked by measured cost rather than by knob value. On the indexes here those give the same
    answer because cost rises with the knob, but the knob value is an implementation detail and
    the distance count is what the deployment pays, so cost is the correct thing to minimise.
    """
    if not 0.0 < target <= 1.0:
        raise ConfigError(f"a recall target of {target} is not a target")
    clearing = [setting for setting in settings if setting.recall >= target]
    if not clearing:
        return None
    return min(clearing, key=lambda setting: setting.distances)


def tune(
    index: Index,
    corpus: torch.Tensor,
    queries: torch.Tensor,
    truth: Neighbours,
    values: Sequence[int],
    apply: Callable[[Index, int], None],
    target: float = 0.9,
    k: int = 10,
) -> Tuning:
    """Sweep a parameter and pick the cheapest setting that meets a recall target."""
    settings = sweep_setting(index, corpus, queries, truth, values, apply, k=k)
    return Tuning(target=target, sweep=settings, chosen=cheapest_that_clears(settings, target))


WIDE_PROBES = (1, 2, 4, 8, 16, 32, 48, 64, 80)
PROBE_VALUES = (1, 2, 4, 8, 16, 32, 64)
BEAM_VALUES = (10, 12, 16, 24, 32, 48, 64, 96, 128)


def set_probe(index: Index, value: int) -> None:
    """Apply a probe count to an inverted file."""
    index.probe = value


def set_beam(index: Index, value: int) -> None:
    """Apply a beam width to a graph or a hierarchy."""
    index.ef = value


@lru_cache(maxsize=64)
def _tuned_ivf(target: float, count: int, dimension: int) -> tuple:
    """An inverted file tuned to a recall target, cached because it is measured repeatedly."""
    corpus = gaussian(count=count, dimension=dimension)
    searched, probes = held_out(corpus, count=100)
    truth = search(probes, searched.vectors, k=10)
    index = IVFIndex(dimension, partitions=int(count**0.5), probe=1)
    index.build(searched.vectors)
    result = tune(
        index, searched.vectors, probes, truth, (1, 2, 4, 8, 16, 32), set_probe, target=target
    )
    return tuple(setting.as_dict()["value"] for setting in result.sweep), result


def a_target_picks_a_setting(target: float = 0.9) -> dict:
    """The whole point of the module, on the index it is most useful for.

    A recall target goes in and a probe count comes out, with the cost it implies. Nothing here
    is clever and that is the argument for it: the alternative is a default probe count chosen
    by whoever wrote the library, which cannot know the corpus.
    """
    _, result = _tuned_ivf(target, 8192, 32)
    if result.chosen is None:
        return {"target": target, "met": False, "best": result.best_available.as_dict()}
    return {
        "target": target,
        "met": True,
        "probe": result.chosen.value,
        "recall": round(result.chosen.recall, 4),
        "distances": round(result.chosen.distances, 1),
    }


def a_higher_target_costs_more(
    targets: Sequence[float] = (0.5, 0.7, 0.9, 0.99),
) -> list[dict]:
    """How the chosen setting moves as the target rises.

    Upward, and faster than the target does. The last few points of recall are the expensive
    ones because they are the queries whose true neighbours are in a partition the query does
    not look like, and finding those means opening partitions that are almost all useless.
    """
    if not targets:
        raise ConfigError("there is nothing to sweep")
    rows = []
    for target in targets:
        _, result = _tuned_ivf(target, 8192, 32)
        chosen = result.chosen
        rows.append(
            {
                "target": target,
                "met": chosen is not None,
                "probe": None if chosen is None else chosen.value,
                "distances": None if chosen is None else round(chosen.distances, 1),
            }
        )
    return rows


def the_last_points_of_recall_are_the_expensive_ones() -> dict:
    """The shape of that, which is the reason a recall target is the right thing to specify."""
    rows = {row["target"]: row for row in a_higher_target_costs_more()}
    cheap = rows[0.5]
    dear = rows[0.9]
    return {
        "distances_at_half": cheap["distances"],
        "distances_at_nine_tenths": dear["distances"],
        "recall_ratio": 1.8,
        "cost_ratio": None
        if cheap["distances"] is None or dear["distances"] is None
        else round(dear["distances"] / cheap["distances"], 2),
        "cost_grows_faster": dear["distances"] is not None
        and cheap["distances"] is not None
        and dear["distances"] / cheap["distances"] > 1.8,
    }


def the_recall_curve_is_monotone() -> dict:
    """Whether bisection over the sweep would be safe, which is the implementation question.

    It would. This was written expecting small dips, on the reasoning that opening one more
    partition could displace a true neighbour already found with a false one that happened to
    score better. Twenty four settings, zero drops. The reasoning was wrong: opening one more
    partition adds candidates and removes none, so the top k is taken over a superset of the
    previous candidates and every neighbour that was found before is still available.
    """
    corpus = gaussian(count=8192, dimension=32)
    searched, probes = held_out(corpus, count=100)
    truth = search(probes, searched.vectors, k=10)
    index = IVFIndex(32, partitions=90, probe=1)
    index.build(searched.vectors)
    settings = sweep_setting(
        index, searched.vectors, probes, truth, tuple(range(1, 25)), set_probe
    )
    recalls = [setting.recall for setting in settings]
    drops = [
        round(recalls[row] - recalls[row - 1], 5)
        for row in range(1, len(recalls))
        if recalls[row] < recalls[row - 1]
    ]
    return {
        "settings": len(settings),
        "drops": len(drops),
        "worst_drop": min(drops) if drops else 0.0,
        "monotone": not drops,
        "rises_overall": recalls[-1] > recalls[0],
    }


def the_cost_curve_is_monotone() -> dict:
    """And the other axis, where monotonicity was never in doubt.

    The distance count rises with the probe count without exception, because it is a count of
    work done rather than a measurement of quality. Both axes being monotone is what makes the
    sweep table readable as a curve rather than a scatter.
    """
    corpus = gaussian(count=8192, dimension=32)
    searched, probes = held_out(corpus, count=100)
    truth = search(probes, searched.vectors, k=10)
    index = IVFIndex(32, partitions=90, probe=1)
    index.build(searched.vectors)
    settings = sweep_setting(
        index, searched.vectors, probes, truth, tuple(range(1, 25)), set_probe
    )
    costs = [setting.distances for setting in settings]
    return {
        "settings": len(settings),
        "monotone": costs == sorted(costs),
        "first": round(costs[0], 1),
        "last": round(costs[-1], 1),
    }


def the_curve_is_monotone_for_every_query_not_just_on_average(
    queries: int = 200,
) -> dict:
    """The claim above, checked where averaging cannot hide a violation.

    A mean over a hundred queries is monotone whenever the improvements outweigh the losses, so
    a monotone mean is weak evidence and the first version of this module leaned on it. Checked
    per query instead: two hundred queries, thirty settings, five thousand eight hundred
    consecutive pairs on the graph and six thousand two hundred on the inverted file, and not
    one pair where a larger knob returned fewer true neighbours for the same query.

    That is strong enough to rely on. Bisection is safe on both structures and the scan is kept
    for the curve it produces rather than for safety.
    """
    if queries < 2:
        raise ConfigError(f"{queries} queries cannot show a distribution")
    corpus = gaussian(count=4096, dimension=32)
    searched, probes = held_out(corpus, count=queries)
    truth = search(probes, searched.vectors, k=10)
    rows = []
    for label, index, knob, values in (
        ("graph", GraphIndex(32, degree=16, ef=10), "ef", tuple(range(10, 40))),
        ("ivf", IVFIndex(32, partitions=64, probe=1), "probe", tuple(range(1, 33))),
    ):
        index.build(searched.vectors)
        rows.append(_per_query_drops(index, probes, truth, knob, values, label))
    table = {row["index"]: row for row in rows}
    return {
        "graph_pairs": table["graph"]["pairs"],
        "graph_drops": table["graph"]["drops"],
        "ivf_pairs": table["ivf"]["pairs"],
        "ivf_drops": table["ivf"]["drops"],
        "monotone_everywhere": table["graph"]["drops"] == 0 and table["ivf"]["drops"] == 0,
    }


def _per_query_drops(
    index: Index,
    probes: torch.Tensor,
    truth: Neighbours,
    knob: str,
    values: Sequence[int],
    label: str,
) -> dict:
    """Count the consecutive setting pairs where one query's recall fell."""
    rows = []
    for value in values:
        setattr(index, knob, value)
        found, _ = index.search(probes, k=10)
        rows.append(_per_query_recall(found, truth))
    grid = torch.stack(rows)
    steps = grid[1:] - grid[:-1]
    return {
        "index": label,
        "settings": len(values),
        "pairs": int(steps.numel()),
        "drops": int((steps < 0).sum()),
        "queries_with_a_drop": int((steps < 0).any(dim=0).sum()),
    }


def _per_query_recall(found: Neighbours, truth: Neighbours) -> torch.Tensor:
    """One recall number per query rather than one for the batch."""
    hits = torch.zeros(int(found.identifiers.shape[0]))
    for row in range(int(found.identifiers.shape[0])):
        wanted = set(truth.identifiers[row].tolist())
        got = set(found.identifiers[row].tolist())
        hits[row] = len(wanted & got) / float(len(wanted))
    return hits


def a_wider_probe_opens_a_superset(probes_tried: Sequence[int] = (1, 2, 4, 8, 16)) -> dict:
    """The mechanism behind that, checked directly rather than inferred from the recall.

    The partitions opened at probe p plus one contain the ones opened at probe p, because they
    are the nearest centroids by distance and that ordering does not change with how many are
    taken. So the candidate set grows by inclusion, and monotone recall follows without needing
    to be measured. It is measured anyway, because an implementation that sorted centroids
    differently at different probe counts would break it silently.
    """
    if len(probes_tried) < 2:
        raise ConfigError("comparing openings needs at least two probe counts")
    corpus = gaussian(count=4096, dimension=32)
    searched, queries = held_out(corpus, count=32)
    index = IVFIndex(32, partitions=64, probe=1)
    index.build(searched.vectors)
    openings = []
    for value in probes_tried:
        index.probe = value
        openings.append(_partitions_opened(index, queries))
    nested = all(
        openings[row - 1][query] <= openings[row][query]
        for row in range(1, len(openings))
        for query in range(len(openings[row]))
    )
    return {
        "probes": list(probes_tried),
        "nested": nested,
        "opened_at_one": sorted(openings[0][0]),
        "opened_at_sixteen": len(openings[-1][0]),
    }


def _partitions_opened(index: IVFIndex, queries: torch.Tensor) -> list[set]:
    """Which partitions each query would open at the index's current probe count."""
    scores = distances(queries, index._centres, index.metric)
    chosen = torch.topk(
        scores, k=index.probe, dim=1, largest=not index.metric.smaller_is_closer
    ).indices
    return [set(chosen[row].tolist()) for row in range(int(queries.shape[0]))]


def a_setting_tuned_on_one_sample_holds_on_another(
    target: float = 0.9, samples: int = 5
) -> dict:
    """Whether tuning fits the query sample it was tuned on, which is the honest worry.

    It does not fit noise: a probe count tuned on one hundred queries lands within a point or
    two of the target on fresh samples. But it lands on both sides of it, so tuning to exactly
    the target means roughly half of production traffic comes in below the number that was
    promised. The margin below is the fix and it is not free.
    """
    if samples < 2:
        raise ConfigError(f"{samples} samples cannot show variation")
    corpus = gaussian(count=8192, dimension=32)
    searched, probes = held_out(corpus, count=100)
    truth = search(probes, searched.vectors, k=10)
    index = IVFIndex(32, partitions=90, probe=1)
    index.build(searched.vectors)
    result = tune(index, searched.vectors, probes, truth, WIDE_PROBES, set_probe, target=target)
    if result.chosen is None:
        raise ConfigError(f"a target of {target} was not reachable on the tuning sample")
    index.probe = result.chosen.value
    fresh = []
    for seed in range(samples):
        other = gaussian(count=8192, dimension=32, seed=seed + 100)
        _, other_probes = held_out(other, count=100)
        other_truth = search(other_probes, searched.vectors, k=10)
        found, stats = index.search(other_probes, k=10)
        fresh.append(
            evaluate_result(
                index, searched.vectors, other_probes, found, stats, truth=other_truth
            ).recall
        )
    return {
        "tuned_probe": result.chosen.value,
        "tuned_recall": round(result.chosen.recall, 4),
        "fresh": [round(value, 4) for value in fresh],
        "fresh_mean": round(sum(fresh) / len(fresh), 4),
        "fresh_worst": round(min(fresh), 4),
        "samples_below_target": sum(1 for value in fresh if value < target),
        "of": samples,
    }


def tuning_to_the_target_exactly_misses_half_the_time() -> dict:
    """The consequence of that, stated as the number a deployment cares about."""
    result = a_setting_tuned_on_one_sample_holds_on_another(target=0.9)
    return {
        "target": 0.9,
        "fresh_mean": result["fresh_mean"],
        "fresh_worst": result["fresh_worst"],
        "below": result["samples_below_target"],
        "of": result["of"],
        "some_miss": result["samples_below_target"] > 0,
    }


def a_margin_fixes_it(margin: float = 0.03, target: float = 0.9) -> dict:
    """Tuning to a target above the one that was asked for, which is the whole fix.

    Tuning to exactly ninety picks probe thirty two and four of five fresh samples land below
    ninety. Tuning to ninety three picks probe forty eight and every fresh sample clears ninety,
    with the worst at 0.963. The margin costs whatever that step costs and there is no way to
    have the guarantee without paying it, which is the honest version of shipping a default.
    """
    if not 0.0 <= margin < 0.5:
        raise ConfigError(f"a margin of {margin} is not a margin")
    plain = a_setting_tuned_on_one_sample_holds_on_another(target=target)
    padded = a_setting_tuned_on_one_sample_holds_on_another(target=target + margin)
    return {
        "margin": margin,
        "target": target,
        "probe_without": plain["tuned_probe"],
        "probe_with": padded["tuned_probe"],
        "below_without": sum(1 for value in plain["fresh"] if value < target),
        "below_with": sum(1 for value in padded["fresh"] if value < target),
        "worst_without": plain["fresh_worst"],
        "worst_with": padded["fresh_worst"],
        "helps": padded["fresh_worst"] > plain["fresh_worst"],
    }


def the_setting_does_not_transfer_between_indexes(target: float = 0.9) -> list[dict]:
    """Whether a knob value tuned on one structure means anything on another.

    Nothing at all, and the two knobs are not even defined over the same domain. A probe count
    of one is legal; a beam below k cannot return k neighbours and is refused, so the graph's
    sweep starts at ten while the inverted file's starts at one.

    At a target of ninety percent the inverted file needs probe thirty two and pays 2090
    distances per query. The graph needs beam forty eight, a larger number, and pays 566, less
    than a third. So the knob values do not transfer, and neither does the ordering of the
    numbers, and neither does what a given setting costs. A system that changes its index
    structure and keeps its tuning is running at an accuracy nobody measured.
    """
    corpus = gaussian(count=4096, dimension=32)
    searched, probes = held_out(corpus, count=100)
    truth = search(probes, searched.vectors, k=10)
    rows = []

    ivf = IVFIndex(32, partitions=64, probe=1)
    ivf.build(searched.vectors)
    ivf_result = tune(
        ivf, searched.vectors, probes, truth, PROBE_VALUES, set_probe, target=target
    )
    rows.append(_transfer_row("ivf", "probe", ivf_result))

    graph = GraphIndex(32, degree=16, ef=10)
    graph.build(searched.vectors)
    graph_result = tune(
        graph,
        searched.vectors,
        probes,
        truth,
        BEAM_VALUES,
        set_beam,
        target=target,
    )
    rows.append(_transfer_row("graph", "beam", graph_result))

    return rows


def _transfer_row(label: str, knob: str, result: Tuning) -> dict:
    """One structure's tuned setting, or the best it managed."""
    chosen = result.chosen
    return {
        "index": label,
        "knob": knob,
        "met": chosen is not None,
        "value": None if chosen is None else chosen.value,
        "distances": None if chosen is None else round(chosen.distances, 1),
        "best_recall": round(result.best_available.recall, 4),
    }


def the_two_structures_need_different_numbers() -> dict:
    """The two rows of that as one comparison."""
    rows = {row["index"]: row for row in the_setting_does_not_transfer_between_indexes()}
    return {
        "ivf_probe": rows["ivf"]["value"],
        "graph_beam": rows["graph"]["value"],
        "ivf_distances": rows["ivf"]["distances"],
        "graph_distances": rows["graph"]["distances"],
        "different": rows["ivf"]["value"] != rows["graph"]["value"],
    }


def a_clustered_corpus_needs_a_different_setting(target: float = 0.9) -> dict:
    """Whether the corpus shape changes the answer, which is the reason to tune at all.

    It does, by more than anything else in this module. The same corpus size, the same
    dimension, the same partition count and the same recall target: the gaussian corpus needs
    probe thirty two and 2090 distances per query, the clustered one needs probe four and 324. A
    factor of six and a half from the shape of the data alone.

    Which is the argument for the whole module. Any probe count shipped as a library default is
    six times too expensive on one of these corpora or far below its target on the other, and no
    amount of care in choosing that default fixes it, because the default cannot see the data.
    """
    rows = []
    for label, corpus in (
        ("gaussian", gaussian(count=4096, dimension=32)),
        ("clustered", clustered(count=4096, dimension=32, clusters=16)),
    ):
        searched, probes = held_out(corpus, count=100)
        truth = search(probes, searched.vectors, k=10)
        index = IVFIndex(32, partitions=64, probe=1)
        index.build(searched.vectors)
        result = tune(
            index,
            searched.vectors,
            probes,
            truth,
            (1, 2, 4, 8, 16, 32, 64),
            set_probe,
            target=target,
        )
        rows.append(
            {
                "corpus": label,
                "probe": None if result.chosen is None else result.chosen.value,
                "distances": None
                if result.chosen is None
                else round(result.chosen.distances, 1),
            }
        )
    table = {row["corpus"]: row for row in rows}
    return {
        "gaussian_probe": table["gaussian"]["probe"],
        "clustered_probe": table["clustered"]["probe"],
        "gaussian_distances": table["gaussian"]["distances"],
        "clustered_distances": table["clustered"]["distances"],
        "differ": table["gaussian"]["probe"] != table["clustered"]["probe"],
    }


def an_unreachable_target_says_so(target: float = 1.0) -> dict:
    """What happens when no setting in the sweep meets the target.

    It reports that it did not, with the best it found. The alternative, returning the highest
    setting and letting the caller assume it worked, is how a system ends up serving at seventy
    percent recall against a contract that says ninety nine.
    """
    corpus = gaussian(count=4096, dimension=32)
    searched, probes = held_out(corpus, count=100)
    truth = search(probes, searched.vectors, k=10)
    index = IVFIndex(32, partitions=64, probe=1)
    index.build(searched.vectors)
    result = tune(index, searched.vectors, probes, truth, (1, 2, 4), set_probe, target=target)
    return {
        "target": target,
        "met": result.met,
        "best_recall": round(result.best_available.recall, 4),
        "chosen": result.chosen,
        "settings_tried": len(result.sweep),
    }


def tuning_the_hierarchy(target: float = 0.9) -> dict:
    """The same procedure on the hierarchy, to show the knob is the only thing that changes.

    Beam width instead of probe count and everything else identical, which is the argument for
    the parameter being passed in as a function rather than the tuner knowing about index types.
    """
    corpus = gaussian(count=4096, dimension=32)
    searched, probes = held_out(corpus, count=100)
    truth = search(probes, searched.vectors, k=10)
    index = HNSWIndex(32, degree=16, ef=10)
    index.build(searched.vectors)
    result = tune(
        index,
        searched.vectors,
        probes,
        truth,
        BEAM_VALUES,
        set_beam,
        target=target,
    )
    return {
        "target": target,
        "met": result.met,
        "beam": None if result.chosen is None else result.chosen.value,
        "distances": None if result.chosen is None else round(result.chosen.distances, 1),
        "best_recall": round(result.best_available.recall, 4),
    }


def an_empty_sweep_is_refused() -> bool:
    """Whether a sweep with no values to try is caught."""
    corpus = gaussian(count=512, dimension=8)
    searched, probes = held_out(corpus, count=20)
    truth = search(probes, searched.vectors, k=5)
    index = IVFIndex(8, partitions=16, probe=1)
    index.build(searched.vectors)
    try:
        sweep_setting(index, searched.vectors, probes, truth, (), set_probe, k=5)
    except ConfigError:
        return True
    return False


def an_impossible_target_is_refused() -> bool:
    """Whether a recall target above one is caught before the sweep runs."""
    try:
        cheapest_that_clears([Setting(1, 0.5, 10.0)], target=1.5)
    except ConfigError:
        return True
    return False


def a_zero_target_is_refused() -> bool:
    """Whether a target of zero is caught.

    It would return the cheapest setting unconditionally, which is a defensible thing to want
    and not a thing to spell as a recall target of zero, because the caller who typed it meant
    something else.
    """
    try:
        cheapest_that_clears([Setting(1, 0.5, 10.0)], target=0.0)
    except ConfigError:
        return True
    return False


def the_cheapest_clearing_setting_is_not_always_the_smallest() -> dict:
    """Whether picking by knob value and picking by cost agree, which they mostly do.

    The tuner picks by measured cost, not by knob value, and on a well behaved sweep those give
    the same answer because cost rises with the knob. On a sweep where they disagree, cost is
    the one that matters, since the knob value is an implementation detail and the distance
    count is what the deployment pays.
    """
    settings = [
        Setting(value=1, recall=0.5, distances=100.0),
        Setting(value=2, recall=0.92, distances=300.0),
        Setting(value=4, recall=0.91, distances=200.0),
        Setting(value=8, recall=0.99, distances=800.0),
    ]
    chosen = cheapest_that_clears(settings, target=0.9)
    return {
        "chosen_value": chosen.value,
        "chosen_distances": chosen.distances,
        "smallest_clearing_value": 2,
        "picked_by_cost": chosen.value == 4,
    }


def an_empty_tuning_has_no_best() -> bool:
    """Whether asking an empty sweep for its best setting is caught."""
    try:
        _ = Tuning(target=0.9, sweep=[], chosen=None).best_available
    except ConfigError:
        return True
    return False

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from vse.errors import ConfigError, DataError
from vse.index.ivf import IVFIndex
from vse.vectors.dataset import clustered, gaussian, held_out, on_a_subspace
from vse.vectors.exact import Neighbours, search

# Whether an answer can tell you it is wrong, without being told what right was.
#
# serve/router.py asks a related question and asks it early: it looks at a query before the
# search and predicts how hard it will be. The best signal it found correlates minus 0.254
# with difficulty. This module asks after the search, when there is a result to look at, on
# the assumption that the result is downstream of everything the query did and must therefore
# say more.
#
# It says less. At probe 4 the best of the five post search signals correlates minus 0.118
# with per query recall, which is half of what the pre search signal manages. The reason is
# concentration: in thirty two dimensions the ten returned distances look much alike whether
# the search found the right ten or the wrong ten, so the shape of the answer is nearly
# uninformative about its correctness.
#
# Five signals are available with no extra distance computations, because the search already
# paid for them: the distance to the nearest returned neighbour, the distance to the furthest,
# the gap between them, the ratio between them, and the spread of the returned scores. All five
# are properties of the answer rather than of the query.
#
# A weak correlation still makes a usable flag, because a flag only needs the tail. Taking the
# worst fifth by the ratio of nearest to furthest, 0.80 of them really were below half recall
# against a base rate of 0.68, and a random flag lands on 0.66. So the rule works.
#
# And it is not worth deploying, which is the result this module exists for. Re running the
# flagged fifth at four times the probe lifts recall from 0.382 to 0.464 for 67 percent more
# distances. Spending that same 67 percent by raising the probe from 4 to 7 on every query
# reaches 0.520. Selective escalation loses by 0.055.
#
# It is not the signal's fault. Replacing the flag with an oracle that knows exactly which
# queries came back worst reaches 0.487, still below the uniform 0.520. The hard queries are
# hard: four times the probe on the worst fifth buys less than a modest rise everywhere,
# because the marginal return on probe is highest in the middle of the distribution and not
# at its bad end. Selective escalation is the wrong shape of spending for this index, however
# good the flag.

SIGNALS = ("nearest", "furthest", "gap", "ratio", "spread")


@dataclass
class Calibration:
    """One signal, measured against what actually happened."""

    signal: str
    values: list[float]
    recalls: list[float]

    def __post_init__(self) -> None:
        if len(self.values) != len(self.recalls):
            raise DataError(f"{len(self.values)} values against {len(self.recalls)} recalls")
        if len(self.values) < 3:
            raise ConfigError("a correlation needs at least three queries")

    @property
    def correlation(self) -> float:
        """Pearson correlation between the signal and the per query recall."""
        return correlation(self.values, self.recalls)

    @property
    def rank_correlation(self) -> float:
        """The same on ranks, which is what a threshold rule actually uses.

        A threshold on a signal only cares about the order of the values, so the rank
        correlation is the honest predictor of how well a rule built on it will do. The two
        agree here, which they need not.
        """
        return correlation(_ranks(self.values), _ranks(self.recalls))

    @property
    def strength(self) -> float:
        """How far from zero, ignoring which way it points."""
        return abs(self.correlation)

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "signal": self.signal,
            "correlation": round(self.correlation, 4),
            "rank_correlation": round(self.rank_correlation, 4),
            "queries": len(self.values),
        }


@dataclass
class Flagged:
    """A rejection rule and what it caught."""

    share: float
    caught: int
    missed: int
    false_alarms: int
    kept: int

    @property
    def precision(self) -> float:
        """Of the answers flagged, how many were actually bad."""
        total = self.caught + self.false_alarms
        return self.caught / total if total else 0.0

    @property
    def sensitivity(self) -> float:
        """Of the bad answers, how many were flagged."""
        total = self.caught + self.missed
        return self.caught / total if total else 0.0

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "share_flagged": round(self.share, 4),
            "precision": round(self.precision, 4),
            "sensitivity": round(self.sensitivity, 4),
            "caught": self.caught,
            "false_alarms": self.false_alarms,
        }


def correlation(left: Sequence[float], right: Sequence[float]) -> float:
    """Pearson correlation, or zero when either side never varies.

    Zero rather than an error for the constant case, because a signal that never moves is a
    signal that predicts nothing and that is the answer the caller wants rather than a crash.
    """
    if len(left) != len(right):
        raise DataError(f"{len(left)} against {len(right)}")
    if len(left) < 3:
        raise ConfigError("a correlation needs at least three points")
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    covariance = sum(
        (a - left_mean) * (b - right_mean) for a, b in zip(left, right, strict=True)
    )
    left_spread = sum((a - left_mean) ** 2 for a in left) ** 0.5
    right_spread = sum((b - right_mean) ** 2 for b in right) ** 0.5
    if left_spread == 0.0 or right_spread == 0.0:
        return 0.0
    return covariance / (left_spread * right_spread)


def _ranks(values: Sequence[float]) -> list[float]:
    """Ranks with ties averaged, which is what a rank correlation needs."""
    order = sorted(range(len(values)), key=lambda position: values[position])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
            end += 1
        shared = (position + end) / 2.0
        for index in range(position, end + 1):
            ranks[order[index]] = shared
        position = end + 1
    return ranks


def _setup(
    count: int = 4096,
    dimension: int = 32,
    queries: int = 400,
    k: int = 10,
    kind: str = "gaussian",
    seed: int = 0,
):
    """One corpus, one query set, one exact answer."""
    if kind == "gaussian":
        corpus = gaussian(count=count, dimension=dimension, seed=seed)
    elif kind == "clustered":
        corpus = clustered(count=count, dimension=dimension, clusters=16, seed=seed)
    elif kind == "subspace":
        corpus = on_a_subspace(count=count, dimension=dimension, intrinsic=6, seed=seed)
    else:
        raise ConfigError(f"{kind} is not a corpus")
    searched, probes = held_out(corpus, count=queries)
    truth = search(probes, searched.vectors, k=k)
    return searched.vectors, probes, truth


def signals(found: Neighbours) -> dict[str, list[float]]:
    """Every signal, read off the returned scores and nothing else.

    None of these costs a distance computation. They are all functions of the k scores the
    search already produced, which is the constraint that makes the exercise worth doing: a
    signal that needs more work than the escalation it triggers has no reason to exist.
    """
    scores = found.scores
    if scores.ndim != 2:
        raise DataError(f"scores are two dimensional, not {scores.ndim}")
    if int(scores.shape[1]) < 2:
        raise ConfigError("a signal needs at least two returned neighbours")
    nearest = scores[:, 0]
    furthest = scores[:, -1]
    return {
        "nearest": nearest.tolist(),
        "furthest": furthest.tolist(),
        "gap": (furthest - nearest).tolist(),
        "ratio": (nearest / furthest.clamp_min(1e-12)).tolist(),
        "spread": scores.std(dim=1).tolist(),
    }


def per_query_recall(truth: Neighbours, found: Neighbours) -> list[float]:
    """The share of each query's true neighbours that came back."""
    wanted = truth.identifiers
    if int(wanted.shape[0]) != int(found.identifiers.shape[0]):
        raise DataError(
            f"{int(wanted.shape[0])} truths against {int(found.identifiers.shape[0])} answers"
        )
    width = int(wanted.shape[1])
    scores = []
    for row in range(int(wanted.shape[0])):
        present = set(found.identifiers[row].tolist())
        scores.append(sum(1 for one in wanted[row].tolist() if one in present) / width)
    return scores


def measure(
    probe: int = 4,
    partitions: int = 64,
    k: int = 10,
    prepared=None,
):
    """One search, with the answer, the signals and the per query recall lined up."""
    corpus, probes, truth = prepared if prepared is not None else _setup(k=k)
    index = IVFIndex(int(corpus.shape[1]), partitions=partitions, probe=probe)
    index.build(corpus)
    found, stats = index.search(probes, k=k)
    return {
        "found": found,
        "signals": signals(found),
        "recalls": per_query_recall(truth, found),
        "distances": float(stats.distances_per_query),
        "corpus": corpus,
        "probes": probes,
        "truth": truth,
    }


def every_signal(probe: int = 4) -> list[dict]:
    """All five signals ranked by how well they predict the recall.

    At probe 4 they run minus 0.118 for the nearest distance, minus 0.095 for the furthest,
    minus 0.088 for the ratio, 0.080 for the spread and 0.037 for the gap. None of them is worth
    much and they are all within a factor of three of each other, which is the shape of a set of
    signals that are all measuring the same small thing.
    """
    result = measure(probe=probe)
    rows = []
    for name in SIGNALS:
        calibration = Calibration(
            signal=name,
            values=result["signals"][name],
            recalls=result["recalls"],
        )
        rows.append(calibration.as_dict())
    return sorted(rows, key=lambda row: -abs(row["correlation"]))


def no_signal_is_strong(probe: int = 4) -> dict:
    """Name the best one and say how little it is worth.

    The nearest returned distance leads at minus 0.118, ahead of the furthest at minus 0.095.
    The margin between first and second is 0.022, which is smaller than the sampling error on a
    correlation from four hundred points, so the ordering is not real either.

    Every signal points the way it should: a larger nearest distance means a worse answer. The
    directions are right and the magnitudes are not there.
    """
    rows = every_signal(probe=probe)
    best = rows[0]
    return {
        "rows": rows,
        "best": best["signal"],
        "correlation": best["correlation"],
        "it_is_negative": best["correlation"] < 0.0,
        "it_is_weak": abs(best["correlation"]) < 0.2,
        "runner_up": rows[1]["signal"],
        "margin": round(abs(best["correlation"]) - abs(rows[1]["correlation"]), 4),
        "the_margin_is_noise": (abs(best["correlation"]) - abs(rows[1]["correlation"]) < 0.05),
    }


def it_does_not_beat_the_signal_from_before_the_search(probe: int = 4) -> dict:
    """The comparison with serve/router.py, which is the reason this module exists.

    I expected the post search signals to win easily, on the grounds that they see the result
    and the router only sees the query. The best pre search signal reaches 0.254 against
    difficulty; the best post search signal reaches 0.118 against the same thing measured the
    same way, which is less than half.

    What the router's signal has that these do not is a view of where the query sits relative
    to the corpus. The returned scores describe a neighbourhood without saying whether it was
    the right neighbourhood, and in high dimensions that turns out to be most of the question.
    """
    best = no_signal_is_strong(probe=probe)
    before = 0.254
    return {
        "before_the_search": before,
        "after_the_search": round(abs(best["correlation"]), 4),
        "it_is_worse": abs(best["correlation"]) < before,
        "by_more_than_half": abs(best["correlation"]) < before / 2.0,
        "ratio": round(abs(best["correlation"]) / before, 2),
    }


def no_signal_travels_between_corpora(
    kinds: Sequence[str] = ("gaussian", "clustered", "subspace"),
) -> dict:
    """Which signals keep their correlation when the corpus changes.

    A signal is only worth deploying if a threshold set on one corpus means something on
    another. Not one of the five holds its sign across the Gaussian, clustered and subspace
    corpora. The ratio runs minus 0.088, 0.040 and 0.028; the nearest distance minus 0.118,
    0.055 and minus 0.071.

    Sign flips at this magnitude are what a correlation of nothing looks like when it is
    measured three times. The steadiest signal is the ratio only in the sense that its three
    readings are closest together, and all three are close to zero.
    """
    if not kinds:
        raise ConfigError("there is nothing to sweep")
    rows: dict = {name: [] for name in SIGNALS}
    for kind in kinds:
        prepared = _setup(kind=kind)
        result = measure(prepared=prepared)
        for name in SIGNALS:
            calibration = Calibration(
                signal=name,
                values=result["signals"][name],
                recalls=result["recalls"],
            )
            rows[name].append(round(calibration.correlation, 4))
    steadiness = {name: round(max(values) - min(values), 4) for name, values in rows.items()}
    return {
        "rows": rows,
        "steadiness": steadiness,
        "steadiest": min(steadiness, key=lambda name: steadiness[name]),
        "nothing_keeps_its_sign": not any(
            all(value < 0.0 for value in values) or all(value > 0.0 for value in values)
            for values in rows.values()
        ),
        "all_are_near_zero": all(
            abs(value) < 0.2 for values in rows.values() for value in values
        ),
    }


def flag_the_worst(
    values: Sequence[float],
    recalls: Sequence[float],
    share: float = 0.2,
    bad_below: float = 0.5,
    largest_is_worse: bool = True,
) -> Flagged:
    """Flag a share of the queries by signal and score the rule against the truth.

    The threshold is a share rather than a value on purpose. A share is a budget and a caller
    knows what budget they have; a value is a number that has to be retuned for every corpus,
    and the previous function shows how badly that travels.
    """
    if not 0.0 < share < 1.0:
        raise ConfigError(f"{share} is not a share to flag")
    if len(values) != len(recalls):
        raise DataError(f"{len(values)} values against {len(recalls)} recalls")
    count = max(1, round(len(values) * share))
    order = sorted(
        range(len(values)),
        key=lambda position: values[position],
        reverse=largest_is_worse,
    )
    chosen = set(order[:count])
    caught = sum(1 for position in chosen if recalls[position] < bad_below)
    alarms = len(chosen) - caught
    missed = sum(
        1
        for position in range(len(values))
        if position not in chosen and recalls[position] < bad_below
    )
    return Flagged(
        share=count / len(values),
        caught=caught,
        missed=missed,
        false_alarms=alarms,
        kept=len(values) - len(chosen),
    )


def what_a_flag_catches(shares: Sequence[float] = (0.1, 0.2, 0.3, 0.5)) -> list[dict]:
    """Precision and sensitivity of the ratio rule at four budgets.

    Precision falls and sensitivity rises as the flag widens, which is the shape every threshold
    rule has. The number worth reading is the precision at the tightest budget: flagging a
    tenth of the queries, 0.825 of the flagged ones really were below half recall, against a
    base rate of 0.678.

    That is a real lift from a correlation of minus 0.088, which is worth understanding rather
    than explaining away. A correlation is an average over the whole distribution and a flag
    only ever touches its tail, so a signal can be useless on average and still sort the
    extremes. The ratio also makes the best flag of the five despite ranking third by
    correlation, for the same reason.
    """
    if not shares:
        raise ConfigError("there is nothing to sweep")
    result = measure()
    rows = []
    for share in shares:
        flagged = flag_the_worst(result["signals"]["ratio"], result["recalls"], share=share)
        row = flagged.as_dict()
        row["base_rate"] = round(
            sum(1 for value in result["recalls"] if value < 0.5) / len(result["recalls"]), 4
        )
        rows.append(row)
    return rows


def the_precision_beats_the_base_rate() -> dict:
    """The one claim a flag has to clear to be worth anything.

    A rule that flags a fifth of the queries at random would be right about the base rate of the
    time. The ratio rule is right much more often than that, which is what makes it a rule
    rather than a decoration.
    """
    rows = what_a_flag_catches()
    tight = rows[0]
    return {
        "rows": rows,
        "precision": tight["precision"],
        "base_rate": tight["base_rate"],
        "it_beats_the_base_rate": tight["precision"] > tight["base_rate"],
        "by": round(tight["precision"] - tight["base_rate"], 4),
    }


def a_random_flag_is_the_baseline(trials: int = 8) -> dict:
    """And it lands on the base rate, as it must.

    Here for the same reason serve/router.py has one: a rule scored against nothing at all looks
    impressive, and the only way to know how impressive is to score the same measurement on a
    signal that carries no information.
    """
    if trials < 2:
        raise ConfigError(f"{trials} is not enough trials")
    result = measure()
    recalls = result["recalls"]
    generator = torch.Generator().manual_seed(0)
    precisions = []
    for _ in range(trials):
        noise = torch.rand(len(recalls), generator=generator).tolist()
        precisions.append(flag_the_worst(noise, recalls, share=0.2).precision)
    base = sum(1 for value in recalls if value < 0.5) / len(recalls)
    return {
        "random_precision": round(statistics.fmean(precisions), 4),
        "base_rate": round(base, 4),
        "it_lands_on_the_base_rate": abs(statistics.fmean(precisions) - base) < 0.08,
        "the_real_rule_beats_it": (
            the_precision_beats_the_base_rate()["precision"] > statistics.fmean(precisions)
        ),
    }


def escalating_the_flagged_queries(
    share: float = 0.2,
    multiplier: int = 4,
    probe: int = 4,
    partitions: int = 64,
) -> dict:
    """Re run the flagged queries at a higher probe and count what it cost.

    Recall goes from 0.382 to 0.464 for 67 percent more distances. Read on its own that looks
    like a good trade, since the extra work falls on a fifth of the traffic rather than all of
    it. Read against the alternative in the next function it is not one.
    """
    if multiplier < 2:
        raise ConfigError(f"{multiplier} is not an escalation")
    prepared = _setup()
    first = measure(probe=probe, partitions=partitions, prepared=prepared)
    corpus, probes, truth = prepared
    flagged = flag_the_worst(first["signals"]["ratio"], first["recalls"], share=share)
    order = sorted(
        range(len(first["recalls"])),
        key=lambda position: first["signals"]["ratio"][position],
        reverse=True,
    )
    chosen = order[: flagged.caught + flagged.false_alarms]
    second = IVFIndex(int(corpus.shape[1]), partitions=partitions, probe=probe * multiplier)
    second.build(corpus)
    again, stats = second.search(probes[chosen], k=10)
    improved = list(first["recalls"])
    better = per_query_recall(
        Neighbours(truth.identifiers[chosen], truth.scores[chosen]), again
    )
    for slot, position in enumerate(chosen):
        improved[position] = max(improved[position], better[slot])
    extra = stats.distances_per_query * len(chosen) / len(first["recalls"])
    return {
        "before": round(statistics.fmean(first["recalls"]), 4),
        "after": round(statistics.fmean(improved), 4),
        "base_distances": round(first["distances"], 1),
        "extra_distances": round(extra, 1),
        "overhead": round(extra / first["distances"], 4),
        "it_helped": statistics.fmean(improved) > statistics.fmean(first["recalls"]),
    }


def spending_the_same_everywhere_wins(
    share: float = 0.2,
    multiplier: int = 4,
) -> dict:
    """The matched cost comparison, which is the only one that settles it.

    Escalating a flagged fifth costs 67 percent more and reaches 0.464. Spending the same 67
    percent uniformly means raising the probe from 4 to 7, which reaches 0.520. The flag loses
    by 0.055, which is three times the noise floor stability.py established.

    I expected the opposite and the reason I was wrong is in the shape of the recall curve. The
    flagged queries are the ones sitting furthest from any partition boundary the build drew,
    and four times the probe on those buys less than a modest rise on the queries in the
    middle, where the curve is steepest. Concentrating a budget on the worst cases spends it
    where the return is lowest.
    """
    selective = escalating_the_flagged_queries(share=share, multiplier=multiplier)
    target = selective["base_distances"] + selective["extra_distances"]
    prepared = _setup()
    rows = []
    for probe in (4, 5, 6, 7, 8, 10, 12):
        result = measure(probe=probe, prepared=prepared)
        rows.append(
            {
                "probe": probe,
                "recall": round(statistics.fmean(result["recalls"]), 4),
                "distances": round(result["distances"], 1),
            }
        )
    affordable = [row for row in rows if row["distances"] <= target]
    uniform = max(affordable, key=lambda row: row["recall"]) if affordable else rows[0]
    return {
        "selective_recall": selective["after"],
        "uniform_recall": uniform["recall"],
        "uniform_probe": uniform["probe"],
        "budget": round(target, 1),
        "the_flag_loses": selective["after"] < uniform["recall"],
        "margin": round(selective["after"] - uniform["recall"], 4),
        "and_by_more_than_the_noise": (uniform["recall"] - selective["after"] > 0.02),
        "rows": rows,
    }


def even_a_perfect_flag_loses(
    share: float = 0.2,
    multiplier: int = 4,
) -> dict:
    """The oracle, which settles whether the signal was the problem.

    Flag exactly the queries whose recall really was worst, which needs the truth and is
    therefore not a rule anyone can run. It reaches 0.487 against the ratio rule's 0.464, so the
    rule is already capturing 0.79 of everything a perfect flag could get.

    And the oracle is still below the 0.520 that the same budget spent uniformly reaches. So the
    signal was never the problem: selective escalation on this index is the wrong design at any
    flag quality, and a better signal would not rescue it.
    """
    prepared = _setup()
    first = measure(prepared=prepared)
    corpus, probes, truth = prepared
    count = max(1, round(len(first["recalls"]) * share))
    chosen = sorted(range(len(first["recalls"])), key=lambda p: first["recalls"][p])[:count]
    second = IVFIndex(int(corpus.shape[1]), partitions=64, probe=4 * multiplier)
    second.build(corpus)
    again, _ = second.search(probes[chosen], k=10)
    improved = list(first["recalls"])
    better = per_query_recall(
        Neighbours(truth.identifiers[chosen], truth.scores[chosen]), again
    )
    for slot, position in enumerate(chosen):
        improved[position] = max(improved[position], better[slot])
    rule = escalating_the_flagged_queries(share=share, multiplier=multiplier)
    oracle = statistics.fmean(improved)
    gained = rule["after"] - rule["before"]
    available = oracle - rule["before"]
    return {
        "before": rule["before"],
        "with_the_rule": rule["after"],
        "with_the_oracle": round(oracle, 4),
        "share_captured": round(gained / available, 4) if available > 0 else 0.0,
        "the_oracle_is_ahead": oracle > rule["after"],
        "but_only_just": round(oracle - rule["after"], 4) < 0.05,
        "most_of_it_is_captured": gained / max(available, 1e-9) > 0.6,
    }


def the_signal_fades_as_the_index_improves(
    probe_values: Sequence[int] = (1, 2, 4, 8, 16),
) -> list[dict]:
    """A good index gives the signal nothing to find.

    The ratio correlates minus 0.386 at probe 1, minus 0.249 at probe 2, minus 0.088 at probe 4,
    minus 0.067 at probe 8 and minus 0.041 at probe 16. It fades monotonically, and it is
    strongest exactly where the index is worst.

    That is the module's one encouraging number. At probe 1 the post search signal is above the
    router's pre search 0.254, so on a genuinely cheap tier there is something to work with.
    Everywhere else the answer describes a neighbourhood without saying whether it was the right
    one.
    """
    if not probe_values:
        raise ConfigError("there is nothing to sweep")
    prepared = _setup()
    rows = []
    for probe in probe_values:
        result = measure(probe=probe, prepared=prepared)
        calibration = Calibration(
            signal="ratio",
            values=result["signals"]["ratio"],
            recalls=result["recalls"],
        )
        rows.append(
            {
                "probe": probe,
                "recall": round(statistics.fmean(result["recalls"]), 4),
                "correlation": round(calibration.correlation, 4),
                "bad_answers": round(
                    sum(1 for value in result["recalls"] if value < 0.5)
                    / len(result["recalls"]),
                    4,
                ),
            }
        )
    return rows


def there_is_nothing_left_to_catch_at_the_top() -> dict:
    """Say that as a claim, because it decides where a flag belongs in a system.

    The base rate of bad answers falls from 0.98 at probe 1 to 0.02 at probe 16, and the
    correlation falls with it. Both arguments point the same way: a flag on a well tuned index
    is looking for something that is not there, and the only place any of this belongs is the
    cheap tier of a router.
    """
    rows = the_signal_fades_as_the_index_improves()
    return {
        "rows": rows,
        "worst_base_rate": rows[0]["bad_answers"],
        "best_base_rate": rows[-1]["bad_answers"],
        "the_base_rate_collapses": rows[-1]["bad_answers"] < rows[0]["bad_answers"] / 4,
        "the_correlation_collapses_too": (
            abs(rows[-1]["correlation"]) < abs(rows[0]["correlation"]) / 4
        ),
        "it_is_strongest_at_the_cheap_end": (
            abs(rows[0]["correlation"]) == max(abs(row["correlation"]) for row in rows)
        ),
    }


def a_share_outside_the_range_is_refused() -> bool:
    """Flagging everything or nothing is a mistake, not a rule."""
    try:
        flag_the_worst([1.0, 2.0, 3.0], [0.1, 0.2, 0.3], share=1.0)
    except ConfigError:
        return True
    return False


def a_mismatched_flag_is_refused() -> bool:
    """A signal and a truth of different lengths cannot be scored against each other."""
    try:
        flag_the_worst([1.0, 2.0], [0.1, 0.2, 0.3])
    except DataError:
        return True
    return False


def a_correlation_of_two_points_is_refused() -> bool:
    """Two points lie on a line, so a correlation over them says nothing."""
    try:
        correlation([1.0, 2.0], [3.0, 4.0])
    except ConfigError:
        return True
    return False


def a_single_neighbour_has_no_shape() -> bool:
    """And a result of width one has no ratio to take."""
    found = Neighbours(torch.zeros(4, 1, dtype=torch.long), torch.ones(4, 1))
    try:
        signals(found)
    except ConfigError:
        return True
    return False


def summarise() -> dict:
    """The module in one mapping, for the command line and for logging."""
    best = no_signal_is_strong()
    caught = the_precision_beats_the_base_rate()
    matched = spending_the_same_everywhere_wins()
    return {
        "best_signal": best["best"],
        "correlation": best["correlation"],
        "precision_at_a_tenth": caught["precision"],
        "base_rate": caught["base_rate"],
        "selective_recall": matched["selective_recall"],
        "uniform_recall": matched["uniform_recall"],
        "the_flag_loses": matched["the_flag_loses"],
    }

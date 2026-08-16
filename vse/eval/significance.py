from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from vse.errors import ConfigError, DataError
from vse.index.forest import ForestIndex
from vse.index.ivf import IVFIndex
from vse.vectors.dataset import gaussian, held_out
from vse.vectors.exact import Neighbours, search

# How many queries a recall number needs before it means anything.
#
# Every measurement in this package is a mean over a query sample, and every one of them is
# quoted to four decimal places. That is a lie of precision. On an inverted file at probe four
# over a corpus of four thousand:
#
#     queries    standard error    width of a 95 percent interval
#          10             0.049                            0.194
#          50             0.021                            0.082
#         200             0.012                            0.046
#        1000             0.005                            0.021
#
# A hundred queries, which is what most modules here use, gives 0.018, so plus or minus 3.6
# points. The fourth decimal place in every number in this package is noise, the third usually
# is, and two configurations differing by three points are not distinguishable from one sample.
#
# Two things this module was written to argue turned out to be false and the corrections are
# more useful than the claims were.
#
# The first was that recall at ten is badly non binomial, because the ten neighbours of one
# query should share that query's fate. Measured, the binomial estimate over a thousand
# neighbour slots gives 0.0153 against a measured 0.0182, understating by nineteen percent
# rather than by the factor of three the argument predicted. The reason is visible in the
# distribution: the per query recall is not bimodal at all, with 47 percent of queries
# between 0.3 and 0.7 and only 8 percent at either extreme. Queries mostly get some of their
# neighbours, so the slots are closer to independent than the story needed them to be.
#
# The second was that pairing always helps. Two configurations on the same queries share their
# difficulty, so the per query difference should have a smaller error than either alone. It does
# when they are close, and it does not when they are far apart:
#
#     probe    own error    paired error    ratio
#         4        0.018           0.011     1.71
#         8        0.017           0.015     1.10
#        16        0.015           0.017     0.87
#
# Against the probe two baseline, pairing is worth 1.71 at a gap of a few points and costs
# thirteen percent at a gap of fifty. At a large gap the two configurations stop sharing their
# difficulty and start being complementary: the queries the weak one misses are exactly the ones
# the strong one picks up, so the difference varies more than either does.
#
# So the practical rule is narrower than the usual advice. Pair adjacent settings in a sweep,
# where the gaps are small and the shared difficulty is real, and do not assume pairing
# rescues a comparison between two very different structures. The first is what every sweep
# in this package does, which is why the trends in them are trustworthy at a hundred queries
# even though the individual numbers are not.


@dataclass
class Estimate:
    """A mean with the uncertainty around it."""

    mean: float
    error: float
    samples: int

    @property
    def interval(self) -> tuple[float, float]:
        """The ninety five percent interval, at roughly two standard errors."""
        half = 1.96 * self.error
        return (self.mean - half, self.mean + half)

    @property
    def width(self) -> float:
        """How wide that interval is."""
        return 2 * 1.96 * self.error

    def overlaps(self, other: Estimate) -> bool:
        """Whether two estimates are close enough to be indistinguishable.

        A crude test and the one people actually apply. It is conservative for unpaired samples
        and wrong for paired ones, where the difference has its own much smaller error, which is
        the point made at length below.
        """
        low, high = self.interval
        other_low, other_high = other.interval
        return not (high < other_low or other_high < low)

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        low, high = self.interval
        return {
            "mean": round(self.mean, 4),
            "error": round(self.error, 4),
            "samples": self.samples,
            "low": round(low, 4),
            "high": round(high, 4),
        }


def per_query_recall(truth: Neighbours, found: Neighbours) -> torch.Tensor:
    """One recall number per query rather than one for the batch.

    Everything here needs the distribution and not the mean, and the package's recall function
    returns the mean. Recomputing per query is the only way to get at the variance, and the
    variance is the whole subject.
    """
    if truth.identifiers.shape != found.identifiers.shape:
        raise DataError(
            f"{tuple(truth.identifiers.shape)} truth and {tuple(found.identifiers.shape)} found"
        )
    rows = int(truth.identifiers.shape[0])
    hits = torch.zeros(rows)
    for row in range(rows):
        wanted = set(truth.identifiers[row].tolist())
        got = set(found.identifiers[row].tolist())
        hits[row] = len(wanted & got) / float(len(wanted))
    return hits


def estimate(values: torch.Tensor) -> Estimate:
    """A mean and its standard error, from the sample's own spread.

    The standard error is the sample standard deviation over the square root of the count, which
    assumes the queries are independent. They are, because they are drawn independently, and the
    neighbours within a query are not, which is exactly why the per query recall is the unit
    rather than the individual neighbour.
    """
    if int(values.numel()) < 2:
        raise ConfigError("a standard error needs at least two samples")
    return Estimate(
        mean=float(values.mean()),
        error=float(values.std(unbiased=True) / math.sqrt(int(values.numel()))),
        samples=int(values.numel()),
    )


def _measured(queries: int, probe: int = 4, count: int = 4096, seed: int = 0) -> torch.Tensor:
    """Per query recall for one configuration on one query sample."""
    corpus = gaussian(count=count, dimension=32, seed=seed)
    searched, probes = held_out(corpus, count=queries)
    truth = search(probes, searched.vectors, k=10)
    index = IVFIndex(32, partitions=64, probe=probe)
    index.build(searched.vectors)
    found, _ = index.search(probes, k=10)
    return per_query_recall(truth, found)


def the_error_falls_as_one_over_root_n(
    sizes: Sequence[int] = (10, 50, 200, 1000),
) -> list[dict]:
    """How much a larger query sample buys, which is the square root and no more.

    A hundred times the queries for ten times the precision. That is the rate for any mean and
    there is no way around it, so the practical question is never how to get a tighter interval
    cheaply but whether the interval you can afford is tight enough for the comparison you want
    to make.
    """
    if not sizes:
        raise ConfigError("there is nothing to sweep")
    rows = []
    for size in sizes:
        result = estimate(_measured(size))
        rows.append(
            {
                "queries": size,
                "mean": round(result.mean, 4),
                "error": round(result.error, 4),
                "width": round(result.width, 4),
            }
        )
    return rows


def a_hundred_queries_gives_two_points() -> dict:
    """The number this package's own measurements should be read against.

    Most modules here use a hundred queries, which gives a standard error of 0.018 and an
    interval seven points wide, so plus or minus 3.6. Every comparison in the package that
    turns on less than that is not a result from one sample, and is described as noise where
    it appears. Stating the figure once, here, is cheaper than restating it everywhere.
    """
    result = estimate(_measured(100))
    return {
        "queries": 100,
        "mean": round(result.mean, 4),
        "error": round(result.error, 4),
        "plus_or_minus": round(1.96 * result.error, 4),
        "width": round(result.width, 4),
        "about_two_points": 0.01 < 1.96 * result.error < 0.04,
    }


def recall_at_ten_is_not_ten_coin_flips() -> dict:
    """Why the binomial estimate of the error is wrong, and in which direction.

    It is not, but by much less than this module first claimed. The argument was that a query
    either lands where the index looked or it does not, so its ten neighbours share one fate
    and the effective sample size is a hundred rather than a thousand, which would understate
    the error by about a factor of three.

    Measured: the binomial gives 0.0153 and the truth is 0.0182, an understatement of
    nineteen percent. The slots are correlated and only mildly. Worth knowing because the
    usual quick calculation turns out not to be badly wrong, and because the reason it is
    nearly right is the distribution measured in the_per_query_recall_is_not_bimodal.
    """
    values = _measured(100)
    measured = estimate(values)
    mean = float(values.mean())
    binomial = math.sqrt(mean * (1 - mean) / (100 * 10))
    return {
        "queries": 100,
        "neighbours": 1000,
        "measured_error": round(measured.error, 4),
        "binomial_error": round(binomial, 4),
        "ratio": round(measured.error / binomial, 2),
        "binomial_understates": measured.error > binomial,
    }


def the_per_query_recall_is_not_bimodal() -> dict:
    """The distribution behind that, which was expected to be two humps and is one.

    The prediction was that most queries would get nearly all their neighbours or nearly none,
    since what decides is whether the opened partitions contain the neighbourhood, and that the
    middle would be empty. Measured over five hundred queries: 47 percent land between 0.3 and
    0.7, one percent above 0.9 and seven percent below 0.1. The middle is where most of the mass
    is.

    Which explains why the binomial estimate above is nearly right. A query typically gets some
    of its neighbours rather than all or none, so the ten slots are only weakly tied together,
    and the correlation the binomial ignores is small.
    """
    values = _measured(500)
    return {
        "queries": 500,
        "mean": round(float(values.mean()), 4),
        "share_above_nine_tenths": round(float((values >= 0.9).float().mean()), 4),
        "share_below_a_tenth": round(float((values <= 0.1).float().mean()), 4),
        "share_in_the_middle": round(
            float(((values > 0.3) & (values < 0.7)).float().mean()), 4
        ),
        "most_mass_is_in_the_middle": float(((values > 0.3) & (values < 0.7)).float().mean())
        > 0.4,
    }


def the_worst_queries_are_the_ones_to_report(
    quantiles: Sequence[float] = (0.01, 0.05, 0.1, 0.5),
) -> list[dict]:
    """What a mean hides, which is the tail a service actually gets complaints about.

    The mean recall of an index says what happens on average and nothing about the worst case.
    On
    this configuration the bottom five percent of queries get almost nothing, and those queries
    are not random: they are the ones whose neighbourhoods straddle a partition boundary, so the
    same queries fail every time and the same users see it every time.
    """
    if not quantiles:
        raise ConfigError("there is nothing to sweep")
    values = _measured(500)
    return [
        {
            "quantile": quantile,
            "recall": round(float(torch.quantile(values, quantile)), 4),
        }
        for quantile in quantiles
    ]


def the_tail_is_much_worse_than_the_mean() -> dict:
    """The gap between the average and the fifth percentile, stated as one number."""
    rows = {
        row["quantile"]: row["recall"] for row in the_worst_queries_are_the_ones_to_report()
    }
    values = _measured(500)
    return {
        "mean": round(float(values.mean()), 4),
        "median": rows[0.5],
        "fifth_percentile": rows[0.05],
        "first_percentile": rows[0.01],
        "the_tail_is_worse": rows[0.05] < float(values.mean()),
        "gap": round(float(values.mean()) - rows[0.05], 4),
    }


def a_paired_comparison_is_much_more_sensitive(
    left_probe: int = 4, right_probe: int = 5, queries: int = 100
) -> dict:
    """Why the sweeps in this package are trustworthy when the absolute numbers are not.

    Two configurations run on the same queries share their difficulty. A query whose
    neighbourhood is awkward is awkward for both, so the per query difference removes most of
    the
    variance and its standard error is far smaller than either configuration's own.

    Measured on probe four against probe five: means of 0.376 and 0.422, intervals of 0.340 to
    0.412 and 0.387 to 0.457, which overlap. The paired difference is 0.046 with an error of
    0.0067, which is 6.8 standard errors from zero. Same data, opposite conclusions, and the
    paired one is right because the queries really were the same.
    """
    if left_probe == right_probe:
        raise ConfigError("comparing a configuration with itself measures nothing")
    corpus = gaussian(count=4096, dimension=32)
    searched, probes = held_out(corpus, count=queries)
    truth = search(probes, searched.vectors, k=10)
    index = IVFIndex(32, partitions=64, probe=left_probe)
    index.build(searched.vectors)
    left_found, _ = index.search(probes, k=10)
    index.probe = right_probe
    right_found, _ = index.search(probes, k=10)

    left = per_query_recall(truth, left_found)
    right = per_query_recall(truth, right_found)
    left_estimate = estimate(left)
    right_estimate = estimate(right)
    difference = estimate(right - left)
    return {
        "left": left_estimate.as_dict(),
        "right": right_estimate.as_dict(),
        "difference_mean": round(difference.mean, 4),
        "difference_error": round(difference.error, 4),
        "unpaired_intervals_overlap": left_estimate.overlaps(right_estimate),
        "paired_difference_in_errors": round(
            abs(difference.mean) / max(difference.error, 1e-9), 2
        ),
        "paired_is_more_sensitive": difference.error < left_estimate.error,
    }


def pairing_shrinks_the_error(probes: Sequence[int] = (2, 4, 8, 16)) -> list[dict]:
    """How much pairing buys across a sweep, rather than at one point.

    The ratio of the unpaired error to the paired one, against a probe two baseline. It is
    1.71 at probe four, 1.10 at probe eight and 0.87 at probe sixteen, so pairing helps at a
    small gap and hurts at a large one.

    The reason it reverses is that shared difficulty turns into complementarity. At a small gap
    the two configurations succeed and fail on the same queries and the difference is nearly
    constant. At a large gap the queries the weak setting misses are exactly the ones the strong
    setting picks up, so the difference is large where one fails and zero where both succeed,
    and it varies more than either configuration does on its own.
    """
    if len(probes) < 2:
        raise ConfigError("a sweep needs at least two settings")
    corpus = gaussian(count=4096, dimension=32)
    searched, queries = held_out(corpus, count=100)
    truth = search(queries, searched.vectors, k=10)
    index = IVFIndex(32, partitions=64, probe=probes[0])
    index.build(searched.vectors)
    rows = []
    baseline = None
    for probe in probes:
        index.probe = probe
        found, _ = index.search(queries, k=10)
        values = per_query_recall(truth, found)
        alone = estimate(values)
        if baseline is None:
            baseline = values
            rows.append(
                {
                    "probe": probe,
                    "mean": round(alone.mean, 4),
                    "error": round(alone.error, 4),
                    "paired_error": None,
                    "shrink": None,
                }
            )
            continue
        paired = estimate(values - baseline)
        rows.append(
            {
                "probe": probe,
                "mean": round(alone.mean, 4),
                "error": round(alone.error, 4),
                "paired_error": round(paired.error, 4),
                "shrink": round(alone.error / max(paired.error, 1e-9), 2),
            }
        )
    return rows


def pairing_helps_at_a_small_gap_and_hurts_at_a_large_one() -> dict:
    """The shape of that, which narrows the usual advice rather than supporting it.

    Pair adjacent settings in a sweep. Do not assume pairing rescues a comparison between two
    structures that behave completely differently, because there the errors stop cancelling and
    start adding.
    """
    rows = {row["probe"]: row for row in pairing_shrinks_the_error()}
    near, far = rows[4], rows[16]
    return {
        "shrink_at_a_small_gap": near["shrink"],
        "shrink_at_a_large_gap": far["shrink"],
        "helps_when_close": near["shrink"] > 1.2,
        "hurts_when_far": far["shrink"] < 1.0,
        "helps_more_when_close": near["shrink"] > far["shrink"],
    }


def how_many_queries_to_detect_a_gap(
    gaps: Sequence[float] = (0.01, 0.02, 0.05, 0.1),
) -> list[dict]:
    """The question a benchmark should be asked before it is run.

    To see a gap of size d with ninety five percent confidence needs roughly (4 s / d) squared
    queries, where s is the per query standard deviation. At the spread measured here that is
    thousands of queries for a one point difference and dozens for a ten point one, which is why
    a table of a hundred query benchmarks separated by a point is not evidence of anything.
    """
    if not gaps:
        raise ConfigError("there is nothing to sweep")
    values = _measured(500)
    spread = float(values.std(unbiased=True))
    return [
        {
            "gap": gap,
            "unpaired_queries": math.ceil((3.92 * spread / gap) ** 2),
            "spread": round(spread, 4),
        }
        for gap in gaps
    ]


def a_one_point_difference_needs_thousands_of_queries() -> dict:
    """The headline of that table, which is the number nobody budgets for."""
    rows = {row["gap"]: row for row in how_many_queries_to_detect_a_gap()}
    return {
        "for_one_point": rows[0.01]["unpaired_queries"],
        "for_two_points": rows[0.02]["unpaired_queries"],
        "for_ten_points": rows[0.1]["unpaired_queries"],
        "one_point_needs_thousands": rows[0.01]["unpaired_queries"] > 1000,
        "ten_points_needs_dozens": rows[0.1]["unpaired_queries"] < 300,
    }


def two_indexes_that_look_different_may_not_be() -> dict:
    """A comparison from another module, rerun with its uncertainty attached.

    The forest against the inverted file at one pair of settings, chosen without matching
    their costs. The inverted file wins, 0.553 against 0.473, the intervals do not overlap,
    and the paired difference is significant.

    Which proves nothing about either structure, because probe eight costs 575 distances per
    query and eight trees cost 461. index/forest.py compares them at matched cost across six
    budgets and finds the opposite ordering. A comparison at unmatched settings is not made
    trustworthy by a tight interval, and this is the clearest illustration in the package of
    why a claim that one index beats another has to say what it held fixed.
    """
    corpus = gaussian(count=4096, dimension=32)
    searched, probes = held_out(corpus, count=100)
    truth = search(probes, searched.vectors, k=10)

    partitioned = IVFIndex(32, partitions=64, probe=8)
    partitioned.build(searched.vectors)
    ivf_found, _ = partitioned.search(probes, k=10)

    forest = ForestIndex(32, trees=8, leaf_size=64)
    forest.build(searched.vectors)
    forest_found, _ = forest.search(probes, k=10)

    left = estimate(per_query_recall(truth, ivf_found))
    right = estimate(per_query_recall(truth, forest_found))
    paired = estimate(
        per_query_recall(truth, forest_found) - per_query_recall(truth, ivf_found)
    )
    return {
        "ivf": left.as_dict(),
        "forest": right.as_dict(),
        "intervals_overlap": left.overlaps(right),
        "paired_difference": round(paired.mean, 4),
        "paired_error": round(paired.error, 4),
        "paired_is_significant": abs(paired.mean) > 1.96 * paired.error,
        "but_the_costs_are_not_matched": True,
    }


def the_seed_moves_the_answer_by_more_than_the_error(
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
) -> dict:
    """Whether the corpus seed matters as much as the query sample, which it does.

    Every measurement in this package fixes a corpus seed as well as a query sample, and the
    corpus is as much a sample as the queries are. Rerunning one configuration on five different
    corpora gives a spread comparable to the query sampling error, so a number quoted from one
    corpus carries two sources of uncertainty and only one of them is usually acknowledged.

    Which is not a reason to average over corpora. It is a reason to prefer paired comparisons
    on
    one corpus, where the corpus is held fixed and cancels out entirely.
    """
    if len(seeds) < 2:
        raise ConfigError("a spread needs at least two seeds")
    means = torch.tensor([float(_measured(100, seed=seed).mean()) for seed in seeds])
    within = estimate(_measured(100)).error
    return {
        "seeds": len(seeds),
        "means": [round(float(value), 4) for value in means],
        "across_seeds": round(float(means.std(unbiased=True)), 4),
        "within_one_seed": round(within, 4),
        "comparable": abs(float(means.std(unbiased=True)) - within) < within * 2,
    }


def an_estimate_of_one_sample_is_refused() -> bool:
    """Whether a standard error from one observation is caught.

    It would be zero, which reads as perfect certainty from a single measurement, and that is
    the
    most misleading number this module could produce.
    """
    try:
        estimate(torch.tensor([0.5]))
    except ConfigError:
        return True
    return False


def a_mismatched_pair_is_refused() -> bool:
    """Whether scoring a result against a truth of a different shape is caught."""
    truth = Neighbours(torch.zeros(4, 10, dtype=torch.long), torch.zeros(4, 10))
    found = Neighbours(torch.zeros(4, 5, dtype=torch.long), torch.zeros(4, 5))
    try:
        per_query_recall(truth, found)
    except DataError:
        return True
    return False


def comparing_a_configuration_with_itself_is_refused() -> bool:
    """Whether a paired comparison of one setting against itself is caught.

    The difference would be exactly zero with an error of exactly zero, which is a true
    statement
    that answers no question and looks like a very confident null result.
    """
    try:
        a_paired_comparison_is_much_more_sensitive(left_probe=4, right_probe=4)
    except ConfigError:
        return True
    return False


def an_interval_widens_with_the_error() -> dict:
    """That the interval arithmetic does what it says.

    Checked because everything in this module is stated in terms of it, and an interval that was
    one standard error wide rather than two would make every conclusion here about twice as
    confident as it should be.
    """
    tight = Estimate(mean=0.5, error=0.01, samples=100)
    loose = Estimate(mean=0.5, error=0.05, samples=100)
    return {
        "tight_width": round(tight.width, 4),
        "loose_width": round(loose.width, 4),
        "wider": loose.width > tight.width,
        "two_standard_errors_each_side": abs(tight.width - 2 * 1.96 * 0.01) < 1e-9,
        "they_overlap": tight.overlaps(loose),
    }


def two_estimates_far_apart_do_not_overlap() -> dict:
    """And that the overlap test separates things it should."""
    low = Estimate(mean=0.2, error=0.01, samples=100)
    high = Estimate(mean=0.8, error=0.01, samples=100)
    return {
        "low": low.as_dict(),
        "high": high.as_dict(),
        "overlap": low.overlaps(high),
        "separated": not low.overlaps(high),
    }

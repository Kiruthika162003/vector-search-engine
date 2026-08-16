from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from vse.errors import ConfigError
from vse.index.forest import ForestIndex
from vse.index.graph import GraphIndex
from vse.index.ivf import IVFIndex
from vse.index.lsh import LSHIndex
from vse.index.tree import TreeIndex
from vse.vectors.dataset import clustered, gaussian, held_out
from vse.vectors.exact import Neighbours, search

# How much of a recall number is the structure and how much is the seed.
#
# Everything in eval/report.py compares structures by their recall at a matched distance budget,
# and every one of those numbers came from a single build with a single seed. Four of the six
# structures here draw random numbers during the build: the inverted file seeds its k means, the
# forest seeds its projections, the hierarchy seeds its level draw, and the hash seeds its
# planes. The graph and the kd tree draw nothing at all and are bit for bit reproducible.
#
# So the comparisons in report.py rest on an assumption nobody stated, which is that the seed
# moves the recall by less than the gap between the structures. This module measures that
# assumption instead of assuming it.
#
# The assumption holds here, with room to spare. Across eight seeds at a fixed setting the
# inverted file has a standard deviation of 0.012, the forest 0.0095 and the hash 0.0118, and
# the structures are 0.078 and 0.163 apart. Nothing flips on any seed.
#
# Two things about that surprised me. The first is that the three deviations are the same size
# despite the builds drawing wildly different amounts of randomness: the hash draws sixty four
# numbers and the inverted file more than a thousand, and it makes no difference. The second is
# that the seed noise is the same size as the query sampling error, 0.012 against 0.012 on two
# hundred queries. I expected the queries to dominate and they do not. Both are about one point
# of recall, so the noise floor on any single number in this package is roughly two points once
# both are counted, and a gap narrower than that has not been measured.
#
# The noise is not constant across the range. It peaks where the recall is low and vanishes at
# both ends, which is what a bounded quantity has to do: at probe 1 every seed is uniformly bad
# and at probe 32 every seed is exact, and there is only room to disagree in between. The
# consequence for a reader of report.py is that the cheap end of a frontier is the noisy end.


SEEDED = ("ivf", "forest", "lsh")
FIXED = ("graph", "tree")


@dataclass
class Spread:
    """What one structure did across a set of seeds."""

    name: str
    recalls: list[float]
    distances: list[float]

    def __post_init__(self) -> None:
        if not self.recalls:
            raise ConfigError(f"{self.name} was never measured")

    @property
    def mean(self) -> float:
        """The average recall over the seeds."""
        return statistics.fmean(self.recalls)

    @property
    def deviation(self) -> float:
        """The sample standard deviation, or zero from a single seed."""
        if len(self.recalls) < 2:
            return 0.0
        return statistics.stdev(self.recalls)

    @property
    def range(self) -> float:
        """Best minus worst, which is what a reader of one number actually risks."""
        return max(self.recalls) - min(self.recalls)

    @property
    def mean_distances(self) -> float:
        """The average cost, to confirm the seeds are being compared at one budget."""
        return statistics.fmean(self.distances)

    @property
    def cost_range(self) -> float:
        """How much the seed moved the cost, which should be small for a fixed setting."""
        return max(self.distances) - min(self.distances)

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "index": self.name,
            "seeds": len(self.recalls),
            "mean": round(self.mean, 4),
            "deviation": round(self.deviation, 4),
            "range": round(self.range, 4),
            "distances": round(self.mean_distances, 1),
        }


def _setup(
    count: int = 2048,
    dimension: int = 32,
    queries: int = 200,
    k: int = 10,
    kind: str = "gaussian",
    seed: int = 0,
):
    """One corpus, one query set, one exact answer."""
    if kind == "gaussian":
        corpus = gaussian(count=count, dimension=dimension, seed=seed)
    elif kind == "clustered":
        corpus = clustered(count=count, dimension=dimension, clusters=16, seed=seed)
    else:
        raise ConfigError(f"{kind} is not a corpus")
    searched, probes = held_out(corpus, count=queries)
    truth = search(probes, searched.vectors, k=k)
    return searched.vectors, probes, truth


def build(name: str, dimension: int, seed: int):
    """One index of the named kind, seeded where the kind takes a seed.

    The settings are held at roughly matched cost across the kinds so the spreads can be read
    next to each other. They are not tuned, because tuning per seed would be measuring the tuner
    rather than the seed.
    """
    if name == "ivf":
        return IVFIndex(dimension, partitions=32, probe=4, seed=seed)
    if name == "forest":
        return ForestIndex(dimension, trees=8, leaf_size=32, seed=seed)
    if name == "lsh":
        return LSHIndex(dimension, bits=8, tables=8, seed=seed)
    if name == "graph":
        return GraphIndex(dimension, degree=16, ef=32)
    if name == "tree":
        return TreeIndex(dimension, leaf_size=16)
    raise ConfigError(f"{name} is not an index")


def per_query_recall(truth: Neighbours, found: Neighbours) -> list[float]:
    """The share of each query's true neighbours that were returned.

    Needed separately from the mean because the query sampling error is a property of the spread
    across queries, and the whole point of the module is to compare that spread against the one
    across seeds.
    """
    wanted = truth.identifiers
    rows = int(wanted.shape[0])
    width = int(wanted.shape[1])
    scores = []
    for row in range(rows):
        present = set(found.identifiers[row].tolist())
        hits = sum(1 for identifier in wanted[row].tolist() if identifier in present)
        scores.append(hits / width)
    return scores


def query_standard_error(truth: Neighbours, found: Neighbours) -> float:
    """The standard error of a recall number, from the queries alone.

    This is the noise a reader already accepts when they read any recall in this package. Every
    seed spread below is judged against it, because a source of variation smaller than the one
    already present is not worth reporting.
    """
    scores = per_query_recall(truth, found)
    if len(scores) < 2:
        raise ConfigError("a standard error needs at least two queries")
    return statistics.stdev(scores) / (len(scores) ** 0.5)


def across_seeds(
    name: str,
    seeds: Sequence[int] = (0, 1, 2, 3, 4, 5, 6, 7),
    corpus: torch.Tensor | None = None,
    probes: torch.Tensor | None = None,
    truth: Neighbours | None = None,
    k: int = 10,
) -> Spread:
    """Build the same structure under several seeds and record what each one reached.

    The corpus and the queries are held fixed, so the only thing changing is the build. That is
    the paired comparison and it is the only one that isolates the seed.
    """
    if not seeds:
        raise ConfigError("there is nothing to sweep")
    if corpus is None or probes is None or truth is None:
        corpus, probes, truth = _setup()
    recalls = []
    costs = []
    for seed in seeds:
        index = build(name, int(corpus.shape[1]), seed)
        index.build(corpus)
        found, stats = index.search(probes, k=k)
        scores = per_query_recall(truth, found)
        recalls.append(statistics.fmean(scores))
        costs.append(float(stats.distances_per_query))
    return Spread(name=name, recalls=recalls, distances=costs)


def the_deterministic_structures_do_not_move() -> dict:
    """The control.

    The graph and the kd tree take no seed, so eight builds give one number eight times. If that
    ever fails then something in the build is reading a global generator and every other number
    in the package is quietly seed dependent.
    """
    corpus, probes, truth = _setup()
    spreads = {
        name: across_seeds(name, corpus=corpus, probes=probes, truth=truth) for name in FIXED
    }
    return {
        "ranges": {name: spread.range for name, spread in spreads.items()},
        "nothing_moved": all(spread.range == 0.0 for spread in spreads.values()),
        "and_the_costs_did_not_either": all(
            spread.cost_range == 0.0 for spread in spreads.values()
        ),
        "recalls": {name: round(spread.mean, 4) for name, spread in spreads.items()},
    }


def the_seeded_structures_move(
    seeds: Sequence[int] = (0, 1, 2, 3, 4, 5, 6, 7),
) -> list[dict]:
    """How far the three seeded structures wander.

    The ranges are 0.032 for the inverted file, 0.030 for the forest and 0.030 for the hash,
    with standard deviations of 0.012, 0.0095 and 0.0118. They are the same number three times
    over, which is the finding: see the next function.
    """
    if not seeds:
        raise ConfigError("there is nothing to sweep")
    corpus, probes, truth = _setup()
    rows = []
    for name in SEEDED:
        spread = across_seeds(name, seeds=seeds, corpus=corpus, probes=probes, truth=truth)
        rows.append(spread.as_dict())
    return rows


def the_seed_and_the_query_noise_are_the_same_size() -> dict:
    """The comparison that decides whether any of this matters.

    I expected the query sampling error to dominate, on the grounds that two hundred queries is
    a small sample and a build is a whole structure. They come out equal: 0.012 from the seed
    and 0.012 from the queries, on the same measurement.

    So neither is safe to ignore and the two add. A recall reported from one build against one
    query set carries about 0.017 of noise once both are counted, and the honest reading of any
    single number in this package is plus or minus two points.
    """
    corpus, probes, truth = _setup()
    index = build("ivf", int(corpus.shape[1]), seed=0)
    index.build(corpus)
    found, _ = index.search(probes, k=10)
    from_queries = query_standard_error(truth, found)
    spread = across_seeds("ivf", corpus=corpus, probes=probes, truth=truth)
    return {
        "from_the_seed": round(spread.deviation, 4),
        "from_the_queries": round(from_queries, 4),
        "they_are_level": abs(from_queries - spread.deviation) < 0.004,
        "combined": round((from_queries**2 + spread.deviation**2) ** 0.5, 4),
        "ratio": round(from_queries / max(spread.deviation, 1e-9), 2),
    }


def the_number_of_random_draws_does_not_predict_the_spread() -> dict:
    """I expected the structure with the fewest random draws to be the most seed sensitive.

    Eight planes of eight bits is sixty four numbers deciding the whole hash, against thirty two
    centroids of thirty two dimensions for the inverted file, which is more than a thousand.
    A coarser lottery should give a wider spread and it does not: 0.0118 for the hash and 0.0120
    for the inverted file, with the forest between them at 0.0095.

    The reason is that the spread is not set by the build at all. It is set by how many queries
    sit near a boundary the build happens to draw, and that is a property of the corpus and the
    setting rather than of the method. Whatever the structure, a seed moves the same handful of
    borderline queries from one side to the other.
    """
    corpus, probes, truth = _setup()
    rows = {}
    for name in SEEDED:
        index = build(name, int(corpus.shape[1]), seed=0)
        index.build(corpus)
        found, _ = index.search(probes, k=10)
        spread = across_seeds(name, corpus=corpus, probes=probes, truth=truth)
        rows[name] = {
            "from_the_seed": round(spread.deviation, 4),
            "from_the_queries": round(query_standard_error(truth, found), 4),
        }
    seeds = [row["from_the_seed"] for row in rows.values()]
    return {
        "rows": rows,
        "widest": max(rows, key=lambda name: rows[name]["from_the_seed"]),
        "narrowest": min(rows, key=lambda name: rows[name]["from_the_seed"]),
        "they_are_all_within_a_third": max(seeds) < min(seeds) * 1.35,
        "the_hash_is_not_the_worst": (
            rows["lsh"]["from_the_seed"] < max(seeds) + 1e-9
            and rows["lsh"]["from_the_seed"] > min(seeds) - 1e-9
        ),
    }


def does_the_leader_change_with_the_seed(
    seeds: Sequence[int] = (0, 1, 2, 3, 4, 5, 6, 7),
) -> dict:
    """Whether a comparison run on one seed picks the same winner as the mean.

    This is the question report.py implicitly answers yes to, and here the answer is yes on all
    eight seeds. The means are 0.473, 0.395 and 0.231, and the widest seed to seed range is
    0.032, so the structures never come close to changing places. Reassuring, and it says
    nothing about a comparison where the gap is smaller, which is the next two functions.
    """
    if not seeds:
        raise ConfigError("there is nothing to sweep")
    corpus, probes, truth = _setup()
    spreads = {
        name: across_seeds(name, seeds=seeds, corpus=corpus, probes=probes, truth=truth)
        for name in SEEDED
    }
    by_mean = sorted(SEEDED, key=lambda name: -spreads[name].mean)
    per_seed = []
    for position in range(len(seeds)):
        ordering = sorted(SEEDED, key=lambda name: -spreads[name].recalls[position])
        per_seed.append(ordering)
    agreeing = sum(1 for ordering in per_seed if ordering[0] == by_mean[0])
    return {
        "by_mean": by_mean,
        "winners": [ordering[0] for ordering in per_seed],
        "seeds_agreeing_with_the_mean": agreeing,
        "seeds": len(seeds),
        "every_seed_agreed": agreeing == len(seeds),
        "the_narrowest_gap": round(
            min(
                spreads[by_mean[position]].mean - spreads[by_mean[position + 1]].mean
                for position in range(len(by_mean) - 1)
            ),
            4,
        ),
        "means": {name: round(spreads[name].mean, 4) for name in SEEDED},
    }


def close_pairs_flip_and_distant_ones_do_not(
    seeds: Sequence[int] = (0, 1, 2, 3, 4, 5, 6, 7),
) -> list[dict]:
    """For every pair, how often the seed reverses the answer.

    Nothing flips. The three pairs are 0.078, 0.163 and 0.242 apart against deviations near
    0.012, so every pair is six or more standard deviations wide and no seed comes close to
    reversing one. Reporting the gap next to the flip count is still the right habit, because
    the point is that these particular comparisons are decided rather than that comparisons
    generally are. The null in the next function is what a gap of zero looks like.
    """
    if not seeds:
        raise ConfigError("there is nothing to sweep")
    corpus, probes, truth = _setup()
    spreads = {
        name: across_seeds(name, seeds=seeds, corpus=corpus, probes=probes, truth=truth)
        for name in SEEDED
    }
    rows = []
    for left in range(len(SEEDED)):
        for right in range(left + 1, len(SEEDED)):
            first, second = SEEDED[left], SEEDED[right]
            gap = spreads[first].mean - spreads[second].mean
            reversals = sum(
                1
                for position in range(len(seeds))
                if (spreads[first].recalls[position] - spreads[second].recalls[position]) * gap
                < 0
            )
            rows.append(
                {
                    "pair": f"{first} against {second}",
                    "gap": round(abs(gap), 4),
                    "flips": reversals,
                    "of": len(seeds),
                }
            )
    return sorted(rows, key=lambda row: row["gap"])


def the_flip_rate_follows_the_gap() -> dict:
    """State the shape above as a single claim, so a test can hold it.

    No pair here flips, because the narrowest gap is 0.078 against a deviation of 0.012. The
    claim a test can hold is the weaker and true one: a gap of six standard deviations is never
    reversed by a reseed.
    """
    rows = close_pairs_flip_and_distant_ones_do_not()
    return {
        "rows": rows,
        "narrowest_gap": rows[0]["gap"],
        "narrowest_flips": rows[0]["flips"],
        "widest_gap": rows[-1]["gap"],
        "widest_flips": rows[-1]["flips"],
        "the_widest_never_flips": rows[-1]["flips"] == 0,
        "nothing_flips_at_all": all(row["flips"] == 0 for row in rows),
        "and_it_is_wider": rows[-1]["gap"] > rows[0]["gap"],
    }


def averaging_over_seeds_narrows_the_interval(
    counts: Sequence[int] = (1, 2, 4),
) -> list[dict]:
    """The standard error of a mean over m seeds falls as one over the root of m.

    Not a discovery, a check, because the next function uses the law to work out how many seeds
    a comparison needs and a formula used for that should be confirmed against the data it is
    applied to. Measured against predicted over sixteen seeds of the hash: 0.0112 against
    0.0112, 0.0065 against 0.0080, 0.0040 against 0.0056.

    The measured figures run a little below the prediction, and the count stops at four on
    purpose. Averaging eight leaves only two groups to take a deviation over, and a deviation
    from two numbers is not an estimate of anything.
    """
    if not counts:
        raise ConfigError("there is nothing to sweep")
    corpus, probes, truth = _setup()
    spread = across_seeds(
        "lsh",
        seeds=tuple(range(16)),
        corpus=corpus,
        probes=probes,
        truth=truth,
    )
    rows = []
    for count in counts:
        if count > len(spread.recalls):
            raise ConfigError(f"{count} seeds were not measured")
        groups = [
            statistics.fmean(spread.recalls[start : start + count])
            for start in range(0, len(spread.recalls) - count + 1, count)
        ]
        rows.append(
            {
                "seeds_averaged": count,
                "groups": len(groups),
                "deviation": round(statistics.stdev(groups) if len(groups) > 1 else 0.0, 5),
                "predicted": round(spread.deviation / (count**0.5), 5),
            }
        )
    return rows


def how_many_seeds_a_comparison_needs(gap: float = 0.02) -> dict:
    """The seed count that shrinks the seed noise below a gap worth calling.

    Separating two settings two points apart, with the seed noise held at half the gap, takes
    two builds for the inverted file and the hash and one for the forest. That is cheap and
    nobody does it, which is the practical point of the module. It also understates the job,
    since it counts only the seed and the query noise is the same size again.
    """
    if gap <= 0.0:
        raise ConfigError(f"{gap} is not a gap")
    corpus, probes, truth = _setup()
    rows = []
    for name in SEEDED:
        spread = across_seeds(name, corpus=corpus, probes=probes, truth=truth)
        needed = 1
        while needed < 64 and spread.deviation / (needed**0.5) > gap / 2.0:
            needed += 1
        rows.append(
            {
                "index": name,
                "deviation": round(spread.deviation, 4),
                "seeds_needed": needed,
            }
        )
    return {
        "gap": gap,
        "rows": rows,
        "most": max(row["seeds_needed"] for row in rows),
        "all_are_affordable": max(row["seeds_needed"] for row in rows) <= 16,
    }


def the_spread_across_the_probe_range(
    probe_values: Sequence[int] = (1, 2, 4, 8, 16, 32),
) -> list[dict]:
    """The deviation at each point of the frontier, with the recall next to it.

    Recalls of 0.177, 0.299, 0.473, 0.689, 0.897 and 1.000 against deviations of 0.0062, 0.0135,
    0.0120, 0.0096, 0.0080 and 0.0000. The reading is in the next function.
    """
    if not probe_values:
        raise ConfigError("there is nothing to sweep")
    corpus, probes, truth = _setup()
    dimension = int(corpus.shape[1])
    rows = []
    for probe in probe_values:
        recalls = []
        for seed in range(8):
            index = IVFIndex(dimension, partitions=32, probe=probe, seed=seed)
            index.build(corpus)
            found, _ = index.search(probes, k=10)
            recalls.append(statistics.fmean(per_query_recall(truth, found)))
        rows.append(
            {
                "probe": probe,
                "recall": round(statistics.fmean(recalls), 4),
                "deviation": round(statistics.stdev(recalls), 5),
            }
        )
    return rows


def the_noise_is_worst_in_the_middle() -> dict:
    """And the shrinking is not monotone, which is the part I had wrong.

    I expected the deviation to fall with the probe count throughout. It rises first and falls
    afterwards, peaking at probe 2 where the recall is 0.299. That is what a bounded quantity
    does: at probe 1 every seed is uniformly bad and at probe 32 every seed is exact, and there
    is only room to disagree in between. The peak sits below the middle of the recall range
    rather than at it, because the probe count buys recall unevenly.

    The practical form of this is that the noisy part of a frontier is its cheap end, which is
    also the part of it people quote.
    """
    rows = the_spread_across_the_probe_range()
    peak = max(rows, key=lambda row: row["deviation"])
    return {
        "rows": rows,
        "peak_probe": peak["probe"],
        "peak_deviation": peak["deviation"],
        "peak_recall": peak["recall"],
        "the_peak_is_interior": rows[0]["probe"] < peak["probe"] < rows[-1]["probe"],
        "the_ends_are_quieter": (
            peak["deviation"] > rows[0]["deviation"]
            and peak["deviation"] > rows[-1]["deviation"]
        ),
    }


def the_corpus_seed_moves_it_no_further_than_the_index_seed(
    seeds: Sequence[int] = (0, 1, 2, 3, 4, 5, 6, 7),
) -> dict:
    """Redrawing the whole corpus, against reseeding the build on a fixed one.

    I expected the corpus to dominate: a new corpus is a new problem and a new seed is only a
    new fit. It comes out at 0.0099 against 0.0120, so if anything the index seed moves it
    further.

    That is a fact about this corpus rather than about corpora. A Gaussian draw of two thousand
    vectors in thirty two dimensions is very close to every other such draw, so redrawing it
    changes almost nothing. On a corpus with real structure the redraw would change what there
    is to find, and this measurement would go the other way.
    """
    if not seeds:
        raise ConfigError("there is nothing to sweep")
    corpus, probes, truth = _setup()
    from_index = across_seeds("ivf", seeds=seeds, corpus=corpus, probes=probes, truth=truth)
    from_corpus = []
    for seed in seeds:
        other, other_probes, other_truth = _setup(seed=seed)
        index = build("ivf", int(other.shape[1]), seed=0)
        index.build(other)
        found, _ = index.search(other_probes, k=10)
        from_corpus.append(statistics.fmean(per_query_recall(other_truth, found)))
    corpus_deviation = statistics.stdev(from_corpus)
    return {
        "from_the_index_seed": round(from_index.deviation, 4),
        "from_the_corpus_seed": round(corpus_deviation, 4),
        "the_corpus_does_not_dominate": corpus_deviation <= from_index.deviation,
        "they_are_level": abs(corpus_deviation - from_index.deviation) < 0.005,
        "ratio": round(corpus_deviation / max(from_index.deviation, 1e-9), 2),
    }


def a_clustered_corpus_is_less_seed_sensitive() -> dict:
    """Structure in the corpus makes the initial draw matter less, not more.

    I argued the opposite: a k means fit on sixteen real clusters either finds them or does not,
    and which happens is decided by where the initial centroids land. The measurement is a
    deviation of 0.0009 on the clustered corpus against 0.0120 on the Gaussian one.

    Every seed finds the clusters. Thirty two partitions over sixteen real groups, refined by
    Lloyd iterations, is an easy problem, and the recall saturates at 0.9996 where there is no
    room left for a seed to matter. The same bounded quantity effect as the probe sweep, seen
    from a different direction.
    """
    rows = {}
    for kind in ("gaussian", "clustered"):
        corpus, probes, truth = _setup(kind=kind)
        spread = across_seeds("ivf", corpus=corpus, probes=probes, truth=truth)
        rows[kind] = {
            "mean": round(spread.mean, 4),
            "deviation": round(spread.deviation, 4),
            "range": round(spread.range, 4),
        }
    return {
        "rows": rows,
        "the_clustered_corpus_moves_less": (
            rows["clustered"]["deviation"] < rows["gaussian"]["deviation"]
        ),
        "because_it_saturates": rows["clustered"]["mean"] > 0.99,
        "and_scores_higher": rows["clustered"]["mean"] > rows["gaussian"]["mean"],
    }


def the_cost_moves_with_the_seed_too() -> dict:
    """And by more than I assumed, which weakens every matched cost comparison a little.

    I wrote this expecting a fixed setting to cost a fixed amount. Across eight seeds the
    inverted file's distance count moves by 3.7 percent of its mean and the hash by 8.8, since a
    different partitioning gives differently sized partitions and probing four of them scans a
    different number of vectors. Only the forest is steady, at 0.25 percent, because its leaf
    size is fixed by construction.

    So a matched cost comparison in this package is matched to a few percent rather than
    exactly. That is small next to the recall gaps being compared and it is not nothing, and it
    is the sort of thing worth stating rather than discovering later.
    """
    corpus, probes, truth = _setup()
    rows = {}
    for name in SEEDED:
        spread = across_seeds(name, corpus=corpus, probes=probes, truth=truth)
        rows[name] = {
            "mean_distances": round(spread.mean_distances, 1),
            "range": round(spread.cost_range, 1),
            "relative": round(spread.cost_range / max(spread.mean_distances, 1e-9), 4),
        }
    return {
        "rows": rows,
        "the_forest_is_steady": rows["forest"]["relative"] < 0.01,
        "the_others_are_not": rows["ivf"]["relative"] > 0.02,
        "worst": max(rows, key=lambda name: rows[name]["relative"]),
        "worst_relative": max(row["relative"] for row in rows.values()),
    }


def compare_the_structures(seeds: Sequence[int] = (0, 1, 2, 3, 4, 5, 6, 7)) -> list[dict]:
    """Every structure with its spread, which is how report.py should have printed it."""
    if not seeds:
        raise ConfigError("there is nothing to sweep")
    corpus, probes, truth = _setup()
    rows = []
    for name in (*SEEDED, *FIXED):
        spread = across_seeds(name, seeds=seeds, corpus=corpus, probes=probes, truth=truth)
        rows.append(spread.as_dict())
    return sorted(rows, key=lambda row: -row["mean"])


def two_identical_methods_flip_half_the_time(
    trials: int = 8,
) -> dict:
    """The null, which is what a gap of zero looks like.

    None of the pairs above flips, so there is nothing in this module showing what a flip is.
    Split sixteen seeds of one index at one setting into two groups and call them rival methods.
    The true gap is exactly zero, so the winner is a coin toss. It comes out 6 of 8, which eight
    trials cannot tell apart from a half and is the second thing this measures: a rate read off
    eight comparisons is worth about a quarter either way.

    Worth having because it calibrates the rest. A comparison of two real methods whose seeds
    disagree anywhere near half the time is a comparison of two methods that are the same,
    whatever their means happen to say.
    """
    if trials < 2:
        raise ConfigError(f"{trials} is not enough trials")
    corpus, probes, truth = _setup()
    spread = across_seeds(
        "ivf",
        seeds=tuple(range(trials * 2)),
        corpus=corpus,
        probes=probes,
        truth=truth,
    )
    left = spread.recalls[:trials]
    right = spread.recalls[trials:]
    wins = sum(1 for position in range(trials) if left[position] > right[position])
    return {
        "trials": trials,
        "wins": wins,
        "share": round(wins / trials, 4),
        "it_is_near_a_half": abs(wins / trials - 0.5) <= 0.25,
        "the_means_are_level": abs(statistics.fmean(left) - statistics.fmean(right)) < 0.01,
    }


def a_single_trial_null_is_refused() -> bool:
    """One trial cannot show a rate."""
    try:
        two_identical_methods_flip_half_the_time(trials=1)
    except ConfigError:
        return True
    return False


def an_unknown_index_is_refused() -> bool:
    """A name that is not a structure is a mistake, not an empty result."""
    try:
        build("bloom", dimension=8, seed=0)
    except ConfigError:
        return True
    return False


def an_empty_seed_list_is_refused() -> bool:
    """And measuring a spread over no seeds is not a spread."""
    try:
        across_seeds("ivf", seeds=())
    except ConfigError:
        return True
    return False


def a_spread_of_nothing_is_refused() -> bool:
    """Nor is a Spread with no recalls in it."""
    try:
        Spread(name="ivf", recalls=[], distances=[])
    except ConfigError:
        return True
    return False


def summarise() -> dict:
    """The module in one mapping, for the command line and for logging."""
    rows = compare_the_structures()
    leader = does_the_leader_change_with_the_seed()
    against_queries = the_seed_and_the_query_noise_are_the_same_size()
    return {
        "structures": rows,
        "widest_spread": max(rows, key=lambda row: row["range"])["index"],
        "leader_by_mean": leader["by_mean"][0],
        "seeds_agreeing": leader["seeds_agreeing_with_the_mean"],
        "seed_noise": against_queries["from_the_seed"],
        "query_noise": against_queries["from_the_queries"],
    }

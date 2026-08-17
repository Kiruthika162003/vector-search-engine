"""Decide whether one index really beat another, or whether the seed did.

Every recall number in this package carries two kinds of noise. The queries were a sample, so a
different sample would have given a different number, and eval/stability.py puts that at about
0.012 on two hundred queries. The build drew random numbers, so a different seed would have
given a different number too, and that is also about 0.012. They add, so the floor under a
single measurement is around 0.017 and a gap narrower than a couple of points has not been
measured.

This script takes two index settings, runs each under several seeds against the same corpus and
the same queries, and says whether the difference between them survives.

    python examples/is_this_difference_real.py
    python examples/is_this_difference_real.py --left "ivf:32:4" --right "ivf:64:4" --seeds 8

A setting is written kind:parameter:parameter. For the inverted file that is partitions and
probe; for the forest, trees and leaf size; for the hash, bits and tables.
"""

from __future__ import annotations

import argparse
import statistics
import sys

from vse.errors import VectorSearchError
from vse.eval.stability import _setup, per_query_recall
from vse.index.forest import ForestIndex
from vse.index.ivf import IVFIndex
from vse.index.lsh import LSHIndex


def parse(setting: str, dimension: int, seed: int):
    """Turn a kind:parameter:parameter string into an index."""
    parts = setting.split(":")
    if len(parts) != 3:
        raise ValueError(f"{setting} is not a setting, expected kind:first:second")
    kind, first, second = parts[0], int(parts[1]), int(parts[2])
    if kind == "ivf":
        return IVFIndex(dimension, partitions=first, probe=second, seed=seed)
    if kind == "forest":
        return ForestIndex(dimension, trees=first, leaf_size=second, seed=seed)
    if kind == "lsh":
        return LSHIndex(dimension, bits=first, tables=second, seed=seed)
    raise ValueError(f"{kind} is not an index kind")


def run(setting: str, seeds: int, corpus, probes, truth):
    """Recalls and costs for one setting under several seeds."""
    recalls = []
    costs = []
    for seed in range(seeds):
        index = parse(setting, int(corpus.shape[1]), seed)
        index.build(corpus)
        found, stats = index.search(probes, k=10)
        recalls.append(statistics.fmean(per_query_recall(truth, found)))
        costs.append(float(stats.distances_per_query))
    return recalls, costs


def verdict(left: list[float], right: list[float]) -> dict:
    """Whether the gap between two settings survives their spreads.

    A paired comparison, since both settings saw the same corpus and the same queries under
    matched seeds. Paired is the right test here and it is much more sensitive than comparing
    the two means with their own spreads, because the query sample cancels.
    """
    gap = statistics.fmean(left) - statistics.fmean(right)
    paired = [a - b for a, b in zip(left, right, strict=True)]
    spread = statistics.stdev(paired) if len(paired) > 1 else 0.0
    error = spread / (len(paired) ** 0.5) if paired else 0.0
    wins = sum(1 for value in paired if value > 0)
    return {
        "gap": gap,
        "standard_error": error,
        "ratio": abs(gap) / error if error > 0 else float("inf"),
        "wins": wins,
        "of": len(paired),
        "decided": (
            abs(gap) > 2.0 * error if error > 0 else abs(gap) > 0.0 and len(paired) > 1
        ),
    }


def report(left_name, right_name, left, right, left_cost, right_cost, call) -> str:
    """The comparison, with enough detail to disagree with it."""
    lines = [
        f"{left_name} against {right_name}, {call['of']} seeds each",
        "",
        f"{'':<14}{'mean':>8}{'spread':>9}{'worst':>8}{'best':>8}{'cost':>9}",
        f"{left_name:<14}{statistics.fmean(left):>8.4f}"
        f"{statistics.stdev(left) if len(left) > 1 else 0.0:>9.4f}"
        f"{min(left):>8.4f}{max(left):>8.4f}{statistics.fmean(left_cost):>9.0f}",
        f"{right_name:<14}{statistics.fmean(right):>8.4f}"
        f"{statistics.stdev(right) if len(right) > 1 else 0.0:>9.4f}"
        f"{min(right):>8.4f}{max(right):>8.4f}{statistics.fmean(right_cost):>9.0f}",
        "",
        f"paired gap        {call['gap']:+.4f}",
        f"standard error    {call['standard_error']:.4f}",
        f"gap over error    {call['ratio']:.1f}",
        f"seeds won         {call['wins']} of {call['of']}",
        "",
    ]
    if not call["decided"]:
        lines.append("the difference is inside the noise, so these two settings are level")
        lines.append("more seeds would narrow the error as one over the root of the count")
    elif call["gap"] > 0:
        lines.append(f"{left_name} is ahead and the gap survives the seed spread")
    else:
        lines.append(f"{right_name} is ahead and the gap survives the seed spread")
    ratio = statistics.fmean(left_cost) / max(statistics.fmean(right_cost), 1e-9)
    if abs(ratio - 1.0) > 0.15:
        lines.append("")
        lines.append(
            f"note that the costs differ by a factor of {ratio:.2f}, so this is not a "
            "matched cost comparison and the recall gap is partly bought"
        )
    return "\n".join(lines)


def main(argv=None) -> int:
    """Parse arguments, run both settings, print the verdict."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--left", default="ivf:32:4")
    parser.add_argument("--right", default="forest:8:32")
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--count", type=int, default=2048)
    parser.add_argument("--dimension", type=int, default=32)
    parser.add_argument("--queries", type=int, default=200)
    args = parser.parse_args(argv)

    corpus, probes, truth = _setup(
        count=args.count, dimension=args.dimension, queries=args.queries
    )
    try:
        left, left_cost = run(args.left, args.seeds, corpus, probes, truth)
        right, right_cost = run(args.right, args.seeds, corpus, probes, truth)
    except (ValueError, VectorSearchError) as problem:
        print(problem)
        return 1
    call = verdict(left, right)
    print(report(args.left, args.right, left, right, left_cost, right_cost, call))
    return 0


if __name__ == "__main__":
    sys.exit(main())

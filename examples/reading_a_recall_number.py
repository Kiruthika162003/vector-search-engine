"""Attach an interval to a recall number, and say which comparisons it can decide.

Every recall in this package is a mean over a query sample, and eval/significance.py measures
what that mean is worth: at a hundred queries the standard error is about 0.018, so the fourth
decimal is noise and two configurations differing by three points are indistinguishable from
one sample.

This script takes two configurations, reports both with intervals, and then does the paired
comparison, which is far more sensitive because the two ran on the same queries. The usual
outcome is that the intervals overlap and the paired difference is decisive, which are opposite
conclusions from the same data, and the paired one is right.

    python examples/reading_a_recall_number.py
    python examples/reading_a_recall_number.py --left 4 --right 5 --queries 200
"""

from __future__ import annotations

import argparse
import math
import sys

import torch

from vse.eval.significance import estimate, per_query_recall
from vse.index.ivf import IVFIndex
from vse.vectors.dataset import gaussian, held_out
from vse.vectors.exact import search


def measure(corpus, queries, truth, dimension, partitions, probe):
    """Per query recall for one probe count."""
    index = IVFIndex(dimension, partitions=partitions, probe=probe)
    index.build(corpus)
    found, stats = index.search(queries, k=10)
    return per_query_recall(truth, found), stats.distances_per_query


def describe(label: str, values: torch.Tensor, cost: float) -> str:
    """One configuration as a line with its interval."""
    result = estimate(values)
    low, high = result.interval
    return (
        f"{label:<12} recall {result.mean:.3f} "
        f"[{low:.3f}, {high:.3f}]  {cost:.1f} distances per query"
    )


def queries_needed(spread: float, gap: float) -> int:
    """How many queries an unpaired comparison of a given gap would need."""
    if gap <= 0:
        return 0
    return math.ceil((3.92 * spread / gap) ** 2)


def main(argv=None) -> int:
    """Compare two probe counts, unpaired and paired."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--count", type=int, default=4096)
    parser.add_argument("--dimension", type=int, default=32)
    parser.add_argument("--partitions", type=int, default=64)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--left", type=int, default=4)
    parser.add_argument("--right", type=int, default=5)
    args = parser.parse_args(argv)

    if args.left == args.right:
        print("comparing a configuration with itself measures nothing")
        return 1

    corpus = gaussian(count=args.count, dimension=args.dimension)
    searched, probes = held_out(corpus, count=args.queries)
    truth = search(probes, searched.vectors, k=10)

    left, left_cost = measure(
        searched.vectors, probes, truth, args.dimension, args.partitions, args.left
    )
    right, right_cost = measure(
        searched.vectors, probes, truth, args.dimension, args.partitions, args.right
    )

    print(f"{args.queries} queries over a corpus of {int(searched.vectors.shape[0])}")
    print()
    print(describe(f"probe {args.left}", left, left_cost))
    print(describe(f"probe {args.right}", right, right_cost))
    print()

    low = estimate(left)
    high = estimate(right)
    overlap = low.overlaps(high)
    print(
        "the intervals overlap, so read separately these are indistinguishable"
        if overlap
        else "the intervals do not overlap, so read separately these differ"
    )

    paired = estimate(right - left)
    ratio = abs(paired.mean) / max(paired.error, 1e-9)
    print()
    print(
        f"paired difference {paired.mean:+.4f} with an error of {paired.error:.4f}, "
        f"which is {ratio:.1f} standard errors from zero"
    )
    if ratio > 2:
        print("so the paired comparison decides it, and it is the correct one")
        print("the two configurations ran on the same queries, so their errors cancel")
    else:
        print("so even the paired comparison cannot decide it at this sample size")

    spread = float(torch.cat([left, right]).std(unbiased=True))
    print()
    print(f"the per query spread is {spread:.3f}, so an unpaired comparison would need")
    for gap in (0.01, 0.02, 0.05, 0.1):
        print(f"  {queries_needed(spread, gap):>6} queries to see a gap of {gap}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Diagnose a recall drop that is really a change in the queries.

The failure this script is for looks like an index regression and is not one. Recall on the
production dashboard falls, nothing about the index changed, and every offline replay of the old
query set still passes. What moved was the traffic.

vse/vectors/drift.py finds that only one of the three kinds of query drift costs anything, and
that it is the one nobody expects: queries pulled towards the dense middle of the corpus do much
worse than queries pushed out into the empty tail. This script runs that diagnosis on a corpus
you supply the shape of, and prints the three things worth knowing.

    python examples/when_the_queries_move.py
    python examples/when_the_queries_move.py --scale 0.3 --probe 8

The output has three parts. What the drift cost, measured against the drifted queries' own true
neighbours, which is the only fair scoring. What it appears to have cost if you make the common
mistake of scoring against the answer the old queries had. And what it takes to get the recall
back, which is always more probes and never a rebuild.
"""

from __future__ import annotations

import argparse
import sys

from vse.index.ivf import IVFIndex
from vse.vectors.drift import _setup, scale
from vse.vectors.exact import identifier_overlap, search


def run(
    count: int,
    dimension: int,
    queries: int,
    magnitude: float,
    probe: int,
    partitions: int,
):
    """Build once, then search with the undrifted and drifted query sets."""
    corpus, probes = _setup(count=count, dimension=dimension, queries=queries)
    index = IVFIndex(dimension, partitions=partitions, probe=probe)
    index.build(corpus)

    before_truth = search(probes, corpus, k=10)
    before_found, before_stats = index.search(probes, k=10)

    moved = scale(probes, magnitude).queries
    after_truth = search(moved, corpus, k=10)
    after_found, after_stats = index.search(moved, k=10)

    return {
        "corpus": corpus,
        "index": index,
        "moved": moved,
        "baseline": identifier_overlap(before_truth, before_found),
        "baseline_cost": float(before_stats.distances_per_query),
        "drifted": identifier_overlap(after_truth, after_found),
        "drifted_cost": float(after_stats.distances_per_query),
        "against_the_old_truth": identifier_overlap(before_truth, after_found),
        "answer_overlap": identifier_overlap(before_truth, after_truth),
    }


def repair(
    result,
    dimension: int,
    partitions: int,
    probe: int,
) -> list[tuple[int, float, float]]:
    """How much probe budget it takes to get back to the baseline."""
    truth = search(result["moved"], result["corpus"], k=10)
    rows = []
    budget = probe
    while budget <= partitions:
        index = IVFIndex(dimension, partitions=partitions, probe=budget)
        index.build(result["corpus"])
        found, stats = index.search(result["moved"], k=10)
        rows.append(
            (budget, identifier_overlap(truth, found), float(stats.distances_per_query))
        )
        if rows[-1][1] >= result["baseline"]:
            break
        budget *= 2
    return rows


def report(result, rows, magnitude: float) -> str:
    """The three part answer."""
    lines = [
        f"queries scaled to {magnitude} of their radius about the corpus mean",
        "",
        "what it cost",
        f"  before      {result['baseline']:.4f} at {result['baseline_cost']:.0f} distances",
        f"  after       {result['drifted']:.4f} at {result['drifted_cost']:.0f} distances",
        f"  lost        {result['baseline'] - result['drifted']:+.4f}",
        "",
        "what it looks like if you score against the old answer",
        f"  apparent    {result['against_the_old_truth']:.4f}",
        f"  real        {result['drifted']:.4f}",
        f"  invented    {result['drifted'] - result['against_the_old_truth']:+.4f}",
        f"  the true answer itself moved: overlap {result['answer_overlap']:.4f}",
        "",
        "what it takes to get back",
    ]
    for probe, recall, cost in rows:
        marker = " reaches the baseline" if recall >= result["baseline"] else ""
        lines.append(f"  probe {probe:<3} {recall:.4f} at {cost:.0f} distances{marker}")
    if rows[-1][1] < result["baseline"]:
        lines.append("  the baseline was not reached inside the partition count")
    else:
        lines.append("")
        lines.append(
            f"  that is {rows[-1][2] / result['baseline_cost']:.1f} times the original cost, "
            "which is the price of the drift"
        )
    lines.append("")
    lines.append(
        "a rebuild would not have helped: the corpus did not move, only the queries did"
    )
    return "\n".join(lines)


def main(argv=None) -> int:
    """Parse arguments, run the diagnosis, print it."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--count", type=int, default=4096)
    parser.add_argument("--dimension", type=int, default=32)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--scale", type=float, default=0.25, dest="magnitude")
    parser.add_argument("--probe", type=int, default=4)
    parser.add_argument("--partitions", type=int, default=64)
    args = parser.parse_args(argv)

    result = run(
        args.count,
        args.dimension,
        args.queries,
        args.magnitude,
        args.probe,
        args.partitions,
    )
    rows = repair(result, args.dimension, args.partitions, args.probe)
    print(report(result, rows, args.magnitude))
    return 0


if __name__ == "__main__":
    sys.exit(main())

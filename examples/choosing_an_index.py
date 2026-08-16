"""Pick a structure for a corpus by measuring, rather than by reading a table.

The package's own conclusion is that there is no best index: eval/report.py finds three
different leaders across four corpora at the same distance budget. So the only defensible way
to choose is to run the candidates on the corpus that will actually be served, at matched cost,
and read the frontier.

This script does that end to end. Point it at a corpus generator, give it a budget in distances
per query, and it prints what each structure reaches inside that budget and what settings got it
there.

    python examples/choosing_an_index.py
    python examples/choosing_an_index.py --corpus clustered --budget 500
"""

from __future__ import annotations

import argparse
import sys

from vse.errors import VectorSearchError
from vse.index.forest import ForestIndex
from vse.index.graph import GraphIndex
from vse.index.hnsw import HNSWIndex
from vse.index.ivf import IVFIndex
from vse.quantize.binary import BinaryIndex
from vse.vectors.dataset import clustered, gaussian, held_out, on_a_subspace
from vse.vectors.exact import identifier_overlap, search

CORPORA = {
    "gaussian": lambda count, dimension: gaussian(count=count, dimension=dimension),
    "clustered": lambda count, dimension: clustered(
        count=count, dimension=dimension, clusters=16
    ),
    "subspace": lambda count, dimension: on_a_subspace(
        count=count, dimension=dimension, intrinsic=6
    ),
}


def candidates(dimension: int):
    """One entry per structure and setting, spanning a range of costs.

    The settings are chosen to cover a range rather than to flatter anybody, because the budget
    filter below is what makes the comparison fair and it needs points on both sides of the
    budget to interpolate between.
    """
    return (
        [
            (
                "ivf",
                f"probe {probe}",
                lambda probe=probe: IVFIndex(dimension, partitions=64, probe=probe),
            )
            for probe in (1, 2, 4, 8, 16, 32)
        ]
        + [
            (
                "graph",
                f"beam {beam}",
                lambda beam=beam: GraphIndex(dimension, degree=16, ef=beam),
            )
            for beam in (10, 16, 32, 64, 128)
        ]
        + [
            ("hnsw", f"beam {beam}", lambda beam=beam: HNSWIndex(dimension, degree=16, ef=beam))
            for beam in (10, 16, 32, 64, 128)
        ]
        + [
            (
                "forest",
                f"{trees} trees",
                lambda trees=trees: ForestIndex(dimension, trees=trees, leaf_size=64),
            )
            for trees in (2, 4, 8, 16, 32)
        ]
        + [
            (
                "binary",
                f"rerank {depth}",
                lambda depth=depth: BinaryIndex(dimension, rerank=depth),
            )
            for depth in (20, 50, 100, 200, 400)
        ]
    )


def best_within(corpus, queries, truth, budget, dimension):
    """For each structure, the best recall it reaches without exceeding the budget.

    A structure whose cheapest setting is already over the budget appears with nothing, which is
    the honest report: it cannot answer inside that budget and saying so is more useful than
    omitting it.
    """
    best: dict = {}
    for name, setting, make in candidates(dimension):
        index = make()
        try:
            index.build(corpus)
            found, stats = index.search(queries, k=10)
        except VectorSearchError:
            continue
        if stats.distances_per_query > budget:
            continue
        recall = identifier_overlap(truth, found)
        current = best.get(name)
        if current is None or recall > current["recall"]:
            best[name] = {
                "setting": setting,
                "recall": round(recall, 3),
                "distances": round(stats.distances_per_query, 1),
                "memory_bytes": index.memory_bytes(),
            }
    return best


def report(best: dict, budget: float, corpus_size: int) -> str:
    """The table, sorted by what each structure reached."""
    if not best:
        return f"no structure answered inside {budget} distances per query"
    rows = sorted(best.items(), key=lambda pair: -pair[1]["recall"])
    width = max(len(name) for name in best)
    lines = [
        f"budget {budget} distances per query, corpus of {corpus_size}",
        f"a full scan would cost {corpus_size}, so the budget is "
        f"{round(corpus_size / budget, 1)} times cheaper",
        "",
        f"{'index'.ljust(width)}  {'setting':<12}  {'recall':>6}  "
        f"{'spent':>8}  {'megabytes':>9}",
    ]
    for name, row in rows:
        lines.append(
            f"{name.ljust(width)}  {row['setting']:<12}  {row['recall']:>6}  "
            f"{row['distances']:>8}  {row['memory_bytes'] / 1e6:>9.2f}"
        )
    leader, top = rows[0]
    lines.append("")
    lines.append(f"the leader here is {leader} at {top['setting']}, reaching {top['recall']}")
    if len(rows) > 1:
        runner, second = rows[1]
        if top["recall"] - second["recall"] < 0.04:
            lines.append(
                f"but {runner} reaches {second['recall']}, which is inside the noise at a "
                "hundred queries, so treat this as a tie"
            )
    return "\n".join(lines)


def main(argv=None) -> int:
    """Parse arguments, run the comparison, print the table."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--corpus", default="gaussian", choices=sorted(CORPORA))
    parser.add_argument("--count", type=int, default=4096)
    parser.add_argument("--dimension", type=int, default=32)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--budget", type=float, default=1000.0)
    args = parser.parse_args(argv)

    corpus = CORPORA[args.corpus](args.count, args.dimension)
    searched, probes = held_out(corpus, count=args.queries)
    truth = search(probes, searched.vectors, k=10)
    best = best_within(searched.vectors, probes, truth, args.budget, args.dimension)
    print(report(best, args.budget, int(searched.vectors.shape[0])))
    return 0


if __name__ == "__main__":
    sys.exit(main())

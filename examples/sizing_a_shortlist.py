"""Pick a first stage and a shortlist depth for a recall target.

Two stage retrieval has exactly one design variable, which is the shortlist. vse/serve/rerank.py
shows why: an exact rescoring of a candidate set keeps every true neighbour the set contains, so
final recall equals shortlist recall and the rerank cannot be tuned. All that is left is
choosing a cheap scorer and choosing how deep to let it go.

This script sweeps both and prints the cheapest combination that clears a target, priced in full
distance equivalents so the two stages can be added up.

    python examples/sizing_a_shortlist.py
    python examples/sizing_a_shortlist.py --target 0.95 --dimension 64

Watch the totals rather than the recalls. The saving over a full scan is smaller than the
framing of two stage retrieval usually suggests, and at thirty two dimensions the best of these
is a factor of four. It grows with the dimension, since the first stage cost falls relative to a
full precision distance and the exact stage is the same price either way.
"""

from __future__ import annotations

import argparse
import sys

from vse.errors import VectorSearchError
from vse.serve.rerank import (
    projected_shortlist,
    sign_shortlist,
    staged_search,
)
from vse.vectors.dataset import clustered, gaussian, held_out, on_a_subspace
from vse.vectors.exact import search

CORPORA = {
    "gaussian": lambda count, dimension: gaussian(count=count, dimension=dimension),
    "clustered": lambda count, dimension: clustered(
        count=count, dimension=dimension, clusters=16
    ),
    "subspace": lambda count, dimension: on_a_subspace(
        count=count, dimension=dimension, intrinsic=6
    ),
}


def stages(dimension: int):
    """The first stages to try, with a label for each."""
    rows = [("sign", "1 bit per dimension", lambda depth, q, c: sign_shortlist(q, c, depth))]
    rank = 2
    while rank <= dimension:
        rows.append(
            (
                f"rank {rank}",
                f"projection to {rank} of {dimension}",
                lambda depth, q, c, rank=rank: projected_shortlist(q, c, depth, rank=rank),
            )
        )
        rank *= 2
    return rows


def sweep(corpus, probes, truth, dimension: int, target: float, depths):
    """For each first stage, the cheapest depth that clears the target."""
    results = []
    for name, description, make in stages(dimension):
        cheapest = None
        for depth in depths:
            if depth > int(corpus.shape[0]):
                break
            try:
                staged = staged_search(probes, corpus, truth, make(depth, probes, corpus), k=10)
            except VectorSearchError:
                break
            if staged.final_recall >= target:
                cheapest = {
                    "stage": name,
                    "description": description,
                    "depth": depth,
                    "recall": staged.final_recall,
                    "scan": staged.first_stage_cost,
                    "rerank": staged.rerank_cost,
                    "total": staged.total_cost,
                }
                break
        results.append(
            cheapest
            or {
                "stage": name,
                "description": description,
                "depth": None,
                "recall": None,
                "scan": None,
                "rerank": None,
                "total": None,
            }
        )
    return results


def report(rows, target: float, corpus_size: int) -> str:
    """The table, cheapest first, with the misses at the bottom."""
    priced = sorted([row for row in rows if row["total"] is not None], key=lambda r: r["total"])
    missed = [row for row in rows if row["total"] is None]
    lines = [
        f"target {target} at ten neighbours, corpus of {corpus_size}",
        f"a full scan costs {corpus_size} distance equivalents",
        "",
        f"{'stage':<10}{'depth':>7}{'recall':>9}{'scan':>9}"
        f"{'rerank':>9}{'total':>9}{'saving':>9}",
    ]
    for row in priced:
        lines.append(
            f"{row['stage']:<10}{row['depth']:>7}{row['recall']:>9.4f}"
            f"{row['scan']:>9.0f}{row['rerank']:>9.0f}{row['total']:>9.0f}"
            f"{corpus_size / row['total']:>8.1f}x"
        )
    for row in missed:
        lines.append(f"{row['stage']:<10}{'never reached the target':>43}")
    if not priced:
        lines.append("")
        lines.append("nothing cleared the target, so lower it or widen the depths")
        return "\n".join(lines)
    best = priced[0]
    lines.append("")
    lines.append(
        f"cheapest is {best['stage']} at a shortlist of {best['depth']}, "
        f"{best['description']}"
    )
    lines.append(
        f"the scan is {best['scan'] / best['total']:.0%} of the bill and the rerank is "
        f"{best['rerank'] / best['total']:.0%}"
    )
    if best["rerank"] > best["scan"]:
        lines.append(
            "the rerank costs more than the scan, so a finer first stage would be the next "
            "thing to try"
        )
    else:
        lines.append(
            "the scan costs more than the rerank, so a cheaper first stage would be the next "
            "thing to try"
        )
    return "\n".join(lines)


def main(argv=None) -> int:
    """Parse arguments, run the sweep, print the table."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--corpus", default="gaussian", choices=sorted(CORPORA))
    parser.add_argument("--count", type=int, default=4096)
    parser.add_argument("--dimension", type=int, default=32)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--target", type=float, default=0.9)
    args = parser.parse_args(argv)

    corpus = CORPORA[args.corpus](args.count, args.dimension)
    searched, probes = held_out(corpus, count=args.queries)
    truth = search(probes, searched.vectors, k=10)
    depths = [10, 20, 50, 100, 200, 400, 800, 1600, 3200]
    rows = sweep(searched.vectors, probes, truth, args.dimension, args.target, depths)
    print(report(rows, args.target, int(searched.vectors.shape[0])))
    return 0


if __name__ == "__main__":
    sys.exit(main())

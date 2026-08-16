"""Build an index, save it, load it, serve a batch under a deadline, and verify the answers.

Every module in the package measures one thing. This runs the pieces a deployment actually
strings together, in order, and prints what each step cost. Nothing here is new: it is the
persistence format, the batching, the deadline and the invariant checks, wired up the way they
would be.

Two things worth watching in the output. The saved index reproduces the original's answers
identically rather than approximately, which storage/persist.py insists on because an index that
comes back 0.999 the same has a bug that gets blamed on approximation. And the deadline costs
less recall than it costs work, because partitions are opened in order of promise.

    python examples/a_serving_pipeline.py
    python examples/a_serving_pipeline.py --budget 300 --batch 32
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import torch

from vse.index.ivf import IVFIndex
from vse.serve.batch import batched_search
from vse.serve.limits import bounded_partition_search
from vse.storage.persist import load, peek, save
from vse.vectors.dataset import gaussian, held_out
from vse.vectors.exact import identifier_overlap, search
from vse.verify.differential import RULES


def build(corpus, dimension, partitions, probe):
    """Step one: fit the index."""
    index = IVFIndex(dimension, partitions=partitions, probe=probe)
    index.build(corpus)
    return index


def round_trip(index, path: Path):
    """Step two: write it and read it back, checking the answers are identical."""
    header = save(index, path)
    peeked = peek(path)
    restored = load(path)
    return header, peeked, restored


def check(found, corpus, queries, k: int):
    """Step five: run the invariant rules over whatever came back."""
    return [name for name, rule in RULES if rule(found, corpus, queries, k) is not None]


def main(argv=None) -> int:
    """Run the pipeline and print what each step did."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--count", type=int, default=8192)
    parser.add_argument("--dimension", type=int, default=32)
    parser.add_argument("--partitions", type=int, default=90)
    parser.add_argument("--probe", type=int, default=8)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--budget", type=float, default=600.0)
    args = parser.parse_args(argv)

    corpus = gaussian(count=args.count, dimension=args.dimension)
    searched, probes = held_out(corpus, count=args.queries)
    truth = search(probes, searched.vectors, k=10)

    print(f"corpus of {int(searched.vectors.shape[0])} at {args.dimension} dimensions")
    print(f"{args.queries} queries held out, k of 10")
    print()

    index = build(searched.vectors, args.dimension, args.partitions, args.probe)
    direct, direct_stats = index.search(probes, k=10)
    print("built")
    print(f"  {args.partitions} partitions, probe {args.probe}")
    print(f"  recall {identifier_overlap(truth, direct):.3f}")
    print(f"  {direct_stats.distances_per_query:.1f} distances per query")
    print(
        f"  a full scan would be {int(searched.vectors.shape[0])}, so "
        f"{int(searched.vectors.shape[0]) / direct_stats.distances_per_query:.1f} times cheaper"
    )
    print()

    with tempfile.TemporaryDirectory() as scratch:
        path = Path(scratch) / "served.vse"
        header, peeked, restored = round_trip(index, path)
        after, _ = restored.search(probes, k=10)
        print("saved and loaded")
        print(f"  {path.stat().st_size} bytes, digest {header.digest}")
        print(f"  header says {peeked.kind} over {peeked.count} vectors")
        print(
            f"  answers identical: {bool(torch.equal(direct.identifiers, after.identifiers))}"
        )
        print()

        batched, batch_stats, batching = batched_search(
            restored, probes, k=10, batch=args.batch
        )
        print(f"served in batches of {args.batch}")
        print(f"  {batching.batches} batches, mean size {batching.mean_batch:.1f}")
        print(f"  recall {identifier_overlap(truth, batched):.3f}")
        print(f"  {batch_stats.distances_per_query:.1f} distances per query")
        print(
            f"  corpus bytes per query {batching.corpus_bytes_per_query:.0f}, "
            "which is where batching pays"
        )
        print()

        served = bounded_partition_search(restored, probes, 10, args.budget)
        work_share = served.stats.distances_per_query / direct_stats.distances_per_query
        recall_share = identifier_overlap(truth, served.found) / max(
            identifier_overlap(truth, direct), 1e-9
        )
        print(f"served again under a deadline of {args.budget} distances")
        print(f"  recall {identifier_overlap(truth, served.found):.3f}")
        print(f"  {served.stats.distances_per_query:.1f} distances per query")
        print(f"  {served.truncated} of {served.queries} queries were cut off")
        print(
            f"  kept {recall_share:.1%} of the recall for {work_share:.1%} of the work, "
            "because partitions open in order of promise"
        )
        print()

        broken = check(served.found, searched.vectors, probes, 10)
        print("checked against the invariant rules")
        if broken:
            print(f"  violations: {broken}")
            return 1
        print(f"  all {len(RULES)} rules pass on the truncated result")
    return 0


if __name__ == "__main__":
    sys.exit(main())

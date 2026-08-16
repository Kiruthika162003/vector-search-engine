"""Find the cheapest setting that meets a recall target, and say what it will really deliver.

eval/autotune.py measures two things a deployment needs and neither is obvious. Tuning to
exactly the target leaves most fresh query samples below it, because the tuning sample is a
sample; three points of margin fixes that and costs a step of the knob. And a setting tuned on
one corpus means nothing on another: the same target needs probe 32 on a gaussian corpus and
probe 4 on a clustered one, a factor of six and a half in work.

This script does the tuning with the margin built in, then verifies on held out samples the
tuner never saw, which is the part usually skipped.

    python examples/tuning_to_a_target.py
    python examples/tuning_to_a_target.py --target 0.95 --margin 0.03
"""

from __future__ import annotations

import argparse
import sys

from vse.errors import VectorSearchError
from vse.eval.autotune import set_probe, sweep_setting
from vse.index.ivf import IVFIndex
from vse.vectors.dataset import clustered, gaussian, held_out
from vse.vectors.exact import identifier_overlap, search

CORPORA = {
    "gaussian": lambda count, dimension: gaussian(count=count, dimension=dimension),
    "clustered": lambda count, dimension: clustered(
        count=count, dimension=dimension, clusters=16
    ),
}

PROBES = (1, 2, 4, 8, 16, 24, 32, 48, 64)


def tune(corpus, queries, truth, dimension, partitions, target):
    """Sweep the probe count and return the cheapest setting clearing the target."""
    index = IVFIndex(dimension, partitions=partitions, probe=1)
    index.build(corpus)
    settings = sweep_setting(index, corpus, queries, truth, PROBES, set_probe, k=10)
    clearing = [setting for setting in settings if setting.recall >= target]
    if not clearing:
        return None, settings
    return min(clearing, key=lambda setting: setting.distances), settings


def verify(corpus, dimension, partitions, probe, samples, count, seed_from):
    """Run the chosen setting on query samples the tuner never saw."""
    index = IVFIndex(dimension, partitions=partitions, probe=probe)
    index.build(corpus)
    recalls = []
    for offset in range(samples):
        fresh = gaussian(count=count, dimension=dimension, seed=seed_from + offset)
        _, probes = held_out(fresh, count=100)
        truth = search(probes, corpus, k=10)
        found, _ = index.search(probes, k=10)
        recalls.append(identifier_overlap(truth, found))
    return recalls


def main(argv=None) -> int:
    """Tune with a margin, then check the tuning held."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--corpus", default="gaussian", choices=sorted(CORPORA))
    parser.add_argument("--count", type=int, default=8192)
    parser.add_argument("--dimension", type=int, default=32)
    parser.add_argument("--partitions", type=int, default=90)
    parser.add_argument("--target", type=float, default=0.9)
    parser.add_argument("--margin", type=float, default=0.03)
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args(argv)

    corpus = CORPORA[args.corpus](args.count, args.dimension)
    searched, probes = held_out(corpus, count=100)
    truth = search(probes, searched.vectors, k=10)

    plain, settings = tune(
        searched.vectors, probes, truth, args.dimension, args.partitions, args.target
    )
    padded, _ = tune(
        searched.vectors,
        probes,
        truth,
        args.dimension,
        args.partitions,
        args.target + args.margin,
    )

    print(f"sweeping probe over {list(PROBES)} on a {args.corpus} corpus of {args.count}")
    print()
    print(f"{'probe':>6}  {'recall':>7}  {'distances':>10}")
    for setting in settings:
        print(f"{setting.value:>6}  {setting.recall:>7.3f}  {setting.distances:>10.1f}")
    print()

    if plain is None:
        print(f"no setting in the sweep reaches {args.target}")
        print(f"the best available is {max(s.recall for s in settings):.3f}")
        return 1

    print(
        f"tuned to exactly {args.target}: probe {plain.value}, {plain.distances:.1f} distances"
    )
    if padded is None:
        print(f"no setting reaches {args.target + args.margin}, so no margin is available")
        return 1
    print(
        f"tuned to {args.target + args.margin:.2f} for margin: probe {padded.value}, "
        f"{padded.distances:.1f} distances"
    )
    print(f"the margin costs {padded.distances - plain.distances:.1f} distances per query")
    print()

    for label, setting in (("no margin", plain), ("with margin", padded)):
        try:
            recalls = verify(
                searched.vectors,
                args.dimension,
                args.partitions,
                setting.value,
                args.samples,
                args.count,
                seed_from=100,
            )
        except VectorSearchError as error:
            print(f"{label}: could not verify, {error}")
            continue
        below = sum(1 for value in recalls if value < args.target)
        print(
            f"{label} (probe {setting.value}): fresh samples "
            f"{[round(value, 3) for value in recalls]}"
        )
        print(f"  {below} of {len(recalls)} came in below the target of {args.target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

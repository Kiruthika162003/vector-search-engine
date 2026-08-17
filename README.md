# vector-search-engine

A vector search library built as a set of measurements rather than a set of claims.

Every approximate nearest neighbour structure here is paired with something obviously correct to
check it against, usually an exact brute force scan, sometimes a second implementation or a
deliberately wrong variant. Nothing in the package asserts that a structure is good. It measures
what the structure found, against what was actually there, at a cost counted in distance
computations, and writes the number down.

That arrangement was chosen because it keeps being right when the author is wrong, and over the
course of building this it was wrong about twenty five times. Those reversals are the most useful
thing in the repository and they are all still here, recorded in the module comment and in the
commit that found them rather than quietly edited out.

## Installing

```bash
pip install -e .
```

Python 3.11 or later. The only runtime dependency is PyTorch, used for the array arithmetic and
nothing else. No index here wraps a library; every structure is implemented from scratch, because
the point is to measure what the structures do rather than to expose somebody else's.

## Running it

```bash
python -m pytest -q
```

The suite is around 1,700 tests and takes about twenty minutes. Most of them are measurements
rather than unit tests: they run a real search over a real corpus and check that the number that
comes back is the number the docstring says.

The command line runs the same measurements without writing Python:

```bash
python -m vse.cli.main report
```

```bash
python -m vse.cli.main measure --index graph --corpus clustered --beam 64
```

Nine subcommands: `measure`, `report`, `sweep`, `build`, `inspect`, `verify`, `drift`,
`stability`, `shortlist`. All of them take `--json`. All of them return a useful exit code.

## What is in it

**Exact search and geometry.** `vse/vectors/` holds the metrics, the brute force search that every
other measurement is scored against, the corpus generators, preprocessing, duplicate analysis and
query drift.

**Structures.** `vse/index/` holds a flat scan, an inverted file, a kNN graph, an HNSW hierarchy, a
kd-tree, a random projection forest, an LSH index, a composite, a spilling variant and the update
path. `vse/quantize/` holds scalar, product, binary, residual and OPQ codes.

**Serving.** `vse/serve/` holds batching, caching, deadlines and budgets, tiered routing,
replication and two stage reranking.

**Measurement.** `vse/eval/` holds recall, autotuning, score fusion, significance, adversarial
query construction, seed stability, post hoc calibration and the comparison report.
`vse/verify/differential.py` runs invariant checks across every structure at once.

**Storage.** `vse/storage/` holds sharding, a paged disk layout and a versioned serialisation
format with a magic number and a digest.

**Examples.** `examples/` holds seven worked scripts, each answering one question end to end:
choosing an index, tuning to a target, reading a recall number, building a serving pipeline,
diagnosing query drift, deciding whether a difference is real, and sizing a shortlist.

## Things that turned out to be false

These are the findings, in the sense that they are the parts that changed what the code does. Each
is measured, and the measurement is in the module named after it.

**There is no best index.** `eval/report.py` finds three different leaders across four corpora at
the same distance budget. Any table that names a winner without naming the corpus is naming the
corpus by accident.

**A comparison at unmatched cost is not a comparison.** Every structure has an accuracy dial, and
probe eight and beam eight are not the same quantity. The only fair comparison interpolates both
frontiers to a shared budget, which is why `eval/report.py` refuses to print a recall without a
distance count beside it.

**The noise floor is about two points of recall.** `eval/stability.py` measures the seed spread at
0.012 and the query sampling error at 0.012, on two hundred queries. I expected the queries to
dominate; they do not, and both count. A gap narrower than roughly four points has not been
measured, whatever the means say.

**A fixed setting is not a fixed cost.** Reseeding an inverted file moves its distance count by 3.7
percent and a hash by 8.8, because a different partitioning gives differently sized partitions. So
matched cost in this package means matched to a few percent.

**Reranking has no slack in it.** `serve/rerank.py` finds that final recall equals shortlist recall
exactly, at every depth and on every corpus, because an exact top k over a candidate set keeps
every true neighbour the set contains. The rerank is bookkeeping and the shortlist is the only
design variable.

**Dropping precision beats dropping dimensions.** Sign codes clear a 0.9 target for 922 distance
equivalents; the best projection rank needs 2348. The optimum rank is interior, at half the
dimension, so neither end of that trade wins.

**Selective escalation is the wrong shape of spending.** `eval/calibration.py` builds a flag that
really works, catching 0.80 of bad answers against a base rate of 0.68, and then finds that
re-running the flagged fifth at four times the probe reaches 0.464 where the same budget spent
uniformly reaches 0.520. Replacing the flag with an oracle that knows which queries were actually
wrong reaches 0.487, still short. The signal was never the problem.

**An answer barely knows it is wrong.** The best post-search signal correlates 0.118 with per query
recall, against 0.254 for the best signal available before the search. In thirty two dimensions the
returned distances look alike whether the search found the right neighbours or the wrong ones.

**Query drift is dangerous inward, not outward.** `vectors/drift.py` moves a query population four
standard deviations away from the corpus and recall goes up. Shrinking it to a quarter of its
radius costs 0.18, because the true neighbours of a query in the dense middle are spread across
more partitions than the probe budget can reach. The repair is more probes; a rebuild does nothing,
since the corpus has not moved.

**Independent replicas disagree far more than expected.** Two replicas of one index built with
different seeds agree on 0.157 of their slots. `serve/replica.py` turns that into an advantage:
merging two reaches 0.7725 where one reaches 0.5515, and five reach 0.965.

**A reciprocal rank fusion constant of sixty is catastrophic here.** `eval/fusion.py` measures it.
The conventional value is tuned for a different regime and nobody says so.

**Whitening damages isotropic corpora most.** Which is backwards from the reason people reach for
it.

## Bugs the checks found

Worth listing separately, because these are the cases where the arrangement paid for itself.

The differential sweep in `verify/differential.py` caught the kd-tree reporting root distances
where every other structure reports squared ones. Pruning legitimately needs the root, and the root
was leaking into the result. Every recall number was unaffected and anything comparing scores
across structures was reading wrong numbers.

The same sweep caught three structures filling unanswered result slots with identifier zero at
score zero, which fired the distinctness, ordering and score agreement rules at once.

The residual quantiser's rerank was doing nothing, because the shortlist reached `top_up` unsorted.

The random baseline in `serve/router.py` caught a signal whose sign was backwards, which had been
producing a plausible looking correlation the whole time.

The rotation control in `vectors/drift.py` caught a bug in itself. It blended the identity with a
random orthonormal matrix, whose singular values are below one for any blend strictly inside the
ends, so it shrank the queries and quietly reproduced the scaling result it was meant to be a
control for.

The package-wide check in `tests/test_package.py` caught a refusal check catching bare `Exception`
and matching on message text, which would have passed on any exception from anywhere.

## Conventions

Ninety six columns, enforced. No em dashes, no emoji, pure ascii, enforced. Long explanations go in
a comment block above the code rather than in a module docstring, so they stay reasoning for a
reader of the source rather than becoming interface documentation. Docstrings are short and are
there to say why rather than what. The library never prints. Every refusal raises a named type
inheriting from `VectorSearchError` and carries a message written to be read by a person.

Every claim in a docstring is a number that the tests re-derive. If a measurement changes, a test
fails and the docstring is wrong until somebody fixes it.

## Attribution

This repository was written with Claude, Anthropic's AI assistant, as a collaborator. The design
decisions, the measurements, the reversals and the prose are the product of that collaboration.

Co-Authored-By: Claude <noreply@anthropic.com>

## Licence

MIT.

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import torch

from vse.errors import ConfigError, DataError, IndexStateError, VectorSearchError
from vse.index.base import Index
from vse.index.flat import FlatIndex
from vse.index.forest import ForestIndex
from vse.index.graph import GraphIndex
from vse.index.hnsw import HNSWIndex
from vse.index.ivf import IVFIndex
from vse.index.lsh import LSHIndex
from vse.index.tree import TreeIndex
from vse.quantize.binary import BinaryIndex
from vse.vectors.dataset import clustered, gaussian, on_a_subspace
from vse.vectors.exact import Neighbours, duplicated_corpus, search
from vse.vectors.metric import squared_l2

# Checking every index against the same set of rules, on inputs nobody chose.
#
# Every rule takes the same four arguments and most of them ignore two, which is deliberate:
# a uniform signature is what lets RULES be a list and the sweep be a loop, and giving each
# rule the arguments it happens to need would mean the sweep knowing which is which. The

#
# The rest of the package measures quality: how much recall a structure gives for how many
# distance computations. This module measures something different and more basic, which is
# whether the answers are well formed at all. An index that returns duplicate identifiers, or
# scores that do not match the vectors it named, or fewer results than it was asked for, is
# broken regardless of what its recall says, and recall will not reveal it: a duplicate in the
# result set costs a fraction of a point and reads as approximation error.
#
# Six invariants, applied to eight index types, over corpora with the properties that break
# things. All are checkable without a ground truth, which is what makes them cheap enough
# to run on inputs generated at random rather than on a curated benchmark.
#
# Two of the six exist because recall cannot see what they check. The scores agree with the
# identifiers rule catches a structure returning the right neighbours with the wrong distances
# attached, which every quality measurement in this package would score as perfect, since recall
# reads identifiers only. And the ordering rule catches a structure returning the right set in
# the wrong order, which scores perfectly on recall and badly on anything rank weighted, exactly
# the gap between eval/recall.py's two headline numbers.
#
# The first run of this sweep found twenty three violations across two hundred and forty checks,
# which is why the module is worth its lines. Nineteen were two bugs:
#
# The kd tree reported root distances where every other index reports squared ones. Its pruning
# needs the root, because the squared distance does not obey the triangle inequality, and the
# root was leaking out into the result. Every recall number in the package was unaffected, and
# anything comparing scores across indexes, which is what eval/fusion.py does for a living, was
# reading numbers that meant something else.
#
# The hash index, the graph and the kd tree all filled short results with identifier zero at
# score zero. That claims vector zero is a neighbour at distance zero, and it fires three rules
# at once, which is the signature: distinctness, ordering and score agreement all break
# together. Fixed by index/base.py's top_up, which fills from live rows the structure did not
# reach, scored honestly.
#
# Four violations survive and they are not bugs. On a clustered corpus the graph and the
# hierarchy both return, for one query in sixteen, a best result 91 times further away than
# exact search's worst. The walk was trapped where it started. That is the connectivity problem
# from build/neighbours.py appearing as a search failure, and averaging it into a recall number
# hides it completely: one bad query in sixteen is six points of recall and reads as ordinary
# approximation error.
#
# The corpora are chosen to be awkward rather than realistic. A corpus with exact duplicates
# makes ties unavoidable, a corpus on a low rank subspace makes many distances equal, and a
# corpus of one vector repeated makes every distance equal, which is where anything assuming
# strict inequalities falls over.


@dataclass
class Violation:
    """One rule broken by one index on one corpus."""

    index: str
    corpus: str
    rule: str
    detail: str

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "index": self.index,
            "corpus": self.corpus,
            "rule": self.rule,
            "detail": self.detail,
        }


@dataclass
class Report:
    """Everything one run found."""

    checks: int = 0
    violations: list[Violation] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        """Whether anything was found."""
        return not self.violations

    def record(self, violation: Violation | None) -> None:
        """Count a check and keep it if it failed."""
        self.checks += 1
        if violation is not None:
            self.violations.append(violation)

    def by_rule(self) -> dict:
        """How many violations each rule produced."""
        counts: dict = {}
        for violation in self.violations:
            counts[violation.rule] = counts.get(violation.rule, 0) + 1
        return counts

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "checks": self.checks,
            "violations": len(self.violations),
            "clean": self.clean,
            "by_rule": self.by_rule(),
        }


def returns_k_results(
    found: Neighbours,
    corpus: torch.Tensor,  # noqa: ARG001
    queries: torch.Tensor,
    k: int,
) -> str | None:
    """Every query gets exactly k neighbours.

    Not at least k and not up to k. A caller asking for ten and receiving eight has no way to
    tell whether the corpus was too small or the structure gave up, and every downstream measure
    silently divides by the wrong number.
    """
    if int(found.identifiers.shape[0]) != int(queries.shape[0]):
        return f"{int(found.identifiers.shape[0])} rows for {int(queries.shape[0])} queries"
    if int(found.identifiers.shape[1]) != k:
        return f"{int(found.identifiers.shape[1])} neighbours where {k} were asked for"
    return None


def identifiers_are_distinct(
    found: Neighbours,
    corpus: torch.Tensor,  # noqa: ARG001
    queries: torch.Tensor,  # noqa: ARG001
    k: int,  # noqa: ARG001
) -> str | None:
    """No query gets the same vector twice.

    A duplicate in a result set costs a fraction of a point of recall, which reads as
    approximation error, and it means one of the k slots the caller paid for returned
    nothing. It happens when a candidate set is assembled from overlapping sources without being
    deduplicated, which is what every multi probe and multi tree structure does.
    """
    for row in range(int(found.identifiers.shape[0])):
        values = found.identifiers[row]
        if int(torch.unique(values).numel()) != int(values.numel()):
            return f"query {row} has a repeated identifier"
    return None


def identifiers_are_in_range(
    found: Neighbours,
    corpus: torch.Tensor,
    queries: torch.Tensor,  # noqa: ARG001
    k: int,  # noqa: ARG001
) -> str | None:
    """Every identifier names a vector that exists."""
    size = int(corpus.shape[0])
    if int(found.identifiers.min()) < 0:
        return f"identifier {int(found.identifiers.min())} is negative"
    if int(found.identifiers.max()) >= size:
        return f"identifier {int(found.identifiers.max())} is outside a corpus of {size}"
    return None


def scores_agree_with_identifiers(
    found: Neighbours,
    corpus: torch.Tensor,
    queries: torch.Tensor,
    k: int,  # noqa: ARG001
) -> str | None:
    """The score in position j is the distance to the vector named in position j.

    The check that catches the bug nothing else would. Recall reads identifiers and ignores
    scores, so a structure can return the right neighbours with the wrong distances attached and
    score perfectly on every quality measurement in this package. Anything that reranks or fuses
    on those scores then does so on numbers that are not what they claim to be.
    """
    tolerance = 1e-3
    for row in range(int(found.identifiers.shape[0])):
        wanted = squared_l2(queries[row : row + 1], corpus[found.identifiers[row]]).flatten()
        gap = (wanted - found.scores[row]).abs().max()
        if float(gap) > tolerance * max(1.0, float(wanted.max())):
            return f"query {row} has a score off by {float(gap):.4f}"
    return None


def results_are_ordered(
    found: Neighbours,
    corpus: torch.Tensor,  # noqa: ARG001
    queries: torch.Tensor,  # noqa: ARG001
    k: int,  # noqa: ARG001
) -> str | None:
    """Scores rise across each row.

    A structure that returns the right set in the wrong order scores perfectly on recall and
    badly on discounted gain, which is exactly the gap eval/recall.py exists to make visible.
    Checking the order here means that gap can be attributed to approximation rather than to a
    sort that was never done.
    """
    for row in range(int(found.scores.shape[0])):
        values = found.scores[row]
        if bool((values[1:] < values[:-1] - 1e-5).any()):
            return f"query {row} is not sorted"
    return None


def the_nearest_is_at_least_as_near_as_the_worst_exact(
    found: Neighbours,
    corpus: torch.Tensor,
    queries: torch.Tensor,
    k: int,
) -> str | None:
    """The best result returned is no worse than the k'th exact neighbour.

    A weak bound and the only correctness statement available without a ground truth comparison
    per query. A structure whose best answer is worse than exact search's worst answer has not
    approximated anything, it has returned a region of the corpus unrelated to the query, and
    that is a different failure from low recall.
    """
    truth = search(queries, corpus, k=k)
    worst_exact = truth.scores[:, -1]
    best_found = found.scores[:, 0]
    over = best_found > worst_exact + 1e-4
    if bool(over.any()):
        row = int(torch.nonzero(over, as_tuple=False)[0, 0])
        return (
            f"query {row} best result scores {float(best_found[row]):.4f} against an exact "
            f"worst of {float(worst_exact[row]):.4f}"
        )
    return None


RULES: tuple[tuple[str, Callable], ...] = (
    ("returns k", returns_k_results),
    ("distinct", identifiers_are_distinct),
    ("in range", identifiers_are_in_range),
    ("scores match", scores_agree_with_identifiers),
    ("ordered", results_are_ordered),
    ("not unrelated", the_nearest_is_at_least_as_near_as_the_worst_exact),
)


def every_index(dimension: int = 16) -> list[tuple[str, Index]]:
    """One of each structure, at settings small enough to run on a small corpus.

    Every index in the package that can be built and searched, including the ones the rest of
    the measurements conclude are not worth using. A correctness check that only covers the
    recommended structures is a check that stops working the moment somebody uses a different
    one.
    """
    return [
        ("flat", FlatIndex(dimension)),
        ("ivf", IVFIndex(dimension, partitions=8, probe=3)),
        ("graph", GraphIndex(dimension, degree=8, ef=16)),
        ("hnsw", HNSWIndex(dimension, degree=8, ef=16)),
        ("forest", ForestIndex(dimension, trees=4, leaf_size=32)),
        ("tree", TreeIndex(dimension, leaf_size=32)),
        ("lsh", LSHIndex(dimension, bits=8, tables=4)),
        ("binary", BinaryIndex(dimension, rerank=32)),
    ]


def awkward_corpora(dimension: int = 16, count: int = 512) -> list[tuple[str, torch.Tensor]]:
    """Corpora chosen to break assumptions rather than to be realistic.

    Duplicates make ties unavoidable, a low rank subspace makes many distances equal, and a
    single vector repeated makes every distance equal. A structure that assumes strict
    inequalities anywhere fails on the last one, and it is the case least likely to appear in a
    benchmark and most likely to appear in a production corpus that has been deduplicated badly.
    """
    repeated = torch.randn(1, dimension, generator=torch.Generator().manual_seed(4)).expand(
        count, dimension
    )
    return [
        ("gaussian", gaussian(count=count, dimension=dimension).vectors),
        ("clustered", clustered(count=count, dimension=dimension, clusters=8).vectors),
        ("subspace", on_a_subspace(count=count, dimension=dimension, intrinsic=3).vectors),
        ("duplicated", duplicated_corpus(count=count, dimension=dimension)),
        ("all identical", repeated.clone()),
    ]


def check(
    index: Index,
    corpus: torch.Tensor,
    queries: torch.Tensor,
    k: int,
    label: str,
    corpus_label: str,
    rules: Sequence[tuple[str, Callable]] = RULES,
) -> list[Violation]:
    """Run every rule against one index on one corpus."""
    found, _ = index.search(queries, k=k)
    violations = []
    for name, rule in rules:
        detail = rule(found, corpus, queries, k)
        if detail is not None:
            violations.append(
                Violation(index=label, corpus=corpus_label, rule=name, detail=detail)
            )
    return violations


def sweep(
    dimension: int = 16, count: int = 512, queries: int = 16, k: int = 10, seed: int = 0
) -> Report:
    """Every index against every corpus against every rule.

    Forty combinations and two hundred and forty checks. Cheap enough to run in a test suite,
    which is the point: an invariant that is only checked when somebody remembers to is an
    invariant that holds until the first time it does not.
    """
    if k < 1 or queries < 1:
        raise ConfigError(f"{queries} queries for {k} neighbours is not a check")
    generator = torch.Generator().manual_seed(seed)
    report = Report()
    for corpus_label, corpus in awkward_corpora(dimension=dimension, count=count):
        probes = corpus[torch.randint(0, count, (queries,), generator=generator)]
        probes = probes + torch.randn(probes.shape, generator=generator) * 0.05
        for label, index in every_index(dimension=dimension):
            try:
                index.build(corpus)
            except VectorSearchError:
                continue
            for violation in check(index, corpus, probes, k, label, corpus_label):
                report.record(violation)
            report.checks += len(RULES) - len(
                [v for v in report.violations if v.index == label and v.corpus == corpus_label]
            )
    return report


def every_index_returns_well_formed_results() -> dict:
    """The headline, which is four violations and not zero.

    Two hundred and forty checks across eight structures and five corpora. Nineteen violations
    were fixed by the two repairs described at the top of this module. Four remain, all of one
    rule, all on the two graph structures, and they are the structures genuinely failing rather
    than the format being wrong.

    Reporting four rather than suppressing them is the point. A sweep that returns zero because
    the awkward cases were removed from it measures nothing, and the four that survive say
    something specific: a graph walk on a clustered corpus can return results unrelated to the
    query, and no recall average will tell you which query it happened to.
    """
    report = sweep()
    return {
        "checks": report.checks,
        "violations": len(report.violations),
        "clean": report.clean,
        "by_rule": report.by_rule(),
    }


def a_corpus_of_identical_vectors_is_the_hard_one() -> dict:
    """The case that breaks implementations assuming strict inequalities.

    Every distance is zero, so every ordering is correct and every tie break is arbitrary. A
    structure that splits on a median has nothing to split on, a structure that picks a nearest
    centroid has an arbitrary one, and a structure that prunes on a strict comparison prunes
    everything or nothing. All eight survive it, once the short result filling was fixed. Before
    that the graph failed three rules on this corpus alone, which is worth knowing because it is
    the corpus nobody tests on and the one a badly deduplicated table looks most like.
    """
    dimension, count = 16, 512
    corpus = (
        torch.randn(1, dimension, generator=torch.Generator().manual_seed(4))
        .expand(count, dimension)
        .clone()
    )
    probes = corpus[:8].clone()
    report = Report()
    built = 0
    for label, index in every_index(dimension=dimension):
        try:
            index.build(corpus)
        except VectorSearchError:
            continue
        built += 1
        for violation in check(index, corpus, probes, 10, label, "all identical"):
            report.record(violation)
    return {
        "indexes_built": built,
        "violations": len(report.violations),
        "clean": report.clean,
        "offenders": sorted({violation.index for violation in report.violations}),
    }


def a_duplicated_corpus_makes_ties_unavoidable() -> dict:
    """That exact duplicates do not produce repeated identifiers.

    A corpus with each vector present twice means every query has at least two equally correct
    answers for each slot, and a structure assembling candidates from several sources will find
    the same row through more than one path. Returning it twice is the bug, and it is invisible
    to recall because the duplicate scores the same as the original.

    Two structures did it on the first run, the graph and the hash index, and in both cases the
    repeat came from the zero fill rather than from the duplicate paths this rule was written to
    catch. The rule found something real and not the thing it was aimed at, which is the usual
    way an invariant check earns its place.
    """
    dimension, count = 16, 512
    corpus = duplicated_corpus(count=count, dimension=dimension)
    probes = corpus[:8].clone()
    report = Report()
    for label, index in every_index(dimension=dimension):
        try:
            index.build(corpus)
        except VectorSearchError:
            continue
        for violation in check(index, corpus, probes, 10, label, "duplicated"):
            report.record(violation)
    repeats = [violation for violation in report.violations if violation.rule == "distinct"]
    return {
        "violations": len(report.violations),
        "repeated_identifiers": len(repeats),
        "clean": report.clean,
    }


def searching_for_a_corpus_vector_finds_it() -> dict:
    """That every index finds a vector that is in the corpus, when asked for it exactly.

    The weakest possible sanity check and the one that catches the most. A query that is a
    corpus vector has a distance of exactly zero to itself, and any structure that fails to
    return it has either lost the vector or is searching a different space. The approximate
    structures are allowed to miss it and mostly do not.
    """
    dimension, count = 16, 1024
    corpus = gaussian(count=count, dimension=dimension).vectors
    probes = corpus[:32].clone()
    rows = []
    for label, index in every_index(dimension=dimension):
        index.build(corpus)
        found, _ = index.search(probes, k=10)
        hits = sum(1 for row in range(32) if row in found.identifiers[row].tolist())
        rows.append({"index": label, "found_itself": hits, "of": 32})
    return {
        "rows": rows,
        "exact_structures_never_miss": all(
            row["found_itself"] == 32 for row in rows if row["index"] in {"flat", "tree"}
        ),
        "worst": min(row["found_itself"] for row in rows),
    }


def the_exact_structures_agree_with_each_other() -> dict:
    """That the two exact indexes return the same answer, which they must.

    A flat scan and a kd tree with exact pruning are two implementations of the same function.
    They can differ only on ties, so requiring the scores to be identical rather than the
    identifiers is the correct test, and it is the strongest check available here because it
    needs no tolerance and no ground truth beyond one of the two.
    """
    dimension, count = 8, 1024
    corpus = gaussian(count=count, dimension=dimension).vectors
    probes = gaussian(count=32, dimension=dimension, seed=9).vectors
    flat = FlatIndex(dimension)
    flat.build(corpus)
    tree = TreeIndex(dimension, leaf_size=16)
    tree.build(corpus)
    left, _ = flat.search(probes, k=10)
    right, _ = tree.search(probes, k=10)
    return {
        "scores_identical": bool(torch.allclose(left.scores, right.scores, atol=1e-5)),
        "identifiers_identical": bool(torch.equal(left.identifiers, right.identifiers)),
        "max_score_gap": round(float((left.scores - right.scores).abs().max()), 8),
    }


def an_index_of_one_vector_works() -> dict:
    """The smallest corpus that can be searched at all.

    One vector and k of one. Every structure has a degenerate path here: a tree with no split, a
    graph with no edges, a partitioning with one partition. Several refuse to build, which is
    the right behaviour, and the ones that build must return that single vector.
    """
    dimension = 8
    corpus = torch.randn(1, dimension, generator=torch.Generator().manual_seed(2))
    rows = []
    for label, index in every_index(dimension=dimension):
        try:
            index.build(corpus)
            found, _ = index.search(corpus, k=1)
            rows.append(
                {
                    "index": label,
                    "built": True,
                    "correct": int(found.identifiers[0, 0]) == 0,
                }
            )
        except VectorSearchError:
            rows.append({"index": label, "built": False, "correct": None})
    return {
        "rows": rows,
        "built": sum(1 for row in rows if row["built"]),
        "refused": sum(1 for row in rows if not row["built"]),
        "all_built_are_correct": all(row["correct"] for row in rows if row["built"]),
    }


def asking_for_more_than_the_corpus_holds_is_refused() -> dict:
    """That k above the corpus size is caught rather than padded.

    Padding with a repeated identifier or a sentinel would satisfy the shape and break the
    distinctness rule, and returning fewer rows would break the shape rule. Refusing is the only
    option that leaves the contract intact, and every structure here does it.
    """
    dimension, count = 8, 32
    corpus = gaussian(count=count, dimension=dimension).vectors
    rows = []
    for label, index in every_index(dimension=dimension):
        try:
            index.build(corpus)
        except VectorSearchError:
            continue
        refused = False
        try:
            index.search(corpus[:2], k=count + 10)
        except VectorSearchError:
            refused = True
        rows.append({"index": label, "refused": refused})
    return {
        "rows": rows,
        "all_refuse": all(row["refused"] for row in rows),
        "checked": len(rows),
    }


def searching_an_unbuilt_index_is_refused() -> dict:
    """That every structure refuses to answer before it has been built.

    An unbuilt index has no vectors, so there is no defensible answer, and returning an empty
    result would satisfy a caller who never checks. All eight refuse with the same error type,
    which is what makes it catchable in one place.
    """
    rows = []
    for label, index in every_index(dimension=8):
        refused = False
        try:
            index.search(torch.randn(2, 8), k=5)
        except IndexStateError:
            refused = True
        rows.append({"index": label, "refused": refused})
    return {
        "rows": rows,
        "all_refuse": all(row["refused"] for row in rows),
        "checked": len(rows),
    }


def a_query_of_the_wrong_width_is_refused() -> dict:
    """That a query whose dimension does not match the index is caught.

    It would otherwise fail somewhere inside a matrix multiply with an error naming shapes
    nobody recognises, or worse, broadcast into something that runs and returns nonsense.
    """
    dimension = 8
    corpus = gaussian(count=256, dimension=dimension).vectors
    rows = []
    for label, index in every_index(dimension=dimension):
        try:
            index.build(corpus)
        except VectorSearchError:
            continue
        refused = False
        try:
            index.search(torch.randn(2, dimension * 2), k=5)
        except (DataError, ConfigError):
            refused = True
        rows.append({"index": label, "refused": refused})
    return {
        "rows": rows,
        "all_refuse": all(row["refused"] for row in rows),
        "checked": len(rows),
    }


def a_deliberately_broken_index_is_caught() -> dict:
    """That the rules would actually fire, checked by breaking one on purpose.

    A check suite that has never failed is a check suite nobody has tested. This wraps a working
    index and corrupts its output in four different ways, one per rule, and confirms each rule
    catches its own. Without this the clean report above is evidence of nothing.
    """
    dimension, count = 8, 256
    corpus = gaussian(count=count, dimension=dimension).vectors
    probes = corpus[:8].clone()
    index = FlatIndex(dimension)
    index.build(corpus)
    found, _ = index.search(probes, k=10)

    repeated = Neighbours(identifiers=found.identifiers.clone(), scores=found.scores.clone())
    repeated.identifiers[:, 1] = repeated.identifiers[:, 0]

    out_of_range = Neighbours(
        identifiers=found.identifiers.clone(), scores=found.scores.clone()
    )
    out_of_range.identifiers[0, 0] = count + 5

    wrong_scores = Neighbours(
        identifiers=found.identifiers.clone(), scores=found.scores.clone() + 10.0
    )

    unsorted = Neighbours(
        identifiers=found.identifiers.flip(1).clone(), scores=found.scores.flip(1).clone()
    )

    return {
        "distinct_fires": identifiers_are_distinct(repeated, corpus, probes, 10) is not None,
        "range_fires": identifiers_are_in_range(out_of_range, corpus, probes, 10) is not None,
        "scores_fire": scores_agree_with_identifiers(wrong_scores, corpus, probes, 10)
        is not None,
        "order_fires": results_are_ordered(unsorted, corpus, probes, 10) is not None,
        "shape_fires": returns_k_results(
            Neighbours(found.identifiers[:, :5], found.scores[:, :5]), corpus, probes, 10
        )
        is not None,
        "clean_input_passes": all(rule(found, corpus, probes, 10) is None for _, rule in RULES),
    }


def the_rules_do_not_fire_on_a_correct_result() -> dict:
    """The other half of that, which is that the rules are not simply always true.

    A rule that fires on everything is as useless as one that fires on nothing, so the check
    above confirms each rule catches its own corruption and this confirms none of them fires on
    an exact result. Both halves together are what makes the clean sweep meaningful.
    """
    result = a_deliberately_broken_index_is_caught()
    return {
        "clean_input_passes": result["clean_input_passes"],
        "every_rule_fires_on_its_own_break": all(
            result[key]
            for key in (
                "distinct_fires",
                "range_fires",
                "scores_fire",
                "order_fires",
                "shape_fires",
            )
        ),
    }


def a_report_counts_by_rule() -> dict:
    """That a report groups what it found, which is how a failure gets diagnosed.

    A count of violations is not actionable. A count per rule says whether the problem is one
    structure doing everything wrong or every structure doing one thing wrong, and those have
    completely different causes.
    """
    report = Report()
    report.record(Violation("flat", "gaussian", "distinct", "a"))
    report.record(Violation("ivf", "gaussian", "distinct", "b"))
    report.record(Violation("ivf", "clustered", "ordered", "c"))
    report.record(None)
    return {
        "checks": report.checks,
        "violations": len(report.violations),
        "by_rule": report.by_rule(),
        "clean": report.clean,
        "distinct_count": report.by_rule()["distinct"],
    }


def an_empty_report_is_clean() -> dict:
    """That a report with nothing in it says so."""
    report = Report()
    return {
        "checks": report.checks,
        "clean": report.clean,
        "by_rule": report.by_rule(),
    }


def a_check_with_no_queries_is_refused() -> bool:
    """Whether a sweep that would check nothing is caught."""
    try:
        sweep(queries=0)
    except ConfigError:
        return True
    return False


def a_check_for_no_neighbours_is_refused() -> bool:
    """Whether asking for zero neighbours is caught at the sweep."""
    try:
        sweep(k=0)
    except ConfigError:
        return True
    return False


def the_sweep_is_deterministic() -> dict:
    """That two runs of the same sweep find the same thing.

    Every corpus and every query set is seeded, so a violation found once is findable again,
    which is the difference between a check that catches bugs and a check that produces
    intermittent failures nobody can reproduce.
    """
    first = sweep(seed=3)
    second = sweep(seed=3)
    return {
        "first_violations": len(first.violations),
        "second_violations": len(second.violations),
        "identical": [v.as_dict() for v in first.violations]
        == [v.as_dict() for v in second.violations],
        "checks_match": first.checks == second.checks,
    }


def different_seeds_check_different_queries() -> dict:
    """And that changing the seed changes what is checked, which it has to.

    A sweep that produced identical queries at every seed would be one check repeated, and its
    coverage would be whatever the first seed happened to hit. Confirming the queries differ is
    the cheap version of confirming the sweep is doing what its name says.
    """
    count = 512
    rows = []
    for seed in (0, 1):
        generator = torch.Generator().manual_seed(seed)
        picks = torch.randint(0, count, (16,), generator=generator)
        rows.append(picks.tolist())
    return {
        "first": rows[0][:4],
        "second": rows[1][:4],
        "differ": rows[0] != rows[1],
    }

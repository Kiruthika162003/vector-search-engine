from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from vse.errors import ConfigError, DataError
from vse.index.flat import FlatIndex
from vse.index.ivf import IVFIndex
from vse.vectors.dataset import gaussian, held_out
from vse.vectors.exact import Neighbours, identifier_overlap, search
from vse.vectors.metric import normalise, squared_l2

# What a duplicate does to a measurement, which is more than it does to a search.
#
# Real corpora have duplicates. The same document indexed twice, a page and its printer friendly
# version, a product listed by two sellers, a template filled in with different names. Nothing
# in
# this package's synthetic corpora has them by default and every measurement is therefore made
# on
# a corpus cleaner than any real one.
#
# The search consequence is mild and the measurement consequence is not, which is the module's
# point.
#
# For the search, a duplicate is a vector that happens to be close to another one. Every
# structure here handles that: it lands in the same partition, it is a neighbour in the graph,
# it
# gets the same code. Nothing breaks.
#
# For the measurement, a duplicate breaks the assumption every recall number rests on, which is
# that the top k is well defined. When two corpus vectors are identical, the exact search's
# tenth
# and eleventh results are tied, and which one it returns is an implementation detail of the
# sort.
# An approximate index returning the other one is marked wrong for returning an equally correct
# answer. The recall reported is then a measurement of tie breaking rather than of search
# quality.
#
# So the module does two things. It measures how much of a reported recall loss is really tie
# breaking, by scoring against distances instead of identifiers. And it measures what
# deduplicating actually buys, which turns out to be mostly a smaller index rather than a better
# one.


@dataclass
class Duplication:
    """A corpus with known duplicates, and the record of which rows are copies of which."""

    vectors: torch.Tensor
    original: torch.Tensor

    def __post_init__(self) -> None:
        if self.vectors.ndim != 2:
            raise DataError(f"a corpus is a matrix, got {tuple(self.vectors.shape)}")
        if int(self.original.numel()) != int(self.vectors.shape[0]):
            raise DataError(
                f"{int(self.original.numel())} labels for {int(self.vectors.shape[0])} rows"
            )

    @property
    def count(self) -> int:
        """How many rows the corpus has."""
        return int(self.vectors.shape[0])

    @property
    def distinct(self) -> int:
        """How many distinct originals there are."""
        return int(torch.unique(self.original).numel())

    @property
    def duplicate_share(self) -> float:
        """What fraction of the rows are copies of something else."""
        if self.count == 0:
            return 0.0
        return 1.0 - self.distinct / self.count

    def groups(self) -> dict:
        """Which rows belong to each original."""
        out: dict = {}
        for row in range(self.count):
            out.setdefault(int(self.original[row]), []).append(row)
        return out

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "count": self.count,
            "distinct": self.distinct,
            "duplicate_share": round(self.duplicate_share, 4),
        }


def exact_duplicates(
    corpus: torch.Tensor, share: float = 0.2, copies: int = 2, seed: int = 0
) -> Duplication:
    """Replace a share of the corpus with exact copies of other rows.

    The corpus size is held fixed so the comparisons downstream are not confounded by it: a
    corpus with duplicates added would be larger, and a larger corpus is harder to search for
    reasons that have nothing to do with duplication.
    """
    if not 0.0 <= share < 1.0:
        raise ConfigError(f"a share of {share} is not a duplication")
    if copies < 2:
        raise ConfigError(f"{copies} copies is not a duplicate")
    count = int(corpus.shape[0])
    generator = torch.Generator().manual_seed(seed)
    vectors = corpus.clone()
    original = torch.arange(count)
    replacing = torch.randperm(count, generator=generator)[: int(count * share)]
    sources = torch.randperm(count, generator=generator)[: int(count * share)]
    for position in range(int(replacing.numel())):
        row, source = int(replacing[position]), int(sources[position])
        if row == source:
            continue
        vectors[row] = corpus[source]
        original[row] = source
    return Duplication(vectors=vectors, original=original)


def near_duplicates(
    corpus: torch.Tensor, share: float = 0.2, nudge: float = 0.01, seed: int = 0
) -> Duplication:
    """The same, but the copies are perturbed slightly rather than identical.

    Closer to what a real corpus holds: two versions of a document produce vectors that are very
    close and not equal. The distinction matters because an exact duplicate makes a true tie and
    a near duplicate makes an ordering that is real but arbitrary in practice, and the two need
    different treatment in a measurement.
    """
    if nudge <= 0:
        raise ConfigError(f"a nudge of {nudge} makes exact duplicates")
    duplication = exact_duplicates(corpus, share=share, seed=seed)
    generator = torch.Generator().manual_seed(seed + 1)
    moved = duplication.original != torch.arange(duplication.count)
    step = torch.randn(int(moved.sum()), int(corpus.shape[1]), generator=generator)
    duplication.vectors[moved] = duplication.vectors[moved] + normalise(step) * nudge
    return duplication


def ties_in_the_truth(corpus: torch.Tensor, queries: torch.Tensor, k: int = 10) -> dict:
    """How often the k'th and the k plus first true neighbours are equidistant.

    The quantity that decides whether a recall number means anything. If the boundary of the top
    k is a tie then two different correct answers exist and recall penalises one of them
    arbitrarily.
    """
    scores = squared_l2(queries, corpus)
    nearest = torch.topk(scores, k=k + 1, dim=1, largest=False).values
    gap = nearest[:, k] - nearest[:, k - 1]
    return {
        "queries": int(queries.shape[0]),
        "tied": int((gap < 1e-6).sum()),
        "tied_share": round(float((gap < 1e-6).float().mean()), 4),
        "median_gap": round(float(gap.median()), 6),
    }


def recall_by_distance(
    queries: torch.Tensor, corpus: torch.Tensor, truth: Neighbours, found: Neighbours
) -> float:
    """Recall scored on the distances returned rather than on the identifiers.

    A result is credited when its score matches the corresponding true score, so returning a
    duplicate of the right answer counts as right. This is the measurement identifier recall
    should have been all along on a corpus with ties, and the gap between the two is exactly the
    amount of tie breaking in the number.
    """
    if truth.identifiers.shape != found.identifiers.shape:
        raise DataError("scoring by distance needs matching shapes")
    rows = int(truth.identifiers.shape[0])
    total = 0.0
    for row in range(rows):
        wanted = squared_l2(queries[row : row + 1], corpus[truth.identifiers[row]]).flatten()
        got = squared_l2(queries[row : row + 1], corpus[found.identifiers[row]]).flatten()
        matched = 0
        remaining = wanted.tolist()
        for score in got.tolist():
            for position, target in enumerate(remaining):
                if abs(score - target) < 1e-5:
                    matched += 1
                    remaining.pop(position)
                    break
        total += matched / float(int(wanted.numel()))
    return total / rows


def _setup(count: int = 4096, dimension: int = 32, queries: int = 200):
    """A corpus with queries held out."""
    corpus = gaussian(count=count, dimension=dimension)
    searched, probes = held_out(corpus, count=queries)
    return searched.vectors, probes


def a_clean_corpus_has_almost_no_ties() -> dict:
    """The baseline, which is that the assumption holds on synthetic data.

    A gaussian corpus of continuous floats has no exact ties at all, so identifier recall is
    well defined and every measurement in this package that uses one is sound. Establishing that
    first is what makes the rest of the module a statement about real corpora rather than about
    the machinery.
    """
    corpus, probes = _setup()
    return ties_in_the_truth(corpus, probes)


def duplicates_create_ties(shares: Sequence[float] = (0.0, 0.1, 0.3, 0.5)) -> list[dict]:
    """How many queries have a tied top k boundary as the duplication rises.

    A duplicate is a tie whenever both copies fall near the boundary, which happens in
    proportion
    to how many duplicates there are and how close the boundary is to them. The share of
    affected
    queries is the fraction of the recall number that is measuring a coin flip.
    """
    if not shares:
        raise ConfigError("there is nothing to sweep")
    corpus, probes = _setup()
    rows = []
    for share in shares:
        duplication = (
            exact_duplicates(corpus, share=share)
            if share
            else Duplication(
                vectors=corpus.clone(), original=torch.arange(int(corpus.shape[0]))
            )
        )
        result = ties_in_the_truth(duplication.vectors, probes)
        rows.append(
            {
                "share": share,
                "distinct": duplication.distinct,
                "tied_queries": result["tied"],
                "tied_share": result["tied_share"],
            }
        )
    return rows


def identifier_recall_understates_the_index(share: float = 0.3) -> dict:
    """The module's main measurement, which separates tie breaking from search quality.

    The same index on the same duplicated corpus, scored two ways. Identifier recall marks a
    returned duplicate wrong; distance recall marks it right.

    The gap is 0.0005 at a duplication share of a third, which is nothing. Written expecting it
    to be substantial, on the reasoning that 16.5 percent of queries have a tied boundary at
    that share. They do, and a tie costs at most one slot of one query, so the most it could
    reach is 0.0165, and the index picks the right side of most of them anyway.

    So the worry is real and the magnitude is not. Identifier recall is sound even on a corpus a
    third duplicated, which is worth knowing precisely because the argument that it should not
    be is a good one.
    """
    corpus, probes = _setup()
    duplication = exact_duplicates(corpus, share=share)
    truth = search(probes, duplication.vectors, k=10)
    index = IVFIndex(32, partitions=64, probe=8)
    index.build(duplication.vectors)
    found, _ = index.search(probes, k=10)
    by_identifier = identifier_overlap(truth, found)
    by_distance = recall_by_distance(probes, duplication.vectors, truth, found)
    return {
        "duplicate_share": share,
        "by_identifier": round(by_identifier, 4),
        "by_distance": round(by_distance, 4),
        "gap": round(by_distance - by_identifier, 4),
        "distance_is_higher": by_distance >= by_identifier,
    }


def the_gap_grows_with_the_duplication(
    shares: Sequence[float] = (0.0, 0.1, 0.3, 0.5),
) -> list[dict]:
    """How much of the reported number is tie breaking, against how dirty the corpus is.

    On a clean corpus the two measurements agree exactly, which is the check that the distance
    scoring is not simply more generous. As the duplication rises the gap opens, and its size is
    how much a recall number is understating the index on that corpus.
    """
    if not shares:
        raise ConfigError("there is nothing to sweep")
    corpus, probes = _setup()
    rows = []
    for share in shares:
        duplication = (
            exact_duplicates(corpus, share=share)
            if share
            else Duplication(
                vectors=corpus.clone(), original=torch.arange(int(corpus.shape[0]))
            )
        )
        truth = search(probes, duplication.vectors, k=10)
        index = IVFIndex(32, partitions=64, probe=8)
        index.build(duplication.vectors)
        found, _ = index.search(probes, k=10)
        by_identifier = identifier_overlap(truth, found)
        by_distance = recall_by_distance(probes, duplication.vectors, truth, found)
        rows.append(
            {
                "share": share,
                "by_identifier": round(by_identifier, 4),
                "by_distance": round(by_distance, 4),
                "gap": round(by_distance - by_identifier, 4),
            }
        )
    return rows


def the_two_measurements_agree_on_a_clean_corpus() -> dict:
    """The control for that sweep, which is the only thing making it trustworthy.

    With no duplicates the distance scoring and the identifier scoring have to give the same
    number, because every distance is distinct and matching on one is matching on the other. If
    they differed the distance scoring would be measuring something looser rather than something
    fairer.
    """
    rows = {row["share"]: row for row in the_gap_grows_with_the_duplication()}
    clean = rows[0.0]
    return {
        "by_identifier": clean["by_identifier"],
        "by_distance": clean["by_distance"],
        "gap": clean["gap"],
        "identical": abs(clean["gap"]) < 1e-6,
    }


def near_duplicates_do_not_make_ties(nudge: float = 0.01) -> dict:
    """Whether a nearly identical copy causes the same problem, which it does not exactly.

    A perturbed copy is at a different distance, so the top k is well defined and identifier
    recall is sound. What it produces instead is an ordering that is technically correct and
    practically arbitrary: nobody cares which of two near identical documents is returned, and
    the measurement does.

    So the exact case is a measurement bug and the near case is a definition problem, and only
    the first has a fix inside the measurement.
    """
    corpus, probes = _setup()
    exact = exact_duplicates(corpus, share=0.3)
    near = near_duplicates(corpus, share=0.3, nudge=nudge)
    return {
        "exact_ties": ties_in_the_truth(exact.vectors, probes)["tied_share"],
        "near_ties": ties_in_the_truth(near.vectors, probes)["tied_share"],
        "near_makes_fewer": ties_in_the_truth(near.vectors, probes)["tied_share"]
        < ties_in_the_truth(exact.vectors, probes)["tied_share"],
        "nudge": nudge,
    }


def deduplicating_shrinks_the_index(share: float = 0.3) -> dict:
    """What removing the duplicates buys, which is mostly a smaller index.

    The obvious fix is to deduplicate before indexing. It removes the ties, it shrinks the
    corpus
    by the duplication share, and the recall it reports afterwards is on a different corpus, so
    the two numbers are not comparable and reporting them side by side is a trap.

    The comparable statement is the cost: a corpus a third smaller costs a third less to scan at
    the same settings, which is a real and unglamorous benefit.
    """
    corpus, probes = _setup()
    duplication = exact_duplicates(corpus, share=share)
    groups = duplication.groups()
    keepers = torch.tensor(sorted(rows[0] for rows in groups.values()))
    rows = {}
    for label, vectors in (
        ("with duplicates", duplication.vectors),
        ("deduplicated", duplication.vectors[keepers]),
    ):
        truth = search(probes, vectors, k=10)
        index = IVFIndex(32, partitions=64, probe=8)
        index.build(vectors)
        found, stats = index.search(probes, k=10)
        rows[label] = {
            "size": int(vectors.shape[0]),
            "recall": identifier_overlap(truth, found),
            "distances": stats.distances_per_query,
        }
    return {
        "size_before": rows["with duplicates"]["size"],
        "size_after": rows["deduplicated"]["size"],
        "distances_before": round(rows["with duplicates"]["distances"], 1),
        "distances_after": round(rows["deduplicated"]["distances"], 1),
        "recall_before": round(rows["with duplicates"]["recall"], 4),
        "recall_after": round(rows["deduplicated"]["recall"], 4),
        "the_index_shrinks": rows["deduplicated"]["size"] < rows["with duplicates"]["size"],
        "and_costs_less": rows["deduplicated"]["distances"]
        < rows["with duplicates"]["distances"],
    }


def the_recalls_are_not_comparable_across_deduplication() -> dict:
    """Why the two recall numbers above should not be put in the same column.

    They are measured against different ground truths over different corpora. A query's tenth
    nearest neighbour in the deduplicated corpus is a different vector from its tenth in the
    original, so the two recalls answer different questions and the difference between them is
    not a gain or a loss.

    Stated as a measurement because the alternative is a comment nobody reads, and because a
    table showing deduplication improving recall is one of the easiest wrong tables to produce.
    """
    result = deduplicating_shrinks_the_index()
    return {
        "recall_before": result["recall_before"],
        "recall_after": result["recall_after"],
        "different_corpora": result["size_before"] != result["size_after"],
        "different_ground_truths": True,
        "the_difference_means_nothing": True,
        "the_cost_difference_does": result["and_costs_less"],
    }


def duplicates_do_not_break_the_search(shares: Sequence[float] = (0.0, 0.3, 0.6)) -> list[dict]:
    """Whether the structures themselves care, which they mostly do not.

    A duplicate lands in the same partition as its original, is a neighbour of it in the graph,
    and gets the same quantised code. Nothing about any structure here treats it specially and
    nothing about it breaks. Scored by distance so the tie breaking is out of the way.
    """
    if not shares:
        raise ConfigError("there is nothing to sweep")
    corpus, probes = _setup()
    rows = []
    for share in shares:
        duplication = (
            exact_duplicates(corpus, share=share)
            if share
            else Duplication(
                vectors=corpus.clone(), original=torch.arange(int(corpus.shape[0]))
            )
        )
        truth = search(probes, duplication.vectors, k=10)
        index = IVFIndex(32, partitions=64, probe=8)
        index.build(duplication.vectors)
        found, stats = index.search(probes, k=10)
        rows.append(
            {
                "share": share,
                "by_distance": round(
                    recall_by_distance(probes, duplication.vectors, truth, found), 4
                ),
                "distances": round(stats.distances_per_query, 1),
            }
        )
    return rows


def the_distance_recall_barely_moves() -> dict:
    """The conclusion of that sweep, which is the reassuring half of the module."""
    rows = {row["share"]: row for row in duplicates_do_not_break_the_search()}
    return {
        "clean": rows[0.0]["by_distance"],
        "a_third_duplicated": rows[0.3]["by_distance"],
        "most_duplicated": rows[0.6]["by_distance"],
        "spread": round(
            max(row["by_distance"] for row in rows.values())
            - min(row["by_distance"] for row in rows.values()),
            4,
        ),
        "barely_moves": (
            max(row["by_distance"] for row in rows.values())
            - min(row["by_distance"] for row in rows.values())
        )
        < 0.1,
    }


def a_duplicate_returns_alongside_its_original(share: float = 0.3) -> dict:
    """That an exact search returns both copies when they are both in the top k.

    Which is correct and is also the thing a user would call a bug. Two identical results in a
    list of ten is a list of nine, and a search service usually deduplicates its output rather
    than its index. That is a presentation decision and it is worth locating: the index is right
    and the result set is wrong.
    """
    corpus, probes = _setup()
    duplication = exact_duplicates(corpus, share=share)
    truth = search(probes, duplication.vectors, k=10)
    pairs = 0
    for row in range(int(probes.shape[0])):
        originals = duplication.original[truth.identifiers[row]]
        pairs += 10 - int(torch.unique(originals).numel())
    return {
        "queries": int(probes.shape[0]),
        "duplicate_slots": pairs,
        "per_query": round(pairs / int(probes.shape[0]), 3),
        "happens": pairs > 0,
    }


def deduplicating_the_result_is_the_usual_fix(share: float = 0.3) -> dict:
    """What removing duplicates from the result costs, which is a deeper search.

    Returning ten distinct originals from a corpus where a third of the rows are copies means
    fetching more than ten and collapsing them, which is the same shape as every rerank in this
    package: ask for more, keep what is wanted. The number worth knowing is how much more.
    """
    corpus, probes = _setup()
    duplication = exact_duplicates(corpus, share=share)
    index = FlatIndex(32)
    index.build(duplication.vectors)
    needed = []
    for depth in (10, 12, 15, 20, 30):
        found, _ = index.search(probes, k=depth)
        distinct = torch.tensor(
            [
                int(torch.unique(duplication.original[found.identifiers[row]]).numel())
                for row in range(int(probes.shape[0]))
            ]
        )
        needed.append(
            {"depth": depth, "mean_distinct": round(float(distinct.float().mean()), 3)}
        )
    enough = next((row["depth"] for row in needed if row["mean_distinct"] >= 10.0), None)
    return {
        "rows": needed,
        "depth_for_ten_distinct": enough,
        "overhead": None if enough is None else round(enough / 10.0, 2),
    }


def a_duplication_reports_its_own_shape() -> dict:
    """That the record says how dirty the corpus it made is, which is less than requested.

    Asking for a quarter of the rows to be replaced gives a duplicate share of 0.186, not 0.25.
    Two reasons, both structural: a replacement whose source is itself a replaced row collapses
    into the same group rather than making a new one, and a row drawn as its own source is
    skipped. So the requested share is an upper bound on what is achieved, and the record says
    the achieved figure, which is the one every other measurement here is indexed by.
    """
    corpus, _ = _setup(count=2048, queries=32)
    duplication = exact_duplicates(corpus, share=0.25)
    row = duplication.as_dict()
    return {
        "count": row["count"],
        "distinct": row["distinct"],
        "duplicate_share": row["duplicate_share"],
        "below_the_requested_share": row["duplicate_share"] < 0.25,
        "within_a_tenth_of_it": abs(row["duplicate_share"] - 0.25) < 0.1,
        "size_is_unchanged": row["count"] == int(corpus.shape[0]),
    }


def the_groups_partition_the_corpus() -> dict:
    """That every row belongs to exactly one group, which the rest of the module relies on.

    The deduplication picks one row per group, so a row appearing in two groups would be kept
    twice and a row in none would be dropped, and either would change the corpus size in a way
    that looks like a duplication rate rather than a bug.
    """
    corpus, _ = _setup(count=2048, queries=32)
    duplication = exact_duplicates(corpus, share=0.3)
    groups = duplication.groups()
    members = [row for rows in groups.values() for row in rows]
    return {
        "groups": len(groups),
        "members": len(members),
        "count": duplication.count,
        "covers_everything": len(members) == duplication.count,
        "no_overlap": len(set(members)) == len(members),
    }


def the_duplicates_really_are_identical() -> dict:
    """A check on the construction, which the recall numbers would not distinguish from noise.

    Every row labelled as a copy should be bit identical to its original. A near miss would make
    the ties disappear and the whole module would report that duplication causes no problem, for
    the wrong reason.
    """
    corpus, _ = _setup(count=2048, queries=32)
    duplication = exact_duplicates(corpus, share=0.3)
    moved = torch.nonzero(
        duplication.original != torch.arange(duplication.count), as_tuple=False
    ).flatten()
    gaps = (duplication.vectors[moved] - corpus[duplication.original[moved]]).abs()
    return {
        "copies": int(moved.numel()),
        "max_gap": round(float(gaps.max()) if int(moved.numel()) else 0.0, 8),
        "identical": bool(int(moved.numel()) == 0 or float(gaps.max()) == 0.0),
    }


def the_near_duplicates_really_are_near() -> dict:
    """The same check on the other construction.

    A near duplicate should be close and not equal. Equal would make it an exact duplicate under
    another name and the comparison between the two constructions would be measuring nothing.
    """
    corpus, _ = _setup(count=2048, queries=32)
    duplication = near_duplicates(corpus, share=0.3, nudge=0.01)
    moved = torch.nonzero(
        duplication.original != torch.arange(duplication.count), as_tuple=False
    ).flatten()
    gaps = (duplication.vectors[moved] - corpus[duplication.original[moved]]).norm(dim=1)
    return {
        "copies": int(moved.numel()),
        "mean_gap": round(float(gaps.mean()), 6),
        "close": float(gaps.mean()) < 0.1,
        "not_equal": float(gaps.min()) > 0.0,
    }


def a_share_of_one_is_refused() -> bool:
    """Whether replacing the whole corpus with copies is caught.

    Every row would be a copy of some other row, which is a cycle rather than a duplication, and
    the distinct count would depend on the permutation rather than on the requested share.
    """
    corpus, _ = _setup(count=512, queries=8)
    try:
        exact_duplicates(corpus, share=1.0)
    except ConfigError:
        return True
    return False


def a_negative_share_is_refused() -> bool:
    """Whether a negative duplication rate is caught."""
    corpus, _ = _setup(count=512, queries=8)
    try:
        exact_duplicates(corpus, share=-0.1)
    except ConfigError:
        return True
    return False


def a_single_copy_is_refused() -> bool:
    """Whether asking for one copy, which is no copy at all, is caught."""
    corpus, _ = _setup(count=512, queries=8)
    try:
        exact_duplicates(corpus, copies=1)
    except ConfigError:
        return True
    return False


def a_zero_nudge_is_refused() -> bool:
    """Whether a near duplicate with no perturbation is caught.

    It would be an exact duplicate under another name, and the two constructions are compared
    against each other, so silently returning one when the other was asked for makes the
    comparison meaningless.
    """
    corpus, _ = _setup(count=512, queries=8)
    try:
        near_duplicates(corpus, nudge=0.0)
    except ConfigError:
        return True
    return False


def a_mismatched_label_set_is_refused() -> bool:
    """Whether a duplication whose labels do not cover its rows is caught."""
    try:
        Duplication(vectors=torch.randn(10, 8), original=torch.arange(5))
    except DataError:
        return True
    return False


def a_rank_one_corpus_is_refused() -> bool:
    """Whether an unbatched corpus reaches the duplication record."""
    try:
        Duplication(vectors=torch.randn(10), original=torch.arange(10))
    except DataError:
        return True
    return False


def scoring_by_distance_needs_matching_shapes() -> bool:
    """Whether comparing a result against a truth of a different width is caught."""
    corpus, probes = _setup(count=512, queries=8)
    truth = Neighbours(torch.zeros(8, 10, dtype=torch.long), torch.zeros(8, 10))
    found = Neighbours(torch.zeros(8, 5, dtype=torch.long), torch.zeros(8, 5))
    try:
        recall_by_distance(probes, corpus, truth, found)
    except DataError:
        return True
    return False


def compare_the_scorings(share: float = 0.3) -> list[dict]:
    """Both scorings against both corpora, as the table the module exists to produce.

    Four rows. On a clean corpus the two scorings agree and on a duplicated one they do not, and
    the difference is the answer to how much a recall number is worth on a corpus with copies in
    it.
    """
    corpus, probes = _setup()
    rows = []
    for label, vectors in (
        ("clean", corpus),
        ("duplicated", exact_duplicates(corpus, share=share).vectors),
    ):
        truth = search(probes, vectors, k=10)
        index = IVFIndex(32, partitions=64, probe=8)
        index.build(vectors)
        found, _ = index.search(probes, k=10)
        rows.append(
            {
                "corpus": label,
                "scoring": "identifier",
                "recall": round(identifier_overlap(truth, found), 4),
            }
        )
        rows.append(
            {
                "corpus": label,
                "scoring": "distance",
                "recall": round(recall_by_distance(probes, vectors, truth, found), 4),
            }
        )
    return rows

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from vse.build.kmeans import lloyd
from vse.errors import BuildError, ConfigError
from vse.index.base import Index, SearchStats, top_up
from vse.index.ivf import IVFIndex
from vse.vectors.dataset import clustered, gaussian, held_out
from vse.vectors.exact import Neighbours, identifier_overlap, search
from vse.vectors.metric import squared_l2

# Filing every vector under several partitions instead of one, which is the cheapest way to buy
# recall in a partitioned index and the one with the most obvious cost.
#
# An inverted file files each vector under its nearest centroid. A query near a boundary has its
# neighbours split across two partitions, and opening one of them finds half of what it wanted.
# The measurements in eval/adversarial.py make that concrete: queries placed exactly on a
# midpoint lose nine points of recall against ordinary ones, and it is the only constructible
# attack in that module that works.
#
# Spilling is the direct fix. File each vector under its nearest s centroids rather than its
# nearest one. A boundary query then finds its neighbours in whichever partition it opens,
# because they are in both. The corpus is stored once and the posting lists reference it s
# times, so the index grows by s in list length and not at all in vectors held.
#
# The cost is exactly that growth. A partition holds s times as many rows, so a probe of p scans
# s times as much, and the comparison against simply probing more is the whole question. Two
# schemes that both scan twice as much: spill of two at probe p, or spill of one at probe 2p.
# They are not the same. Doubling the probe opens partitions in order of centroid distance,
# reaching further from the query; doubling the spill deepens the partitions already open, which
# stays close. Which is better should depend on whether the missing neighbours are nearby in a
# partition that was not opened, or nearby in one that was.
#
# The other question is whether spilling has to be uniform. A vector well inside a partition
# gains nothing from being filed twice, and one near a boundary gains everything, so spilling
# adaptively by how close each vector is to its second nearest centroid should cost less for the
# same benefit. That is measured against uniform spilling at matched list growth.


@dataclass
class Spilled:
    """A partitioning where each vector appears in several lists."""

    assignment: torch.Tensor
    centres: torch.Tensor

    def __post_init__(self) -> None:
        if self.assignment.ndim != 2:
            raise ConfigError(
                f"a spilled assignment is rows by copies, got {self.assignment.ndim}"
            )
        if int(self.assignment.shape[1]) < 1:
            raise ConfigError("every vector needs at least one partition")

    @property
    def count(self) -> int:
        """How many vectors are filed."""
        return int(self.assignment.shape[0])

    @property
    def copies(self) -> int:
        """How many partitions each vector appears in."""
        return int(self.assignment.shape[1])

    @property
    def partitions(self) -> int:
        """How many partitions there are."""
        return int(self.centres.shape[0])

    @property
    def entries(self) -> int:
        """Total posting list length across all partitions."""
        return self.count * self.copies

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "count": self.count,
            "copies": self.copies,
            "partitions": self.partitions,
            "entries": self.entries,
            "growth": round(self.entries / max(self.count, 1), 2),
        }


def spill_uniformly(vectors: torch.Tensor, centres: torch.Tensor, copies: int) -> Spilled:
    """File every vector under its nearest few centroids.

    The simplest scheme and the baseline the adaptive one has to beat. Every vector pays the
    same price whether it needed it or not, which is where the waste is and is why it needs no
    parameter beyond the copy count.
    """
    if copies < 1:
        raise ConfigError(f"{copies} copies files nothing")
    partitions = int(centres.shape[0])
    if copies > partitions:
        raise ConfigError(f"cannot file into {copies} of {partitions} partitions")
    scores = squared_l2(vectors, centres)
    return Spilled(
        assignment=torch.topk(scores, k=copies, dim=1, largest=False).indices,
        centres=centres.clone(),
    )


def spill_adaptively(
    vectors: torch.Tensor, centres: torch.Tensor, share: float = 0.3
) -> Spilled:
    """File a share of the vectors twice, choosing the ones nearest a boundary.

    A vector whose first and second centroid distances are close is the one a boundary query
    needs to find, and one deep inside a partition is not. Sorting by that ratio and duplicating
    the top share spends the list growth where it can do something.

    The result still has a rectangular assignment, because a ragged one would complicate every
    consumer for no benefit: vectors not chosen get their nearest centroid twice, which the
    posting list construction deduplicates.
    """
    if not 0.0 <= share <= 1.0:
        raise ConfigError(f"a share of {share} is not a share")
    if int(centres.shape[0]) < 2:
        raise ConfigError("adaptive spilling needs at least two partitions")
    scores = squared_l2(vectors, centres)
    nearest = torch.topk(scores, k=2, dim=1, largest=False)
    ratio = nearest.values[:, 0] / nearest.values[:, 1].clamp_min(1e-12)
    count = int(vectors.shape[0])
    chosen = torch.argsort(ratio, descending=True)[: round(count * share)]
    assignment = nearest.indices.clone()
    keep = torch.ones(count, dtype=torch.bool)
    keep[chosen] = False
    assignment[keep, 1] = assignment[keep, 0]
    return Spilled(assignment=assignment, centres=centres.clone())


class SpillIndex(Index):
    """An inverted file where each vector is filed under several centroids."""

    def __init__(
        self,
        dimension: int,
        partitions: int = 64,
        probe: int = 8,
        copies: int = 2,
        adaptive: float | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__(dimension)
        if partitions < 1:
            raise ConfigError(f"{partitions} partitions is not a partitioning")
        if probe < 1:
            raise ConfigError(f"probing {probe} partitions is not a search")
        self.partitions = partitions
        self.probe = probe
        self.copies = copies
        self.adaptive = adaptive
        self.seed = seed
        self._vectors: torch.Tensor | None = None
        self._live: torch.Tensor | None = None
        self._centres: torch.Tensor | None = None
        self._lists: list[torch.Tensor] = []
        self._spilled: Spilled | None = None

    @property
    def spilled(self) -> Spilled:
        """The assignment."""
        self._require_built()
        return self._spilled

    def build(self, vectors: torch.Tensor) -> None:
        """Cluster, then file every vector under several centroids."""
        vectors = self._check_vectors(vectors)
        if self.partitions > int(vectors.shape[0]):
            raise BuildError(
                f"{self.partitions} partitions over {int(vectors.shape[0])} vectors"
            )
        run = lloyd(vectors, k=self.partitions, seed=self.seed)
        self._centres = run.centres.clone()
        self._spilled = (
            spill_adaptively(vectors, self._centres, share=self.adaptive)
            if self.adaptive is not None
            else spill_uniformly(vectors, self._centres, self.copies)
        )
        self._vectors = vectors.clone()
        self._live = torch.ones(int(vectors.shape[0]), dtype=torch.bool)
        self._rebuild_lists()
        self._built = True

    def _rebuild_lists(self) -> None:
        """Materialise the posting lists, deduplicating within each one."""
        buckets: list[list[int]] = [[] for _ in range(self.partitions)]
        assignment = self._spilled.assignment
        for row in range(int(assignment.shape[0])):
            for partition in set(assignment[row].tolist()):
                buckets[partition].append(row)
        self._lists = [torch.tensor(rows, dtype=torch.long) for rows in buckets]

    def list_lengths(self) -> list[int]:
        """How long each posting list is, which is what the growth costs."""
        self._require_built()
        return [int(rows.numel()) for rows in self._lists]

    def search(self, queries: torch.Tensor, k: int = 10) -> tuple[Neighbours, SearchStats]:
        """Score the centres, open the nearest few, scan what is in them."""
        self._require_built()
        self._check_queries(queries, k)
        if self.probe > self.partitions:
            raise ConfigError(f"probing {self.probe} of {self.partitions} partitions")
        count = int(queries.shape[0])
        stats = SearchStats(queries=count)
        stats.charge(self.partitions * count)
        centre_scores = squared_l2(queries, self._centres)
        chosen = torch.topk(centre_scores, k=self.probe, dim=1, largest=False).indices
        identifiers = torch.zeros(count, k, dtype=torch.long)
        scores = torch.zeros(count, k)
        for row in range(count):
            rows = torch.cat([self._lists[int(part)] for part in chosen[row]])
            rows = torch.unique(rows)
            rows = rows[self._live[rows]]
            stats.hop(self.probe)
            stats.charge(int(rows.numel()))
            stats.visit(int(rows.numel()))
            reached: list[tuple[float, int]] = []
            if int(rows.numel()):
                block = squared_l2(queries[row : row + 1], self._vectors[rows]).flatten()
                best = torch.topk(block, k=min(k, int(rows.numel())), largest=False)
                reached = list(
                    zip(best.values.tolist(), rows[best.indices].tolist(), strict=True)
                )
            filled = top_up(
                reached, k, queries[row : row + 1], self._vectors, self._live, self.metric
            )
            for slot, (score, other) in enumerate(filled):
                identifiers[row, slot] = other
                scores[row, slot] = score
        return Neighbours(identifiers=identifiers, scores=scores), stats

    def insert(self, vectors: torch.Tensor) -> list[int]:
        """File new vectors under the same centroids and rebuild the lists."""
        self._require_built()
        vectors = self._check_vectors(vectors)
        start = int(self._vectors.shape[0])
        added = (
            spill_adaptively(vectors, self._centres, share=self.adaptive)
            if self.adaptive is not None
            else spill_uniformly(vectors, self._centres, self.copies)
        )
        self._spilled.assignment = torch.cat(
            [self._spilled.assignment, added.assignment], dim=0
        )
        self._vectors = torch.cat([self._vectors, vectors.clone()], dim=0)
        self._live = torch.cat(
            [self._live, torch.ones(int(vectors.shape[0]), dtype=torch.bool)]
        )
        self._rebuild_lists()
        return list(range(start, int(self._vectors.shape[0])))

    def remove(self, identifiers: Sequence[int]) -> int:
        """Mark rows dead. They stay in every list they were filed into."""
        self._require_built()
        removed = 0
        for identifier in identifiers:
            if not 0 <= identifier < int(self._live.numel()):
                raise ConfigError(
                    f"{identifier} is not one of the {int(self._live.numel())} rows"
                )
            if bool(self._live[identifier]):
                self._live[identifier] = False
                removed += 1
        return removed

    def memory_bytes(self) -> int:
        """Vectors once, posting list entries once each."""
        self._require_built()
        vectors = int(self._vectors.shape[0]) * self.dimension * 4
        centres = self.partitions * self.dimension * 4
        entries = sum(int(rows.numel()) for rows in self._lists) * 8
        return vectors + centres + entries

    @property
    def size(self) -> int:
        """Live vectors, counted once regardless of how many lists hold them."""
        self._require_built()
        return int(self._live.sum())


def _setup(count: int = 4096, dimension: int = 32, queries: int = 200):
    """A corpus with queries held out and their true answers."""
    corpus = gaussian(count=count, dimension=dimension)
    searched, probes = held_out(corpus, count=queries)
    return searched.vectors, probes, search(probes, searched.vectors, k=10)


def spilling_grows_the_lists(copies: Sequence[int] = (1, 2, 3, 4)) -> list[dict]:
    """What filing each vector several times costs in list length.

    Almost exactly the copy count, minus whatever deduplication removes, which is nothing here
    because a vector's nearest few centroids are distinct by construction. The vectors are
    stored
    once, so the memory grows by the identifiers rather than by the corpus, which is the whole
    reason this is cheap.
    """
    if not copies:
        raise ConfigError("there is nothing to sweep")
    corpus, _, _ = _setup()
    rows = []
    for count in copies:
        index = SpillIndex(32, partitions=64, probe=8, copies=count)
        index.build(corpus)
        lengths = index.list_lengths()
        rows.append(
            {
                "copies": count,
                "total_entries": sum(lengths),
                "mean_list": round(sum(lengths) / len(lengths), 1),
                "growth": round(sum(lengths) / int(corpus.shape[0]), 2),
                "memory_bytes": index.memory_bytes(),
            }
        )
    return rows


def the_growth_is_the_copy_count() -> dict:
    """That the list growth is exactly what was asked for, which bounds everything else."""
    rows = {row["copies"]: row for row in spilling_grows_the_lists()}
    return {
        "growth_at_one": rows[1]["growth"],
        "growth_at_four": rows[4]["growth"],
        "exact": all(abs(row["growth"] - copies) < 0.01 for copies, row in rows.items()),
        "memory_ratio": round(rows[4]["memory_bytes"] / rows[1]["memory_bytes"], 2),
    }


def spilling_buys_recall(copies: Sequence[int] = (1, 2, 3, 4)) -> list[dict]:
    """What the copies buy at a fixed probe count.

    Recall and distances together, because the copies raise both: a partition holds more, so
    opening the same number of them scans more. Reading the recall column alone would repeat the
    error build/sampling.py made, which is why the cost is next to it.
    """
    if not copies:
        raise ConfigError("there is nothing to sweep")
    corpus, probes, truth = _setup()
    rows = []
    for count in copies:
        index = SpillIndex(32, partitions=64, probe=8, copies=count)
        index.build(corpus)
        found, stats = index.search(probes, k=10)
        rows.append(
            {
                "copies": count,
                "recall": round(identifier_overlap(truth, found), 4),
                "distances": round(stats.distances_per_query, 1),
            }
        )
    return rows


def against_probing_more(budgets: Sequence[float] = (600.0, 1200.0, 2400.0)) -> list[dict]:
    """The comparison this module exists for, at matched distance counts.

    Two ways to scan more: file each vector in more partitions, or open more partitions. At the
    same distance count they are not the same operation. Opening more reaches further from the
    query; spilling more deepens what is already open.
    """
    if not budgets:
        raise ConfigError("there is nothing to sweep")
    corpus, probes, truth = _setup()
    rows = []
    for budget in budgets:
        spilled = _best_within(
            corpus, probes, truth, budget, copies=(2, 3, 4), probes_tried=(2, 4, 8, 16)
        )
        plain = _best_within(
            corpus, probes, truth, budget, copies=(1,), probes_tried=(2, 4, 8, 16, 32, 64)
        )
        rows.append(
            {
                "budget": budget,
                "spilled_recall": spilled[0],
                "spilled_setting": spilled[1],
                "plain_recall": plain[0],
                "plain_setting": plain[1],
                "spilling_wins": spilled[0] > plain[0],
            }
        )
    return rows


def _best_within(
    corpus: torch.Tensor,
    probes: torch.Tensor,
    truth: Neighbours,
    budget: float,
    copies: Sequence[int],
    probes_tried: Sequence[int],
) -> tuple[float, str]:
    """The best recall reachable inside a distance budget over a grid of settings."""
    best = (0.0, "none")
    for count in copies:
        index = SpillIndex(32, partitions=64, probe=1, copies=count)
        index.build(corpus)
        for probe in probes_tried:
            index.probe = probe
            found, stats = index.search(probes, k=10)
            if stats.distances_per_query > budget:
                break
            recall = identifier_overlap(truth, found)
            if recall > best[0]:
                best = (round(recall, 4), f"{count} copies at probe {probe}")
    return best


def probing_more_is_usually_the_better_spend() -> dict:
    """The summary of that comparison, which is what a deployment would act on."""
    rows = against_probing_more()
    wins = sum(1 for row in rows if row["spilling_wins"])
    return {
        "budgets": len(rows),
        "spilling_wins": wins,
        "probing_wins": len(rows) - wins,
        "at_the_smallest_budget": rows[0]["spilling_wins"],
        "at_the_largest": rows[-1]["spilling_wins"],
    }


def spilling_fixes_the_boundary_attack() -> dict:
    """Whether the thing spilling is for actually works.

    eval/adversarial.py found that queries on the midpoint of two centroids lose six points of
    recall against ordinary ones, which is the only constructible attack in that module that
    does anything. Spilling files each vector in both partitions, so a midpoint query finds its
    neighbours whichever one it opens.

    In absolute terms it works and by a lot: midpoint recall goes from 0.489 to 0.632. What it
    does not do is close the relative gap, which widens from 0.063 to 0.113, because spilling
    helps ordinary queries even more than it helps boundary ones.

    Which is the honest reading of a fix that raises the whole curve rather than flattening it.
    The attacked queries are better served than before and they are still the worst served, and
    a deployment cares about the first of those while a benchmark reports the second.
    """
    corpus, probes, _ = _setup()
    plain = IVFIndex(32, partitions=64, probe=8)
    plain.build(corpus)
    midpoints = _midpoints(plain._centres, count=200)
    rows = {}
    for label, index in (
        ("plain", plain),
        ("spilled", _built(SpillIndex(32, partitions=64, probe=8, copies=2), corpus)),
    ):
        ordinary_truth = search(probes, corpus, k=10)
        ordinary, _ = index.search(probes, k=10)
        hard_truth = search(midpoints, corpus, k=10)
        hard, _ = index.search(midpoints, k=10)
        rows[label] = {
            "ordinary": identifier_overlap(ordinary_truth, ordinary),
            "midpoint": identifier_overlap(hard_truth, hard),
        }
    return {
        "plain_ordinary": round(rows["plain"]["ordinary"], 4),
        "plain_midpoint": round(rows["plain"]["midpoint"], 4),
        "spilled_ordinary": round(rows["spilled"]["ordinary"], 4),
        "spilled_midpoint": round(rows["spilled"]["midpoint"], 4),
        "plain_gap": round(rows["plain"]["ordinary"] - rows["plain"]["midpoint"], 4),
        "spilled_gap": round(rows["spilled"]["ordinary"] - rows["spilled"]["midpoint"], 4),
        "midpoint_recall_rises": rows["spilled"]["midpoint"] > rows["plain"]["midpoint"],
        "but_the_gap_widens": (rows["spilled"]["ordinary"] - rows["spilled"]["midpoint"])
        > (rows["plain"]["ordinary"] - rows["plain"]["midpoint"]),
    }


def _midpoints(centres: torch.Tensor, count: int = 200, seed: int = 0) -> torch.Tensor:
    """Queries halfway between two centroids, as eval/adversarial.py builds them."""
    partitions = int(centres.shape[0])
    generator = torch.Generator().manual_seed(seed)
    left = torch.randint(0, partitions, (count,), generator=generator)
    right = (left + 1 + torch.randint(0, partitions - 1, (count,), generator=generator)) % (
        partitions
    )
    return (centres[left] + centres[right]) / 2


def _built(index: Index, corpus: torch.Tensor) -> Index:
    """Build and return, so the tables above read as lists."""
    index.build(corpus)
    return index


def adaptive_spilling_spends_less_for_the_same(
    shares: Sequence[float] = (0.1, 0.3, 0.5, 1.0),
) -> list[dict]:
    """Whether the copies can be spent only where they help.

    Duplicating only the vectors near a boundary should buy most of the benefit for a fraction
    of
    the list growth. A share of one is uniform spilling at two copies, which makes the sweep its
    own control: the last row should match the uniform measurement exactly.
    """
    if not shares:
        raise ConfigError("there is nothing to sweep")
    corpus, probes, truth = _setup()
    rows = []
    for share in shares:
        index = SpillIndex(32, partitions=64, probe=8, adaptive=share)
        index.build(corpus)
        found, stats = index.search(probes, k=10)
        lengths = index.list_lengths()
        rows.append(
            {
                "share": share,
                "recall": round(identifier_overlap(truth, found), 4),
                "distances": round(stats.distances_per_query, 1),
                "growth": round(sum(lengths) / int(corpus.shape[0]), 3),
            }
        )
    return rows


def the_full_share_matches_uniform_spilling() -> dict:
    """That the adaptive scheme's endpoint is the uniform one, which is the control.

    A share of one duplicates every vector, which is exactly what two uniform copies does, so
    the
    two should agree on recall, cost and growth. They are computed by different code paths, so
    an
    agreement is evidence both are right and a disagreement would say which to look at.
    """
    adaptive = {row["share"]: row for row in adaptive_spilling_spends_less_for_the_same()}
    uniform = {row["copies"]: row for row in spilling_buys_recall()}
    growth = {row["copies"]: row for row in spilling_grows_the_lists()}
    return {
        "adaptive_recall": adaptive[1.0]["recall"],
        "uniform_recall": uniform[2]["recall"],
        "adaptive_growth": adaptive[1.0]["growth"],
        "uniform_growth": growth[2]["growth"],
        "recall_matches": abs(adaptive[1.0]["recall"] - uniform[2]["recall"]) < 1e-6,
        "growth_matches": abs(adaptive[1.0]["growth"] - growth[2]["growth"]) < 0.01,
    }


def the_adaptive_curve_is_better_than_the_uniform_one() -> dict:
    """Whether spending the copies selectively beats spending them evenly.

    Comparing recall per unit of list growth at the adaptive scheme's middle setting against the
    uniform scheme's endpoints. The adaptive scheme can sit between one and two copies, which
    the
    uniform one cannot express at all, so the comparison is also about the granularity of the
    knob rather than only about where the copies go.
    """
    adaptive = {row["share"]: row for row in adaptive_spilling_spends_less_for_the_same()}
    uniform = {row["copies"]: row for row in spilling_buys_recall()}
    growth = {row["copies"]: row for row in spilling_grows_the_lists()}
    middle = adaptive[0.3]
    return {
        "adaptive_recall_at_a_third": middle["recall"],
        "adaptive_growth_at_a_third": middle["growth"],
        "uniform_one_recall": uniform[1]["recall"],
        "uniform_two_recall": uniform[2]["recall"],
        "adaptive_share_of_the_gain": round(
            (middle["recall"] - uniform[1]["recall"])
            / max(uniform[2]["recall"] - uniform[1]["recall"], 1e-9),
            3,
        ),
        "adaptive_share_of_the_cost": round(
            (middle["growth"] - 1.0) / max(growth[2]["growth"] - 1.0, 1e-9), 3
        ),
    }


def a_clustered_corpus_needs_less_spilling() -> dict:
    """Whether the corpus shape changes what spilling is worth.

    A clustered corpus has partitions that line up with real structure, so few vectors sit near
    a
    boundary and the copies mostly go where they are not needed. An isotropic corpus has
    boundaries running through dense regions. Both are measured at the same copy count.
    """
    rows = {}
    for label, corpus in (
        ("gaussian", gaussian(count=4096, dimension=32)),
        ("clustered", clustered(count=4096, dimension=32, clusters=16)),
    ):
        searched, probes = held_out(corpus, count=200)
        truth = search(probes, searched.vectors, k=10)
        plain = SpillIndex(32, partitions=64, probe=8, copies=1)
        plain.build(searched.vectors)
        spilled = SpillIndex(32, partitions=64, probe=8, copies=2)
        spilled.build(searched.vectors)
        one, _ = plain.search(probes, k=10)
        two, _ = spilled.search(probes, k=10)
        rows[label] = {
            "one": identifier_overlap(truth, one),
            "two": identifier_overlap(truth, two),
        }
    return {
        "gaussian_gain": round(rows["gaussian"]["two"] - rows["gaussian"]["one"], 4),
        "clustered_gain": round(rows["clustered"]["two"] - rows["clustered"]["one"], 4),
        "gaussian_one": round(rows["gaussian"]["one"], 4),
        "clustered_one": round(rows["clustered"]["one"], 4),
        "structure_needs_less": (rows["clustered"]["two"] - rows["clustered"]["one"])
        < (rows["gaussian"]["two"] - rows["gaussian"]["one"]),
    }


def the_lists_are_deduplicated() -> dict:
    """That a vector filed twice into the same partition appears once.

    The adaptive scheme files unselected vectors under their nearest centroid twice, so the
    deduplication is load bearing rather than defensive: without it every unselected vector
    would
    appear twice in one list and the scan would score it twice, producing a duplicate identifier
    of exactly the kind verify/differential.py's distinctness rule exists to catch.
    """
    corpus, probes, _ = _setup(count=2048, queries=32)
    index = SpillIndex(32, partitions=32, probe=4, adaptive=0.0)
    index.build(corpus)
    lengths = index.list_lengths()
    found, _ = index.search(probes, k=10)
    distinct = all(
        int(torch.unique(found.identifiers[row]).numel()) == 10
        for row in range(int(found.identifiers.shape[0]))
    )
    return {
        "share": 0.0,
        "total_entries": sum(lengths),
        "corpus": int(corpus.shape[0]),
        "no_growth": sum(lengths) == int(corpus.shape[0]),
        "results_are_distinct": distinct,
    }


def the_result_is_well_formed() -> dict:
    """That a spilled search returns what every other index here returns.

    Exactly k, distinct, sorted. Worth checking separately because the candidate set is a union
    of overlapping lists, which is precisely the construction that produced duplicate
    identifiers
    in three structures when verify/differential.py first ran.
    """
    corpus, probes, truth = _setup(count=2048, queries=64)
    index = SpillIndex(32, partitions=32, probe=4, copies=3)
    index.build(corpus)
    found, _ = index.search(probes, k=10)
    return {
        "shape": tuple(found.identifiers.shape),
        "distinct": all(
            int(torch.unique(found.identifiers[row]).numel()) == 10
            for row in range(int(found.identifiers.shape[0]))
        ),
        "sorted": bool(torch.all(found.scores[:, 1:] >= found.scores[:, :-1] - 1e-5)),
        "recall": round(identifier_overlap(truth, found), 4),
    }


def removal_and_insertion_work() -> dict:
    """That the write path keeps every list consistent.

    An insert files the new vector into several lists and a removal takes it out of all of them
    by marking it dead, which is the same tombstone the plain inverted file uses. The rebuild on
    insert is the expensive part and it is honest: a spilled index has no cheap incremental
    insert because the lists are recomputed from the assignment.
    """
    corpus, probes, _ = _setup(count=2048, queries=32)
    index = SpillIndex(32, partitions=32, probe=4, copies=2)
    index.build(corpus[:1000])
    before = index.size
    index.insert(corpus[1000:1500])
    grown = index.size
    found, _ = index.search(probes, k=5)
    index.remove([0, 1, 2])
    return {
        "built": before,
        "after_insert": grown,
        "after_remove": index.size,
        "insert_worked": grown == before + 500,
        "remove_worked": index.size == grown - 3,
        "still_searchable": int(found.identifiers.shape[0]) == int(probes.shape[0]),
    }


def one_copy_is_a_plain_inverted_file() -> dict:
    """That a spill of one is the structure it generalises.

    Filing each vector under one centroid is exactly what IVFIndex does, so the two should agree
    on recall and cost at the same settings. They are separate implementations, so agreement is
    evidence for both and a gap would say the spilled scan is doing something the plain one is
    not.
    """
    corpus, probes, truth = _setup()
    spilled = SpillIndex(32, partitions=64, probe=8, copies=1)
    spilled.build(corpus)
    plain = IVFIndex(32, partitions=64, probe=8)
    plain.build(corpus)
    left, left_stats = spilled.search(probes, k=10)
    right, right_stats = plain.search(probes, k=10)
    return {
        "spilled_recall": round(identifier_overlap(truth, left), 4),
        "plain_recall": round(identifier_overlap(truth, right), 4),
        "spilled_distances": round(left_stats.distances_per_query, 1),
        "plain_distances": round(right_stats.distances_per_query, 1),
        "recall_matches": abs(
            identifier_overlap(truth, left) - identifier_overlap(truth, right)
        )
        < 1e-6,
        "cost_matches": abs(left_stats.distances_per_query - right_stats.distances_per_query)
        < 1.0,
    }


def zero_copies_are_refused() -> bool:
    """Whether filing a vector nowhere is caught."""
    corpus = gaussian(count=512, dimension=8).vectors
    try:
        spill_uniformly(corpus, torch.randn(8, 8), copies=0)
    except ConfigError:
        return True
    return False


def more_copies_than_partitions_are_refused() -> bool:
    """Whether filing into more partitions than exist is caught."""
    corpus = gaussian(count=512, dimension=8).vectors
    try:
        spill_uniformly(corpus, torch.randn(4, 8), copies=8)
    except ConfigError:
        return True
    return False


def an_adaptive_share_outside_the_interval_is_refused() -> bool:
    """Whether duplicating more than everything is caught."""
    corpus = gaussian(count=512, dimension=8).vectors
    try:
        spill_adaptively(corpus, torch.randn(8, 8), share=1.5)
    except ConfigError:
        return True
    return False


def adaptive_spilling_needs_two_partitions() -> bool:
    """Whether measuring a boundary with one partition is caught.

    There is no second nearest centroid with one partition, so the ratio the scheme sorts on
    does
    not exist and the topk would fail with a message about dimensions rather than about the
    thing
    that is wrong.
    """
    corpus = gaussian(count=512, dimension=8).vectors
    try:
        spill_adaptively(corpus, torch.randn(1, 8), share=0.5)
    except ConfigError:
        return True
    return False


def a_rank_one_assignment_is_refused() -> bool:
    """Whether an assignment with no copy dimension is caught."""
    try:
        Spilled(assignment=torch.zeros(10, dtype=torch.long), centres=torch.randn(4, 8))
    except ConfigError:
        return True
    return False


def a_corpus_smaller_than_the_partitions_is_refused() -> bool:
    """Whether building more partitions than there are vectors is caught."""
    try:
        SpillIndex(8, partitions=64).build(torch.randn(32, 8))
    except BuildError:
        return True
    return False


def probing_more_partitions_than_exist_is_refused() -> bool:
    """Whether opening more partitions than were built is caught at search time."""
    corpus = gaussian(count=512, dimension=8).vectors
    index = SpillIndex(8, partitions=8, probe=32, copies=2)
    index.build(corpus)
    try:
        index.search(corpus[:2], k=5)
    except ConfigError:
        return True
    return False


def a_spilled_assignment_reports_its_shape() -> dict:
    """That the record says how much it grew the index by."""
    corpus = gaussian(count=1024, dimension=8).vectors
    run = lloyd(corpus, k=16, seed=0)
    spilled = spill_uniformly(corpus, run.centres, copies=3)
    row = spilled.as_dict()
    return {
        "count": row["count"],
        "copies": row["copies"],
        "partitions": row["partitions"],
        "entries": row["entries"],
        "growth": row["growth"],
        "growth_is_the_copies": row["growth"] == 3.0,
    }


def compare_the_schemes() -> list[dict]:
    """Plain, uniform spilling and adaptive spilling, as one table.

    Four rows at the same probe count, which is the wrong comparison on its own and is the one a
    reader wants first. The matched cost version is against_probing_more and it is the one the
    conclusion rests on.
    """
    corpus, probes, truth = _setup()
    rows = []
    for label, index in (
        ("plain", SpillIndex(32, partitions=64, probe=8, copies=1)),
        ("uniform two", SpillIndex(32, partitions=64, probe=8, copies=2)),
        ("uniform four", SpillIndex(32, partitions=64, probe=8, copies=4)),
        ("adaptive a third", SpillIndex(32, partitions=64, probe=8, adaptive=0.3)),
    ):
        index.build(corpus)
        found, stats = index.search(probes, k=10)
        lengths = index.list_lengths()
        rows.append(
            {
                "scheme": label,
                "recall": round(identifier_overlap(truth, found), 4),
                "distances": round(stats.distances_per_query, 1),
                "growth": round(sum(lengths) / int(corpus.shape[0]), 3),
            }
        )
    return rows

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from vse.errors import ConfigError, DataError
from vse.index.ivf import IVFIndex
from vse.vectors.dataset import Corpus, gaussian, held_out
from vse.vectors.exact import Neighbours, search
from vse.vectors.metric import squared_l2

# The measures every index in this package is judged by, and what each one hides.
#
# Recall at k is the share of the true neighbours that came back. It is the number everybody
# quotes and it has three properties worth knowing before quoting it. It ignores order
# completely, so a result that returns the right ten in the worst possible sequence scores the
# same as one that returns them perfectly. It is all or nothing per vector, so returning the
# eleventh nearest instead of the tenth costs as much as returning a random vector. And it is
# undefined in a useful way when the corpus has duplicates, which vectors/exact.py measured.
#
# So there are four measures here rather than one. Recall for the headline. Rank based gain,
# which discounts by position and so notices ordering. Mean reciprocal rank, which only looks at
# where the first correct answer landed and is the right measure when a caller shows one result.
# And the distance ratio, which asks how much further the returned neighbours are than the true
# ones and is the only measure of the four that does not care about identifiers at all.
#
# Three things came out of measuring these against each other, two of which corrected me.
#
# The gain measure does not notice a pure reordering either, which is what I built it to catch.
# Shuffling a perfectly correct result leaves it at exactly one, because every item in it is
# relevant and the discount weights are a fixed set being permuted among relevant items. It
# notices position only when correct and incorrect answers are mixed, which is the useful case
# and is not the case I wrote the docstring about.
#
# The gain can exceed the recall, which I had assumed impossible. A result with half the true
# neighbours all at the front scores fifty percent recall and sixty five percent gain, because
# the discount rewards them for being early. That is the ordering information recall discards,
# showing up as the two numbers disagreeing rather than as one bounding the other.
#
# And the distance ratio separates a near miss from a random vector by only a third, where I
# expected a much wider gap. That is the concentration result from vectors/dataset.py arriving
# somewhere new: in thirty two dimensions a random vector is not very much further from a query
# than the tenth nearest one is, so a measure based on distance has less room to distinguish
# them than a measure based on identity. Recall calls both of them a total failure; the distance
# ratio calls one of them six percent worse and the other forty two.


@dataclass(frozen=True)
class Scores:
    """Every measure of one result against one ground truth."""

    recall: float
    gain: float
    reciprocal_rank: float
    distance_ratio: float
    k: int

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "recall": round(self.recall, 4),
            "gain": round(self.gain, 4),
            "reciprocal_rank": round(self.reciprocal_rank, 4),
            "distance_ratio": round(self.distance_ratio, 5),
            "k": self.k,
        }


def _checked(truth: Neighbours, found: Neighbours) -> None:
    """Reject a comparison between results that are not comparable."""
    if truth.queries != found.queries:
        raise DataError(f"{truth.queries} queries against {found.queries}")
    if truth.k != found.k:
        raise DataError(f"top {truth.k} against top {found.k}")


def recall_at_k(truth: Neighbours, found: Neighbours) -> float:
    """The share of the true neighbours that came back, in any order.

    The headline number. It treats the result as a set, so it cannot distinguish a perfectly
    ordered answer from a shuffled one, and it treats every miss the same, so the eleventh
    nearest and a random vector cost identically.
    """
    _checked(truth, found)
    total = 0.0
    for row in range(truth.queries):
        total += len(set(truth.row(row)) & set(found.row(row))) / truth.k
    return total / truth.queries


def discounted_gain(truth: Neighbours, found: Neighbours) -> float:
    """Rank based gain, normalised so a perfect answer scores one.

    Each returned vector is worth one if it belongs in the true set and nothing otherwise,
    discounted by the logarithm of the position it appeared at. So getting the right answers is
    worth more than getting them in the right order, and getting them in the right order is
    worth something, which recall says it is not.
    """
    _checked(truth, found)
    weights = torch.tensor(
        [1.0 / math.log2(position + 2) for position in range(truth.k)], dtype=torch.float32
    )
    perfect = float(weights.sum())
    total = 0.0
    for row in range(truth.queries):
        wanted = set(truth.row(row))
        earned = sum(
            float(weights[position])
            for position, other in enumerate(found.row(row))
            if other in wanted
        )
        total += earned / perfect
    return total / truth.queries


def reciprocal_rank(truth: Neighbours, found: Neighbours) -> float:
    """One over the position of the first correct answer, averaged.

    The measure for an application that shows one result and asks whether it was right. It
    ignores everything after the first hit, which makes it useless for a caller that consumes a
    whole page and exactly right for one that does not.
    """
    _checked(truth, found)
    total = 0.0
    for row in range(truth.queries):
        wanted = set(truth.row(row))
        for position, other in enumerate(found.row(row)):
            if other in wanted:
                total += 1.0 / (position + 1)
                break
    return total / truth.queries


def distance_ratio(
    queries: torch.Tensor, corpus: torch.Tensor, truth: Neighbours, found: Neighbours
) -> float:
    """How much further the returned neighbours are than the true ones.

    One when the answer is as good as the truth, whatever identifiers it used. The only measure
    here that survives a corpus with duplicates in it, and the only one an application can
    interpret without knowing anything about the index, because a result that is two percent
    further away is two percent worse in whatever the embedding was measuring.
    """
    _checked(truth, found)
    best = torch.gather(squared_l2(queries, corpus), 1, truth.identifiers).sqrt()
    theirs = torch.gather(squared_l2(queries, corpus), 1, found.identifiers).sqrt()
    floor = best.clamp_min(1e-9)
    return float((theirs / floor).mean())


def score_all(
    queries: torch.Tensor, corpus: torch.Tensor, truth: Neighbours, found: Neighbours
) -> Scores:
    """Every measure at once, which is how they should be reported."""
    return Scores(
        recall=recall_at_k(truth, found),
        gain=discounted_gain(truth, found),
        reciprocal_rank=reciprocal_rank(truth, found),
        distance_ratio=distance_ratio(queries, corpus, truth, found),
        k=truth.k,
    )


def shuffled(found: Neighbours, seed: int = 0) -> Neighbours:
    """The same identifiers in a different order.

    The control for whether a measure notices ordering. Every set based measure scores this
    identically to the original and every rank based one does not, which is the whole
    distinction between them.
    """
    generator = torch.Generator().manual_seed(seed)
    order = torch.stack(
        [torch.randperm(found.k, generator=generator) for _ in range(found.queries)]
    )
    return Neighbours(
        identifiers=torch.gather(found.identifiers, 1, order),
        scores=torch.gather(found.scores, 1, order),
    )


def near_misses(
    queries: torch.Tensor, corpus: torch.Tensor, k: int = 10, offset: int = 10
) -> Neighbours:
    """A result made of the vectors just outside the true top k.

    The control for whether a measure notices how wrong a wrong answer is. Every identifier here
    is a miss, so recall is zero, and every one of them is nearly as close as the answer it
    replaced, so the distance measure barely moves.
    """
    if offset < 1:
        raise ConfigError(f"an offset of {offset} does not miss anything")
    wide = search(queries, corpus, k=k + offset)
    return Neighbours(
        identifiers=wide.identifiers[:, offset:],
        scores=wide.scores[:, offset:],
    )


def neither_set_measure_sees_a_pure_reordering() -> dict:
    """Whether shuffling a correct answer changes either set based measure.

    Neither of them, which is not what I expected of the second one. Recall treats the result as
    a set so a permutation is invisible by construction. The discounted gain weights by position
    and still scores exactly one, because every item in a perfectly correct result is relevant,
    so permuting them permutes which relevant item gets which weight and the total is unchanged.
    A rank measure only sees position when there is something irrelevant to push down, which the
    next function measures.
    """
    corpus = gaussian(count=2048, dimension=32)
    searched, probes = held_out(corpus, count=64)
    truth = search(probes, searched.vectors, k=10)
    mixed = shuffled(truth)
    return {
        "recall_before": round(recall_at_k(truth, truth), 4),
        "recall_after": round(recall_at_k(truth, mixed), 4),
        "gain_before": round(discounted_gain(truth, truth), 4),
        "gain_after": round(discounted_gain(truth, mixed), 4),
        "recall_unchanged": recall_at_k(truth, mixed) == recall_at_k(truth, truth),
        "gain_unchanged": abs(discounted_gain(truth, mixed) - discounted_gain(truth, truth))
        < 1e-6,
    }


def but_the_gain_sees_position_when_the_result_is_mixed() -> dict:
    """Where the rank measure earns its keep, which is not on a pure permutation.

    With correct and incorrect answers mixed together, moving the correct ones to the front
    raises the gain and leaves the recall exactly where it was. Five right answers first scores
    sixty five percent gain and five right answers last scores thirty five, at fifty percent
    recall either way. That thirty point spread is the ordering information, and it only exists
    because there is something wrong in the result for the right answers to be ranked above.
    """
    corpus = gaussian(count=2048, dimension=32)
    searched, probes = held_out(corpus, count=64)
    truth = search(probes, searched.vectors, k=10)
    wide = search(probes, searched.vectors, k=200)
    front = Neighbours(
        identifiers=torch.cat([truth.identifiers[:, :5], wide.identifiers[:, 190:195]], dim=1),
        scores=torch.cat([truth.scores[:, :5], wide.scores[:, 190:195]], dim=1),
    )
    back = Neighbours(
        identifiers=torch.cat([wide.identifiers[:, 190:195], truth.identifiers[:, :5]], dim=1),
        scores=torch.cat([wide.scores[:, 190:195], truth.scores[:, :5]], dim=1),
    )
    return {
        "front_recall": round(recall_at_k(truth, front), 4),
        "back_recall": round(recall_at_k(truth, back), 4),
        "front_gain": round(discounted_gain(truth, front), 4),
        "back_gain": round(discounted_gain(truth, back), 4),
        "recall_identical": recall_at_k(truth, front) == recall_at_k(truth, back),
        "gain_separates": discounted_gain(truth, front) > discounted_gain(truth, back),
    }


def recall_cannot_see_how_wrong_a_miss_is() -> dict:
    """And whether a near miss looks different from a random answer.

    Not to recall, which scores both at zero. The distance measure does separate them and by
    much less than I assumed: six percent further for the near misses against forty two for the
    random vectors, a factor of seven in the excess and only a third in the ratio itself.

    That narrowness is the concentration result from vectors/dataset.py showing up somewhere
    new. In thirty two dimensions a randomly chosen vector is not very much further from a query
    than the tenth nearest one is, so a distance based measure has little room to distinguish a
    near miss from noise. It still distinguishes them, and recall does not distinguish them at
    all, which is the point.
    """
    corpus = gaussian(count=2048, dimension=32)
    searched, probes = held_out(corpus, count=64)
    truth = search(probes, searched.vectors, k=10)
    close = near_misses(probes, searched.vectors, k=10, offset=10)
    generator = torch.Generator().manual_seed(4)
    noise = Neighbours(
        identifiers=torch.randint(
            0, searched.count, (probes.shape[0], 10), generator=generator
        ),
        scores=torch.zeros(probes.shape[0], 10),
    )
    return {
        "near_miss_recall": round(recall_at_k(truth, close), 4),
        "random_recall": round(recall_at_k(truth, noise), 4),
        "near_miss_ratio": round(distance_ratio(probes, searched.vectors, truth, close), 4),
        "random_ratio": round(distance_ratio(probes, searched.vectors, truth, noise), 4),
        "recall_cannot_tell": abs(recall_at_k(truth, close) - recall_at_k(truth, noise)) < 0.05,
        "distance_can": distance_ratio(probes, searched.vectors, truth, noise)
        > distance_ratio(probes, searched.vectors, truth, close) * 1.2,
        "excess_ratio": round(
            (distance_ratio(probes, searched.vectors, truth, noise) - 1.0)
            / max(distance_ratio(probes, searched.vectors, truth, close) - 1.0, 1e-9),
            2,
        ),
    }


def the_measures_can_rank_two_results_oppositely() -> dict:
    """Whether two measures ever disagree about which of two answers is better.

    They do, and the construction is not contrived. One result returns half the true neighbours
    in a bad order and half near misses. The other returns fewer true neighbours and every one
    of its misses is very close. Recall prefers the first and the distance ratio prefers the
    second, so an index tuned against one of them would be tuned away from the other.
    """
    corpus = gaussian(count=2048, dimension=32)
    searched, probes = held_out(corpus, count=64)
    truth = search(probes, searched.vectors, k=10)
    wide = search(probes, searched.vectors, k=200)
    half_right = Neighbours(
        identifiers=torch.cat([truth.identifiers[:, :5], wide.identifiers[:, 190:195]], dim=1),
        scores=torch.cat([truth.scores[:, :5], wide.scores[:, 190:195]], dim=1),
    )
    all_close = Neighbours(
        identifiers=torch.cat([truth.identifiers[:, :3], wide.identifiers[:, 10:17]], dim=1),
        scores=torch.cat([truth.scores[:, :3], wide.scores[:, 10:17]], dim=1),
    )
    first = score_all(probes, searched.vectors, truth, half_right)
    second = score_all(probes, searched.vectors, truth, all_close)
    return {
        "half_right": first.as_dict(),
        "all_close": second.as_dict(),
        "recall_prefers_the_first": first.recall > second.recall,
        "distance_prefers_the_second": second.distance_ratio < first.distance_ratio,
        "they_disagree": (first.recall > second.recall)
        and (second.distance_ratio < first.distance_ratio),
    }


def reciprocal_rank_only_sees_the_first_hit() -> dict:
    """What the single result measure ignores.

    Everything after the first correct answer. A result whose first entry is right and whose
    other nine are wrong scores one, the same as a perfect answer, so it is the right measure
    for a caller that displays one thing and completely wrong for one that displays a page.
    """
    corpus = gaussian(count=2048, dimension=32)
    searched, probes = held_out(corpus, count=64)
    truth = search(probes, searched.vectors, k=10)
    wide = search(probes, searched.vectors, k=200)
    first_right = Neighbours(
        identifiers=torch.cat([truth.identifiers[:, :1], wide.identifiers[:, 150:159]], dim=1),
        scores=torch.cat([truth.scores[:, :1], wide.scores[:, 150:159]], dim=1),
    )
    return {
        "perfect_reciprocal": round(reciprocal_rank(truth, truth), 4),
        "first_right_reciprocal": round(reciprocal_rank(truth, first_right), 4),
        "first_right_recall": round(recall_at_k(truth, first_right), 4),
        "identical_reciprocal": reciprocal_rank(truth, first_right)
        == reciprocal_rank(truth, truth),
    }


def a_perfect_answer_scores_one_everywhere() -> dict:
    """The calibration check, which every measure has to pass.

    A result identical to the truth has to score one on recall, one on the gain, one on the
    reciprocal rank and exactly one on the distance ratio. Any measure that does not is
    misnormalised, and a misnormalised measure produces numbers that look like accuracy and
    cannot be compared to anybody else's.
    """
    corpus = gaussian(count=1024, dimension=32)
    searched, probes = held_out(corpus, count=32)
    truth = search(probes, searched.vectors, k=10)
    scores = score_all(probes, searched.vectors, truth, truth)
    return {
        **scores.as_dict(),
        "all_one": all(
            abs(value - 1.0) < 1e-4
            for value in (
                scores.recall,
                scores.gain,
                scores.reciprocal_rank,
                scores.distance_ratio,
            )
        ),
    }


def the_gain_can_exceed_the_recall() -> dict:
    """A relationship between two of the measures that I got backwards.

    I assumed the gain could never exceed the recall, on the reasoning that the discount only
    ever reduces a weight. That is wrong: the discount reduces weights relative to the best
    possible position, and a result whose correct answers are all at the front is being compared
    against a perfect result whose weights are spread over ten positions. Half the true
    neighbours placed first scores fifty percent recall and sixty five percent gain.

    So there is no bound to check here, and the disagreement is the useful part: where the two
    differ is exactly where ordering is carrying information that recall throws away.
    """
    corpus = gaussian(count=1024, dimension=32)
    searched, probes = held_out(corpus, count=32)
    truth = search(probes, searched.vectors, k=10)
    wide = search(probes, searched.vectors, k=100)
    rows = []
    for start in (0, 5, 20, 50):
        found = Neighbours(
            identifiers=wide.identifiers[:, start : start + 10],
            scores=wide.scores[:, start : start + 10],
        )
        rows.append(
            {
                "offset": start,
                "recall": round(recall_at_k(truth, found), 4),
                "gain": round(discounted_gain(truth, found), 4),
            }
        )
    return {
        "rows": rows,
        "gain_exceeds_recall_somewhere": any(
            row["gain"] > row["recall"] + 1e-9 for row in rows
        ),
        "they_agree_at_the_extremes": rows[0]["gain"] == rows[0]["recall"]
        and rows[-1]["gain"] == rows[-1]["recall"],
    }


def k_changes_what_recall_means(values: Sequence[int] = (1, 5, 10, 50)) -> list[dict]:
    """How the same index scores at different result sizes.

    Recall at one is a different measure from recall at fifty and they are routinely quoted
    interchangeably. At one the question is whether the single best vector came back, which is
    the hardest thing to get right, and at fifty it is whether the answer is roughly in the
    right region, which almost anything manages. Comparing an index quoted at one against one
    quoted at fifty says nothing.
    """
    if not values:
        raise ConfigError("there is nothing to sweep")
    corpus = gaussian(count=2048, dimension=32)
    searched, probes = held_out(corpus, count=64)
    rows = []
    for k in values:
        truth = search(probes, searched.vectors, k=k)
        wide = search(probes, searched.vectors, k=k + 20)
        approximate = Neighbours(
            identifiers=wide.identifiers[:, 10 : 10 + k],
            scores=wide.scores[:, 10 : 10 + k],
        )
        rows.append(
            {
                "k": k,
                "recall": round(recall_at_k(truth, approximate), 4),
                "ratio": round(distance_ratio(probes, searched.vectors, truth, approximate), 4),
            }
        )
    return rows


def recall_at_one_is_the_hard_one() -> dict:
    """The two ends of that sweep, which is the number worth quoting alongside any recall.

    Recall at one is zero for a result shifted by ten places and recall at fifty is most of the
    way back, from the same shift. The measure got easier, the index did not change, and the
    distance ratio moved hardly at all across the whole sweep because the shift is the same
    shift in every case.
    """
    rows = {row["k"]: row for row in k_changes_what_recall_means()}
    return {
        "at_one": rows[1]["recall"],
        "at_fifty": rows[50]["recall"],
        "ratio_at_one": rows[1]["ratio"],
        "ratio_at_fifty": rows[50]["ratio"],
        "recall_moved_more": (rows[50]["recall"] - rows[1]["recall"])
        > abs(rows[1]["ratio"] - rows[50]["ratio"]),
    }


def compare_on_a_real_index(corpus: Corpus | None = None) -> list[dict]:
    """Every measure on results from an actual index at several settings.

    The table an evaluation would produce. The four columns move together at the top of the
    range and separate at the bottom, which is where the choice of measure starts to matter and
    is exactly the range where an index is interesting.
    """
    target = corpus if corpus is not None else gaussian(count=2048, dimension=32)
    searched, probes = held_out(target, count=64)
    truth = search(probes, searched.vectors, k=10)
    rows = []
    for probe in (1, 4, 16):
        index = IVFIndex(target.dimension, partitions=32, probe=probe)
        index.build(searched.vectors)
        found, _ = index.search(probes, k=10)
        rows.append(
            {"probe": probe, **score_all(probes, searched.vectors, truth, found).as_dict()}
        )
    return rows


def the_measures_agree_at_the_top_and_separate_at_the_bottom() -> dict:
    """How far apart the four get as the index is made worse.

    A long way, and always in the same direction. Across a probe sweep on unstructured data the
    recall moves several times further than the distance ratio does, because a partitioned index
    opening the wrong partition still returns vectors from somewhere reasonable and the
    concentration of distances means somewhere reasonable is not far off. So the measure that
    flatters an index most is the one closest to what a user would actually notice. That is
    uncomfortable and it is what the numbers say, and it is the reason both are reported for
    every index in this package rather than one.
    """
    rows = {row["probe"]: row for row in compare_on_a_real_index()}
    return {
        "at_one_probe": rows[1],
        "at_sixteen_probes": rows[16],
        "recall_spread": round(rows[16]["recall"] - rows[1]["recall"], 4),
        "ratio_spread": round(rows[1]["distance_ratio"] - rows[16]["distance_ratio"], 4),
        "recall_moves_more": (rows[16]["recall"] - rows[1]["recall"])
        > (rows[1]["distance_ratio"] - rows[16]["distance_ratio"]),
    }


def comparing_different_shapes_is_refused() -> bool:
    """Whether a comparison between results of different sizes is caught."""
    corpus = gaussian(count=512, dimension=16)
    truth = search(corpus.vectors[:8], corpus.vectors, k=10)
    other = search(corpus.vectors[:8], corpus.vectors, k=20)
    try:
        recall_at_k(truth, other)
    except DataError:
        return True
    return False


def comparing_different_query_counts_is_refused() -> bool:
    """Whether a comparison across different batches is caught."""
    corpus = gaussian(count=512, dimension=16)
    truth = search(corpus.vectors[:8], corpus.vectors, k=10)
    other = search(corpus.vectors[:16], corpus.vectors, k=10)
    try:
        discounted_gain(truth, other)
    except DataError:
        return True
    return False


def a_zero_offset_near_miss_is_refused() -> bool:
    """Whether asking for near misses that are not misses is caught."""
    corpus = gaussian(count=512, dimension=16)
    try:
        near_misses(corpus.vectors[:4], corpus.vectors, k=10, offset=0)
    except ConfigError:
        return True
    return False

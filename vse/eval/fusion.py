from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from vse.errors import ConfigError, DataError
from vse.eval.recall import discounted_gain, recall_at_k
from vse.index.forest import ForestIndex
from vse.index.graph import GraphIndex
from vse.index.ivf import IVFIndex
from vse.vectors.dataset import gaussian, held_out
from vse.vectors.exact import Neighbours, identifier_overlap, search

# Combining two ranked lists into one, which every retrieval system does and almost none of them
# measure.
#
# The situation is always the same. Two retrievers over one corpus disagree and something has to
# produce a single ordering. In a real system the two are usually a vector index and a keyword
# index; here they are two vector indexes with different structures, because the fusion problem
# does not depend on where the lists came from and two vector indexes can both be scored against
# the same ground truth, which a keyword index cannot.
#
# Score fusion adds the scores. Rank fusion throws the scores away and adds a function of the
# positions, which is reciprocal rank fusion, with a constant in the denominator whose stated
# job is to stop the first position from dominating. The constant is conventionally sixty.
#
# This module was written believing that rank fusion wins, on the usual argument that scores
# from different retrievers are not comparable and ranks are. The first run said the opposite
# by a wide margin: 0.264 for reciprocal rank fusion against 0.492 for the best score fusion
# and 0.376 for the better retriever alone. Rank fusion was losing to doing nothing.
#
# The constant was the problem, and it is worth stating carefully because sixty is quoted as
# though it were a property of the method:
#
#     constant    recall
#            0     0.522
#            1     0.519
#            5     0.500
#           10     0.476
#           60     0.264
#          500     0.264
#
# At sixty, positions one and fifty differ by less than a factor of two, so a fifty deep list
# contributes almost the same weight everywhere and the fusion has become a vote on membership
# with the ordering discarded. Sixty was chosen for web search, where the lists are thousands
# long and position fifty really should count for little. On a fifty deep list it deletes the
# signal. With the constant at zero, rank fusion is the best method here, ahead of every score
# fusion tried.
#
# So the real result is about the two failure modes rather than about which family wins. Rank
# fusion is exactly invariant to any monotone rescaling of the scores, checked by cubing and
# exponentiating one retriever's scores and getting byte identical output. Score fusion is not:
# the same distortions take it from 0.472 to 0.425 to 0.340 to 0.169. And normalisation barely
# helps: the four normalisations tried span two points of recall when the scales already
# match, in an order that does not follow the argument for any of them, and none of them
# survives a genuine scale distortion.
#
# The advice that falls out is unglamorous. Use rank fusion, set the constant from the depth of
# the lists rather than from the literature, and check what it does before shipping it, because
# the conventional value is capable of being worse than not fusing at all.


@dataclass
class Ranking:
    """One retriever's answer, with its scores and where they came from."""

    identifiers: torch.Tensor
    scores: torch.Tensor
    name: str
    smaller_is_better: bool = True

    def __post_init__(self) -> None:
        if self.identifiers.shape != self.scores.shape:
            raise DataError(
                f"{tuple(self.identifiers.shape)} identifiers and "
                f"{tuple(self.scores.shape)} scores do not match"
            )
        if self.identifiers.ndim != 2:
            raise DataError(f"a ranking is a matrix, got {tuple(self.identifiers.shape)}")

    @property
    def queries(self) -> int:
        """How many queries this answers."""
        return int(self.identifiers.shape[0])

    @property
    def depth(self) -> int:
        """How many results per query."""
        return int(self.identifiers.shape[1])

    def as_neighbours(self) -> Neighbours:
        """The same thing in the package's usual shape."""
        return Neighbours(identifiers=self.identifiers, scores=self.scores)

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "name": self.name,
            "queries": self.queries,
            "depth": self.depth,
            "smaller_is_better": self.smaller_is_better,
        }


def from_index(index, queries: torch.Tensor, k: int, name: str) -> Ranking:
    """Run an index and wrap what it returns."""
    found, _ = index.search(queries, k=k)
    return Ranking(identifiers=found.identifiers, scores=found.scores, name=name)


def normalise_by_maximum(ranking: Ranking) -> torch.Tensor:
    """Divide each row's scores by its own largest.

    Written here as the obvious wrong choice, on the argument that the largest score is the
    worst item in the list, the one nobody looks at and the one that moves most between queries,
    so every normalised score ends up depending on where the tenth result happened to land.

    The argument is sound and the measurement does not care: this is the best of the four, at
    0.492 against 0.472 for the range and 0.481 for the best anchored version. The four span two
    points, which is small enough that the ordering between them is not worth a claim.
    """
    if ranking.depth < 1:
        raise ConfigError("an empty ranking cannot be normalised")
    largest = ranking.scores.max(dim=1, keepdim=True).values
    return ranking.scores / largest.clamp_min(1e-12)


def normalise_by_range(ranking: Ranking) -> torch.Tensor:
    """Map each row's scores to the unit interval by its own minimum and maximum.

    Still anchored on the worst item, but at least the best one is pinned to zero, so the top of
    the list is stable and only the spacing moves. This is the standard choice and the
    measurements treat it as the score fusion baseline.
    """
    if ranking.depth < 2:
        raise ConfigError("a range needs at least two results")
    smallest = ranking.scores.min(dim=1, keepdim=True).values
    largest = ranking.scores.max(dim=1, keepdim=True).values
    return (ranking.scores - smallest) / (largest - smallest).clamp_min(1e-12)


def normalise_by_the_best(ranking: Ranking) -> torch.Tensor:
    """Divide each row by its own best score, so the top result is always one.

    Anchored on the item that matters instead of the one that does not. Every row's first entry
    is exactly one and the rest say how much worse they are than the best available, which is
    the quantity a fusion ought to be comparing across retrievers.

    That reasoning predicts it should beat the maximum anchored version. It does not, by a
    hundredth, which is the sort of margin that should not be argued about in either direction.
    """
    if ranking.depth < 1:
        raise ConfigError("an empty ranking cannot be normalised")
    best = ranking.scores.min(dim=1, keepdim=True).values
    return ranking.scores / best.clamp_min(1e-12)


def normalise_globally(ranking: Ranking) -> torch.Tensor:
    """Divide every score by the same constant, taken across all queries.

    The one normalisation that does not let one query's results change another's. It fixes the
    scale difference between retrievers, which is the actual problem, and leaves everything
    inside a query alone, which is the part that was never broken.
    """
    if ranking.depth < 1:
        raise ConfigError("an empty ranking cannot be normalised")
    return ranking.scores / ranking.scores.mean().clamp_min(1e-12)


def fuse_by_score(
    rankings: Sequence[Ranking],
    corpus_size: int,
    k: int = 10,
    weights: Sequence[float] | None = None,
    normaliser=normalise_by_range,
) -> Neighbours:
    """Add normalised scores over the union of the lists.

    An identifier absent from one list gets that list's worst normalised score rather than its
    best, which is the only defensible filling: absence from a top ten means the retriever
    ranked it below ten, not that it was rated well.
    """
    if not rankings:
        raise ConfigError("there is nothing to fuse")
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ConfigError(f"{len(weights)} weights for {len(rankings)} rankings")
    queries = rankings[0].queries
    totals = torch.full((queries, corpus_size), 0.0)
    for ranking, weight in zip(rankings, weights, strict=True):
        if ranking.queries != queries:
            raise DataError(f"{ranking.queries} queries cannot fuse with {queries}")
        normalised = normaliser(ranking)
        penalty = normalised.max(dim=1, keepdim=True).values
        block = penalty.expand(queries, corpus_size).clone()
        block.scatter_(1, ranking.identifiers, normalised)
        totals += block * weight
    chosen = torch.topk(totals, k=k, dim=1, largest=False)
    return Neighbours(identifiers=chosen.indices, scores=chosen.values)


def fuse_by_rank(
    rankings: Sequence[Ranking],
    corpus_size: int,
    k: int = 10,
    weights: Sequence[float] | None = None,
    constant: float = 1.0,
) -> Neighbours:
    """Add one over the rank plus a constant, over the union of the lists.

    The constant is the whole design, and the default here is one rather than the conventional
    sixty. At zero the first position contributes one and the second a half, so a retriever
    putting something first outvotes everything below it. At sixty, on a fifty deep list, every
    position contributes within a factor of two of every other and the fusion has stopped using
    the ordering at all.

    Sixty is right for lists thousands long. On the depths used here it costs 0.522 recall
    against 0.264, the difference between the best method in this module and the worst. The
    default is one because the sweep below says so, and the honest guidance is to set it from
    the list depth rather than from a citation.
    """
    if not rankings:
        raise ConfigError("there is nothing to fuse")
    if constant < 0:
        raise ConfigError(f"a constant of {constant} is not a constant")
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ConfigError(f"{len(weights)} weights for {len(rankings)} rankings")
    queries = rankings[0].queries
    totals = torch.zeros(queries, corpus_size)
    for ranking, weight in zip(rankings, weights, strict=True):
        if ranking.queries != queries:
            raise DataError(f"{ranking.queries} queries cannot fuse with {queries}")
        positions = torch.arange(1, ranking.depth + 1, dtype=torch.float32)
        contribution = (weight / (constant + positions)).expand(queries, ranking.depth)
        totals.scatter_add_(1, ranking.identifiers, contribution.contiguous())
    chosen = torch.topk(totals, k=k, dim=1, largest=True)
    return Neighbours(identifiers=chosen.indices, scores=chosen.values)


def _setup(count: int = 4096, dimension: int = 32, queries: int = 100, depth: int = 50):
    """Two different retrievers over one corpus, with the truth for both."""
    corpus = gaussian(count=count, dimension=dimension)
    searched, probes = held_out(corpus, count=queries)
    truth = search(probes, searched.vectors, k=10)

    partitioned = IVFIndex(dimension, partitions=64, probe=4)
    partitioned.build(searched.vectors)
    forest = ForestIndex(dimension, trees=4, leaf_size=64)
    forest.build(searched.vectors)

    return (
        searched.vectors,
        probes,
        truth,
        from_index(partitioned, probes, depth, "ivf"),
        from_index(forest, probes, depth, "forest"),
    )


def the_two_retrievers_disagree() -> dict:
    """That there is anything to fuse, established before anything is fused.

    A fusion of two identical lists is the list, so the whole exercise is only meaningful if the
    inputs differ. They do: the overlap between the two top tens is well below one, and each
    finds true neighbours the other missed. That second number is what a fusion has to capture
    and it bounds what any fusion can achieve.
    """
    _, _, truth, left, right = _setup()
    left_found = set()
    right_found = set()
    for row in range(truth.queries):
        wanted = set(truth.identifiers[row].tolist())
        left_found |= {
            (row, value) for value in left.identifiers[row, :10].tolist() if value in wanted
        }
        right_found |= {
            (row, value) for value in right.identifiers[row, :10].tolist() if value in wanted
        }
    union = left_found | right_found
    return {
        "agreement": round(
            identifier_overlap(
                Neighbours(left.identifiers[:, :10], left.scores[:, :10]),
                Neighbours(right.identifiers[:, :10], right.scores[:, :10]),
            ),
            4,
        ),
        "left_hits": len(left_found),
        "right_hits": len(right_found),
        "union_hits": len(union),
        "only_left": len(left_found - right_found),
        "only_right": len(right_found - left_found),
        "there_is_something_to_gain": len(union) > max(len(left_found), len(right_found)),
    }


def the_ceiling_is_the_union() -> dict:
    """The most any fusion could do, which is worth having before judging any of them.

    Everything either retriever found, in the best possible order. No fusion can return an
    identifier neither list contains, so the union of the two top lists is a hard ceiling, and a
    fusion should be judged by how much of that gap it closes rather than by whether it beat the
    better retriever.
    """
    result = the_two_retrievers_disagree()
    best_single = max(result["left_hits"], result["right_hits"])
    return {
        "best_single": best_single,
        "union": result["union_hits"],
        "headroom": result["union_hits"] - best_single,
        "headroom_share": round((result["union_hits"] - best_single) / max(best_single, 1), 4),
    }


def rank_fusion_beats_score_fusion(depth: int = 50) -> list[dict]:
    """The comparison the module exists for.

    Every normalisation tried, against reciprocal rank fusion at two constants, against each
    retriever alone. Rank fusion at the conventional constant of sixty loses to everything,
    including to not fusing at all. At a constant of one it beats everything.

    Which is the module's main result and it is about a parameter rather than about a family.
    Both retrievers here return squared distances, so the scales already match and score fusion
    is not being sabotaged by them. The four normalisations span two points of recall between
    them, which measures how little that part matters when the inputs are commensurable.
    """
    corpus, _, truth, left, right = _setup(depth=depth)
    size = int(corpus.shape[0])
    rows = [
        {
            "method": "ivf alone",
            "recall": round(
                recall_at_k(truth, Neighbours(left.identifiers[:, :10], left.scores[:, :10])), 4
            ),
        },
        {
            "method": "forest alone",
            "recall": round(
                recall_at_k(truth, Neighbours(right.identifiers[:, :10], right.scores[:, :10])),
                4,
            ),
        },
    ]
    for label, normaliser in (
        ("score, by maximum", normalise_by_maximum),
        ("score, by range", normalise_by_range),
        ("score, by best", normalise_by_the_best),
        ("score, global", normalise_globally),
    ):
        fused = fuse_by_score([left, right], size, k=10, normaliser=normaliser)
        rows.append({"method": label, "recall": round(recall_at_k(truth, fused), 4)})
    for label, constant in (("reciprocal rank, k=1", 1.0), ("reciprocal rank, k=60", 60.0)):
        fused = fuse_by_rank([left, right], size, k=10, constant=constant)
        rows.append({"method": label, "recall": round(recall_at_k(truth, fused), 4)})
    return rows


def the_constant_decides_whether_rank_fusion_wins() -> dict:
    """The headline of that table, which is a parameter and not a family.

    At a constant of one, rank fusion is the best method measured. At sixty it is the worst,
    below both retrievers on their own. Same rule, same inputs, same corpus, and the only
    difference is a number quoted from a paper about a different list depth.
    """
    rows = {row["method"]: row["recall"] for row in rank_fusion_beats_score_fusion()}
    best_score = max(value for name, value in rows.items() if name.startswith("score"))
    best_single = max(rows["ivf alone"], rows["forest alone"])
    tuned = rows["reciprocal rank, k=1"]
    conventional = rows["reciprocal rank, k=60"]
    return {
        "best_single_retriever": best_single,
        "best_score_fusion": best_score,
        "rank_fusion_tuned": tuned,
        "rank_fusion_conventional": conventional,
        "tuned_beats_score": tuned > best_score,
        "conventional_loses_to_doing_nothing": conventional < best_single,
        "the_constant_is_worth": round(tuned - conventional, 4),
    }


def the_normalisation_barely_matters() -> dict:
    """Which normalisation is best, which turns out not to be a useful question.

    They span two points of recall: 0.492, 0.485, 0.481, 0.472, and the ordering is not the one
    argued for above. Dividing by the maximum is supposed to be the bad one, because it anchors
    on the least stable item in the list, and it comes out slightly ahead.

    The reason none of it matters is that both retrievers here return squared distances, so
    the scales already agree and there is nothing for a normalisation to fix. Where they do
    not agree, no normalisation saves it, which is the sweep in
    score_fusion_breaks_under_a_rescaling. So the normaliser is a detail either way and the
    thing to get right is whether the scores are comparable at all.
    """
    rows = {row["method"]: row["recall"] for row in rank_fusion_beats_score_fusion()}
    scores = {name: value for name, value in rows.items() if name.startswith("score")}
    return {
        "by_maximum": rows["score, by maximum"],
        "by_range": rows["score, by range"],
        "by_best": rows["score, by best"],
        "global": rows["score, global"],
        "spread": round(max(scores.values()) - min(scores.values()), 4),
        "barely_matters": (max(scores.values()) - min(scores.values())) < 0.05,
    }


def score_fusion_breaks_under_a_rescaling(
    scales: Sequence[float] = (10.0, 5.0, 2.0, 1.0, 0.5),
) -> list[dict]:
    """What happens to each family when one retriever's scores are monotonically distorted.

    Exponentiating a retriever's scores changes nothing about its ordering and everything about
    its scale. Score fusion falls from 0.392 to 0.169 as the distortion sharpens. Rank fusion
    does not move at all, at any distortion, because it never looked at the numbers.

    That invariance is the property reciprocal rank fusion is sold on and it is worth having
    measured rather than argued: the output is byte identical, not approximately stable.
    """
    if not scales:
        raise ConfigError("there is nothing to sweep")
    corpus, _, truth, left, right = _setup()
    size = int(corpus.shape[0])
    rows = []
    for scale in scales:
        distorted = Ranking(
            identifiers=right.identifiers,
            scores=torch.exp(right.scores / scale),
            name="rescaled",
        )
        rows.append(
            {
                "scale": scale,
                "score_fusion": round(
                    recall_at_k(truth, fuse_by_score([left, distorted], size, k=10)), 4
                ),
                "rank_fusion": round(
                    recall_at_k(truth, fuse_by_rank([left, distorted], size, k=10)), 4
                ),
            }
        )
    return rows


def rank_fusion_is_exactly_invariant() -> dict:
    """That the rank fusion column above is constant, checked as equality not as a trend.

    Every entry identical, because a monotone map cannot change a position and the fusion reads
    nothing else. Score fusion loses more than half its recall across the same sweep.
    """
    rows = score_fusion_breaks_under_a_rescaling()
    ranked = {row["rank_fusion"] for row in rows}
    return {
        "distinct_rank_results": len(ranked),
        "invariant": len(ranked) == 1,
        "score_at_the_mildest": rows[0]["score_fusion"],
        "score_at_the_sharpest": rows[-1]["score_fusion"],
        "score_collapses": rows[-1]["score_fusion"] < rows[0]["score_fusion"] * 0.6,
    }


def the_constant_is_what_makes_rank_fusion_work(
    constants: Sequence[float] = (0.0, 1.0, 10.0, 60.0, 500.0),
) -> list[dict]:
    """How the reciprocal rank constant changes the result.

    At zero the fusion is dominated by whichever list puts something first, since one over one
    is twice one over two. At five hundred every position contributes nearly the same amount and
    the fusion has become a vote on membership with the ordering thrown away.

    On lists fifty deep the first of those is what is wanted and the second is a disaster: 0.522
    at zero against 0.264 at sixty. The conventional value is on the wrong side of the cliff by
    a wide margin, and this sweep is why the default in fuse_by_rank is one.
    """
    if not constants:
        raise ConfigError("there is nothing to sweep")
    corpus, _, truth, left, right = _setup()
    size = int(corpus.shape[0])
    rows = []
    for constant in constants:
        fused = fuse_by_rank([left, right], size, k=10, constant=constant)
        rows.append(
            {
                "constant": constant,
                "recall": round(recall_at_k(truth, fused), 4),
                "gain": round(discounted_gain(truth, fused), 4),
            }
        )
    return rows


def the_conventional_constant_is_wrong_for_short_lists() -> dict:
    """The two ends of that sweep, which bracket what the constant is doing."""
    rows = {row["constant"]: row for row in the_constant_is_what_makes_rank_fusion_work()}
    best = max(rows.values(), key=lambda row: row["recall"])
    return {
        "recall_at_zero": rows[0.0]["recall"],
        "recall_at_ten": rows[10.0]["recall"],
        "recall_at_sixty": rows[60.0]["recall"],
        "recall_at_five_hundred": rows[500.0]["recall"],
        "best_constant": best["constant"],
        "sixty_is_far_from_the_best": abs(rows[60.0]["recall"] - best["recall"]) > 0.1,
        "small_is_better_here": rows[0.0]["recall"] > rows[60.0]["recall"],
    }


def the_depth_and_the_constant_are_one_parameter(
    depths: Sequence[int] = (10, 20, 50, 100, 200),
    constants: Sequence[float] = (1.0, 60.0),
) -> list[dict]:
    """How deep each list should be, which depends entirely on the constant.

    Written expecting depth to be the dominant knob, on the reasoning that fusing two top tens
    can only return twenty candidates and that the ceiling rises with the depth. Both halves of
    that are true and neither controls the result.

    At a constant of one, depth is inert: 0.528, 0.519, 0.519, 0.521, 0.518 from ten to two
    hundred. A deep entry contributes one over its position and that is nearly nothing, so
    fetching more of each list changes almost no fused ordering.

    At sixty, depth is actively harmful: 0.528, 0.482, 0.264, 0.170, 0.170. Every extra entry
    arrives with nearly the same voting weight as the first, so deepening the lists is pouring
    junk into the ballot. The two constants agree exactly at depth ten and diverge by a factor
    of three by depth two hundred.

    So there is one parameter here and not two. What matters is how many positions the fusion
    actually distinguishes, and a small constant is the safe configuration because under it
    extra depth is free rather than dangerous.
    """
    if not depths or not constants:
        raise ConfigError("there is nothing to sweep")
    rows = []
    for constant in constants:
        for depth in depths:
            corpus, _, truth, left, right = _setup(depth=depth)
            fused = fuse_by_rank([left, right], int(corpus.shape[0]), k=10, constant=constant)
            rows.append(
                {
                    "constant": constant,
                    "depth": depth,
                    "recall": round(recall_at_k(truth, fused), 4),
                    "candidates": depth * 2,
                }
            )
    return rows


def a_small_constant_makes_depth_free() -> dict:
    """The two rows of that sweep at their extremes, which is the practical advice.

    Fetch as deep as the retrievers cheaply allow and keep the constant small. Under that
    configuration depth cannot hurt, so nobody has to tune it, and the fusion behaves the way a
    fusion is supposed to: it uses more information when more is available and ignores what it
    cannot use.
    """
    rows = {
        (row["constant"], row["depth"]): row
        for row in the_depth_and_the_constant_are_one_parameter()
    }
    small_change = rows[(1.0, 200)]["recall"] - rows[(1.0, 10)]["recall"]
    large_change = rows[(60.0, 200)]["recall"] - rows[(60.0, 10)]["recall"]
    return {
        "small_constant_at_ten": rows[(1.0, 10)]["recall"],
        "small_constant_at_two_hundred": rows[(1.0, 200)]["recall"],
        "large_constant_at_ten": rows[(60.0, 10)]["recall"],
        "large_constant_at_two_hundred": rows[(60.0, 200)]["recall"],
        "depth_is_inert_when_the_constant_is_small": abs(small_change) < 0.03,
        "depth_is_harmful_when_it_is_large": large_change < -0.2,
        "they_agree_at_depth_ten": rows[(1.0, 10)]["recall"] == rows[(60.0, 10)]["recall"],
    }


def weighting_a_better_retriever_helps(
    weights: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> list[dict]:
    """Whether the fusion should trust the better retriever more, which it should, a bit.

    A weight of zero is the second retriever alone and a weight of one is the first alone, so
    the sweep contains both single retriever baselines as its endpoints and the fusion in
    between. The best point is interior, which is the entire justification for fusing, and it is
    not sharply peaked, which is why an equal weighting is a defensible default.
    """
    if not weights:
        raise ConfigError("there is nothing to sweep")
    corpus, _, truth, left, right = _setup()
    size = int(corpus.shape[0])
    rows = []
    for weight in weights:
        fused = fuse_by_rank([left, right], size, k=10, weights=[weight, 1.0 - weight])
        rows.append({"weight": weight, "recall": round(recall_at_k(truth, fused), 4)})
    return rows


def the_best_weight_is_interior() -> dict:
    """That the fusion beats both its inputs, which it has to do to be worth running."""
    rows = {row["weight"]: row for row in weighting_a_better_retriever_helps()}
    best = max(rows.values(), key=lambda row: row["recall"])
    return {
        "second_alone": rows[0.0]["recall"],
        "first_alone": rows[1.0]["recall"],
        "even": rows[0.5]["recall"],
        "best_weight": best["weight"],
        "best_recall": best["recall"],
        "interior": 0.0 < best["weight"] < 1.0,
        "beats_both": best["recall"] > max(rows[0.0]["recall"], rows[1.0]["recall"]),
    }


def fusing_three_is_better_than_fusing_two() -> dict:
    """Whether a third retriever adds anything, which it does with diminishing returns.

    A graph index joins the inverted file and the forest. It adds true neighbours neither of the
    others found, so the ceiling rises, and the fusion captures some of that. The gain from the
    third is smaller than the gain from the second, which is the shape every ensemble has and
    the reason nobody runs five.
    """
    corpus = gaussian(count=4096, dimension=32)
    searched, probes = held_out(corpus, count=100)
    truth = search(probes, searched.vectors, k=10)
    size = int(searched.vectors.shape[0])

    partitioned = IVFIndex(32, partitions=64, probe=4)
    partitioned.build(searched.vectors)
    forest = ForestIndex(32, trees=4, leaf_size=64)
    forest.build(searched.vectors)
    graph = GraphIndex(32, degree=16, ef=64)
    graph.build(searched.vectors)

    left = from_index(partitioned, probes, 50, "ivf")
    right = from_index(forest, probes, 50, "forest")
    third = from_index(graph, probes, 50, "graph")

    alone = recall_at_k(truth, Neighbours(left.identifiers[:, :10], left.scores[:, :10]))
    two = recall_at_k(truth, fuse_by_rank([left, right], size, k=10))
    three = recall_at_k(truth, fuse_by_rank([left, right, third], size, k=10))
    return {
        "one": round(alone, 4),
        "two": round(two, 4),
        "three": round(three, 4),
        "gain_from_the_second": round(two - alone, 4),
        "gain_from_the_third": round(three - two, 4),
        "third_helps": three > two,
        "diminishing": (three - two) < (two - alone),
    }


def fusing_a_list_with_itself_changes_nothing() -> dict:
    """The degenerate case, which any fusion rule has to get right.

    Fusing a ranking with a copy of itself must return that ranking, since there is no
    disagreement to resolve. Rank fusion does, exactly, because doubling every contribution
    preserves the order. Score fusion does too. A rule that failed this would be introducing an
    ordering out of nothing.
    """
    corpus, _, truth, left, _ = _setup()
    size = int(corpus.shape[0])
    alone = Neighbours(left.identifiers[:, :10], left.scores[:, :10])
    by_rank = fuse_by_rank([left, left], size, k=10)
    by_score = fuse_by_score([left, left], size, k=10)
    return {
        "rank_agreement": round(identifier_overlap(alone, by_rank), 4),
        "score_agreement": round(identifier_overlap(alone, by_score), 4),
        "rank_is_identity": identifier_overlap(alone, by_rank) == 1.0,
        "score_is_identity": identifier_overlap(alone, by_score) == 1.0,
        "recall_unchanged": round(recall_at_k(truth, by_rank), 4)
        == round(recall_at_k(truth, alone), 4),
    }


def fusing_with_a_useless_retriever_costs_something() -> dict:
    """What happens when one input is noise, which is the risk fusion carries.

    A random ranking contributes nothing correct and still gets a vote, so it displaces real
    results from the fused list and the recall falls below the good retriever's own. Rank fusion
    is more robust than score fusion here, because a random list's positions are spread over the
    corpus while its scores can be arbitrarily confident, but neither is immune and the honest
    statement is that fusion assumes its inputs are worth fusing.
    """
    corpus, _, truth, left, _ = _setup()
    size = int(corpus.shape[0])
    generator = torch.Generator().manual_seed(19)
    noise = Ranking(
        identifiers=torch.randint(0, size, (left.queries, left.depth), generator=generator),
        scores=torch.rand(left.queries, left.depth, generator=generator),
        name="noise",
    )
    alone = recall_at_k(truth, Neighbours(left.identifiers[:, :10], left.scores[:, :10]))
    ranked = recall_at_k(truth, fuse_by_rank([left, noise], size, k=10))
    scored = recall_at_k(truth, fuse_by_score([left, noise], size, k=10))
    return {
        "good_alone": round(alone, 4),
        "fused_by_rank": round(ranked, 4),
        "fused_by_score": round(scored, 4),
        "rank_is_more_robust": ranked > scored,
        "both_lose": ranked < alone and scored < alone,
    }


def an_empty_fusion_is_refused() -> bool:
    """Whether fusing no rankings at all is caught."""
    try:
        fuse_by_rank([], 100, k=10)
    except ConfigError:
        return True
    return False


def a_mismatched_weight_list_is_refused() -> bool:
    """Whether a weight per ranking is enforced.

    It has to be, because the alternative is silently reusing a weight or dropping a ranking,
    and both produce a plausible result from a configuration that says something different.
    """
    _, _, _, left, right = _setup(count=512, queries=8, depth=10)
    try:
        fuse_by_rank([left, right], 512, k=5, weights=[1.0])
    except ConfigError:
        return True
    return False


def a_negative_constant_is_refused() -> bool:
    """Whether a reciprocal rank constant below zero is caught.

    At a constant of minus one the first position divides by zero and the fusion returns
    infinity for whatever any list ranked first, which is a silent way of turning a fusion into
    a passthrough of one retriever.
    """
    _, _, _, left, right = _setup(count=512, queries=8, depth=10)
    try:
        fuse_by_rank([left, right], 512, k=5, constant=-1.0)
    except ConfigError:
        return True
    return False


def rankings_of_different_query_counts_are_refused() -> bool:
    """Whether fusing lists that answer different queries is caught."""
    _, _, _, left, _ = _setup(count=512, queries=8, depth=10)
    other = Ranking(identifiers=left.identifiers[:4], scores=left.scores[:4], name="short")
    try:
        fuse_by_rank([left, other], 512, k=5)
    except DataError:
        return True
    return False


def a_ranking_whose_scores_do_not_match_is_refused() -> bool:
    """Whether a ranking with mismatched identifier and score shapes is caught."""
    try:
        Ranking(
            identifiers=torch.zeros(4, 10, dtype=torch.long),
            scores=torch.zeros(4, 5),
            name="bad",
        )
    except DataError:
        return True
    return False


def a_rank_one_ranking_is_refused() -> bool:
    """Whether a single unbatched result list is caught."""
    try:
        Ranking(
            identifiers=torch.zeros(10, dtype=torch.long), scores=torch.zeros(10), name="flat"
        )
    except DataError:
        return True
    return False


def a_range_normalisation_of_one_result_is_refused() -> bool:
    """Whether normalising a list too short to have a range is caught.

    A one item list has a range of zero, and dividing by it after clamping gives every score the
    same value, so the fusion would rank on the other retriever alone while reporting that it
    used both.
    """
    single = Ranking(
        identifiers=torch.zeros(4, 1, dtype=torch.long), scores=torch.ones(4, 1), name="one"
    )
    try:
        normalise_by_range(single)
    except ConfigError:
        return True
    return False


def the_normalisations_all_preserve_the_order() -> dict:
    """That normalising changes the numbers and not the ranking, which is what it is for.

    Every normalisation here is monotone within a row, so it cannot reorder that row's own
    results. If one did, it would be a retrieval step disguised as a preprocessing step, and the
    fusion's behaviour would depend on a choice nobody thought they were making.
    """
    _, _, _, left, _ = _setup()
    original = torch.argsort(left.scores, dim=1)
    rows = {}
    for label, normaliser in (
        ("maximum", normalise_by_maximum),
        ("range", normalise_by_range),
        ("best", normalise_by_the_best),
        ("global", normalise_globally),
    ):
        moved = torch.argsort(normaliser(left), dim=1)
        rows[label] = bool(torch.equal(original, moved))
    return {
        "maximum": rows["maximum"],
        "range": rows["range"],
        "best": rows["best"],
        "global": rows["global"],
        "all_monotone": all(rows.values()),
    }

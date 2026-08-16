from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from vse.errors import ConfigError, DataError

# The three metrics, and the two places the relationships between them stop holding.
#
# Everything downstream is written against one distance function and it matters which. Squared
# euclidean, negative inner product and cosine are usually described as interchangeable after a
# normalisation, and they are, but the normalisation is doing more work than the description
# suggests and the failure is not where people expect it.
#
# On normalised vectors the euclidean and inner product orderings are identical, exactly, and
# this file proves it rather than asserting it. On unnormalised vectors they are not merely
# different, they disagree about the top neighbour for a large share of queries, because inner
# product rewards magnitude and euclidean punishes it. That is the first place a caller loses.
#
# The second is about pruning. Inner product is not a distance at all: the score of a vector
# against itself is its squared length, so on a corpus with varying lengths a short vector is
# not its own best match and a longer one pointing the same way beats it. A structure that
# assumes a point is its own nearest neighbour is wrong at the first step, which rules out every
# branch and bound argument a tree or pivot based index relies on.
#
# And a third that this file was not written to find. The squared euclidean distance is not a
# metric either. Nothing here takes the root, because the root is monotone and changes no
# ranking, and that reasoning is correct for ranking and wrong for pruning: on random triples in
# sixteen dimensions the squared distance breaks the triangle inequality about one time in
# seventy. An index that computes squared distances for speed and then prunes with a bound
# derived on the true distance discards vectors that were in range, and the result it returns
# looks entirely reasonable. The bound has to be taken on the root even when the ranking is not.

METRICS = ("l2", "ip", "cosine")


@dataclass(frozen=True)
class Metric:
    """One distance, and what an index is allowed to assume about it."""

    name: str
    smaller_is_closer: bool
    is_a_metric: bool

    def __post_init__(self) -> None:
        if self.name not in METRICS:
            raise ConfigError(f"unknown metric {self.name!r}, expected one of {METRICS}")

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "metric": self.name,
            "smaller_is_closer": self.smaller_is_closer,
            "triangle_inequality": self.is_a_metric,
        }


L2 = Metric(name="l2", smaller_is_closer=True, is_a_metric=True)
INNER_PRODUCT = Metric(name="ip", smaller_is_closer=False, is_a_metric=False)
COSINE = Metric(name="cosine", smaller_is_closer=False, is_a_metric=False)
BY_NAME = {metric.name: metric for metric in (L2, INNER_PRODUCT, COSINE)}


def metric_named(name: str) -> Metric:
    """One of the three, by name."""
    if name not in BY_NAME:
        raise ConfigError(f"unknown metric {name!r}, expected one of {sorted(BY_NAME)}")
    return BY_NAME[name]


def _checked(vectors: torch.Tensor, name: str = "vectors") -> torch.Tensor:
    """Reject anything that is not a two dimensional float tensor with rows in it."""
    if vectors.ndim != 2:
        raise DataError(f"{name} has to be a matrix of rows, got rank {vectors.ndim}")
    if vectors.shape[0] == 0:
        raise DataError(f"{name} is empty")
    if vectors.shape[1] == 0:
        raise DataError(f"{name} has zero dimensions per row")
    if not vectors.dtype.is_floating_point:
        raise DataError(f"{name} has to be floating point, got {vectors.dtype}")
    return vectors


def squared_l2(queries: torch.Tensor, corpus: torch.Tensor) -> torch.Tensor:
    """Squared euclidean distance from every query to every vector.

    Computed as the expansion rather than by materialising the differences, because the
    difference tensor is the product of the three sizes and the expansion is a matrix product.
    The square root is never taken: it is monotone, so it changes no ordering, and taking it
    costs a transcendental per pair for nothing.
    """
    _checked(queries, "queries")
    _checked(corpus, "corpus")
    if queries.shape[1] != corpus.shape[1]:
        raise DataError(
            f"queries have {queries.shape[1]} dimensions and the corpus has {corpus.shape[1]}"
        )
    query_norms = queries.pow(2).sum(dim=1, keepdim=True)
    corpus_norms = corpus.pow(2).sum(dim=1)
    cross = queries @ corpus.transpose(0, 1)
    return (query_norms + corpus_norms - 2.0 * cross).clamp_min(0.0)


def inner_product(queries: torch.Tensor, corpus: torch.Tensor) -> torch.Tensor:
    """Inner product from every query to every vector. Larger is closer."""
    _checked(queries, "queries")
    _checked(corpus, "corpus")
    if queries.shape[1] != corpus.shape[1]:
        raise DataError(
            f"queries have {queries.shape[1]} dimensions and the corpus has {corpus.shape[1]}"
        )
    return queries @ corpus.transpose(0, 1)


def normalise(vectors: torch.Tensor, epsilon: float = 1e-12) -> torch.Tensor:
    """Scale every row to unit length.

    The epsilon is not decoration. A zero row has no direction, and dividing it by its own norm
    produces a row of nans that then poisons every distance computed against it, so it is
    clamped and left at zero instead, which at least keeps the rest of the batch usable.
    """
    _checked(vectors)
    if epsilon <= 0:
        raise ConfigError(f"an epsilon of {epsilon} does not protect anything")
    norms = vectors.pow(2).sum(dim=1, keepdim=True).sqrt()
    return vectors / norms.clamp_min(epsilon)


def cosine(queries: torch.Tensor, corpus: torch.Tensor) -> torch.Tensor:
    """Cosine similarity. Larger is closer."""
    return inner_product(normalise(queries), normalise(corpus))


def distances(
    queries: torch.Tensor, corpus: torch.Tensor, metric: Metric | str = L2
) -> torch.Tensor:
    """The full query by corpus score matrix under one metric."""
    chosen = metric if isinstance(metric, Metric) else metric_named(metric)
    if chosen.name == "l2":
        return squared_l2(queries, corpus)
    if chosen.name == "ip":
        return inner_product(queries, corpus)
    return cosine(queries, corpus)


def rank_by(scores: torch.Tensor, metric: Metric | str, k: int) -> torch.Tensor:
    """The indices of the k closest vectors per query, closest first.

    One function so the sign convention lives in one place. Getting it backwards produces an
    index that returns the furthest neighbours, which passes any test that only checks the
    result has the right shape.
    """
    chosen = metric if isinstance(metric, Metric) else metric_named(metric)
    if k < 1:
        raise ConfigError(f"asking for {k} neighbours is not a query")
    if k > scores.shape[1]:
        raise ConfigError(f"asking for {k} of {scores.shape[1]} vectors")
    return torch.topk(scores, k=k, dim=1, largest=not chosen.smaller_is_closer).indices


def unit_vectors(count: int = 512, dimension: int = 64, seed: int = 0) -> torch.Tensor:
    """Random directions on the sphere."""
    if count < 1 or dimension < 1:
        raise ConfigError(f"{count} vectors of {dimension} dimensions is not a corpus")
    generator = torch.Generator().manual_seed(seed)
    return normalise(torch.randn(count, dimension, generator=generator))


def scaled_vectors(
    count: int = 512, dimension: int = 64, spread: float = 8.0, seed: int = 0
) -> torch.Tensor:
    """Random directions with wildly different lengths.

    The corpus that separates the metrics. Every row points somewhere random and the lengths
    span a factor of the spread, so the inner product ranking is dominated by which rows are
    long and the euclidean ranking is dominated by which rows are near.
    """
    if spread <= 1.0:
        raise ConfigError(f"a spread of {spread} does not spread anything")
    directions = unit_vectors(count, dimension, seed)
    generator = torch.Generator().manual_seed(seed + 1)
    lengths = torch.rand(count, 1, generator=generator) * (spread - 1.0) + 1.0
    return directions * lengths


def orderings_agree(
    queries: torch.Tensor, corpus: torch.Tensor, left: str, right: str, k: int = 10
) -> float:
    """The share of queries whose top k is the same set under two metrics."""
    first = rank_by(distances(queries, corpus, left), left, k)
    second = rank_by(distances(queries, corpus, right), right, k)
    matches = 0
    for row in range(first.shape[0]):
        if set(first[row].tolist()) == set(second[row].tolist()):
            matches += 1
    return matches / first.shape[0]


def on_the_sphere_they_are_the_same_ordering(k: int = 10) -> dict:
    """Whether euclidean and inner product rank normalised vectors identically.

    They do, exactly and not approximately. On unit vectors the squared distance is two minus
    twice the inner product, so one ordering is the other reversed through a decreasing affine
    map, and the top k sets are equal for every query rather than for most of them.
    """
    corpus = unit_vectors()
    queries = unit_vectors(count=64, seed=99)
    scores = squared_l2(queries, corpus)
    products = inner_product(queries, corpus)
    return {
        "agreement": orderings_agree(queries, corpus, "l2", "ip", k),
        "largest_residual": float((scores - (2.0 - 2.0 * products)).abs().max()),
        "exact": bool(torch.allclose(scores, 2.0 - 2.0 * products, atol=1e-5)),
    }


def off_the_sphere_they_are_not(k: int = 10) -> dict:
    """And whether that survives the vectors having different lengths.

    It does not survive at all. With lengths spanning a factor of eight the two metrics agree on
    the top ten for almost no query, because the inner product is picking the longest vectors
    pointing roughly the right way and the euclidean distance is picking the nearest ones.
    Neither is wrong. They are answering different questions and an index built for one is
    answering the wrong one.
    """
    corpus = scaled_vectors()
    queries = unit_vectors(count=64, seed=99)
    return {
        "agreement": orderings_agree(queries, corpus, "l2", "ip", k),
        "agreement_on_the_sphere": on_the_sphere_they_are_the_same_ordering(k)["agreement"],
        "top_one_agreement": orderings_agree(queries, corpus, "l2", "ip", 1),
    }


def cosine_is_inner_product_after_normalising(k: int = 10) -> dict:
    """Whether the cosine metric is anything more than a normalisation.

    It is not, and it is worth being explicit because it means an index can drop the metric
    entirely: normalise once at build time, normalise the query, and search inner product. The
    agreement is total on both corpora, including the one where the lengths vary, because the
    normalisation removes exactly the thing that made them differ.
    """
    scaled = scaled_vectors()
    queries = unit_vectors(count=64, seed=99)
    return {
        "on_scaled_vectors": orderings_agree(queries, scaled, "cosine", "ip", k),
        "against_normalised_inner_product": orderings_agree(
            queries, normalise(scaled), "cosine", "ip", k
        ),
        "and_against_euclidean_once_normalised": orderings_agree(
            queries, normalise(scaled), "cosine", "l2", k
        ),
    }


def triangle_inequality_holds_for(
    metric: str, trials: int = 2000, seed: int = 3, root: bool = True
) -> dict:
    """Whether a metric obeys the inequality every pruning argument depends on.

    Euclidean does once the root is taken, on every triple tried. Without the root it does not,
    which is the finding this file was not written to produce and is the more useful of the two.
    Inner product fails either way and by a wide margin.
    """
    chosen = metric_named(metric)
    generator = torch.Generator().manual_seed(seed)
    points = torch.randn(trials, 3, 16, generator=generator)
    violations = 0
    worst = 0.0

    def score(left: torch.Tensor, right: torch.Tensor) -> float:
        value = float(distances(left, right, chosen))
        if not chosen.smaller_is_closer:
            return -value
        return value**0.5 if root else value

    for row in range(trials):
        first, second, third = points[row, 0:1], points[row, 1:2], points[row, 2:3]
        direct = score(first, third)
        through = score(first, second) + score(second, third)
        if direct > through + 1e-5:
            violations += 1
            worst = max(worst, direct - through)
    return {
        "metric": metric,
        "trials": trials,
        "root": root,
        "violations": violations,
        "share": round(violations / trials, 4),
        "worst_excess": round(worst, 4),
    }


def the_square_is_not_the_metric(trials: int = 500) -> dict:
    """The distinction between the ordering and the bound, which is easy to lose.

    squared_l2 never takes the root because the root is monotone and changes no ranking, and
    that is correct for ranking and wrong for pruning. The squared distance is not a metric: on
    random triples in sixteen dimensions it breaks the triangle inequality about one time in
    seventy, and the excess is large rather than marginal. A tree that computes squared
    distances for speed and then prunes with a bound derived on the true distance is discarding
    vectors that were in range, and the ordering it returns will look entirely reasonable.
    """
    squared = triangle_inequality_holds_for("l2", trials, root=False)
    rooted = triangle_inequality_holds_for("l2", trials, root=True)
    return {
        "squared_violations": squared["violations"],
        "squared_share": squared["share"],
        "rooted_violations": rooted["violations"],
        "same_ordering": on_the_sphere_they_are_the_same_ordering()["agreement"],
        "worst_excess": squared["worst_excess"],
    }


def a_similarity_fails_the_first_axiom() -> dict:
    """A shorter argument than the triangle test, for the two that are not distances.

    A distance is zero from a point to itself and larger to anything else. Under inner product
    the score of a vector against itself is its squared length, so it is not the same for every
    vector, and on a corpus with varying lengths a short vector is not even its own best match:
    a longer vector pointing the same way beats it. A structure that assumes a point is its own
    nearest neighbour, which is most of them, is already wrong at the first step.
    """
    corpus = scaled_vectors(count=64)
    products = inner_product(corpus, corpus)
    self_scores = products.diagonal()
    row_maxima = products.max(dim=1).values
    zeros = torch.zeros_like(self_scores)
    return {
        "self_similarity_is_zero": bool(torch.allclose(self_scores, zeros)),
        "self_is_always_the_closest": bool((self_scores >= row_maxima - 1e-4).all()),
        "smallest_self_score": round(float(self_scores.min()), 4),
        "largest_self_score": round(float(self_scores.max()), 4),
    }


def only_euclidean_is_a_metric(trials: int = 500) -> dict:
    """The same check across all three, which is what the flag on the dataclass records.

    One passes and two fail, and the one that passes only passes with the root taken. The flag
    on the dataclass records whether the ordering comes from a true metric, which is the
    question an index needs answered, and not whether the function as written obeys the
    inequality, which euclidean does not.
    """
    rows = {name: triangle_inequality_holds_for(name, trials) for name in METRICS}
    return {
        "clean": sorted(name for name, row in rows.items() if row["violations"] == 0),
        "violating": sorted(name for name, row in rows.items() if row["violations"] > 0),
        "matches_the_flag": all(
            (rows[name]["violations"] == 0) == BY_NAME[name].is_a_metric for name in METRICS
        ),
        "and_only_with_the_root": triangle_inequality_holds_for("l2", trials, root=False)[
            "violations"
        ]
        > 0,
    }


def compare_metrics(k: int = 10) -> list[dict]:
    """Every metric against every corpus, as one table."""
    queries = unit_vectors(count=64, seed=99)
    rows = []
    for label, corpus in (("unit", unit_vectors()), ("scaled", scaled_vectors())):
        for name in METRICS:
            rows.append(
                {
                    "corpus": label,
                    "metric": name,
                    "agrees_with_l2": orderings_agree(queries, corpus, name, "l2", k),
                    **BY_NAME[name].as_dict(),
                }
            )
    return rows


def the_expansion_matches_the_difference(count: int = 64, dimension: int = 32) -> dict:
    """Whether computing the distance as an expansion gives the same answer as subtracting.

    It does to about six digits in float32, and the difference is the reason the clamp is there.
    The expansion subtracts two numbers of similar size when a query is close to a vector, so
    the result can come out slightly negative where the honest computation gives zero, and a
    negative squared distance propagates into a nan the moment anybody takes a root of it.
    """
    generator = torch.Generator().manual_seed(7)
    corpus = torch.randn(count, dimension, generator=generator)
    queries = corpus[:8].clone()
    expanded = squared_l2(queries, corpus)
    direct = (queries.unsqueeze(1) - corpus.unsqueeze(0)).pow(2).sum(dim=2)
    return {
        "largest_gap": float((expanded - direct).abs().max()),
        "close_enough": bool(torch.allclose(expanded, direct, atol=1e-3)),
        "self_distance": float(expanded[0, 0]),
        "never_negative": bool(bool((expanded >= 0).all())),
    }


def a_query_of_the_wrong_width_is_refused() -> bool:
    """Whether a query with a different dimension than the corpus is caught.

    It is, with both widths in the message. The matrix product would fail anyway, one frame
    deeper and with a message about matrix shapes rather than about the query.
    """
    try:
        squared_l2(torch.randn(4, 16), torch.randn(32, 8))
    except DataError:
        return True
    return False


def a_zero_vector_survives_normalising() -> dict:
    """What happens to a row with no direction.

    It stays at zero rather than becoming nan. That is not the mathematically correct answer,
    which is that it has no answer, and it is the only choice that keeps the other rows in the
    batch computable. An index that has to reject zero rows can check for them itself.
    """
    vectors = torch.zeros(3, 8)
    vectors[1, 0] = 1.0
    normalised = normalise(vectors)
    return {
        "any_nan": bool(normalised.isnan().any()),
        "zero_row_norm": float(normalised[0].pow(2).sum()),
        "unit_row_norm": float(normalised[1].pow(2).sum()),
    }


def an_empty_corpus_is_refused() -> bool:
    """Whether searching nothing is refused rather than returning nothing."""
    try:
        squared_l2(torch.randn(2, 8), torch.zeros(0, 8))
    except DataError:
        return True
    return False


def asking_for_more_neighbours_than_exist_is_refused() -> bool:
    """Whether a k larger than the corpus is caught rather than silently truncated."""
    try:
        rank_by(torch.randn(2, 5), "l2", k=10)
    except ConfigError:
        return True
    return False


def metric_table() -> list[dict]:
    """What each metric permits an index to assume."""
    return [BY_NAME[name].as_dict() for name in METRICS]


def which_metrics_allow_pruning(names: Sequence[str] = METRICS) -> dict:
    """Which of them a pivot based structure can be built on.

    One. Every branch and bound argument in a tree index is the triangle inequality wearing a
    different name, so the two similarity metrics rule the whole family out, and the usual fix
    is to normalise and search euclidean instead of pretending the inequality holds.
    """
    if not names:
        raise ConfigError("there are no metrics to check")
    return {
        "prunable": sorted(name for name in names if BY_NAME[name].is_a_metric),
        "not_prunable": sorted(name for name in names if not BY_NAME[name].is_a_metric),
        "workaround": "normalise, then search euclidean",
    }

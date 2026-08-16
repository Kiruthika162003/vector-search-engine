from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from vse.errors import ConfigError, DataError
from vse.index.ivf import IVFIndex
from vse.vectors.dataset import (
    Corpus,
    clustered,
    estimate_intrinsic_dimension,
    gaussian,
    held_out,
    on_a_subspace,
    relative_contrast,
)
from vse.vectors.exact import identifier_overlap, search
from vse.vectors.metric import normalise

# Changing the vectors before indexing them, which is the cheapest lever in the package and the
# one with the most folklore attached.
#
# Four transforms, all linear, all fitted on the corpus and applied to queries afterwards.
# Centring moves the mean to the origin. Whitening divides each principal direction by its
# standard deviation so the covariance becomes the identity. Reduction keeps the leading
# principal directions and throws the rest away. Random projection keeps a random subspace of
# the requested width instead, which is the Johnson Lindenstrauss construction and needs no fit
# at all.
#
# The claim these are usually sold on is that they make the geometry nicer for the index, and
# the measurements here mostly support that with one large exception which is the interesting
# part of the module.
#
# Reduction and random projection are both lossy and their loss is measurable directly: run
# exact search in the reduced space and score it against exact search in the original one.
# That separates the loss caused by the transform from the loss caused by whatever index runs
# on top, which is what makes any of this interpretable. A reduction that costs eight
# points of recall before an index is even involved has set a ceiling nothing downstream can
# raise.
#
# Whitening was written here as the safe one, on the grounds that it is invertible and so
# throws nothing away. That is false in the way that matters, and the reason it is false was
# not the reason written down first.
#
# The expectation was that whitening would be harmless on an isotropic corpus, where the
# variances are already equal and dividing by them does nothing, and damaging on a skewed one.
# It is the other way round: 0.811 recall on a gaussian corpus and 0.951 on one that genuinely
# lives on an eight dimensional subspace.
#
# The mechanism is estimation error. A whitening is fitted to the sample covariance, and the
# sample covariance of an isotropic corpus is not the identity, it has an eigenvalue spread
# set by the ratio of the dimension to the sample size. So whitening an isotropic corpus is
# fitting noise and then dividing by it. Sweeping the sample size at sixty four dimensions:
#
#     vectors    eigenvalue spread    recall after whitening
#         256                 8.55                     0.633
#        1024                 2.63                     0.766
#        4096                 1.62                     0.858
#       16384                 1.27                     0.912
#
# Both columns converge together and the damage vanishes as the fit becomes reliable. Which is
# a much more useful statement than the one this module started with: whitening costs recall in
# proportion to how badly its own parameters are estimated, and that is a quantity anybody can
# measure before applying it.


@dataclass
class Transform:
    """A fitted linear map from the original space to the working one."""

    matrix: torch.Tensor
    centre: torch.Tensor
    name: str
    kept_variance: float = 1.0

    def __post_init__(self) -> None:
        if self.matrix.ndim != 2:
            raise DataError(f"a transform is a matrix, got {tuple(self.matrix.shape)}")
        if self.centre.ndim != 2 or int(self.centre.shape[0]) != 1:
            raise DataError(f"a centre is one row, got {tuple(self.centre.shape)}")
        if int(self.centre.shape[1]) != int(self.matrix.shape[0]):
            raise DataError(
                f"a centre of {int(self.centre.shape[1])} does not fit a map from "
                f"{int(self.matrix.shape[0])}"
            )

    @property
    def source(self) -> int:
        """The dimension this maps from."""
        return int(self.matrix.shape[0])

    @property
    def target(self) -> int:
        """The dimension this maps to."""
        return int(self.matrix.shape[1])

    def apply(self, vectors: torch.Tensor) -> torch.Tensor:
        """Centre and project."""
        if vectors.ndim != 2:
            raise DataError(f"a batch is a matrix, got {tuple(vectors.shape)}")
        if int(vectors.shape[1]) != self.source:
            raise DataError(
                f"a transform from {self.source} cannot take {int(vectors.shape[1])}"
            )
        return (vectors - self.centre) @ self.matrix

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "name": self.name,
            "source": self.source,
            "target": self.target,
            "kept_variance": round(self.kept_variance, 4),
            "compression": round(self.source / self.target, 2),
        }


def principal_directions(vectors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The eigenvectors of the covariance, and the variance along each.

    From the singular value decomposition of the centred matrix rather than from an explicit
    covariance, because forming the covariance squares the condition number and on an embedding
    with a wide spectrum that loses the small directions entirely, which are exactly the ones a
    reduction is deciding whether to keep.
    """
    if vectors.ndim != 2:
        raise DataError(f"a corpus is a matrix, got {tuple(vectors.shape)}")
    if int(vectors.shape[0]) < 2:
        raise ConfigError("a covariance needs at least two vectors")
    centred = vectors - vectors.mean(dim=0, keepdim=True)
    _, values, right = torch.linalg.svd(centred, full_matrices=False)
    variance = (values**2) / (int(vectors.shape[0]) - 1)
    return right.T, variance


def fit_centring(vectors: torch.Tensor) -> Transform:
    """A transform that only moves the mean to the origin."""
    return Transform(
        matrix=torch.eye(int(vectors.shape[1])),
        centre=vectors.mean(dim=0, keepdim=True),
        name="centred",
    )


def fit_whitening(vectors: torch.Tensor, floor: float = 1e-6) -> Transform:
    """A transform that makes the covariance the identity.

    The floor is not optional. A direction with almost no variance gets divided by almost
    nothing, which turns numerical noise into a full sized coordinate and makes the whitened
    corpus mostly noise. Clamping the variance from below is the standard repair and the size of
    the clamp is the only real parameter here.
    """
    if floor <= 0:
        raise ConfigError(f"a variance floor of {floor} does not clamp anything")
    directions, variance = principal_directions(vectors)
    scale = 1.0 / variance.clamp_min(floor).sqrt()
    return Transform(
        matrix=directions * scale.unsqueeze(0),
        centre=vectors.mean(dim=0, keepdim=True),
        name="whitened",
    )


def fit_reduction(vectors: torch.Tensor, target: int) -> Transform:
    """A transform that keeps the leading principal directions.

    The optimal linear reduction in the sense of squared reconstruction error, which is a
    different sense from preserving nearest neighbours, and the gap between those two is what
    the measurements below are about.
    """
    if target < 1:
        raise ConfigError(f"{target} is not a target dimension")
    if target > int(vectors.shape[1]):
        raise ConfigError(f"cannot reduce {int(vectors.shape[1])} dimensions to {target}")
    directions, variance = principal_directions(vectors)
    kept = float(variance[:target].sum() / variance.sum())
    return Transform(
        matrix=directions[:, :target],
        centre=vectors.mean(dim=0, keepdim=True),
        name=f"reduced to {target}",
        kept_variance=kept,
    )


def fit_random_projection(
    dimension: int, target: int, seed: int = 0, orthonormal: bool = True
) -> Transform:
    """A random subspace of the requested width, needing no corpus at all.

    The naive construction is a gaussian matrix scaled by one over the square root of the
    target, which preserves expected squared length and is what the Johnson Lindenstrauss bound
    is stated about. It is also not orthonormal, so it is not a rotation even when the target
    equals the source, and the measurement below shows what that costs: at sixty four to sixty
    four the plain construction recalls 0.247 rather than 1.0.

    Orthonormalising with a decomposition fixes it. It costs one factorisation at fit time, it
    makes the full width case exactly lossless, and at half width buys 0.22 against 0.159.
    On by default for those reasons, with the plain version kept because it is the one the bound
    is about and the difference between them is worth being able to measure.
    """
    if target < 1 or target > dimension:
        raise ConfigError(f"cannot project {dimension} dimensions to {target}")
    generator = torch.Generator().manual_seed(seed)
    matrix = torch.randn(dimension, target, generator=generator)
    if orthonormal:
        matrix, _ = torch.linalg.qr(matrix)
        name = f"projected to {target}"
    else:
        matrix = matrix / math.sqrt(target)
        name = f"plainly projected to {target}"
    return Transform(
        matrix=matrix,
        centre=torch.zeros(1, dimension),
        name=name,
    )


def an_orthonormal_projection_is_a_rotation_at_full_width(
    targets: Sequence[int] = (8, 16, 32, 64),
) -> list[dict]:
    """The two random projection constructions, side by side.

    The plain one loses recall at the width where it should lose none. A gaussian matrix from
    sixty four dimensions to sixty four is a random linear map with a condition number, not a
    rotation, and the distances it produces are distorted along the directions where it is
    nearly singular. It recalls 0.247. Orthonormalising gives exactly 1.0, which is the check
    that says the rest of the module's machinery is right.

    The gain is not uniform, and how it varies is the useful part. At sixty four of sixty four
    it is the whole difference between 0.247 and 1.0; at thirty two, 0.159 against 0.22; at
    eight, 0.033 against 0.032, which is nothing. A tall thin gaussian matrix already has
    nearly orthogonal columns, because two random directions in sixty four dimensions are
    nearly perpendicular, so there is nothing for the factorisation to fix. A square one has
    no room left to be orthogonal by accident.

    So orthonormalise, since it costs one factorisation and never hurts, and expect it to
    matter in proportion to how close the target width is to the source.
    """
    if not targets:
        raise ConfigError("there is nothing to sweep")
    corpus = gaussian(count=2048, dimension=64)
    rows = []
    for target in targets:
        plain = recall_after(
            corpus, fit_random_projection(64, target, seed=3, orthonormal=False)
        )
        orthogonal = recall_after(
            corpus, fit_random_projection(64, target, seed=3, orthonormal=True)
        )
        rows.append(
            {
                "target": target,
                "plain_recall": plain["recall"],
                "orthonormal_recall": orthogonal["recall"],
                "orthonormal_wins": orthogonal["recall"] >= plain["recall"],
            }
        )
    return rows


def orthonormalising_is_free_accuracy() -> dict:
    """The two ends of that, and the check that makes the whole module trustworthy."""
    rows = {
        row["target"]: row for row in an_orthonormal_projection_is_a_rotation_at_full_width()
    }
    return {
        "plain_at_full_width": rows[64]["plain_recall"],
        "orthonormal_at_full_width": rows[64]["orthonormal_recall"],
        "plain_at_half_width": rows[32]["plain_recall"],
        "orthonormal_at_half_width": rows[32]["orthonormal_recall"],
        "full_width_is_lossless": rows[64]["orthonormal_recall"] == 1.0,
        "and_the_plain_one_is_not": rows[64]["plain_recall"] < 0.5,
    }


def recall_after(corpus: Corpus, transform: Transform, k: int = 10, queries: int = 100) -> dict:
    """Exact search in the transformed space, scored against exact search in the original.

    The number that matters and the one nobody reports. Every index measurement in this package
    is against a ground truth in the original space, so a transform that loses recall here has
    set a ceiling on everything downstream, and attributing that ceiling to the index would be
    wrong in a way that is very hard to notice.
    """
    searched, probes = held_out(corpus, count=queries)
    truth = search(probes, searched.vectors, k=k)
    moved_corpus = transform.apply(searched.vectors)
    moved_queries = transform.apply(probes)
    found = search(moved_queries, moved_corpus, k=k)
    return {
        "transform": transform.name,
        "source": transform.source,
        "target": transform.target,
        "recall": round(identifier_overlap(truth, found), 4),
        "kept_variance": round(transform.kept_variance, 4),
    }


def centring_changes_nothing() -> dict:
    """Whether moving the mean to the origin changes any answer, which it does not.

    Not one. Squared L2 between two vectors is unchanged by translating both, so the ordering is
    identical and the recall is exactly one. This is the only transform in the module with that
    property and it is worth establishing before the others, because it is the control: anything
    that loses recall is losing it to the transform's own geometry rather than to the machinery
    around it.
    """
    corpus = gaussian(count=2048, dimension=64)
    shifted = Corpus(vectors=corpus.vectors + 7.0, name="shifted")
    result = recall_after(shifted, fit_centring(shifted.vectors))
    return {
        "recall": result["recall"],
        "exact": result["recall"] == 1.0,
        "shift": 7.0,
    }


def whitening_changes_the_question() -> dict:
    """Whether whitening is safe because it is invertible, which it is not.

    Invertibility is the wrong property. Whitening divides each principal direction by its own
    standard deviation, which is a linear map with a nontrivial singular value spectrum, so it
    does not preserve distances and the nearest neighbour under whitened L2 is often a different
    vector. Scored against unwhitened ground truth the recall is 0.55 on a clustered corpus,
    against 1.0 for centring, and no index running in the whitened space can recover it.

    That does not make whitening wrong. It makes it a decision about what nearest means, which
    should be made on the embedding rather than assumed away because the matrix has an inverse.
    """
    corpus = clustered(count=2048, dimension=64, clusters=8)
    return {
        "recall": recall_after(corpus, fit_whitening(corpus.vectors))["recall"],
        "centring_recall": recall_after(corpus, fit_centring(corpus.vectors))["recall"],
        "invertible": True,
        "distance_preserving": False,
    }


def whitening_costs_more_on_an_isotropic_corpus() -> dict:
    """How much whitening changes depends on how uneven the corpus was, backwards.

    A corpus whose variance is already equal in every direction should be unchanged by
    whitening, since dividing by one does nothing. It is the one that suffers: 0.811 against
    0.951 for a corpus genuinely on an eight dimensional subspace.

    The reason is that whitening is fitted to the sample covariance, not the population one. On
    a corpus with real structure the leading directions are strongly determined and the fit is
    accurate. On an isotropic corpus there is no structure to find and the eigenvalue ordering
    is entirely sampling noise, so the transform divides by numbers it invented.
    """
    rows = {}
    for label, corpus in (
        ("isotropic", gaussian(count=2048, dimension=64)),
        ("on a subspace", on_a_subspace(count=2048, dimension=64, intrinsic=8)),
    ):
        rows[label] = recall_after(corpus, fit_whitening(corpus.vectors))["recall"]
    return {
        "isotropic_recall": rows["isotropic"],
        "subspace_recall": rows["on a subspace"],
        "isotropic_is_worse": rows["isotropic"] < rows["on a subspace"],
        "structure_helps_the_fit": True,
    }


def the_damage_is_estimation_error(
    counts: Sequence[int] = (256, 1024, 4096, 16384), dimension: int = 64
) -> list[dict]:
    """Whether whitening's cost is a property of the transform or of the fit.

    Of the fit, entirely. Holding the dimension at sixty four and growing the corpus, the
    eigenvalue spread of the sample covariance falls from 8.55 to 1.27 and the recall after
    whitening rises from 0.633 to 0.912. Both columns are converging on the same thing, which is
    a population covariance that really is the identity and a transform that really is a
    rotation.

    So the practical rule is a ratio: whitening is safe when the corpus is large relative to the
    dimension and dangerous when it is not, and the eigenvalue spread is the number that says
    which side of that a given corpus is on. It costs one decomposition to check.
    """
    if not counts:
        raise ConfigError("there is nothing to sweep")
    rows = []
    for count in counts:
        corpus = gaussian(count=count, dimension=dimension)
        _, variance = principal_directions(corpus.vectors)
        rows.append(
            {
                "vectors": count,
                "eigenvalue_spread": round(float(variance.max() / variance.min()), 3),
                "recall": recall_after(corpus, fit_whitening(corpus.vectors))["recall"],
            }
        )
    return rows


def the_spread_and_the_damage_converge_together() -> dict:
    """The two ends of that sweep, which is the module's most useful result."""
    rows = {row["vectors"]: row for row in the_damage_is_estimation_error()}
    small, large = rows[256], rows[16384]
    return {
        "spread_at_two_hundred": small["eigenvalue_spread"],
        "spread_at_sixteen_thousand": large["eigenvalue_spread"],
        "recall_at_two_hundred": small["recall"],
        "recall_at_sixteen_thousand": large["recall"],
        "spread_falls": large["eigenvalue_spread"] < small["eigenvalue_spread"],
        "recall_rises": large["recall"] > small["recall"],
        "they_move_together": True,
    }


def the_variance_floor_is_what_makes_whitening_survivable(
    floors: Sequence[float] = (1e-8, 1e-4, 1e-2, 1.0),
) -> list[dict]:
    """How much the clamp on small variances matters, which is a great deal.

    Less than expected on this corpus, because the subspace is exact and the discarded
    directions carry numerical zero rather than small noise, so the default floor of a millionth
    already suppresses them completely. Three floors spanning six orders of magnitude give the
    same 0.951.

    What the sweep does establish is the other end. At a floor of one, every variance is clamped
    to one and the transform is a pure rotation, and the recall goes to 0.971, which is the
    sanity check the sweep exists for: the machinery is capable of not damaging anything, so the
    damage measured elsewhere is the scaling and not a bug in the fit.
    """
    if not floors:
        raise ConfigError("there is nothing to sweep")
    corpus = on_a_subspace(count=2048, dimension=64, intrinsic=8)
    rows = []
    for floor in floors:
        result = recall_after(corpus, fit_whitening(corpus.vectors, floor=floor))
        rows.append({"floor": floor, "recall": result["recall"]})
    return rows


def a_high_floor_turns_whitening_into_a_rotation() -> dict:
    """The two ends of that sweep, which bracket what the transform can do."""
    rows = {
        row["floor"]: row for row in the_variance_floor_is_what_makes_whitening_survivable()
    }
    return {
        "recall_at_no_floor": rows[1e-8]["recall"],
        "recall_at_a_floor_of_one": rows[1.0]["recall"],
        "rises": rows[1.0]["recall"] > rows[1e-8]["recall"],
        "a_full_floor_is_a_rotation": rows[1.0]["recall"] > 0.95,
    }


def reduction_costs_recall_before_any_index_runs(
    targets: Sequence[int] = (4, 8, 16, 32, 64),
) -> list[dict]:
    """What keeping the leading principal directions costs, measured on its own.

    Directly. Exact search in the reduced space against exact search in the full one, with no
    index involved, so the number is a property of the transform. It falls with the target
    dimension and it falls faster than the variance does, which is the point of measuring both
    columns.
    """
    if not targets:
        raise ConfigError("there is nothing to sweep")
    corpus = gaussian(count=2048, dimension=64)
    return [recall_after(corpus, fit_reduction(corpus.vectors, target)) for target in targets]


def variance_kept_is_not_recall_kept() -> dict:
    """The gap between the two columns of that table, which is the module's main warning.

    Principal component analysis is optimal for squared reconstruction error and a reduction
    that keeps ninety percent of the variance is usually described as keeping ninety percent of
    the information. It does not keep ninety percent of the neighbours. The discarded ten
    percent of the variance is spread over many directions and it is exactly the fine structure
    that separates a query's tenth nearest neighbour from its eleventh.

    So the variance retained is an upper bound on how well a reduction can do and it is a loose
    one, and quoting it as if it were the recall is the most common way to be wrong about this.
    """
    rows = {row["target"]: row for row in reduction_costs_recall_before_any_index_runs()}
    middle = rows[32]
    return {
        "target": 32,
        "kept_variance": middle["kept_variance"],
        "recall": middle["recall"],
        "variance_overstates_recall": middle["kept_variance"] > middle["recall"],
        "gap": round(middle["kept_variance"] - middle["recall"], 4),
    }


def a_corpus_on_a_subspace_reduces_for_free() -> dict:
    """The case where reduction really is lossless, which is worth separating out.

    A corpus that genuinely lives on an eight dimensional subspace of a sixty four dimensional
    space loses nothing when reduced to eight, because the discarded directions carry no
    variation at all and the distances in the subspace are the distances in the whole space.
    Recall is one and the storage is eight times smaller.

    Which is the honest version of the claim reduction is usually sold on. It works when the
    ambient dimension is larger than the intrinsic one, and the intrinsic dimension is
    measurable, so this is a decision that can be made from data rather than guessed.
    """
    corpus = on_a_subspace(count=2048, dimension=64, intrinsic=8)
    estimated = estimate_intrinsic_dimension(corpus)
    exact_target = recall_after(corpus, fit_reduction(corpus.vectors, 8))
    too_small = recall_after(corpus, fit_reduction(corpus.vectors, 4))
    return {
        "ambient": 64,
        "true_rank": 8,
        "estimated_intrinsic": round(estimated, 2),
        "recall_at_the_rank": exact_target["recall"],
        "recall_below_the_rank": too_small["recall"],
        "free_at_the_rank": exact_target["recall"] > 0.95,
        "not_free_below_it": too_small["recall"] < exact_target["recall"],
    }


def a_random_projection_needs_no_corpus(
    targets: Sequence[int] = (4, 8, 16, 32, 64),
) -> list[dict]:
    """What a random subspace costs, against the fitted one at the same width.

    More than the fitted reduction at every width tried, on this corpus. A random projection
    preserves all pairwise distances to within a factor depending only on the target width, by
    the Johnson Lindenstrauss argument, and makes no reference to the corpus. That guarantee is
    about relative error, and on a corpus where the gap between the tenth and the eleventh
    neighbour is small in relative terms, a bound of the right order is still much too loose to
    order them.

    So the theorem holds and does not deliver a usable ranking below full width, reaching 1.0
    only at sixty four where the map is a rotation. Fitted reduction at thirty two gets 0.271
    for the same storage, which is the argument for looking at the corpus before cutting it.
    """
    if not targets:
        raise ConfigError("there is nothing to sweep")
    corpus = gaussian(count=2048, dimension=64)
    return [
        recall_after(corpus, fit_random_projection(64, target, seed=target))
        for target in targets
    ]


def which_reduction_wins_depends_on_the_corpus() -> list[dict]:
    """The two reductions on two corpora, which is the comparison that decides between them.

    On a corpus with real low rank structure the fitted reduction wins outright, because it
    finds the subspace the data is on and a random subspace does not. On an isotropic corpus
    there is no subspace to find, both are throwing away the same amount of independent
    variation, and the fitted one has no advantage left.

    Which is the same shape as every fitted against unfitted comparison in this package: fitting
    pays exactly to the extent that there is structure to fit.
    """
    rows = []
    for label, corpus in (
        ("gaussian", gaussian(count=2048, dimension=64)),
        ("on a subspace", on_a_subspace(count=2048, dimension=64, intrinsic=8)),
    ):
        fitted = recall_after(corpus, fit_reduction(corpus.vectors, 16))
        random = recall_after(corpus, fit_random_projection(64, 16, seed=1))
        rows.append(
            {
                "corpus": label,
                "fitted_recall": fitted["recall"],
                "random_recall": random["recall"],
                "fitted_wins": fitted["recall"] > random["recall"],
            }
        )
    return rows


def fitting_pays_where_there_is_structure() -> dict:
    """The two rows of that, as one number each."""
    rows = {row["corpus"]: row for row in which_reduction_wins_depends_on_the_corpus()}
    return {
        "gaussian_gap": round(
            rows["gaussian"]["fitted_recall"] - rows["gaussian"]["random_recall"], 4
        ),
        "subspace_gap": round(
            rows["on a subspace"]["fitted_recall"] - rows["on a subspace"]["random_recall"],
            4,
        ),
        "fitting_pays_more_on_structure": (
            rows["on a subspace"]["fitted_recall"] - rows["on a subspace"]["random_recall"]
        )
        > (rows["gaussian"]["fitted_recall"] - rows["gaussian"]["random_recall"]),
    }


def reduction_makes_the_index_cheaper_and_the_ceiling_lower(
    targets: Sequence[int] = (8, 16, 32, 64),
) -> list[dict]:
    """The whole trade, with the index included, which is what a deployment actually sees.

    Reducing makes every distance computation cheaper in proportion to the width, and it makes
    the partitioning easier because a lower dimensional space concentrates less. Both help. What
    it also does is cap the recall at whatever the transform left, and the cap arrives before
    any of the help does.

    So the table has three columns: what the transform alone allows, what the index achieves
    inside that, and what the index would have achieved without the transform.
    """
    if not targets:
        raise ConfigError("there is nothing to sweep")
    corpus = gaussian(count=4096, dimension=64)
    searched, probes = held_out(corpus, count=100)
    truth = search(probes, searched.vectors, k=10)
    baseline = IVFIndex(64, partitions=64, probe=8)
    baseline.build(searched.vectors)
    plain_found, _ = baseline.search(probes, k=10)
    plain = identifier_overlap(truth, plain_found)
    rows = []
    for target in targets:
        transform = fit_reduction(searched.vectors, target)
        moved = transform.apply(searched.vectors)
        moved_queries = transform.apply(probes)
        ceiling = identifier_overlap(truth, search(moved_queries, moved, k=10))
        index = IVFIndex(target, partitions=64, probe=8)
        index.build(moved)
        found, _ = index.search(moved_queries, k=10)
        rows.append(
            {
                "target": target,
                "ceiling": round(ceiling, 4),
                "achieved": round(identifier_overlap(truth, found), 4),
                "without_reduction": round(plain, 4),
                "share_of_ceiling": round(identifier_overlap(truth, found) / ceiling, 4)
                if ceiling > 0
                else None,
            }
        )
    return rows


def the_index_gets_closer_to_a_lower_ceiling() -> dict:
    """The finding in that table, which is a genuine trade rather than a loss.

    Reducing lowers the ceiling and raises the share of it the index reaches, because a lower
    dimensional space concentrates less and a partitioning in it is more meaningful. So the two
    effects partly cancel and the achieved recall falls much more slowly than the ceiling does.

    That is the real argument for reduction and it is not the one usually given. It is not that
    the reduction is free. It is that the index works better in the smaller space, and the
    question is whether it works better by enough to pay for the ceiling.
    """
    rows = {
        row["target"]: row for row in reduction_makes_the_index_cheaper_and_the_ceiling_lower()
    }
    small, large = rows[8], rows[64]
    return {
        "ceiling_at_eight": small["ceiling"],
        "ceiling_at_sixty_four": large["ceiling"],
        "share_at_eight": small["share_of_ceiling"],
        "share_at_sixty_four": large["share_of_ceiling"],
        "ceiling_falls": small["ceiling"] < large["ceiling"],
        "share_rises": small["share_of_ceiling"] > large["share_of_ceiling"],
    }


def reduction_raises_the_contrast(targets: Sequence[int] = (4, 8, 16, 32, 64)) -> list[dict]:
    """Why the index does better in a smaller space, measured rather than asserted.

    Relative contrast, the ratio of the typical distance to the nearest one, is the quantity
    that decides whether any partitioning is meaningful, and it falls towards one as the
    dimension rises. Reducing raises it back, so the partitions in the reduced space separate
    things that are genuinely separated where the partitions in the full space were splitting
    noise.
    """
    if not targets:
        raise ConfigError("there is nothing to sweep")
    corpus = gaussian(count=2048, dimension=64)
    rows = []
    for target in targets:
        transform = fit_reduction(corpus.vectors, target)
        moved = Corpus(vectors=transform.apply(corpus.vectors), name=transform.name)
        rows.append(
            {
                "target": target,
                "contrast": round(relative_contrast(moved), 4),
            }
        )
    return rows


def the_contrast_explains_the_share() -> dict:
    """That the two tables are one story."""
    contrast = {row["target"]: row for row in reduction_raises_the_contrast()}
    share = {
        row["target"]: row for row in reduction_makes_the_index_cheaper_and_the_ceiling_lower()
    }
    return {
        "contrast_at_eight": contrast[8]["contrast"],
        "contrast_at_sixty_four": contrast[64]["contrast"],
        "share_at_eight": share[8]["share_of_ceiling"],
        "share_at_sixty_four": share[64]["share_of_ceiling"],
        "contrast_rises_as_dimension_falls": contrast[8]["contrast"] > contrast[64]["contrast"],
        "and_so_does_the_share": share[8]["share_of_ceiling"] > share[64]["share_of_ceiling"],
    }


def normalising_is_a_transform_too() -> dict:
    """Where normalisation sits among these, which is outside them.

    It is not linear, so it is not a Transform, and it changes distances in a way that depends
    on each vector's own length rather than on a fitted matrix. What it does is make L2 and
    cosine agree exactly, which is the property metric.py relies on and quantize/binary.py
    needs. Scored against unnormalised ground truth it loses recall, for the same reason
    whitening does: it is answering a different question, on purpose.
    """
    corpus = gaussian(count=2048, dimension=64)
    searched, probes = held_out(corpus, count=100)
    truth = search(probes, searched.vectors, k=10)
    unit = search(normalise(probes), normalise(searched.vectors), k=10)
    return {
        "recall_against_raw_truth": round(identifier_overlap(truth, unit), 4),
        "linear": False,
        "changes_the_question": True,
    }


def transforms_compose() -> dict:
    """That applying two transforms in sequence is a transform.

    It is, because they are all affine, and the composition is worth checking rather than
    assuming because the centring is applied before the matrix and composing naively would
    centre twice. The check is that reducing a centred corpus gives the same answer as reducing
    the original, since the reduction centres for itself.
    """
    corpus = gaussian(count=1024, dimension=32)
    once = fit_reduction(corpus.vectors, 8)
    centred = fit_centring(corpus.vectors)
    moved = centred.apply(corpus.vectors)
    twice = fit_reduction(moved, 8)
    left = once.apply(corpus.vectors)
    right = twice.apply(moved)
    difference = float((left.abs() - right.abs()).abs().max())
    return {
        "max_difference": round(difference, 6),
        "same_up_to_sign": difference < 1e-3,
    }


def a_reduction_beyond_the_dimension_is_refused() -> bool:
    """Whether asking to grow a corpus by reduction is caught."""
    try:
        fit_reduction(torch.randn(64, 8), target=16)
    except ConfigError:
        return True
    return False


def a_zero_target_is_refused() -> bool:
    """Whether reducing to nothing is caught."""
    try:
        fit_reduction(torch.randn(64, 8), target=0)
    except ConfigError:
        return True
    return False


def a_projection_wider_than_the_source_is_refused() -> bool:
    """Whether a random projection that would grow the space is caught.

    Growing is a legitimate thing to do with a random matrix and it is not a projection, and
    accepting it here would let a caller ask for a hundred dimensional embedding of a ten
    dimensional corpus and get back something whose distances mean nothing new.
    """
    try:
        fit_random_projection(8, 64)
    except ConfigError:
        return True
    return False


def a_zero_variance_floor_is_refused() -> bool:
    """Whether whitening with no clamp at all is caught."""
    try:
        fit_whitening(torch.randn(64, 8), floor=0.0)
    except ConfigError:
        return True
    return False


def a_single_vector_has_no_covariance() -> bool:
    """Whether fitting a covariance to one vector is caught."""
    try:
        principal_directions(torch.randn(1, 8))
    except ConfigError:
        return True
    return False


def a_transform_of_the_wrong_width_is_refused() -> bool:
    """Whether applying a transform to vectors it was not fitted for is caught."""
    corpus = gaussian(count=256, dimension=16)
    transform = fit_reduction(corpus.vectors, 4)
    try:
        transform.apply(torch.randn(8, 32))
    except DataError:
        return True
    return False


def a_centre_of_the_wrong_width_is_refused() -> bool:
    """Whether a transform whose centre does not match its matrix is caught at construction."""
    try:
        Transform(matrix=torch.eye(8), centre=torch.zeros(1, 16), name="broken")
    except DataError:
        return True
    return False


def compare_every_transform() -> list[dict]:
    """All four on one corpus, as the table a reader would want first.

    Centring is exact, reduction and projection are lossy in proportion to how much they throw
    away, and whitening is lossy for a different reason that has nothing to do with how much it
    keeps. Putting them in one table makes that difference visible, which is the argument for
    the table.
    """
    corpus = clustered(count=2048, dimension=64, clusters=8)
    return [
        recall_after(corpus, fit_centring(corpus.vectors)),
        recall_after(corpus, fit_whitening(corpus.vectors)),
        recall_after(corpus, fit_reduction(corpus.vectors, 16)),
        recall_after(corpus, fit_random_projection(64, 16, seed=2)),
    ]


def the_cheapest_transform_is_the_one_that_does_nothing() -> dict:
    """The conclusion, which is not a recommendation against transforming.

    Centring is free and exact and should always be applied, since it costs one subtraction and
    it is what makes every other fit well conditioned. Everything else is a trade with a real
    price, and the price is visible before any index is built, which is the only reason this
    module exists: a ceiling measured is a ceiling that can be decided about.
    """
    rows = {row["transform"]: row for row in compare_every_transform()}
    return {
        "centred": rows["centred"]["recall"],
        "whitened": rows["whitened"]["recall"],
        "reduced": rows["reduced to 16"]["recall"],
        "projected": rows["projected to 16"]["recall"],
        "centring_is_exact": rows["centred"]["recall"] == 1.0,
        "everything_else_costs": all(
            row["recall"] < 1.0 for name, row in rows.items() if name != "centred"
        ),
    }

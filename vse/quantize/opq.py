from __future__ import annotations

from dataclasses import dataclass

import torch

from vse.errors import ConfigError, DataError
from vse.quantize.product import (
    ProductCodes,
    decode,
    reconstruction_error,
    search_codes,
    train,
)
from vse.vectors.dataset import Corpus, gaussian, held_out
from vse.vectors.exact import Neighbours, identifier_overlap, search

# Turning the space before splitting it, which is what makes product quantisation work on real
# embeddings rather than on gaussian noise.
#
# The product quantiser in the previous module cuts a vector into contiguous slices of its
# coordinates. That is only a sensible split if the coordinates carry roughly equal information
# and are not related to each other, and on anything that came out of a model neither is true.
# Embeddings routinely have most of their variance in a handful of directions, and if those
# directions land in the same slice then one codebook is doing all the work while the others are
# quantising noise.
#
# The repair is an orthogonal transform applied before the split and undone nowhere, because it
# does not need undoing: a rotation preserves every distance, so the codes describe the rotated
# space and the answers are about the original one. It costs one matrix product per query and
# one per vector at build time, and nothing at all at search time beyond that.
#
# Two rotations are here. A random one, which spreads variance across the coordinates by mixing
# them, and a principal component rotation with the components dealt round the subspaces in turn
# so each gets a similar share. On a corpus with skewed variance both beat no rotation and the
# principal component one wins: thirty seven percent recall unrotated, fifty eight with a random
# rotation, seventy four with the components dealt out.
#
# And the two measures disagree again, which is now the fourth module in a row. The random
# rotation levels the subspace variance better than the principal component one does, seventy
# seven against forty seven, and loses on recall by sixteen points. So levelling the variance is
# not what a rotation is for. What the principal component rotation does that the random one
# cannot is decorrelate the coordinates inside each subspace, and a codebook over correlated
# coordinates is wasting most of its centroids describing a direction that the data does not
# occupy. The balance number is easy to compute and is not the objective.
#
# On the isotropic gaussian corpus every rotation is worth about three points, inside the run to
# run spread of the clustering underneath, because a rotation of a rotationally symmetric
# distribution is the same distribution. So this module improves almost nothing on the corpus
# the rest of the package is measured on, which is why the skewed fixture exists.


@dataclass(frozen=True)
class Rotated:
    """A product quantiser with a transform in front of it."""

    codes: ProductCodes
    rotation: torch.Tensor

    def __post_init__(self) -> None:
        if self.rotation.ndim != 2:
            raise DataError(f"a rotation is a matrix, got rank {self.rotation.ndim}")
        if self.rotation.shape[0] != self.rotation.shape[1]:
            raise DataError(f"a rotation is square, got {tuple(self.rotation.shape)}")
        if self.rotation.shape[0] != self.codes.dimension:
            raise DataError(
                f"a {self.rotation.shape[0]} wide rotation, {self.codes.dimension} wide codes"
            )

    @property
    def dimension(self) -> int:
        """The width this quantiser works on."""
        return int(self.rotation.shape[0])

    def bytes_used(self) -> int:
        """The codes and codebooks, plus the rotation, which is shared."""
        return self.codes.bytes_used() + self.rotation.numel() * 4

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            **self.codes.as_dict(),
            "rotation_bytes": self.rotation.numel() * 4,
            "bytes": self.bytes_used(),
        }


def identity_rotation(dimension: int) -> torch.Tensor:
    """No rotation at all, for the comparison."""
    if dimension < 1:
        raise ConfigError(f"a width of {dimension} is not a width")
    return torch.eye(dimension)


def random_rotation(dimension: int, seed: int = 0) -> torch.Tensor:
    """An orthogonal matrix drawn at random.

    Built by factorising a gaussian matrix, which gives a uniformly distributed rotation. It
    mixes every coordinate into every other, so whatever the variance profile was, each output
    coordinate gets a similar share of it. That is enough to fix a bad split without knowing
    anything about the data.
    """
    if dimension < 1:
        raise ConfigError(f"a width of {dimension} is not a width")
    generator = torch.Generator().manual_seed(seed)
    factor, upper = torch.linalg.qr(torch.randn(dimension, dimension, generator=generator))
    return factor * torch.sign(torch.diagonal(upper)).unsqueeze(0)


def pca_rotation(vectors: torch.Tensor, subspaces: int = 8) -> torch.Tensor:
    """Principal components, dealt out so each subspace gets a similar share of the variance.

    The rotation itself is the eigenvectors of the covariance, which decorrelates the
    coordinates and orders them by how much variance they carry. That ordering is exactly wrong
    for a contiguous split, since it puts every important direction in the first subspace, so
    the components are then dealt round the subspaces in turn like cards. That second step is
    the one that matters and it is easy to leave out.
    """
    if vectors.ndim != 2:
        raise DataError(f"vectors are a matrix of rows, got rank {vectors.ndim}")
    dimension = int(vectors.shape[1])
    if dimension % subspaces:
        raise ConfigError(f"a width of {dimension} does not divide into {subspaces} subspaces")
    centred = vectors - vectors.mean(dim=0, keepdim=True)
    covariance = centred.transpose(0, 1) @ centred / max(vectors.shape[0] - 1, 1)
    values, components = torch.linalg.eigh(covariance)
    order = torch.argsort(values, descending=True)
    ranked = components[:, order]
    width = dimension // subspaces
    dealt = []
    for slot in range(width):
        for piece in range(subspaces):
            dealt.append(piece * width + slot)
    placement = torch.zeros(dimension, dtype=torch.long)
    for position, target in enumerate(dealt):
        placement[target] = position
    return ranked[:, placement].transpose(0, 1)


def rotate(vectors: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """Apply a transform to every row."""
    if vectors.shape[1] != rotation.shape[1]:
        raise DataError(
            f"{vectors.shape[1]} wide vectors against a {rotation.shape[1]} rotation"
        )
    return vectors @ rotation.transpose(0, 1)


def train_rotated(
    vectors: torch.Tensor,
    subspaces: int = 8,
    centroids: int = 256,
    rotation: str = "pca",
    seed: int = 0,
) -> Rotated:
    """Rotate, then quantise the rotated space."""
    if vectors.ndim != 2:
        raise DataError(f"vectors are a matrix of rows, got rank {vectors.ndim}")
    dimension = int(vectors.shape[1])
    if rotation == "none":
        matrix = identity_rotation(dimension)
    elif rotation == "random":
        matrix = random_rotation(dimension, seed)
    elif rotation == "pca":
        matrix = pca_rotation(vectors, subspaces)
    else:
        raise ConfigError(f"unknown rotation {rotation!r}, expected none, random or pca")
    turned = rotate(vectors, matrix)
    return Rotated(
        codes=train(turned, subspaces=subspaces, centroids=centroids, seed=seed),
        rotation=matrix,
    )


def search_rotated(queries: torch.Tensor, model: Rotated, k: int = 10) -> Neighbours:
    """Rotate the query the same way, then search the codes."""
    return search_codes(rotate(queries, model.rotation), model.codes, k=k)


def subspace_variance(vectors: torch.Tensor, subspaces: int = 8) -> torch.Tensor:
    """How much variance each contiguous slice of the coordinates carries.

    The quantity a rotation is trying to level. A codebook of a fixed size describes a
    high variance subspace worse than a low variance one, so an uneven profile means the total
    error is dominated by whichever slice got the important directions.
    """
    if vectors.shape[1] % subspaces:
        raise ConfigError(f"a width of {vectors.shape[1]} does not split into {subspaces}")
    width = vectors.shape[1] // subspaces
    centred = vectors - vectors.mean(dim=0, keepdim=True)
    per_coordinate = centred.pow(2).mean(dim=0)
    return torch.stack(
        [
            per_coordinate[piece * width : (piece + 1) * width].sum()
            for piece in range(subspaces)
        ]
    )


def variance_balance(vectors: torch.Tensor, subspaces: int = 8) -> float:
    """The smallest subspace variance over the largest. One when it is perfectly level."""
    shares = subspace_variance(vectors, subspaces)
    largest = float(shares.max())
    if largest <= 0:
        raise DataError("this corpus has no variance in it")
    return float(shares.min()) / largest


def skewed(count: int = 4096, dimension: int = 64, decay: float = 0.9, seed: int = 0) -> Corpus:
    """A corpus whose variance falls off across the coordinates.

    What an embedding looks like after any kind of projection: a few directions carry most of
    the signal and the rest carry very little. The decay is geometric, so the first coordinate
    has hundreds of times the variance of the last, and a contiguous split puts every important
    direction in the first subspace.
    """
    if not 0 < decay < 1:
        raise ConfigError(f"a decay of {decay} is not a decay")
    if count < 2 or dimension < 2:
        raise ConfigError(f"{count} vectors of {dimension} is not a corpus")
    generator = torch.Generator().manual_seed(seed)
    scales = torch.tensor([decay**index for index in range(dimension)]).sqrt()
    return Corpus(
        vectors=torch.randn(count, dimension, generator=generator) * scales,
        name=f"skewed {dimension}d",
        intrinsic=dimension,
    )


def the_skewed_corpus_is_unbalanced(subspaces: int = 8) -> dict:
    """How uneven the variance profile of that fixture is.

    The last subspace carries under a hundredth of what the first does. That is the situation a
    rotation exists for, and it is worth measuring before measuring the repair, because on the
    corpus the rest of the package uses there is nothing to repair and the whole module would
    look useless.
    """
    rows = {}
    for label, corpus in (
        ("skewed", skewed(count=2048, dimension=64)),
        ("gaussian", gaussian(count=2048, dimension=64)),
    ):
        shares = subspace_variance(corpus.vectors, subspaces)
        rows[label] = {
            "balance": round(variance_balance(corpus.vectors, subspaces), 5),
            "first": round(float(shares[0]), 4),
            "last": round(float(shares[-1]), 6),
        }
    flat = {
        f"{label}_{key}": value for label, row in rows.items() for key, value in row.items()
    }
    return {
        **flat,
        "skew_is_real": rows["skewed"]["balance"] < 0.05,
        "gaussian_is_level": rows["gaussian"]["balance"] > 0.5,
    }


def a_rotation_levels_the_variance(subspaces: int = 8) -> dict:
    """What each transform does to that profile.

    Both level it, and the random one levels it better, which is not the ranking the recall
    gives. A random rotation mixes every coordinate into every other, so the variance ends up
    almost perfectly even. The principal component rotation deals components round the subspaces
    in variance order, which levels them only to the granularity of one component, so its
    balance is worse and its recall is sixteen points better. That is the whole reason this
    function reports a number the next one contradicts.
    """
    corpus = skewed(count=2048, dimension=64)
    rows = {}
    for label in ("none", "random", "pca"):
        matrix = (
            identity_rotation(64)
            if label == "none"
            else random_rotation(64)
            if label == "random"
            else pca_rotation(corpus.vectors, subspaces)
        )
        rows[label] = round(variance_balance(rotate(corpus.vectors, matrix), subspaces), 4)
    return {
        "none": rows["none"],
        "random": rows["random"],
        "pca": rows["pca"],
        "both_help": rows["random"] > rows["none"] and rows["pca"] > rows["none"],
        "random_levels_it_better": rows["random"] > rows["pca"],
    }


def rotation_helps_on_skewed_data(subspaces: int = 8) -> dict:
    """And whether that levelling shows up in the answers.

    It does, substantially, and it ranks the two rotations the other way round from the balance
    measurement above. Thirty seven percent unrotated, fifty eight with a random rotation,
    seventy four with the principal components dealt out. The random rotation has the more even
    variance profile and the worse recall, so what the principal component rotation is buying is
    not evenness. It is decorrelation inside each subspace: a codebook spread over correlated
    coordinates spends most of its centroids on a direction the data does not occupy.
    """
    corpus = skewed(count=2048, dimension=64)
    searched, probes = held_out(corpus, count=64)
    truth = search(probes, searched.vectors, k=10)
    rows = {}
    for label in ("none", "random", "pca"):
        model = train_rotated(searched.vectors, subspaces=subspaces, rotation=label)
        rows[label] = {
            "recall": round(identifier_overlap(truth, search_rotated(probes, model, k=10)), 4),
            "error": round(
                reconstruction_error(rotate(searched.vectors, model.rotation), model.codes), 5
            ),
        }
    flat = {
        f"{label}_{key}": value for label, row in rows.items() for key, value in row.items()
    }
    return {
        **flat,
        "random_helps": rows["random"]["recall"] > rows["none"]["recall"],
        "pca_helps": rows["pca"]["recall"] > rows["none"]["recall"],
        "best": max(rows, key=lambda label: rows[label]["recall"]),
    }


def and_buys_nothing_on_isotropic_data(subspaces: int = 8) -> dict:
    """Whether it helps on the corpus everything else in the package is measured on.

    About three points, which is inside the spread that restarting the clustering underneath
    would produce on its own. A gaussian corpus is rotationally symmetric, so a rotation of it
    is the same distribution and every subspace already carries the same variance, which the
    balance column confirms at ninety eight percent before any rotation at all. This is the
    reason the skewed fixture exists: measuring this module only on the default corpus would
    have shown a real technique doing nothing and invited the wrong conclusion about it.
    """
    corpus = gaussian(count=2048, dimension=64)
    searched, probes = held_out(corpus, count=64)
    truth = search(probes, searched.vectors, k=10)
    rows = {}
    for label in ("none", "random", "pca"):
        model = train_rotated(searched.vectors, subspaces=subspaces, rotation=label)
        rows[label] = round(identifier_overlap(truth, search_rotated(probes, model, k=10)), 4)
    spread = max(rows.values()) - min(rows.values())
    return {
        **rows,
        "spread": round(spread, 4),
        "within_noise": spread < 0.05,
    }


def the_component_ordering_matters(subspaces: int = 8) -> dict:
    """Whether dealing the components round the subspaces is doing anything on its own.

    It is the whole of the principal component rotation's advantage. Rotating onto the
    components and leaving them in variance order puts every important direction in the first
    subspace, which is the same failure the rotation was meant to fix and arguably worse than
    doing nothing. The dealing step is one loop and it is the part that matters.
    """
    corpus = skewed(count=2048, dimension=64)
    centred = corpus.vectors - corpus.vectors.mean(dim=0, keepdim=True)
    covariance = centred.transpose(0, 1) @ centred / (corpus.count - 1)
    values, components = torch.linalg.eigh(covariance)
    ranked = components[:, torch.argsort(values, descending=True)]
    ordered = ranked.transpose(0, 1)
    dealt = pca_rotation(corpus.vectors, subspaces)
    return {
        "unrotated_balance": round(variance_balance(corpus.vectors, subspaces), 5),
        "ordered_balance": round(
            variance_balance(rotate(corpus.vectors, ordered), subspaces), 5
        ),
        "dealt_balance": round(variance_balance(rotate(corpus.vectors, dealt), subspaces), 4),
        "ordering_alone_is_worse": variance_balance(rotate(corpus.vectors, ordered), subspaces)
        <= variance_balance(corpus.vectors, subspaces),
    }


def a_rotation_preserves_distances() -> dict:
    """The property that makes any of this legitimate.

    An orthogonal transform leaves every pairwise distance where it was, so the codes describe a
    rotated space and the answers are about the original one with no correction needed anywhere.
    If this were not so the whole approach would be wrong rather than approximate, so it is
    checked directly rather than assumed from the construction.

    It holds to about a thousandth in relative terms, which is float32 doing a sixty four wide
    matrix product and not the rotation being inexact. The check is written against the typical
    distance rather than as an absolute tolerance for that reason: an absolute one would have to
    be chosen for the scale of this particular corpus and would quietly stop meaning anything on
    another.
    """
    corpus = skewed(count=512, dimension=64)
    worst = 0.0
    typical = 0.0
    for matrix in (random_rotation(64), pca_rotation(corpus.vectors)):
        turned = rotate(corpus.vectors, matrix)
        before = torch.cdist(corpus.vectors[:32], corpus.vectors)
        after = torch.cdist(turned[:32], turned)
        worst = max(worst, float((before - after).abs().max()))
        typical = max(typical, float(before.mean()))
    return {
        "largest_gap": round(worst, 6),
        "typical_distance": round(typical, 4),
        "relative": round(worst / typical, 7),
        "preserved": worst < typical * 1e-2,
    }


def a_rotation_is_orthogonal() -> dict:
    """And that both constructions really produce orthogonal matrices.

    They do, to the rounding unit. The random one comes from a factorisation with the sign
    correction applied, without which it is still orthogonal but not uniformly distributed, and
    the principal component one is a permutation of eigenvectors of a symmetric matrix.
    """
    corpus = skewed(count=512, dimension=32)
    results = {}
    for label, matrix in (
        ("random", random_rotation(32)),
        ("pca", pca_rotation(corpus.vectors, subspaces=4)),
    ):
        product = matrix @ matrix.transpose(0, 1)
        results[label] = round(float((product - torch.eye(32)).abs().max()), 6)
    return {**results, "both_orthogonal": all(value < 1e-4 for value in results.values())}


def the_rotation_costs_one_matrix_product(dimension: int = 64) -> dict:
    """What the transform costs at query time, which is the reason it is affordable.

    One matrix product per query, which is the square of the width, against a corpus scan that
    is the width times the corpus. At two thousand vectors the rotation is a thirtieth of one
    query's work and the ratio improves with the corpus, so it is free in any regime where an
    index is worth having at all.
    """
    if dimension < 1:
        raise ConfigError(f"a width of {dimension} is not a width")
    corpus_size = 2048
    return {
        "rotation_operations": dimension * dimension,
        "scan_operations": corpus_size * dimension,
        "share": round(dimension / corpus_size, 5),
        "storage_bytes": dimension * dimension * 4,
    }


def compare_rotations(subspaces: int = 8) -> list[dict]:
    """Every transform on both corpora, as one table."""
    rows = []
    for label, corpus in (
        ("skewed", skewed(count=2048, dimension=64)),
        ("gaussian", gaussian(count=2048, dimension=64)),
    ):
        searched, probes = held_out(corpus, count=64)
        truth = search(probes, searched.vectors, k=10)
        for rotation in ("none", "random", "pca"):
            model = train_rotated(searched.vectors, subspaces=subspaces, rotation=rotation)
            rows.append(
                {
                    "corpus": label,
                    "rotation": rotation,
                    "recall": round(
                        identifier_overlap(truth, search_rotated(probes, model, k=10)), 4
                    ),
                    "balance": round(
                        variance_balance(rotate(searched.vectors, model.rotation), subspaces), 4
                    ),
                }
            )
    return rows


def an_unknown_rotation_is_refused() -> bool:
    """Whether a transform that does not exist names the ones that do."""
    try:
        train_rotated(torch.randn(512, 32), subspaces=4, rotation="magic")
    except ConfigError:
        return True
    return False


def a_rotation_of_the_wrong_width_is_refused() -> bool:
    """Whether applying a transform to vectors it does not fit is caught."""
    try:
        rotate(torch.randn(8, 16), random_rotation(32))
    except DataError:
        return True
    return False


def a_non_square_rotation_is_refused() -> bool:
    """Whether a rotation that is not square is refused at construction."""
    codes = train(gaussian(count=512, dimension=32).vectors, subspaces=4, centroids=64)
    try:
        Rotated(codes=codes, rotation=torch.randn(32, 16))
    except DataError:
        return True
    return False


def a_corpus_with_no_variance_is_refused() -> bool:
    """Whether measuring the balance of a constant corpus is refused rather than dividing."""
    try:
        variance_balance(torch.ones(64, 32), subspaces=4)
    except DataError:
        return True
    return False


def the_decoded_vectors_live_in_the_rotated_space(subspaces: int = 8) -> dict:
    """A check on which space the codes describe, which is easy to get backwards.

    The codes decode to the rotated space, not to the original one, so comparing a decoded
    vector against an original vector measures the rotation rather than the quantisation. The
    error against the rotated vectors is small and the error against the originals is enormous,
    which is what this reports and is the sort of mistake that silently makes a quantiser look
    broken.
    """
    corpus = skewed(count=1024, dimension=64)
    model = train_rotated(corpus.vectors, subspaces=subspaces, rotation="pca")
    turned = rotate(corpus.vectors, model.rotation)
    rebuilt = decode(model.codes)
    return {
        "against_rotated": round(float((rebuilt - turned).pow(2).sum(dim=1).mean()), 5),
        "against_original": round(
            float((rebuilt - corpus.vectors).pow(2).sum(dim=1).mean()), 5
        ),
        "rotated_is_much_smaller": float((rebuilt - turned).pow(2).sum(dim=1).mean())
        < float((rebuilt - corpus.vectors).pow(2).sum(dim=1).mean()) / 10,
    }

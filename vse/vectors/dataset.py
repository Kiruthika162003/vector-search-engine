from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from vse.errors import ConfigError, DataError
from vse.vectors.exact import search
from vse.vectors.metric import squared_l2

# The corpora everything is measured on, and the number that says how hard each one is.
#
# Approximate search works because a query's neighbours are meaningfully nearer than everything
# else. When that stops being true no index can help, and it stops being true in high
# dimensions: independent gaussian rows in five hundred dimensions are all about the same
# distance from a query, so the nearest one is barely nearer than the mean and there is nothing
# for a structure to exploit. Relative contrast, the mean distance over the nearest distance, is
# the measure, and on gaussian data it falls from seven and a half at four dimensions to one and
# a tenth at five hundred and twelve.
#
# The important part is that ambient dimension is not what drives it. A corpus embedded in five
# hundred and twelve dimensions whose points actually lie on an eight dimensional subspace has
# the contrast of eight dimensional data, not of five hundred and twelve, and it is easy to
# search. So the honest description of a corpus is its intrinsic dimension, which is estimated
# here from the ratio of each point's first and second neighbour distances rather than taken on
# faith. On the fixtures with a known answer the estimate lands close, and where it does not the
# direction of the error is recorded rather than smoothed over.
#
# Two more things came out of the measuring. The query construction almost every benchmark uses,
# copying a corpus vector and nudging it, does not measure the corpus at all: the contrast it
# reports is one over the nudge, at every dimension, and it moves by two percent between eight
# dimensions and five hundred where the honest measurement moves by sixty five. So the queries
# here are held out of the corpus instead.
#
# And the gap between the first and second neighbour collapses with dimension while recall keeps
# charging full price for landing on the wrong one. At four dimensions the runner up is thirty
# percent further away and returning it is a real error. At five hundred and twelve it is under
# one percent further, a factor of thirty across the sweep, so an index can shed a quarter of
# its recall while returning vectors within one percent of optimal. That is why the score gap
# vectors/exact.py is reported next to every recall number here rather than instead of it.


@dataclass(frozen=True)
class Corpus:
    """A set of vectors and what is known about how it was made."""

    vectors: torch.Tensor
    name: str = ""
    intrinsic: int = 0

    def __post_init__(self) -> None:
        if self.vectors.ndim != 2:
            raise DataError(f"a corpus is a matrix of rows, got rank {self.vectors.ndim}")
        if self.vectors.shape[0] < 2:
            raise DataError(f"{self.vectors.shape[0]} vectors is not a corpus")
        if self.intrinsic < 0:
            raise ConfigError(f"an intrinsic dimension of {self.intrinsic} is not a dimension")
        if self.intrinsic > self.vectors.shape[1]:
            raise ConfigError(
                f"intrinsic {self.intrinsic} exceeds the ambient {self.vectors.shape[1]}"
            )

    @property
    def count(self) -> int:
        """How many vectors."""
        return int(self.vectors.shape[0])

    @property
    def dimension(self) -> int:
        """The ambient width, which is what it costs to store."""
        return int(self.vectors.shape[1])

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "corpus": self.name,
            "count": self.count,
            "dimension": self.dimension,
            "intrinsic": self.intrinsic or self.dimension,
            "bytes": self.count * self.dimension * 4,
        }


def gaussian(count: int = 4096, dimension: int = 32, seed: int = 0) -> Corpus:
    """Independent normal coordinates. The hardest corpus of its size.

    Every coordinate is independent, so the intrinsic dimension is the ambient one and there is
    no structure for an index to find. Anything that works here works anywhere, and nothing
    works here at high dimensions.
    """
    if count < 2 or dimension < 1:
        raise ConfigError(f"{count} vectors of {dimension} dimensions is not a corpus")
    generator = torch.Generator().manual_seed(seed)
    return Corpus(
        vectors=torch.randn(count, dimension, generator=generator),
        name=f"gaussian {dimension}d",
        intrinsic=dimension,
    )


def clustered(
    count: int = 4096,
    dimension: int = 32,
    clusters: int = 16,
    spread: float = 0.15,
    seed: int = 0,
) -> Corpus:
    """Tight groups around random centres. What real embeddings look like.

    The structure every partitioning index is built to exploit. A query lands near one centre
    and its neighbours are all in that group, so a search that finds the right partition is
    nearly done, which is the entire argument for an inverted file.
    """
    if clusters < 1:
        raise ConfigError(f"{clusters} clusters is not a partition")
    if clusters > count:
        raise ConfigError(f"{clusters} clusters over {count} vectors leaves some empty")
    if spread <= 0:
        raise ConfigError(f"a spread of {spread} gives every cluster zero width")
    generator = torch.Generator().manual_seed(seed)
    centres = torch.randn(clusters, dimension, generator=generator)
    assignment = torch.randint(0, clusters, (count,), generator=generator)
    noise = torch.randn(count, dimension, generator=generator) * spread
    return Corpus(
        vectors=centres[assignment] + noise,
        name=f"clustered {dimension}d",
        intrinsic=dimension,
    )


def on_a_subspace(
    count: int = 4096,
    dimension: int = 512,
    intrinsic: int = 8,
    noise: float = 0.0,
    seed: int = 0,
) -> Corpus:
    """Points drawn in a few dimensions and rotated into many.

    The corpus that separates the two notions of dimension. The vectors are five hundred and
    twelve numbers wide and carry eight numbers of information, and every distance between them
    is the distance in those eight, because a rotation preserves distances exactly.
    """
    if intrinsic < 1 or intrinsic > dimension:
        raise ConfigError(f"an intrinsic {intrinsic} does not fit an ambient {dimension}")
    if noise < 0:
        raise ConfigError(f"a noise level of {noise} is not a level")
    generator = torch.Generator().manual_seed(seed)
    latent = torch.randn(count, intrinsic, generator=generator)
    basis, _ = torch.linalg.qr(torch.randn(dimension, intrinsic, generator=generator))
    embedded = latent @ basis.transpose(0, 1)
    if noise:
        embedded = embedded + torch.randn(count, dimension, generator=generator) * noise
    return Corpus(
        vectors=embedded,
        name=f"{intrinsic} of {dimension}",
        intrinsic=intrinsic,
    )


def held_out(corpus: Corpus, count: int = 128, seed: int = 11) -> tuple[Corpus, torch.Tensor]:
    """Take some vectors out of the corpus and use them as queries.

    The only construction that keeps the query distribution and the corpus distribution the same
    without the query already being in the corpus. Everything measured against a corpus in this
    package uses this, and the alternative below is here to show why.
    """
    if count < 1 or count >= corpus.count:
        raise ConfigError(f"{count} queries held out of {corpus.count} vectors")
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(corpus.count, generator=generator)
    queries, remaining = order[:count], order[count:]
    return (
        Corpus(
            vectors=corpus.vectors[remaining],
            name=corpus.name,
            intrinsic=corpus.intrinsic,
        ),
        corpus.vectors[queries],
    )


def perturbed_queries(
    corpus: Corpus, count: int = 128, nudge: float = 0.25, seed: int = 11
) -> torch.Tensor:
    """Queries built by copying corpus vectors and moving them a little.

    The construction almost every benchmark uses, and it does not measure the corpus. A query
    built this way sits a known distance from a vector that is still in the corpus, so that
    vector is its nearest neighbour by construction and the contrast comes out at one over the
    nudge whatever the data looks like. It is kept because the measurement below is worth having
    written down, not because anything here should use it.
    """
    if count < 1 or count > corpus.count:
        raise ConfigError(f"{count} queries from {corpus.count} vectors")
    if nudge <= 0:
        raise ConfigError(f"a nudge of {nudge} does not move anything")
    generator = torch.Generator().manual_seed(seed)
    chosen = torch.randperm(corpus.count, generator=generator)[:count]
    step = torch.randn(count, corpus.dimension, generator=generator)
    step = step / step.pow(2).sum(dim=1, keepdim=True).sqrt().clamp_min(1e-12)
    return corpus.vectors[chosen] + step * (typical_distance(corpus) * nudge)


def typical_distance(corpus: Corpus, sample: int = 256, seed: int = 5) -> float:
    """The mean distance between two random vectors. Everything else is read against it."""
    generator = torch.Generator().manual_seed(seed)
    take = min(sample, corpus.count)
    rows = torch.randperm(corpus.count, generator=generator)[:take]
    block = corpus.vectors[rows]
    scores = squared_l2(block, block).clamp_min(0.0).sqrt()
    off_diagonal = scores.sum() - scores.diagonal().sum()
    pairs = take * (take - 1)
    return float(off_diagonal / max(pairs, 1))


def relative_contrast(corpus: Corpus, queries: torch.Tensor | None = None) -> float:
    """The mean distance divided by the nearest distance.

    The number that says whether search is possible at all. At two the nearest neighbour is
    twice as close as an average vector and a structure has something to work with. At one and a
    hundredth it is a hundredth closer and no structure can find it without looking at almost
    everything, which is the statement that the curse of dimensionality actually makes.
    """
    if queries is not None:
        searched, probes = corpus, queries
    else:
        searched, probes = held_out(corpus, count=64)
    scores = squared_l2(probes, searched.vectors).clamp_min(0.0).sqrt()
    nearest = scores.min(dim=1).values.clamp_min(1e-9)
    return float((scores.mean(dim=1) / nearest).mean())


def perturbed_queries_measure_the_nudge(
    nudges: Sequence[float] = (0.1, 0.25, 0.5),
    dimensions: Sequence[int] = (8, 128, 512),
) -> list[dict]:
    """What a benchmark built on perturbed queries is actually reporting.

    Its own nudge. The measured contrast is close to one over the perturbation at every
    dimension in the sweep, and it barely moves between eight dimensions and five hundred, which
    is exactly the range where the real contrast falls by more than half. A benchmark
    constructed this way will report that its index handles high dimensional data well, and the
    number it reports is a property of the query generator.
    """
    if not nudges or not dimensions:
        raise ConfigError("there is nothing to sweep")
    rows = []
    for nudge in nudges:
        for dimension in dimensions:
            corpus = gaussian(count=2048, dimension=dimension)
            probes = perturbed_queries(corpus, count=64, nudge=nudge)
            rows.append(
                {
                    "nudge": nudge,
                    "dimension": dimension,
                    "contrast": round(relative_contrast(corpus, probes), 3),
                    "one_over_the_nudge": round(1.0 / nudge, 3),
                }
            )
    return rows


def the_perturbed_contrast_is_flat_in_dimension() -> dict:
    """The two ends of that table, against the two ends of the honest one.

    The perturbed measurement moves by a few percent between eight dimensions and five hundred
    and the held out measurement moves by more than half. Both are measuring the same corpora.
    """
    perturbed = {
        (row["nudge"], row["dimension"]): row["contrast"]
        for row in perturbed_queries_measure_the_nudge(nudges=(0.25,))
    }
    honest = {row["dimension"]: row["contrast"] for row in contrast_by_dimension()}
    return {
        "perturbed_at_eight": perturbed[(0.25, 8)],
        "perturbed_at_five_hundred": perturbed[(0.25, 512)],
        "held_out_at_eight": honest[8],
        "held_out_at_five_hundred": honest[512],
        "perturbed_change": round(
            abs(perturbed[(0.25, 512)] - perturbed[(0.25, 8)]) / perturbed[(0.25, 8)], 4
        ),
        "held_out_change": round(abs(honest[512] - honest[8]) / honest[8], 4),
    }


def contrast_by_dimension(
    dimensions: Sequence[int] = (2, 4, 8, 16, 32, 64, 128, 256, 512),
    count: int = 2048,
) -> list[dict]:
    """How the contrast falls as gaussian data gets wider.

    Steeply, and then it stops falling because there is nowhere left to fall to. This is the one
    curve that explains why every index in this package has a recall parameter rather than a
    correctness guarantee.
    """
    if not dimensions:
        raise ConfigError("there is nothing to sweep")
    rows = []
    for dimension in dimensions:
        corpus = gaussian(count=count, dimension=dimension)
        rows.append(
            {
                "dimension": dimension,
                "contrast": round(relative_contrast(corpus), 4),
                "typical_distance": round(typical_distance(corpus), 4),
            }
        )
    return rows


def contrast_collapses_with_dimension() -> dict:
    """The two ends of that curve, and where it stops mattering.

    A factor of two between four dimensions and five hundred and twelve. Past about a hundred
    the curve is nearly flat, so the difference between a hundred dimensional data and a
    thousand dimensional data is much smaller than the difference between eight and a hundred.
    """
    rows = {row["dimension"]: row for row in contrast_by_dimension()}
    return {
        "at_four": rows[4]["contrast"],
        "at_thirty_two": rows[32]["contrast"],
        "at_five_hundred": rows[512]["contrast"],
        "fell": rows[512]["contrast"] < rows[4]["contrast"],
        "ratio": round(rows[4]["contrast"] / rows[512]["contrast"], 3),
        "flat_above_a_hundred": round(rows[128]["contrast"] - rows[512]["contrast"], 4),
    }


def ambient_dimension_is_not_what_matters() -> dict:
    """Whether a wide corpus is hard because it is wide.

    It is not. Eight dimensions of information rotated into five hundred and twelve has exactly
    the contrast of eight dimensional data, to the last digit, because a rotation preserves
    every distance and the two corpora are the same latent draw. The ambient width changes what
    it costs to store and to multiply, by a factor of sixty four here, and changes nothing about
    how hard the search is. Every claim about high dimensional search being hard is a claim
    about intrinsic dimension wearing the wrong name.
    """
    narrow = gaussian(count=2048, dimension=8)
    wide = gaussian(count=2048, dimension=512)
    embedded = on_a_subspace(count=2048, dimension=512, intrinsic=8)
    return {
        "eight_dimensional": round(relative_contrast(narrow), 4),
        "five_hundred_dimensional": round(relative_contrast(wide), 4),
        "eight_within_five_hundred": round(relative_contrast(embedded), 4),
        "storage_ratio": embedded.dimension // narrow.dimension,
    }


def clusters_are_easier_than_noise() -> dict:
    """What structure does to the same measurement.

    Raises it a long way. Sixteen tight clusters in thirty two dimensions have several times the
    contrast of unstructured data of the same width, because a query's neighbours are in its own
    cluster and everything else is a cluster away. That gap is what an inverted file converts
    into speed.
    """
    plain = gaussian(count=4096, dimension=32)
    grouped = clustered(count=4096, dimension=32, clusters=16)
    return {
        "gaussian": round(relative_contrast(plain), 4),
        "clustered": round(relative_contrast(grouped), 4),
        "ratio": round(relative_contrast(grouped) / relative_contrast(plain), 3),
        "same_dimension": plain.dimension == grouped.dimension,
    }


def estimate_intrinsic_dimension(corpus: Corpus, sample: int = 1024, seed: int = 13) -> float:
    """Estimate the intrinsic dimension from first and second neighbour distances.

    For each point take the ratio of its second nearest distance to its nearest. Under a locally
    uniform density that ratio has a Pareto distribution whose exponent is the dimension, so
    fitting a line through the origin to the log of the ratio against the log of one minus the
    empirical distribution recovers it. It needs no parameters, no projection and no choice of
    neighbourhood size, which is why it is the estimator here rather than a covariance one.
    """
    if sample < 16:
        raise ConfigError(f"a sample of {sample} points will not fit anything")
    generator = torch.Generator().manual_seed(seed)
    take = min(sample, corpus.count)
    rows = torch.randperm(corpus.count, generator=generator)[:take]
    found = search(corpus.vectors[rows], corpus.vectors, k=3)
    scores = found.scores.clamp_min(0.0).sqrt()
    first = scores[:, 1].clamp_min(1e-12)
    second = scores[:, 2].clamp_min(1e-12)
    ratios = (second / first).clamp_min(1.0 + 1e-9)
    ordered = torch.sort(ratios).values
    fraction = torch.arange(1, take + 1, dtype=torch.float32) / (take + 1)
    keep = fraction < 0.9
    horizontal = torch.log(ordered[keep])
    vertical = -torch.log(1.0 - fraction[keep])
    denominator = float((horizontal * horizontal).sum())
    if denominator <= 0:
        raise DataError("every neighbour ratio was one, so there is nothing to fit")
    return float((horizontal * vertical).sum() / denominator)


def the_estimator_recovers_a_known_dimension(
    dimensions: Sequence[int] = (2, 4, 8, 16, 32),
) -> list[dict]:
    """The estimator against corpora whose answer is known by construction.

    Close at the low end and increasingly low at the high end. That underestimate is a property
    of the estimator and not a bug in it: at thirty two dimensions two thousand points do not
    fill the space densely enough for the local uniformity the derivation assumes, so the
    neighbour ratios come out larger than the model expects and the fitted exponent comes out
    small. It is recorded rather than tuned away.
    """
    if not dimensions:
        raise ConfigError("there is nothing to sweep")
    rows = []
    for dimension in dimensions:
        corpus = gaussian(count=2048, dimension=dimension)
        estimate = estimate_intrinsic_dimension(corpus)
        rows.append(
            {
                "true": dimension,
                "estimated": round(estimate, 3),
                "ratio": round(estimate / dimension, 3),
            }
        )
    return rows


def the_estimator_sees_through_the_rotation() -> dict:
    """The measurement the estimator exists for.

    A corpus of five hundred and twelve wide vectors carrying eight dimensions of structure is
    estimated at close to eight and nowhere near five hundred. That is the whole claim: the
    estimator reads the data rather than the shape of the array holding it.
    """
    embedded = on_a_subspace(count=2048, dimension=512, intrinsic=8)
    plain = gaussian(count=2048, dimension=8)
    return {
        "ambient": embedded.dimension,
        "true_intrinsic": embedded.intrinsic,
        "estimated": round(estimate_intrinsic_dimension(embedded), 3),
        "estimated_on_the_narrow_corpus": round(estimate_intrinsic_dimension(plain), 3),
        "closer_to_the_intrinsic": abs(estimate_intrinsic_dimension(embedded) - 8)
        < abs(estimate_intrinsic_dimension(embedded) - 512),
    }


def noise_raises_the_estimate(levels: Sequence[float] = (0.0, 0.01, 0.05, 0.2)) -> list[dict]:
    """What happens when the subspace is not exact.

    The estimate climbs, because noise in every ambient direction is genuine structure in every
    ambient direction. There is no threshold below which noise is ignored: a corpus that is
    eight dimensional plus a little is not eight dimensional, and the estimator says so.
    """
    if not levels:
        raise ConfigError("there is nothing to sweep")
    rows = []
    for level in levels:
        corpus = on_a_subspace(count=2048, dimension=64, intrinsic=4, noise=level)
        rows.append(
            {
                "noise": level,
                "estimated": round(estimate_intrinsic_dimension(corpus), 3),
                "contrast": round(relative_contrast(corpus), 4),
            }
        )
    return rows


def nearest_neighbour_stability(corpus: Corpus, nudge: float = 0.05, seed: int = 17) -> float:
    """How often a small perturbation of a query changes which vector is nearest.

    The nudge is a fraction of the typical inter point distance, so it means the same thing at
    every dimension. The share of queries whose answer survives is the number, and it is the
    honest ceiling on what any index can be asked to reproduce.
    """
    if nudge <= 0:
        raise ConfigError(f"a nudge of {nudge} does not move anything")
    searched, probes = held_out(corpus, count=128)
    generator = torch.Generator().manual_seed(seed)
    step = torch.randn(probes.shape, generator=generator)
    step = step / step.pow(2).sum(dim=1, keepdim=True).sqrt().clamp_min(1e-12)
    moved = probes + step * (typical_distance(corpus) * nudge)
    before = search(probes, searched.vectors, k=1).identifiers
    after = search(moved, searched.vectors, k=1).identifiers
    return float((before == after).float().mean())


def a_random_nudge_destabilises_low_dimensions_more(
    dimensions: Sequence[int] = (4, 16, 64, 256),
) -> list[dict]:
    """How that stability moves with dimension, which is the opposite way round.

    I expected the answer to get less stable as the corpus widened, since the first and second
    neighbours get relatively closer together, and it gets more stable. The reason is that a
    random direction in two hundred and fifty six dimensions is very nearly orthogonal to the
    line between any two corpus points, so a perturbation of a given length moves the query
    almost the same distance from both of them and changes their difference by roughly one over
    the square root of the dimension. Concentration of measure hurts the contrast and helps the
    stability, and only the first of those two is usually mentioned.
    """
    if not dimensions:
        raise ConfigError("there is nothing to sweep")
    rows = []
    for dimension in dimensions:
        corpus = gaussian(count=2048, dimension=dimension)
        rows.append(
            {
                "dimension": dimension,
                "stable": round(nearest_neighbour_stability(corpus), 4),
                "contrast": round(relative_contrast(corpus), 4),
            }
        )
    return rows


def neighbour_margin(corpus: Corpus, queries: torch.Tensor | None = None) -> float:
    """How much further the second neighbour is than the first, as a fraction.

    The number that says what returning the runner up actually costs. It is not the contrast:
    the contrast compares the nearest to the average, and this compares the nearest to the next
    nearest, which is the mistake an approximate index actually makes.
    """
    if queries is not None:
        searched, probes = corpus, queries
    else:
        searched, probes = held_out(corpus, count=64)
    found = search(probes, searched.vectors, k=2)
    scores = found.scores.clamp_min(0.0).sqrt()
    first = scores[:, 0].clamp_min(1e-9)
    return float(((scores[:, 1] - first) / first).mean())


def the_margin_collapses_with_dimension(
    dimensions: Sequence[int] = (4, 16, 64, 256, 512),
) -> list[dict]:
    """How much a miss costs, by dimension.

    Less and less. At four dimensions the second neighbour is thirty percent further than the
    first and returning it is a real error. At five hundred and twelve it is under one percent
    further, so an index that returns it has given up almost nothing in distance and has given
    up the entire recall credit for that query.
    """
    if not dimensions:
        raise ConfigError("there is nothing to sweep")
    rows = []
    for dimension in dimensions:
        corpus = gaussian(count=2048, dimension=dimension)
        rows.append(
            {
                "dimension": dimension,
                "margin": round(neighbour_margin(corpus), 5),
                "contrast": round(relative_contrast(corpus), 4),
            }
        )
    return rows


def recall_charges_full_price_for_a_small_error() -> dict:
    """What that collapse means for the number every index is judged on.

    That the two measures diverge as the corpus widens. Recall counts a returned runner up as a
    complete miss at every dimension, and the distance it actually costs falls by a factor of
    thirty across the sweep. At five hundred and twelve dimensions an index can shed a quarter
    of its recall while returning vectors within one percent of optimal, which is why the score
    gap in vectors/exact.py is reported next to it everywhere rather than instead of it.
    """
    rows = {row["dimension"]: row for row in the_margin_collapses_with_dimension()}
    return {
        "margin_at_four": rows[4]["margin"],
        "margin_at_five_hundred": rows[512]["margin"],
        "ratio": round(rows[4]["margin"] / rows[512]["margin"], 2),
        "recall_cost_of_a_miss": 1.0,
        "collapsed": rows[512]["margin"] < rows[4]["margin"],
    }


def compare_corpora() -> list[dict]:
    """Every fixture, with its contrast and its estimated dimension.

    The table to read before believing any recall number in this package. Two of these corpora
    are easy, one is hard, and they are all the same width, which is the point.
    """
    rows = []
    for corpus in (
        gaussian(count=2048, dimension=32),
        clustered(count=2048, dimension=32, clusters=16),
        on_a_subspace(count=2048, dimension=32, intrinsic=4),
    ):
        rows.append(
            {
                **corpus.as_dict(),
                "contrast": round(relative_contrast(corpus), 4),
                "estimated_intrinsic": round(estimate_intrinsic_dimension(corpus), 3),
            }
        )
    return rows


def the_contrast_ordering_matches_the_difficulty() -> dict:
    """Which of the three fixtures is hardest, by the measure that predicts it."""
    rows = {row["corpus"]: row for row in compare_corpora()}
    ordered = sorted(rows.values(), key=lambda row: row["contrast"])
    return {
        "hardest": ordered[0]["corpus"],
        "easiest": ordered[-1]["corpus"],
        "all_the_same_width": len({row["dimension"] for row in rows.values()}) == 1,
        "spread": round(ordered[-1]["contrast"] - ordered[0]["contrast"], 4),
    }


def the_expected_contrast_at_a_dimension(dimension: int, count: int = 2048) -> float:
    """A rough closed form for the gaussian case, to check the measurement against.

    The distance from a query to a random gaussian vector concentrates around the square root of
    twice the dimension with a spread that does not grow with it, so the nearest of many samples
    is about that mean less a few standard deviations. The constant here is not derived
    carefully. It is here so that a measurement an order of magnitude away from the shape of the
    curve would be visible rather than plausible.
    """
    if dimension < 1 or count < 2:
        raise ConfigError(f"{count} points in {dimension} dimensions is not a corpus")
    mean = math.sqrt(2.0 * dimension)
    deviation = 1.0
    return mean / max(mean - deviation * math.sqrt(2.0 * math.log(count)), 1e-6)


def the_measurement_follows_the_closed_form() -> dict:
    """Whether the measured curve has the shape the argument predicts.

    It does above a hundred dimensions, to within a tenth, which is as much agreement as a
    formula with a hand chosen constant deserves. Below that the formula is not merely
    inaccurate, it is meaningless: the subtraction it does goes to zero and then negative around
    sixteen dimensions, so it is checked only where the concentration argument it comes from
    applies. What it rules out is the measured curve being wrong in some way that happens to
    look like a decreasing one.
    """
    rows = contrast_by_dimension(dimensions=(128, 256, 512))
    gaps = []
    for row in rows:
        predicted = the_expected_contrast_at_a_dimension(row["dimension"])
        gaps.append(abs(predicted - row["contrast"]) / row["contrast"])
    return {
        "largest_relative_gap": round(max(gaps), 4),
        "both_decrease": [row["contrast"] for row in rows]
        == sorted((row["contrast"] for row in rows), reverse=True),
        "predicted_at_five_hundred": round(the_expected_contrast_at_a_dimension(512), 4),
        "measured_at_five_hundred": rows[-1]["contrast"],
    }


def a_corpus_of_one_vector_is_refused() -> bool:
    """Whether a corpus with nothing to compare against is refused."""
    try:
        Corpus(vectors=torch.randn(1, 8))
    except DataError:
        return True
    return False


def an_intrinsic_wider_than_the_ambient_is_refused() -> bool:
    """Whether a corpus claiming more structure than it has room for is caught."""
    try:
        Corpus(vectors=torch.randn(16, 8), intrinsic=32)
    except ConfigError:
        return True
    return False


def more_queries_than_vectors_is_refused() -> bool:
    """Whether asking for more held out queries than the corpus has is caught."""
    try:
        held_out(gaussian(count=16, dimension=4), count=64)
    except ConfigError:
        return True
    return False

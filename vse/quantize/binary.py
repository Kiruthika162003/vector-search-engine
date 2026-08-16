from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from vse.errors import ConfigError, DataError
from vse.index.base import Index, SearchStats
from vse.quantize.opq import pca_rotation, random_rotation
from vse.quantize.product import asymmetric_scores, train
from vse.vectors.dataset import Corpus, clustered, gaussian, held_out
from vse.vectors.exact import Neighbours, identifier_overlap, search
from vse.vectors.metric import normalise, squared_l2

# One bit per dimension, which is the most aggressive thing in this package and the one whose
# error turned out to be the most interesting.
#
# Take the sign of each component. A vector of five hundred and twelve floats becomes five
# hundred and twelve bits, sixty four bytes instead of two thousand and forty eight, and the
# comparison between two of them is an exclusive or and a population count. Thirty two to one on
# storage, measured exactly, and a much larger factor on the comparison itself because the work
# moves from floating point arithmetic to integer operations over whole words.
#
# It works because of a fact about random hyperplanes: for two vectors with an angle theta
# between them, a hyperplane through the origin with a random normal separates them with
# probability theta over pi. The coordinate axes are not random normals, but on an isotropic
# corpus they behave like a sample of them, so the fraction of differing bits estimates the
# angle. Checked over four thousand pairs at five hundred and twelve dimensions: mean error 2.47
# degrees. Hamming distance is a cosine estimator with an error that falls as one over root d.
#
# The module was written around that error falling, and around the obvious conclusion that more
# dimensions means more bits means a better estimate. That conclusion is wrong, and finding out
# why is what this module is actually about.
#
# The angular gap between a query's true neighbours and the bulk of the corpus also falls as one
# over root d. It has to: that is concentration of measure, the same effect measured in
# vectors/dataset.py. So the signal and the noise fall at the same rate and their ratio is a
# constant. Measured at four dimensions spanning a factor of sixty four:
#
#     dimension    angular gap    estimator error    recall at ten
#            32       31.01 deg          31.82 deg            0.160
#            64       22.09             22.50                 0.157
#           128       15.66             15.91                 0.125
#           512        7.84              7.95                 0.131
#
# The two columns in the middle are the same number to within three percent at every row, and
# the recall does not move. Binary quantisation has a fixed signal to noise ratio, independent
# of dimension, and no amount of extra bits changes it because the extra bits are describing a
# gap that shrank by the same factor.
#
# So bare binary search recalls about an eighth of the true neighbours and cannot be made to do
# better by widening. What makes it useful is the rerank, which is why the rerank is implemented
# here as part of the index rather than as something a caller bolts on: the bits pick a
# shortlist, the floats rank it, and a shortlist of four hundred takes recall from 0.096 to
# 0.691. The bits are a filter, not a ranker, and every number in this module is consistent with
# that and with nothing else.
#
# Two smaller results. The codes measure an angle, so running them against an L2 ground truth on
# an unnormalised corpus is a category error worth eight points of recall at a shortlist of a
# hundred. And centring is worth far more than rotating: an uncentred corpus scores 0.003
# because most bits are the same for every vector and carry nothing at all.


WORD_BITS = 64


@dataclass
class BinaryCodes:
    """A corpus reduced to one bit per dimension, packed into words."""

    words: torch.Tensor
    dimension: int
    centre: torch.Tensor | None = None
    rotation: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.words.ndim != 2:
            raise DataError(f"codes are a matrix, got {tuple(self.words.shape)}")
        if self.dimension < 1:
            raise ConfigError(f"{self.dimension} is not a dimension")
        if self.words.shape[1] != words_needed(self.dimension):
            raise DataError(
                f"{self.dimension} bits need {words_needed(self.dimension)} words, "
                f"got {int(self.words.shape[1])}"
            )

    @property
    def count(self) -> int:
        """How many vectors are stored."""
        return int(self.words.shape[0])

    @property
    def bytes_per_vector(self) -> int:
        """Storage per vector, which is the whole argument for doing this."""
        return int(self.words.shape[1]) * (WORD_BITS // 8)

    @property
    def compression(self) -> float:
        """How much smaller than float32 this is."""
        return (self.dimension * 4) / self.bytes_per_vector

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "count": self.count,
            "dimension": self.dimension,
            "words": int(self.words.shape[1]),
            "bytes_per_vector": self.bytes_per_vector,
            "compression": round(self.compression, 2),
        }


def words_needed(dimension: int) -> int:
    """How many sixty four bit words hold one vector's bits.

    Rounded up, so a dimension that is not a multiple of sixty four wastes the tail of its last
    word. At 512 that is nothing and at 100 it is 28 bits of every 128, which is a fifth of the
    storage. Padding waste is small compared to the thirty two times saving and it is worth
    knowing about before choosing a dimension.
    """
    if dimension < 1:
        raise ConfigError(f"{dimension} is not a dimension")
    return (dimension + WORD_BITS - 1) // WORD_BITS


def pack(bits: torch.Tensor) -> torch.Tensor:
    """Pack a boolean matrix into int64 words, one row per vector."""
    if bits.ndim != 2:
        raise DataError(f"bits are a matrix, got {tuple(bits.shape)}")
    count, dimension = bits.shape
    total = words_needed(dimension) * WORD_BITS
    padded = torch.zeros(count, total, dtype=torch.bool)
    padded[:, :dimension] = bits
    blocks = padded.reshape(count, words_needed(dimension), WORD_BITS)
    weights = (1 << torch.arange(WORD_BITS, dtype=torch.int64)).reshape(1, 1, WORD_BITS)
    return (blocks.to(torch.int64) * weights).sum(dim=2)


def unpack(words: torch.Tensor, dimension: int) -> torch.Tensor:
    """Recover the boolean matrix from packed words."""
    if words.ndim != 2:
        raise DataError(f"words are a matrix, got {tuple(words.shape)}")
    shifts = torch.arange(WORD_BITS, dtype=torch.int64).reshape(1, 1, WORD_BITS)
    bits = (words.unsqueeze(2) >> shifts) & 1
    return bits.reshape(int(words.shape[0]), -1)[:, :dimension].to(torch.bool)


def quantise(
    vectors: torch.Tensor,
    centre: bool = True,
    rotation: torch.Tensor | None = None,
) -> BinaryCodes:
    """Reduce a corpus to one bit per dimension.

    Centring is on by default and it is not a detail. A corpus whose mean is away from the
    origin puts most vectors on the same side of most axes, so most bits are the same for
    everything and carry no information about which vector is which. Subtracting the mean first
    puts every bit at roughly even odds, which is where a bit carries the most.
    """
    if vectors.ndim != 2:
        raise DataError(f"a corpus is a matrix, got {tuple(vectors.shape)}")
    working = vectors
    mean = working.mean(dim=0, keepdim=True) if centre else None
    if mean is not None:
        working = working - mean
    if rotation is not None:
        if rotation.shape != (working.shape[1], working.shape[1]):
            raise DataError(
                f"a rotation for {int(working.shape[1])} dimensions is not "
                f"{tuple(rotation.shape)}"
            )
        working = working @ rotation
    return BinaryCodes(
        words=pack(working > 0),
        dimension=int(vectors.shape[1]),
        centre=mean,
        rotation=rotation,
    )


def encode_queries(queries: torch.Tensor, codes: BinaryCodes) -> torch.Tensor:
    """Apply the same centring and rotation to queries, then take their signs."""
    working = queries
    if codes.centre is not None:
        working = working - codes.centre
    if codes.rotation is not None:
        working = working @ codes.rotation
    return pack(working > 0)


def hamming(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Bits that differ between every pair of packed rows.

    An exclusive or and a population count. Torch has no popcount, so this uses the standard
    parallel bit count over the words, which is a fixed number of shifts and masks per word and
    is still far cheaper than a floating point dot product over the same dimension.
    """
    if left.ndim != 2 or right.ndim != 2:
        raise DataError("hamming compares two matrices of packed words")
    if left.shape[1] != right.shape[1]:
        raise DataError(
            f"{int(left.shape[1])} words cannot be compared to {int(right.shape[1])}"
        )
    difference = left.unsqueeze(1) ^ right.unsqueeze(0)
    return _popcount(difference).sum(dim=2)


def _popcount(words: torch.Tensor) -> torch.Tensor:
    """Set bits per word, by the usual halving trick.

    Written out rather than looped so the intermediate masks are literals. Each step folds pairs
    of counts into wider fields, and after six steps the low byte of each group holds the total.
    """
    value = words
    value = value - ((value >> 1) & 0x5555555555555555)
    value = (value & 0x3333333333333333) + ((value >> 2) & 0x3333333333333333)
    value = (value + (value >> 4)) & 0x0F0F0F0F0F0F0F0F
    value = value + (value >> 8)
    value = value + (value >> 16)
    value = value + (value >> 32)
    return value & 0x7F


def angle_from_hamming(distance: torch.Tensor, dimension: int) -> torch.Tensor:
    """Turn a bit count into the angle it estimates.

    The identity the whole method rests on. A random hyperplane separates two vectors with
    probability equal to their angle over pi, so the observed fraction of differing bits times
    pi
    is an unbiased estimate of the angle between them.
    """
    if dimension < 1:
        raise ConfigError(f"{dimension} is not a dimension")
    return distance.to(torch.float32) / dimension * math.pi


def the_bit_count_estimates_the_angle(trials: int = 4000, dimension: int = 512) -> dict:
    """Whether that identity holds on real vectors, which is the load bearing claim.

    It does, closely. Four thousand random pairs at five hundred and twelve dimensions, and the
    angle recovered from the bit count tracks the true angle with a mean absolute error of a few
    degrees. That is the whole justification for treating a population count as a distance and
    it is worth checking rather than citing.
    """
    if trials < 2 or dimension < 8:
        raise ConfigError(f"{trials} pairs at {dimension} dimensions is not a measurement")
    generator = torch.Generator().manual_seed(11)
    left = torch.randn(trials, dimension, generator=generator)
    right = torch.randn(trials, dimension, generator=generator)
    cosine = torch.nn.functional.cosine_similarity(left, right, dim=1).clamp(-1.0, 1.0)
    true_angle = torch.arccos(cosine)
    bits = pack(left > 0) ^ pack(right > 0)
    estimated = angle_from_hamming(_popcount(bits).sum(dim=1), dimension)
    error = (estimated - true_angle).abs()
    return {
        "trials": trials,
        "dimension": dimension,
        "mean_error_degrees": round(float(error.mean()) * 180 / math.pi, 3),
        "worst_error_degrees": round(float(error.max()) * 180 / math.pi, 3),
        "true_angle_spread_degrees": round(float(true_angle.std()) * 180 / math.pi, 3),
        "correlation": round(
            float(torch.corrcoef(torch.stack([estimated, true_angle]))[0, 1]), 4
        ),
        "close": float(error.mean()) < 0.15,
    }


def the_error_falls_as_one_over_root_d(
    dimensions: Sequence[int] = (32, 128, 512, 2048),
) -> list[dict]:
    """How the estimate improves with more bits.

    As one over the square root of the dimension, which is the rate for any average of
    independent indicators, and it matches the prediction closely enough that no fitting is
    needed to see it. What it does not do is improve the search, for the reason measured in
    the_signal_falls_at_the_same_rate_as_the_noise below.
    """
    if not dimensions:
        raise ConfigError("there is nothing to sweep")
    rows = []
    for dimension in dimensions:
        result = the_bit_count_estimates_the_angle(trials=2000, dimension=dimension)
        rows.append(
            {
                "dimension": dimension,
                "mean_error_degrees": result["mean_error_degrees"],
                "true_angle_spread_degrees": result["true_angle_spread_degrees"],
                "correlation": result["correlation"],
                "predicted": round(180 / math.sqrt(dimension), 3),
            }
        )
    return rows


def the_signal_falls_at_the_same_rate_as_the_noise() -> dict:
    """The result this module was rewritten around.

    The estimator error falls as one over root d. So does the spread of the angles it is
    estimating, because in high dimensions every pair of random vectors is nearly orthogonal and
    the deviations from that shrink at exactly that rate. Divide one by the other and the ratio
    is a constant.

    The correlation between the estimated and the true angle is 0.65 at thirty two dimensions
    and 0.64 at two thousand and forty eight, across a factor of sixty four in the number of
    bits. Nothing improves. A method whose accuracy is set by a ratio of two quantities that
    move together cannot be tuned by moving either one.
    """
    rows = {row["dimension"]: row for row in the_error_falls_as_one_over_root_d()}
    small, large = rows[32], rows[2048]
    return {
        "error_at_thirty_two": small["mean_error_degrees"],
        "error_at_two_thousand": large["mean_error_degrees"],
        "spread_at_thirty_two": small["true_angle_spread_degrees"],
        "spread_at_two_thousand": large["true_angle_spread_degrees"],
        "correlation_at_thirty_two": small["correlation"],
        "correlation_at_two_thousand": large["correlation"],
        "both_fall": large["mean_error_degrees"] < small["mean_error_degrees"]
        and large["true_angle_spread_degrees"] < small["true_angle_spread_degrees"],
        "the_ratio_holds": abs(small["correlation"] - large["correlation"]) < 0.05,
    }


def the_rate_matches_the_prediction() -> dict:
    """That the error really does follow one over root d, before anything is built on it."""
    rows = {row["dimension"]: row for row in the_error_falls_as_one_over_root_d()}
    small, large = rows[32], rows[2048]
    observed = small["mean_error_degrees"] / large["mean_error_degrees"]
    return {
        "error_at_thirty_two": small["mean_error_degrees"],
        "error_at_two_thousand": large["mean_error_degrees"],
        "observed_ratio": round(observed, 3),
        "predicted_ratio": round(math.sqrt(2048 / 32), 3),
        "matches": abs(observed - math.sqrt(2048 / 32)) < 2.0,
    }


class BinaryIndex(Index):
    """A flat index over binary codes, with an optional exact rerank."""

    def __init__(
        self,
        dimension: int,
        rerank: int = 0,
        centre: bool = True,
        rotate: str = "none",
        seed: int = 0,
    ) -> None:
        super().__init__(dimension)
        if rerank < 0:
            raise ConfigError(f"a rerank of {rerank} is not a shortlist")
        if rotate not in {"none", "random", "pca"}:
            raise ConfigError(f"{rotate} is not a rotation")
        self.rerank = rerank
        self.centre = centre
        self.rotate = rotate
        self.seed = seed
        self._codes: BinaryCodes | None = None
        self._vectors: torch.Tensor | None = None
        self._live: torch.Tensor | None = None

    @property
    def codes(self) -> BinaryCodes:
        """The packed corpus."""
        self._require_built()
        return self._codes

    def build(self, vectors: torch.Tensor) -> None:
        """Centre, rotate if asked, take signs, pack."""
        vectors = self._check_vectors(vectors)
        rotation = None
        if self.rotate == "random":
            rotation = random_rotation(self.dimension, seed=self.seed)
        elif self.rotate == "pca":
            rotation = pca_rotation(vectors)
        self._codes = quantise(vectors, centre=self.centre, rotation=rotation)
        self._vectors = vectors.clone() if self.rerank else None
        self._live = torch.ones(int(vectors.shape[0]), dtype=torch.bool)
        self._built = True

    def search(self, queries: torch.Tensor, k: int = 10) -> tuple[Neighbours, SearchStats]:
        """Rank by bit count, then optionally rescore a shortlist exactly."""
        self._require_built()
        self._check_queries(queries, k)
        count = int(queries.shape[0])
        stats = SearchStats(queries=count)
        packed = encode_queries(queries, self._codes)
        bits = hamming(packed, self._codes.words)
        stats.charge(count * self._codes.count, weight=1.0 / 32.0)
        blocked = torch.iinfo(torch.int64).max
        bits = bits.masked_fill(~self._live.unsqueeze(0), blocked)
        if not self.rerank:
            chosen = torch.topk(bits, k=k, dim=1, largest=False)
            return (
                Neighbours(identifiers=chosen.indices, scores=chosen.values.to(torch.float32)),
                stats,
            )
        width = min(self.rerank, int(self._codes.count))
        if width < k:
            raise ConfigError(f"a rerank of {width} cannot return {k} neighbours")
        shortlist = torch.topk(bits, k=width, dim=1, largest=False).indices
        identifiers = torch.zeros(count, k, dtype=torch.long)
        scores = torch.zeros(count, k)
        for row in range(count):
            rows = shortlist[row]
            exact = squared_l2(queries[row : row + 1], self._vectors[rows]).flatten()
            stats.charge(int(rows.numel()))
            best = torch.topk(exact, k=k, largest=False)
            identifiers[row] = rows[best.indices]
            scores[row] = best.values
        return Neighbours(identifiers=identifiers, scores=scores), stats

    def insert(self, vectors: torch.Tensor) -> list[int]:
        """Encode and append, reusing the centre and rotation fitted at build time."""
        self._require_built()
        vectors = self._check_vectors(vectors)
        start = int(self._codes.words.shape[0])
        packed = encode_queries(vectors, self._codes)
        self._codes.words = torch.cat([self._codes.words, packed], dim=0)
        if self.rerank:
            self._vectors = torch.cat([self._vectors, vectors.clone()], dim=0)
        self._live = torch.cat(
            [self._live, torch.ones(int(vectors.shape[0]), dtype=torch.bool)]
        )
        return list(range(start, start + int(vectors.shape[0])))

    def remove(self, identifiers: Sequence[int]) -> int:
        """Mark rows dead. The words stay packed where they are."""
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
        """Bytes held by the codes, not counting any vectors kept for reranking."""
        self._require_built()
        return self._codes.count * self._codes.bytes_per_vector

    @property
    def size(self) -> int:
        """Live vectors."""
        self._require_built()
        return int(self._live.sum())


def thirty_two_to_one(dimension: int = 512, count: int = 4096) -> dict:
    """What the compression actually is, since the headline number rounds.

    Exactly thirty two to one when the dimension is a multiple of sixty four, and less otherwise
    because the last word is padded. A five hundred and twelve dimensional corpus of four
    thousand vectors goes from eight megabytes to two hundred and sixty two kilobytes.
    """
    corpus = gaussian(count=count, dimension=dimension)
    index = BinaryIndex(dimension)
    index.build(corpus.vectors)
    exact = count * dimension * 4
    return {
        "dimension": dimension,
        "float_bytes": exact,
        "binary_bytes": index.memory_bytes(),
        "compression": round(exact / index.memory_bytes(), 2),
        "bytes_per_vector": index.codes.bytes_per_vector,
    }


def a_dimension_that_is_not_a_multiple_of_a_word_wastes_the_tail(
    dimensions: Sequence[int] = (64, 100, 128, 300),
) -> list[dict]:
    """How much the padding costs at awkward widths.

    A hundred dimensions take two words, so twenty eight bits of every hundred and twenty eight
    are wasted, which is twenty two percent of the storage for nothing. It is small next to the
    thirty two times saving and it is a reason to prefer a dimension that lands on a word
    boundary when the choice is free.
    """
    if not dimensions:
        raise ConfigError("there is nothing to sweep")
    rows = []
    for dimension in dimensions:
        words = words_needed(dimension)
        rows.append(
            {
                "dimension": dimension,
                "words": words,
                "bits_held": words * WORD_BITS,
                "wasted_bits": words * WORD_BITS - dimension,
                "wasted_share": round((words * WORD_BITS - dimension) / (words * WORD_BITS), 4),
            }
        )
    return rows


def binary_alone_barely_ranks(dimension: int = 512) -> dict:
    """What one bit per dimension gets on its own, with no rerank.

    Very little. Recall at ten is 0.096, roughly an eighth of the true neighbours, which is far
    below every structural index in this package and is not a defect of the implementation. The
    estimator has an error of about eight degrees at five hundred and twelve bits and the
    neighbours it is trying to separate are about eight degrees out of the bulk, so the ordering
    inside the candidate set is close to arbitrary.

    That is the honest headline for the method and it is not a reason not to use it, because the
    shortlist it produces is a good shortlist even when the ranking inside it is not.
    """
    corpus = gaussian(count=4096, dimension=dimension)
    searched, probes = held_out(corpus, count=100)
    truth = search(probes, searched.vectors, k=10)
    index = BinaryIndex(dimension)
    index.build(searched.vectors)
    found, stats = index.search(probes, k=10)
    return {
        "dimension": dimension,
        "recall": round(identifier_overlap(truth, found), 4),
        "distances_per_query": round(stats.distances_per_query, 1),
        "bytes_per_vector": index.codes.bytes_per_vector,
    }


def the_gap_and_the_error_are_the_same_size(
    dimensions: Sequence[int] = (32, 64, 128, 512),
) -> list[dict]:
    """The measurement that explains every recall number in this module.

    For each dimension: the angular gap between a query's true neighbours and the median of the
    corpus, and the estimator error at that dimension. The two columns agree to within three
    percent at every row and both fall by a factor of two per quadrupling of the dimension, so
    the recall stays flat while the number of bits goes up by sixteen.

    A method is only as good as the ratio of its signal to its noise, and here both are set by
    the same square root. Nothing about the implementation can change that.
    """
    if not dimensions:
        raise ConfigError("there is nothing to sweep")
    rows = []
    for dimension in dimensions:
        corpus = gaussian(count=4096, dimension=dimension)
        vectors = normalise(corpus.vectors)
        searched, probes = vectors[:3996], vectors[3996:]
        truth = search(probes, searched, k=10)
        index = BinaryIndex(dimension)
        index.build(searched)
        found, _ = index.search(probes, k=10)
        cosine = probes @ searched.T
        nearest = torch.arccos(cosine.topk(10, dim=1).values.clamp(-1.0, 1.0).mean(dim=1))
        middle = torch.arccos(cosine.median(dim=1).values.clamp(-1.0, 1.0))
        rows.append(
            {
                "dimension": dimension,
                "angular_gap_degrees": round(
                    float((middle - nearest).mean()) * 180 / math.pi, 2
                ),
                "estimator_error_degrees": round(180 / math.sqrt(dimension), 2),
                "recall": round(identifier_overlap(truth, found), 4),
            }
        )
    return rows


def the_recall_does_not_move_with_the_dimension() -> dict:
    """The consequence of that, stated as the number a reader would check first."""
    rows = {row["dimension"]: row for row in the_gap_and_the_error_are_the_same_size()}
    small, large = rows[32], rows[512]
    return {
        "recall_at_thirty_two": small["recall"],
        "recall_at_five_hundred": large["recall"],
        "gap_at_thirty_two": small["angular_gap_degrees"],
        "gap_at_five_hundred": large["angular_gap_degrees"],
        "bits_went_up_sixteenfold": True,
        "recall_is_flat": abs(small["recall"] - large["recall"]) < 0.06,
        "gap_fell_by_four": small["angular_gap_degrees"] > large["angular_gap_degrees"] * 3,
    }


def the_codes_measure_an_angle_so_the_corpus_should_be_normalised(
    shortlists: Sequence[int] = (0, 100, 400),
) -> list[dict]:
    """Whether it matters that binary codes throw away magnitude, which it does.

    A sign vector carries direction and nothing else, so binary search is a cosine method. Run
    against an L2 ground truth on a corpus that is not normalised, it is being asked for a
    quantity it never computed, and the recall shows it: 0.353 against 0.511 at a shortlist of a
    hundred, and 0.652 against 0.800 at four hundred.

    Normalising the corpus makes L2 and cosine give the same ordering, which is the same
    argument metric.py makes for inner product, and it is the same fix.
    """
    if not shortlists:
        raise ConfigError("there is nothing to sweep")
    rows = []
    for label, transform in (("raw", False), ("normalised", True)):
        corpus = gaussian(count=4096, dimension=512)
        vectors = normalise(corpus.vectors) if transform else corpus.vectors
        searched, probes = vectors[:3996], vectors[3996:]
        truth = search(probes, searched, k=10)
        entry = {"corpus": label}
        for shortlist in shortlists:
            index = BinaryIndex(512, rerank=shortlist)
            index.build(searched)
            found, _ = index.search(probes, k=10)
            entry[f"rerank_{shortlist}"] = round(identifier_overlap(truth, found), 4)
        rows.append(entry)
    return rows


def normalising_is_worth_more_than_any_amount_of_bits() -> dict:
    """The two rows of that, next to the dimension sweep it should be compared against."""
    rows = {
        row["corpus"]: row
        for row in the_codes_measure_an_angle_so_the_corpus_should_be_normalised()
    }
    return {
        "raw_at_a_hundred": rows["raw"]["rerank_100"],
        "normalised_at_a_hundred": rows["normalised"]["rerank_100"],
        "raw_at_four_hundred": rows["raw"]["rerank_400"],
        "normalised_at_four_hundred": rows["normalised"]["rerank_400"],
        "gain_at_a_hundred": round(
            rows["normalised"]["rerank_100"] - rows["raw"]["rerank_100"], 4
        ),
        "helps": rows["normalised"]["rerank_100"] > rows["raw"]["rerank_100"],
    }


def a_rerank_is_part_of_the_method(
    shortlists: Sequence[int] = (0, 20, 50, 100, 400),
) -> list[dict]:
    """How much an exact rescore of a binary shortlist buys.

    Nearly everything. A shortlist of a hundred rescored exactly recovers most of the recall the
    bits threw away, for a hundred float distance computations per query on top of a scan that
    was thirty two times cheaper than a float scan. This is not an optimisation bolted onto
    binary search, it is the method: the bits choose the candidates and the floats rank them.
    """
    if not shortlists:
        raise ConfigError("there is nothing to sweep")
    corpus = gaussian(count=4096, dimension=512)
    searched, probes = held_out(corpus, count=100)
    truth = search(probes, searched.vectors, k=10)
    rows = []
    for shortlist in shortlists:
        index = BinaryIndex(512, rerank=shortlist)
        index.build(searched.vectors)
        found, stats = index.search(probes, k=10)
        rows.append(
            {
                "shortlist": shortlist,
                "recall": round(identifier_overlap(truth, found), 4),
                "distances_per_query": round(stats.distances_per_query, 1),
            }
        )
    return rows


def the_rerank_recovers_most_of_the_loss() -> dict:
    """The two ends of that sweep."""
    rows = {row["shortlist"]: row for row in a_rerank_is_part_of_the_method()}
    return {
        "recall_without_rerank": rows[0]["recall"],
        "recall_at_a_hundred": rows[100]["recall"],
        "recall_at_four_hundred": rows[400]["recall"],
        "cost_without_rerank": rows[0]["distances_per_query"],
        "cost_at_a_hundred": rows[100]["distances_per_query"],
        "recovers": rows[100]["recall"] > rows[0]["recall"] * 1.5,
    }


def centring_matters_more_than_rotating() -> dict:
    """The two preprocessing steps, measured against each other.

    Centring is worth much more and the size of the difference was the surprise. An uncentred
    corpus shifted three units off the origin scores 0.003, which is not degraded, it is broken:
    almost every vector is on the same side of almost every axis, so almost every bit is the
    same for everything and the codes carry nearly no information about which vector is which.
    Centring takes it to 0.085, a factor of twenty eight.

    Rotating without centring gets 0.031, which helps and does not fix it, because a rotation
    moves the axes and not the mean. Centring and rotating together give 0.083, the same as
    centring alone on this isotropic corpus, and the rotation earns its place elsewhere.
    """
    corpus = gaussian(count=4096, dimension=256)
    shifted = Corpus(vectors=corpus.vectors + 3.0, name="shifted")
    searched, probes = held_out(shifted, count=100)
    truth = search(probes, searched.vectors, k=10)
    rows = {}
    for label, centre, rotate in (
        ("raw", False, "none"),
        ("centred", True, "none"),
        ("rotated", False, "random"),
        ("both", True, "random"),
    ):
        index = BinaryIndex(256, centre=centre, rotate=rotate)
        index.build(searched.vectors)
        found, _ = index.search(probes, k=10)
        rows[label] = round(identifier_overlap(truth, found), 4)
    return {
        "raw": rows["raw"],
        "centred": rows["centred"],
        "rotated": rows["rotated"],
        "both": rows["both"],
        "centring_helps": rows["centred"] > rows["raw"],
        "centring_beats_rotating": rows["centred"] > rows["rotated"],
    }


def a_rotation_helps_a_clustered_corpus_and_not_a_gaussian_one() -> list[dict]:
    """Where the rotation earns its place, which is the case it was invented for.

    A gaussian corpus is isotropic, so the coordinate axes are already as good as random
    hyperplanes and rotating them changes nothing: 0.085 without and 0.083 with, a gap of two
    thousandths in the wrong direction. A clustered corpus has directions carrying nearly all
    the variation, and rotating spreads it across the axes: 0.092 to 0.109.

    Small in absolute terms because bare binary search is weak on both, and the point is the
    sign of the difference rather than its size. The rotation fixes a specific problem, and
    applying it where that problem is absent costs a matrix multiply for nothing.
    """
    rows = []
    for label, corpus in (
        ("gaussian", gaussian(count=4096, dimension=256)),
        ("clustered", clustered(count=4096, dimension=256, clusters=16)),
    ):
        searched, probes = held_out(corpus, count=100)
        truth = search(probes, searched.vectors, k=10)
        entry = {"corpus": label}
        for rotate in ("none", "random", "pca"):
            index = BinaryIndex(256, rotate=rotate)
            index.build(searched.vectors)
            found, _ = index.search(probes, k=10)
            entry[rotate] = round(identifier_overlap(truth, found), 4)
        rows.append(entry)
    return rows


def the_rotation_is_worth_more_on_structure() -> dict:
    """The two rows of that, as one comparison."""
    rows = {
        row["corpus"]: row
        for row in a_rotation_helps_a_clustered_corpus_and_not_a_gaussian_one()
    }
    gaussian_gain = rows["gaussian"]["random"] - rows["gaussian"]["none"]
    clustered_gain = rows["clustered"]["random"] - rows["clustered"]["none"]
    return {
        "gaussian_without": rows["gaussian"]["none"],
        "gaussian_with": rows["gaussian"]["random"],
        "clustered_without": rows["clustered"]["none"],
        "clustered_with": rows["clustered"]["random"],
        "gaussian_gain": round(gaussian_gain, 4),
        "clustered_gain": round(clustered_gain, 4),
        "worth_more_on_structure": clustered_gain > gaussian_gain,
    }


def binary_beats_product_quantisation_on_speed_and_loses_on_accuracy() -> dict:
    """The comparison that decides which compression to use, at matched storage.

    Product quantisation with eight bit codes over sixty four subspaces is sixty four bytes per
    vector, the same as five hundred and twelve bits, so the two are directly comparable. The
    product codes carry much more information per byte, because each one indexes a fitted
    codebook rather than reporting the sign of a coordinate, so its recall is higher.

    What binary has is the comparison. A product quantised distance is a sum of sixty four table
    lookups; a binary one is eight exclusive ors and eight population counts, and no table has
    to be built per query. Which of those matters depends entirely on whether the workload is
    bound by accuracy or by throughput.
    """
    corpus = gaussian(count=4096, dimension=512)
    searched, probes = held_out(corpus, count=100)
    truth = search(probes, searched.vectors, k=10)

    binary = BinaryIndex(512)
    binary.build(searched.vectors)
    binary_found, _ = binary.search(probes, k=10)

    codes = train(searched.vectors, subspaces=64, centroids=256)
    scores = asymmetric_scores(probes, codes)
    product_found = Neighbours(
        identifiers=torch.topk(scores, k=10, dim=1, largest=False).indices,
        scores=torch.topk(scores, k=10, dim=1, largest=False).values,
    )
    return {
        "binary_bytes": binary.codes.bytes_per_vector,
        "product_bytes": 64,
        "binary_recall": round(identifier_overlap(truth, binary_found), 4),
        "product_recall": round(identifier_overlap(truth, product_found), 4),
        "product_is_more_accurate": identifier_overlap(truth, product_found)
        > identifier_overlap(truth, binary_found),
        "storage_is_matched": binary.codes.bytes_per_vector == 64,
    }


def packing_round_trips(count: int = 64, dimension: int = 300) -> dict:
    """Whether the bits come back out the way they went in.

    The only thing in this module that is a correctness check rather than a measurement, and it
    is here because an off by one in the padding would corrupt the tail of every vector at
    dimensions that are not word aligned, which is exactly where nobody looks.
    """
    generator = torch.Generator().manual_seed(3)
    bits = torch.rand(count, dimension, generator=generator) > 0.5
    recovered = unpack(pack(bits), dimension)
    return {
        "count": count,
        "dimension": dimension,
        "identical": bool(torch.equal(bits, recovered)),
        "words": words_needed(dimension),
    }


def the_population_count_is_right(trials: int = 512) -> dict:
    """Whether the bit counting trick agrees with counting the bits one at a time.

    Checked against the slow version rather than trusted, since the halving trick is six lines
    of masks and a wrong constant would give an answer that is close enough to look plausible on
    random data and wrong on the structured cases.
    """
    generator = torch.Generator().manual_seed(5)
    words = torch.randint(-(2**62), 2**62, (trials, 4), generator=generator, dtype=torch.int64)
    fast = _popcount(words)
    slow = torch.zeros_like(fast)
    for bit in range(WORD_BITS):
        slow = slow + ((words >> bit) & 1)
    return {
        "trials": trials,
        "identical": bool(torch.equal(fast, slow)),
        "max_disagreement": int((fast - slow).abs().max()),
    }


def hamming_is_a_metric(trials: int = 2000, dimension: int = 128) -> dict:
    """Whether the triangle inequality holds for bit counts, which it does exactly.

    Hamming distance is a genuine metric, not an approximation of one, so a structure that
    prunes on the triangle inequality is exactly correct over binary codes. That is a stronger
    guarantee than anything the float metrics in this package offer for inner product, and it is
    a real reason to consider a tree over binary codes where a tree over floats would be
    refused.
    """
    generator = torch.Generator().manual_seed(13)
    rows = pack(torch.randn(trials, dimension, generator=generator) > 0)
    triples = torch.randint(0, trials, (trials, 3), generator=generator)
    violations = 0
    for row in range(trials):
        a, b, c = (int(value) for value in triples[row])
        left = int(hamming(rows[a : a + 1], rows[b : b + 1])[0, 0])
        right = int(hamming(rows[b : b + 1], rows[c : c + 1])[0, 0])
        direct = int(hamming(rows[a : a + 1], rows[c : c + 1])[0, 0])
        if direct > left + right + 1e-9:
            violations += 1
    return {
        "trials": trials,
        "violations": violations,
        "holds": violations == 0,
    }


def a_mismatched_rotation_is_refused() -> bool:
    """Whether a rotation of the wrong size is caught at quantisation time."""
    try:
        quantise(torch.randn(16, 8), rotation=torch.eye(4))
    except DataError:
        return True
    return False


def a_zero_dimension_is_refused() -> bool:
    """Whether asking how many words hold nothing is caught."""
    try:
        words_needed(0)
    except ConfigError:
        return True
    return False


def a_rerank_below_k_is_refused() -> bool:
    """Whether a shortlist too short to fill the result is caught.

    It has to be. Returning fewer than k neighbours and padding the rest would look like a
    correct result with a few bad rows, which scores as a recall loss rather than as the bug it
    is.
    """
    corpus = gaussian(count=512, dimension=64)
    index = BinaryIndex(64, rerank=5)
    index.build(corpus.vectors)
    try:
        index.search(corpus.vectors[:4], k=10)
    except ConfigError:
        return True
    return False


def a_negative_rerank_is_refused() -> bool:
    """Whether a negative shortlist is caught at construction."""
    try:
        BinaryIndex(64, rerank=-1)
    except ConfigError:
        return True
    return False


def an_unknown_rotation_is_refused() -> bool:
    """Whether a rotation nobody implemented is caught at construction rather than at build."""
    try:
        BinaryIndex(64, rotate="hadamard")
    except ConfigError:
        return True
    return False


def codes_of_the_wrong_width_are_refused() -> bool:
    """Whether a word count that does not match the dimension is caught."""
    try:
        BinaryCodes(words=torch.zeros(4, 2, dtype=torch.int64), dimension=512)
    except DataError:
        return True
    return False


def removal_takes_a_row_out_of_the_result() -> dict:
    """That a removed vector stops coming back, which packing makes easy to get wrong.

    The words stay where they are and a liveness mask blocks them before the top k is taken. The
    alternative, repacking the whole corpus on every removal, would make a single delete cost a
    full pass, and the mask costs one comparison per candidate.
    """
    corpus = gaussian(count=1024, dimension=64)
    index = BinaryIndex(64)
    index.build(corpus.vectors)
    query = corpus.vectors[:1]
    before, _ = index.search(query, k=5)
    index.remove([int(before.identifiers[0, 0])])
    after, _ = index.search(query, k=5)
    return {
        "removed": int(before.identifiers[0, 0]),
        "still_present": int(before.identifiers[0, 0]) in after.identifiers[0].tolist(),
        "size_fell": index.size == 1023,
        "still_returns_k": int(after.identifiers.shape[1]) == 5,
    }


def insertion_reuses_the_fitted_centre() -> dict:
    """That inserted vectors are encoded the same way the built ones were.

    They have to be, or their bits mean something different from everybody else's and they never
    match anything. Recomputing the mean over the grown corpus would be more accurate for the
    new vectors and would silently reinterpret every code already stored, which is worse.
    """
    corpus = gaussian(count=1024, dimension=64)
    searched, probes = held_out(corpus, count=32)
    index = BinaryIndex(64)
    index.build(searched.vectors[:512])
    fitted = index.codes.centre.clone()
    index.insert(searched.vectors[512:])
    found, _ = index.search(probes, k=5)
    return {
        "centre_unchanged": bool(torch.equal(fitted, index.codes.centre)),
        "size": index.size,
        "returns_results": int(found.identifiers.shape[0]) == int(probes.shape[0]),
    }

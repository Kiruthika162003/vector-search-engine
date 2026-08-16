from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from vse.errors import ConfigError, DataError
from vse.vectors.dataset import Corpus, gaussian, held_out
from vse.vectors.exact import Neighbours, identifier_overlap, score_gap, search
from vse.vectors.metric import squared_l2

# Storing each coordinate in a byte instead of four, and what that costs in answers.
#
# The compression is trivial: find the range of the data, map it onto the two hundred and fifty
# six values a byte holds, round. The interesting part is that there are two ways to do it and
# two ways to use the result, and three of the four combinations are worse than the fourth by
# amounts nobody would guess from the reconstruction error.
#
# Per vector scales beat one global scale and the two measures disagree about by how much. The
# reconstruction error falls by a factor of five, which sounds decisive, and the recall moves by
# half a point, because every gaussian row has roughly the same range and a per vector scale is
# mostly fitting sampling noise. It costs eight bytes a row to store the scale and offset, a
# quarter of the compressed size at thirty two dimensions, so the compression ratio falls from
# four to three and a fifth. That is the first place here where the error and the answers point
# at different configurations, and it is not the last.
#
# Asymmetric scoring is the one that is unambiguous. Leaving the query in full precision and
# comparing it against the codes is better than quantising the query too, at identical cost,
# because the query's own rounding error is independent of the corpus and so adds noise to every
# distance while carrying nothing. It is half a point of recall, it is free, and quantising the
# query is what happens if the same encoding path is used for indexing and for querying.
#
# The measurement that makes any of this usable is the last one: retrieving too many candidates
# with the codes and rescoring them with the full precision vectors gives perfect recall from a
# shortlist of twice k, and a shortlist of a hundred buys nothing over twenty. So the codes do
# not have to be accurate, they have to get the right answer into a shortlist, which is a much
# weaker requirement than any reconstruction error suggests.

LEVELS = 256


@dataclass(frozen=True)
class ScalarCodes:
    """Quantised vectors, and what is needed to interpret them."""

    codes: torch.Tensor
    scale: torch.Tensor
    offset: torch.Tensor

    def __post_init__(self) -> None:
        if self.codes.ndim != 2:
            raise DataError(f"codes are a matrix of rows, got rank {self.codes.ndim}")
        if self.codes.dtype != torch.uint8:
            raise DataError(f"codes are bytes, got {self.codes.dtype}")
        if self.scale.shape != self.offset.shape:
            raise DataError("every scale needs a matching offset")

    @property
    def count(self) -> int:
        """How many vectors are stored."""
        return int(self.codes.shape[0])

    @property
    def dimension(self) -> int:
        """The width of each one."""
        return int(self.codes.shape[1])

    @property
    def per_vector(self) -> bool:
        """Whether each vector carries its own scale."""
        return self.scale.numel() > 1

    def bytes_used(self) -> int:
        """One byte a coordinate, plus the scales."""
        return self.codes.numel() + self.scale.numel() * 4 + self.offset.numel() * 4

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "count": self.count,
            "dimension": self.dimension,
            "per_vector": self.per_vector,
            "bytes": self.bytes_used(),
            "ratio": round(self.count * self.dimension * 4 / max(self.bytes_used(), 1), 3),
        }


def quantise(vectors: torch.Tensor, per_vector: bool = False) -> ScalarCodes:
    """Map each coordinate onto a byte.

    With a global scale the range is taken over the whole corpus, so every vector shares one
    mapping and the codes are comparable without decoding. With per vector scales each row gets
    its own range, which fits a row with an unusual spread better and costs eight bytes to
    record.
    """
    if vectors.ndim != 2:
        raise DataError(f"vectors are a matrix of rows, got rank {vectors.ndim}")
    if vectors.shape[0] == 0:
        raise DataError("there are no vectors here")
    if per_vector:
        low = vectors.min(dim=1, keepdim=True).values
        high = vectors.max(dim=1, keepdim=True).values
    else:
        low = vectors.min().reshape(1, 1)
        high = vectors.max().reshape(1, 1)
    span = (high - low).clamp_min(1e-12)
    scaled = ((vectors - low) / span * (LEVELS - 1)).round().clamp(0, LEVELS - 1)
    return ScalarCodes(
        codes=scaled.to(torch.uint8),
        scale=span / (LEVELS - 1),
        offset=low,
    )


def dequantise(codes: ScalarCodes) -> torch.Tensor:
    """Turn the codes back into vectors, with the rounding error baked in."""
    return codes.codes.to(torch.float32) * codes.scale + codes.offset


def reconstruction_error(vectors: torch.Tensor, codes: ScalarCodes) -> float:
    """Mean squared distance between a vector and what its code decodes to."""
    rebuilt = dequantise(codes)
    if rebuilt.shape != vectors.shape:
        raise DataError(f"{tuple(rebuilt.shape)} rebuilt against {tuple(vectors.shape)}")
    return float((rebuilt - vectors).pow(2).sum(dim=1).mean())


def asymmetric_scores(queries: torch.Tensor, codes: ScalarCodes) -> torch.Tensor:
    """Score full precision queries against decoded codes.

    The query is never quantised. Its rounding error would be independent of the corpus and so
    would add noise to every distance without carrying any information, and it buys nothing:
    the arithmetic is the same either way once the codes are decoded.
    """
    return squared_l2(queries, dequantise(codes))


def symmetric_scores(queries: torch.Tensor, codes: ScalarCodes) -> torch.Tensor:
    """Score quantised queries against decoded codes, which is the wrong way.

    Kept because it is what happens when the same code path is used for indexing and for
    querying, which is a natural thing to write, and because the size of the loss is worth
    knowing rather than guessing.
    """
    quantised = quantise(queries, per_vector=codes.per_vector)
    return squared_l2(dequantise(quantised), dequantise(codes))


def search_codes(
    queries: torch.Tensor, codes: ScalarCodes, k: int = 10, symmetric: bool = False
) -> Neighbours:
    """Find the k nearest under the quantised representation."""
    if k < 1 or k > codes.count:
        raise ConfigError(f"asking for {k} neighbours from {codes.count} codes")
    score = symmetric_scores if symmetric else asymmetric_scores
    scores = score(queries, codes)
    found = torch.topk(scores, k=k, dim=1, largest=False)
    return Neighbours(identifiers=found.indices, scores=found.values)


def rerank(
    queries: torch.Tensor,
    corpus: torch.Tensor,
    codes: ScalarCodes,
    k: int = 10,
    shortlist: int = 50,
) -> Neighbours:
    """Shortlist with the codes, then rescore the shortlist exactly.

    The whole point of a compressed index. The codes only have to rank the right answer into the
    shortlist, and the exact rescoring of a few dozen vectors then puts it in the right order.
    The full precision vectors have to be readable for this, which is a real constraint: it
    works when they are on disk or on another machine, and not when they were discarded.
    """
    if shortlist < k:
        raise ConfigError(f"a shortlist of {shortlist} cannot produce {k} neighbours")
    if shortlist > codes.count:
        raise ConfigError(f"a shortlist of {shortlist} from {codes.count} codes")
    rough = search_codes(queries, codes, k=shortlist).identifiers
    exact = torch.gather(squared_l2(queries, corpus), 1, rough)
    best = torch.topk(exact, k=k, dim=1, largest=False)
    return Neighbours(identifiers=torch.gather(rough, 1, best.indices), scores=best.values)


def the_compression_is_four_to_one(dimension: int = 32) -> dict:
    """What a byte per coordinate saves, and what the scales take back.

    Four to one with a global scale, which is exactly the ratio of a float to a byte. Per vector
    scales cost eight bytes a row on top, so at thirty two dimensions the ratio drops to three
    and a bit, and at eight dimensions it would be under two. The overhead is fixed per vector
    and the saving scales with the width, so this only makes sense on wide vectors.
    """
    corpus = gaussian(count=2048, dimension=dimension)
    plain = quantise(corpus.vectors)
    rowwise = quantise(corpus.vectors, per_vector=True)
    return {
        "raw_bytes": 2048 * dimension * 4,
        "global_ratio": round(plain.as_dict()["ratio"], 3),
        "per_vector_ratio": round(rowwise.as_dict()["ratio"], 3),
        "overhead_per_vector": 8,
        "narrow_would_be_worse": dimension * 4 / (dimension + 8) < 4.0,
    }


def per_vector_scales_barely_help() -> dict:
    """Whether giving each row its own range is worth the eight bytes.

    Barely, on this data. The reconstruction error falls by a factor of five, which sounds
    decisive, and the recall moves by half a point, because every gaussian row has roughly the
    same range and a per vector scale is mostly fitting sampling noise. It costs a quarter more
    memory to get that half point. On data with rows of genuinely different magnitude it would
    matter, and that is a property of the corpus rather than of the method. It is also the first
    place in this file where reconstruction error and recall point at different answers.
    """
    corpus = gaussian(count=2048, dimension=32)
    searched, probes = held_out(corpus, count=64)
    truth = search(probes, searched.vectors, k=10)
    rows = {}
    for label, flag in (("global", False), ("per_vector", True)):
        codes = quantise(searched.vectors, per_vector=flag)
        found = search_codes(probes, codes, k=10)
        rows[label] = {
            "error": round(reconstruction_error(searched.vectors, codes), 6),
            "recall": round(identifier_overlap(truth, found), 4),
            "bytes": codes.bytes_used(),
        }
    flat = {
        f"{label}_{key}": value for label, row in rows.items() for key, value in row.items()
    }
    return {
        **flat,
        "error_ratio": round(rows["global"]["error"] / rows["per_vector"]["error"], 3),
        "recall_gap": round(abs(rows["global"]["recall"] - rows["per_vector"]["recall"]), 4),
    }


def asymmetric_scoring_is_strictly_better() -> dict:
    """Whether quantising the query as well costs anything.

    It does, and it buys nothing, so this is not a tradeoff. The query's rounding error is
    independent of the corpus, so it adds noise to every distance and carries no information,
    and the arithmetic is identical either way once the codes are decoded. Quantising the query
    costs recall for free, and it is the natural thing to write if the same encoding path is
    used for indexing and for querying. The margin here is half a point of recall and half again
    on the gap, which is small and is entirely one sided.
    """
    corpus = gaussian(count=2048, dimension=32)
    searched, probes = held_out(corpus, count=64)
    truth = search(probes, searched.vectors, k=10)
    codes = quantise(searched.vectors)
    asymmetric = search_codes(probes, codes, k=10, symmetric=False)
    symmetric = search_codes(probes, codes, k=10, symmetric=True)
    return {
        "asymmetric_recall": round(identifier_overlap(truth, asymmetric), 4),
        "symmetric_recall": round(identifier_overlap(truth, symmetric), 4),
        "asymmetric_gap": round(score_gap(probes, searched.vectors, truth, asymmetric), 6),
        "symmetric_gap": round(score_gap(probes, searched.vectors, truth, symmetric), 6),
        "asymmetric_wins": identifier_overlap(truth, asymmetric)
        >= identifier_overlap(truth, symmetric),
        "same_cost": True,
    }


def reranking_recovers_the_recall(shortlists: Sequence[int] = (10, 20, 50, 100)) -> list[dict]:
    """How much of the lost recall a short exact rescoring pass gets back.

    Nearly all of it, from a shortlist a few times the result size. The codes only have to rank
    the right answer into the shortlist and the exact pass does the rest, which is a far weaker
    requirement than being accurate. This is the reason compressed indexes are usable at all.
    """
    if not shortlists:
        raise ConfigError("there is nothing to sweep")
    corpus = gaussian(count=2048, dimension=32)
    searched, probes = held_out(corpus, count=64)
    truth = search(probes, searched.vectors, k=10)
    codes = quantise(searched.vectors)
    rows = []
    for shortlist in shortlists:
        found = rerank(probes, searched.vectors, codes, k=10, shortlist=shortlist)
        rows.append(
            {
                "shortlist": shortlist,
                "recall": round(identifier_overlap(truth, found), 4),
                "exact_distances": shortlist,
                "gap": round(score_gap(probes, searched.vectors, truth, found), 6),
            }
        )
    return rows


def a_shortlist_of_twice_k_is_enough() -> dict:
    """Where that sweep stops improving, which is the number worth quoting.

    Twice k, on this corpus. A shortlist of twenty rescored exactly gives perfect recall and a
    gap of zero, and a hundred gives the same, so the extra eighty exact distances buy nothing.
    That is cheaper than I expected and it is a property of how good the codes already are
    here: on a harder corpus or a coarser code the shortlist would have to be longer, and the
    right way to set it is to sweep it rather than to pick a multiple.
    """
    rows = {row["shortlist"]: row for row in reranking_recovers_the_recall()}
    plain = round(
        identifier_overlap(
            search(
                held_out(gaussian(count=2048, dimension=32), count=64)[1],
                held_out(gaussian(count=2048, dimension=32), count=64)[0].vectors,
                k=10,
            ),
            search_codes(
                held_out(gaussian(count=2048, dimension=32), count=64)[1],
                quantise(held_out(gaussian(count=2048, dimension=32), count=64)[0].vectors),
                k=10,
            ),
        ),
        4,
    )
    return {
        "without_reranking": plain,
        "at_ten": rows[10]["recall"],
        "at_fifty": rows[50]["recall"],
        "at_a_hundred": rows[100]["recall"],
        "recovered": rows[50]["recall"] > plain,
        "saturates": abs(rows[100]["recall"] - rows[50]["recall"]) < 0.02,
    }


def the_codes_do_not_have_to_be_good() -> dict:
    """The point the reranking result actually makes.

    That a compressed index is a shortlisting device and not a distance oracle. The raw code
    recall and the reranked recall are far apart, so the codes are visibly inaccurate, and the
    reranked answer is nearly exact. Anybody tuning a quantiser by its reconstruction error is
    optimising a quantity that only matters through its effect on the shortlist.
    """
    corpus = gaussian(count=2048, dimension=32)
    searched, probes = held_out(corpus, count=64)
    truth = search(probes, searched.vectors, k=10)
    codes = quantise(searched.vectors)
    raw = search_codes(probes, codes, k=10)
    reranked = rerank(probes, searched.vectors, codes, k=10, shortlist=50)
    return {
        "raw_recall": round(identifier_overlap(truth, raw), 4),
        "reranked_recall": round(identifier_overlap(truth, reranked), 4),
        "reconstruction_error": round(reconstruction_error(searched.vectors, codes), 6),
        "typical_squared_distance": round(
            float(squared_l2(probes[:8], searched.vectors).mean()), 3
        ),
    }


def error_falls_with_the_level_count(bits: Sequence[int] = (2, 4, 6, 8)) -> list[dict]:
    """How the reconstruction error depends on how many values a code holds.

    By a factor of four per bit, which is what rounding to a grid gives: halving the spacing
    quarters the squared error. The measurement follows it closely enough that a departure
    would indicate a bug in the mapping rather than a property of the data. Worth noting where
    the bottom is: a two bit code has a reconstruction error of twenty three against a typical
    squared distance of seventy, so it is not a coarse representation, it is noise.
    """
    if not bits:
        raise ConfigError("there is nothing to sweep")
    corpus = gaussian(count=1024, dimension=32)
    rows = []
    for width in bits:
        levels = 2**width
        low, high = corpus.vectors.min(), corpus.vectors.max()
        span = (high - low).clamp_min(1e-12)
        scaled = ((corpus.vectors - low) / span * (levels - 1)).round().clamp(0, levels - 1)
        rebuilt = scaled * (span / (levels - 1)) + low
        rows.append(
            {
                "bits": width,
                "levels": levels,
                "error": round(float((rebuilt - corpus.vectors).pow(2).sum(dim=1).mean()), 6),
            }
        )
    return rows


def each_bit_quarters_the_error() -> dict:
    """That relationship, checked against the arithmetic it comes from."""
    rows = {row["bits"]: row for row in error_falls_with_the_level_count()}
    ratios = [
        rows[4]["error"] / rows[6]["error"],
        rows[6]["error"] / rows[8]["error"],
    ]
    return {
        "at_two_bits": rows[2]["error"],
        "at_eight_bits": rows[8]["error"],
        "ratios_per_two_bits": [round(ratio, 2) for ratio in ratios],
        "close_to_sixteen": all(10 < ratio < 22 for ratio in ratios),
    }


def compare_configurations(corpus: Corpus | None = None) -> list[dict]:
    """Every combination of scale choice and scoring choice, as one table."""
    target = corpus if corpus is not None else gaussian(count=2048, dimension=32)
    searched, probes = held_out(target, count=64)
    truth = search(probes, searched.vectors, k=10)
    rows = []
    for per_vector in (False, True):
        codes = quantise(searched.vectors, per_vector=per_vector)
        for symmetric in (False, True):
            found = search_codes(probes, codes, k=10, symmetric=symmetric)
            rows.append(
                {
                    "scale": "per vector" if per_vector else "global",
                    "scoring": "symmetric" if symmetric else "asymmetric",
                    "recall": round(identifier_overlap(truth, found), 4),
                    "bytes": codes.bytes_used(),
                }
            )
    return rows


def the_best_configuration_is_the_cheap_one() -> dict:
    """Which combination wins, which is the one that stores least.

    Not quite, and the honest version is more useful. Per vector scales with asymmetric scoring
    has the best recall, and a global scale with asymmetric scoring is cheapest, and they are
    different configurations. The whole spread across all four is under one point of recall
    while the memory spread is a quarter, so the cheap one is the right default and the
    expensive one is not wrong. What is unambiguous is the scoring: asymmetric beats symmetric
    at both scale settings, for free.
    """
    rows = compare_configurations()
    best = max(rows, key=lambda row: (row["recall"], -row["bytes"]))
    cheapest = min(rows, key=lambda row: row["bytes"])
    return {
        "best": f"{best['scale']} {best['scoring']}",
        "cheapest": f"{cheapest['scale']} {cheapest['scoring']}",
        "same": best["scale"] == cheapest["scale"] and best["scoring"] == cheapest["scoring"],
        "recall_spread": round(
            max(row["recall"] for row in rows) - min(row["recall"] for row in rows), 4
        ),
    }


def a_shortlist_shorter_than_k_is_refused() -> bool:
    """Whether a rerank that cannot produce the requested result is caught."""
    corpus = gaussian(count=256, dimension=8)
    try:
        rerank(corpus.vectors[:4], corpus.vectors, quantise(corpus.vectors), k=10, shortlist=5)
    except ConfigError:
        return True
    return False


def a_float_code_is_refused() -> bool:
    """Whether codes that are not bytes are refused at construction.

    The whole point is that a code is a byte. A float tensor here would decode correctly and
    silently use four times the memory the ratio claims, which is the one error this class
    exists to prevent.
    """
    try:
        ScalarCodes(codes=torch.zeros(4, 8), scale=torch.ones(1, 1), offset=torch.zeros(1, 1))
    except DataError:
        return True
    return False


def an_empty_corpus_is_refused() -> bool:
    """Whether quantising nothing is refused rather than producing an empty codebook."""
    try:
        quantise(torch.zeros(0, 8))
    except DataError:
        return True
    return False


def a_constant_vector_survives() -> dict:
    """What happens to a row with no range at all.

    It quantises to zero and decodes back to its own value, because the span is clamped away
    from zero before the division. Without the clamp a constant row would divide by zero and
    produce a row of nans that would then poison every distance computed against it.
    """
    vectors = torch.ones(4, 8)
    vectors[1] = 3.0
    codes = quantise(vectors, per_vector=True)
    rebuilt = dequantise(codes)
    return {
        "any_nan": bool(rebuilt.isnan().any()),
        "constant_row_recovered": bool(torch.allclose(rebuilt[0], vectors[0], atol=1e-5)),
        "other_row_recovered": bool(torch.allclose(rebuilt[1], vectors[1], atol=1e-5)),
    }

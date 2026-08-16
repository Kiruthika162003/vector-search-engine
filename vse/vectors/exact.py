from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from vse.errors import ConfigError, DataError
from vse.vectors.metric import L2, Metric, distances, metric_named

# Exact search, which is the thing every other index in this package is measured against.
#
# It is a matrix product and a top k, and there is no cleverness in it on purpose. Its whole job
# is to be obviously right, so that when an approximate index disagrees with it the disagreement
# is the approximate index. Everything here is written to be checkable by reading it.
#
# The one piece of engineering is that it batches. A full score matrix is the query count times
# the corpus count times four bytes, which for a hundred thousand vectors and a thousand queries
# is four hundred megabytes for a result that is a thousand rows of ten integers. Batching over
# the corpus and merging the partial top k lists keeps the peak at the batch size and produces
# bit identical results, which is checked below rather than assumed. It is not free: the merge
# costs a second top k over the partial results, so a batched search does a little more work
# than an unbatched one and uses a small fraction of the memory.
#
# Ties are the part that causes trouble later. When two vectors are exactly the same distance
# from a query, which one comes back is an implementation detail, and an approximate index that
# returns the other one is not wrong. Recall measured on identifiers punishes it anyway, so
# there is a distance based comparison here as well, and the gap between the two numbers on a
# corpus with duplicates in it is worth knowing before trusting either.


@dataclass(frozen=True)
class Neighbours:
    """The result of a search: identifiers and their scores, closest first."""

    identifiers: torch.Tensor
    scores: torch.Tensor

    def __post_init__(self) -> None:
        if self.identifiers.shape != self.scores.shape:
            raise DataError(
                f"{tuple(self.identifiers.shape)} identifiers and "
                f"{tuple(self.scores.shape)} scores"
            )
        if self.identifiers.ndim != 2:
            raise DataError(f"a result is a matrix of rows, got rank {self.identifiers.ndim}")

    @property
    def queries(self) -> int:
        """How many queries this answers."""
        return int(self.identifiers.shape[0])

    @property
    def k(self) -> int:
        """How many neighbours per query."""
        return int(self.identifiers.shape[1])

    def row(self, query: int) -> list[int]:
        """One query's neighbours, as identifiers."""
        if not 0 <= query < self.queries:
            raise ConfigError(f"query {query} is not one of the {self.queries} asked")
        return [int(value) for value in self.identifiers[query]]

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {"queries": self.queries, "k": self.k}


def _checked(vectors: torch.Tensor, name: str) -> torch.Tensor:
    """Reject anything that is not a matrix of float rows."""
    if vectors.ndim != 2:
        raise DataError(f"{name} has to be a matrix of rows, got rank {vectors.ndim}")
    if vectors.shape[0] == 0:
        raise DataError(f"{name} is empty")
    if not vectors.dtype.is_floating_point:
        raise DataError(f"{name} has to be floating point, got {vectors.dtype}")
    return vectors


def search(
    queries: torch.Tensor,
    corpus: torch.Tensor,
    k: int = 10,
    metric: Metric | str = L2,
) -> Neighbours:
    """Every distance, then the k best. The definition, written out."""
    _checked(queries, "queries")
    _checked(corpus, "corpus")
    chosen = metric if isinstance(metric, Metric) else metric_named(metric)
    if k < 1:
        raise ConfigError(f"asking for {k} neighbours is not a query")
    if k > corpus.shape[0]:
        raise ConfigError(f"asking for {k} neighbours from {corpus.shape[0]} vectors")
    scores = distances(queries, corpus, chosen)
    found = torch.topk(scores, k=k, dim=1, largest=not chosen.smaller_is_closer)
    return Neighbours(identifiers=found.indices, scores=found.values)


def search_batched(
    queries: torch.Tensor,
    corpus: torch.Tensor,
    k: int = 10,
    metric: Metric | str = L2,
    batch: int = 4096,
) -> Neighbours:
    """The same answer, without ever holding the full score matrix.

    Each corpus batch produces its own top k, the partial results are concatenated, and one more
    top k over that picks the winners. The identifiers have to be shifted by the batch offset
    before the merge, which is the only place this can go wrong and the reason the test compares
    against the unbatched version rather than against itself.
    """
    _checked(queries, "queries")
    _checked(corpus, "corpus")
    chosen = metric if isinstance(metric, Metric) else metric_named(metric)
    if batch < 1:
        raise ConfigError(f"a batch of {batch} vectors is not a batch")
    if k > corpus.shape[0]:
        raise ConfigError(f"asking for {k} neighbours from {corpus.shape[0]} vectors")
    partial_ids = []
    partial_scores = []
    for start in range(0, corpus.shape[0], batch):
        block = corpus[start : start + batch]
        width = min(k, block.shape[0])
        scores = distances(queries, block, chosen)
        found = torch.topk(scores, k=width, dim=1, largest=not chosen.smaller_is_closer)
        partial_ids.append(found.indices + start)
        partial_scores.append(found.values)
    merged_ids = torch.cat(partial_ids, dim=1)
    merged_scores = torch.cat(partial_scores, dim=1)
    best = torch.topk(merged_scores, k=k, dim=1, largest=not chosen.smaller_is_closer)
    return Neighbours(identifiers=torch.gather(merged_ids, 1, best.indices), scores=best.values)


def scores_for(
    queries: torch.Tensor,
    corpus: torch.Tensor,
    identifiers: torch.Tensor,
    metric: Metric | str = L2,
) -> torch.Tensor:
    """The score each query gives to a specific set of vectors.

    Needed for comparing two results by distance rather than by identity, and written against
    the same distance function so that a result and its rescoring cannot disagree for any reason
    other than the identifiers being different.
    """
    chosen = metric if isinstance(metric, Metric) else metric_named(metric)
    if identifiers.shape[0] != queries.shape[0]:
        raise DataError(
            f"{identifiers.shape[0]} identifier rows for {queries.shape[0]} queries"
        )
    full = distances(queries, corpus, chosen)
    return torch.gather(full, 1, identifiers)


def duplicated_corpus(
    count: int = 256, dimension: int = 16, copies: int = 4, seed: int = 0
) -> torch.Tensor:
    """A corpus where every vector appears several times.

    The fixture that makes ties unavoidable. With four copies of each vector the top ten of any
    query contains at least two exact ties, so any two correct implementations can return
    different identifier sets and both be right.
    """
    if copies < 2:
        raise ConfigError(f"{copies} copies does not duplicate anything")
    if count % copies:
        raise ConfigError(f"{count} vectors does not divide into {copies} copies")
    generator = torch.Generator().manual_seed(seed)
    distinct = torch.randn(count // copies, dimension, generator=generator)
    return distinct.repeat(copies, 1)


def random_corpus(count: int = 4096, dimension: int = 32, seed: int = 0) -> torch.Tensor:
    """Independent gaussian rows. The default corpus everything is measured on."""
    if count < 1 or dimension < 1:
        raise ConfigError(f"{count} vectors of {dimension} dimensions is not a corpus")
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(count, dimension, generator=generator)


def random_queries(count: int = 128, dimension: int = 32, seed: int = 7) -> torch.Tensor:
    """Queries drawn from the same distribution as the corpus."""
    return random_corpus(count, dimension, seed)


def identifier_overlap(left: Neighbours, right: Neighbours) -> float:
    """The average share of identifiers two results have in common.

    This is recall when the left argument is the exact answer, and it is the number everybody
    reports. It counts a tie broken differently as a miss, which is the reason for the second
    measure below.
    """
    if left.queries != right.queries:
        raise DataError(f"{left.queries} queries against {right.queries}")
    if left.k != right.k:
        raise DataError(f"top {left.k} against top {right.k}")
    total = 0.0
    for row in range(left.queries):
        total += len(set(left.row(row)) & set(right.row(row))) / left.k
    return total / left.queries


def score_gap(
    queries: torch.Tensor,
    corpus: torch.Tensor,
    left: Neighbours,
    right: Neighbours,
    metric: Metric | str = L2,
) -> float:
    """How much worse the right result is than the left, by distance rather than by name.

    Zero when the two results are equally good, whatever identifiers they used. The measure that
    does not punish a tie broken the other way, and the one an application actually cares about,
    since nobody notices which of two identical vectors came back.
    """
    chosen = metric if isinstance(metric, Metric) else metric_named(metric)
    theirs = scores_for(queries, corpus, right.identifiers, chosen)
    ours = scores_for(queries, corpus, left.identifiers, chosen)
    difference = theirs - ours
    if not chosen.smaller_is_closer:
        difference = -difference
    return float(difference.mean())


def batching_changes_nothing(batch: int = 512) -> dict:
    """Whether the memory saving costs any accuracy.

    None at all. The batched merge returns the same identifiers in the same order as the single
    matrix version, because the partial top k lists between them contain every candidate that
    could have made the final list and the merge is a stable selection over the same scores.
    """
    corpus = random_corpus()
    queries = random_queries(count=64)
    plain = search(queries, corpus, k=10)
    chunked = search_batched(queries, corpus, k=10, batch=batch)
    return {
        "identical_identifiers": bool(torch.equal(plain.identifiers, chunked.identifiers)),
        "identical_scores": bool(torch.allclose(plain.scores, chunked.scores, atol=1e-6)),
        "overlap": identifier_overlap(plain, chunked),
        "batches": (corpus.shape[0] + batch - 1) // batch,
    }


def batching_saves_the_matrix(batch: int = 512) -> dict:
    """And what it saves, in bytes.

    The full score matrix against the largest one a batched search holds. At four thousand
    vectors it is a factor of eight, and the factor is the number of batches, so on a corpus
    large enough for the memory to matter it is arbitrarily large.
    """
    corpus = random_corpus()
    queries = random_queries(count=64)
    if batch < 1:
        raise ConfigError(f"a batch of {batch} vectors is not a batch")
    full = queries.shape[0] * corpus.shape[0] * 4
    held = queries.shape[0] * min(batch, corpus.shape[0]) * 4
    return {
        "full_matrix_bytes": full,
        "batched_bytes": held,
        "ratio": round(full / held, 3),
        "batches": (corpus.shape[0] + batch - 1) // batch,
    }


def batching_costs_a_second_selection(batch: int = 512) -> dict:
    """What it costs, in candidates considered.

    A second top k over the merged partials. With eight batches and a top ten the merge selects
    ten from eighty, so the extra work is a selection over eight times k rather than over the
    corpus, which is small and is not nothing.
    """
    corpus = random_corpus()
    if batch < 1:
        raise ConfigError(f"a batch of {batch} vectors is not a batch")
    batches = (corpus.shape[0] + batch - 1) // batch
    return {
        "batches": batches,
        "candidates_merged": batches * 10,
        "against_the_corpus": corpus.shape[0],
        "share_of_the_corpus": round(batches * 10 / corpus.shape[0], 5),
    }


def ties_make_identifiers_ambiguous(copies: int = 4) -> dict:
    """What a corpus with duplicates does to a comparison by identifier.

    Breaks it, but only when k cuts through a tie group. With four copies of every vector a top
    eight is unambiguous, because eight is two whole groups, and a top six is not: the last two
    slots are any two of four identical vectors. Searching the corpus and a permuted view of it
    then returns different identifiers, the overlap falls short of one, and the score gap stays
    at zero. Both answers are exactly correct and only the identifier measure disagrees.
    """
    corpus = duplicated_corpus(copies=copies)
    queries = corpus[:32]
    order = torch.randperm(corpus.shape[0], generator=torch.Generator().manual_seed(4))
    permuted = corpus[order]
    ours = search(queries, corpus, k=6)
    theirs = search(queries, permuted, k=6)
    mapped = Neighbours(identifiers=order[theirs.identifiers], scores=theirs.scores)
    whole_groups = Neighbours(
        identifiers=order[search(queries, permuted, k=8).identifiers],
        scores=search(queries, permuted, k=8).scores,
    )
    return {
        "overlap": round(identifier_overlap(ours, mapped), 4),
        "score_gap": round(score_gap(queries, corpus, ours, mapped), 8),
        "both_are_correct": abs(score_gap(queries, corpus, ours, mapped)) < 1e-5,
        "overlap_on_whole_groups": round(
            identifier_overlap(search(queries, corpus, k=8), whole_groups), 4
        ),
    }


def without_ties_the_two_measures_agree() -> dict:
    """And that this is a property of the corpus rather than of the measure.

    On a corpus with no duplicates in it the same experiment gives an overlap of one and a gap
    of zero, so the identifier measure is not broken, it is measuring something the corpus made
    ambiguous.
    """
    corpus = random_corpus(count=512, dimension=16)
    queries = corpus[:32]
    order = torch.randperm(corpus.shape[0], generator=torch.Generator().manual_seed(4))
    permuted = corpus[order]
    ours = search(queries, corpus, k=8)
    theirs = search(queries, permuted, k=8)
    mapped = Neighbours(identifiers=order[theirs.identifiers], scores=theirs.scores)
    return {
        "overlap": round(identifier_overlap(ours, mapped), 4),
        "score_gap": round(score_gap(queries, corpus, ours, mapped), 8),
    }


def a_vector_is_its_own_nearest_neighbour(metric: str = "l2") -> dict:
    """The sanity check that catches a sign error in one line.

    Searching a corpus against itself has to return each vector first with a distance of zero.
    An index that has the comparison backwards fails this immediately, which is worth having
    before any of the approximate structures are written.
    """
    corpus = random_corpus(count=256, dimension=16)
    found = search(corpus, corpus, k=1, metric=metric)
    return {
        "metric": metric,
        "all_self": found.identifiers.flatten().tolist() == list(range(corpus.shape[0])),
        "largest_score": round(float(found.scores.abs().max()), 6),
    }


def k_sweep(values: Sequence[int] = (1, 10, 50, 100)) -> list[dict]:
    """How the result size changes with k, and what it costs to return.

    The search cost does not change with k in any way worth measuring: the distances dominate
    and the selection is a fraction of them. That is the reason a caller is better off asking
    for more neighbours than it needs and filtering afterwards than issuing a second query.
    """
    if not values:
        raise ConfigError("there is nothing to sweep")
    corpus = random_corpus()
    queries = random_queries(count=32)
    rows = []
    for k in values:
        found = search(queries, corpus, k=k)
        rows.append(
            {
                "k": k,
                "results": found.queries * found.k,
                "distances_computed": queries.shape[0] * corpus.shape[0],
                "share_returned": round(found.k / corpus.shape[0], 5),
            }
        )
    return rows


def asking_for_more_than_the_corpus_is_refused() -> bool:
    """Whether a k larger than the corpus is refused rather than quietly shortened.

    A result with fewer neighbours than requested is the kind of thing that propagates into a
    recall denominator and makes an index look better than it is.
    """
    corpus = random_corpus(count=8, dimension=16)
    try:
        search(random_queries(count=4, dimension=16), corpus, k=32)
    except ConfigError:
        return True
    return False


def comparing_different_shapes_is_refused() -> bool:
    """Whether comparing a top ten against a top twenty is caught."""
    corpus = random_corpus(count=128, dimension=16)
    queries = random_queries(count=8, dimension=16)
    try:
        identifier_overlap(search(queries, corpus, k=10), search(queries, corpus, k=20))
    except DataError:
        return True
    return False


def a_mismatched_result_is_refused() -> bool:
    """Whether identifiers and scores of different shapes are caught at construction."""
    try:
        Neighbours(identifiers=torch.zeros(4, 10, dtype=torch.long), scores=torch.zeros(4, 5))
    except DataError:
        return True
    return False

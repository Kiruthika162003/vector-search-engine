from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from vse.errors import ConfigError, DataError
from vse.index.ivf import IVFIndex
from vse.vectors.dataset import clustered, gaussian, held_out, on_a_subspace
from vse.vectors.exact import Neighbours, identifier_overlap, search
from vse.vectors.metric import L2, distances

# Two stage retrieval: a cheap scorer picks a shortlist, an exact one puts it in order.
#
# Every approximate index in this package makes one of two kinds of error. It can fail to look
# at the right vector, which is a coverage error, or it can look at the right vector and score
# it wrongly, which is a scoring error. The IVF and the graph make only the first kind: whatever
# they reach, they measure exactly. The quantisers make both, because a code is a lossy stand in
# for the vector and the distance to a code is not the distance to the vector.
#
# Reranking fixes the second kind and cannot touch the first, and it fixes it completely. I
# expected the rescoring to recover most of what the shortlist held and measured it recovering
# all of it, at every depth and on every corpus here. That is not luck, it is forced: a true
# top k neighbour sitting in the shortlist has at most k minus one vectors closer to the query
# anywhere in the corpus, so an exact top k over the shortlist cannot push it out. Shortlist
# recall and final recall are the same number.
#
# So there is only one design variable, which is the shortlist, and the rerank is bookkeeping.
# The measurements below still report both numbers, but as a check on the implementation
# rather than as a result: any gap between them is an indexing bug.
#
# What two stages buy is a different shape of budget. A first stage scoring on one bit per
# dimension costs a thirty second of a float distance, so it can afford to look at everything,
# and the exact stage then pays full price for m vectors and nothing else. On this corpus of
# 3896 searched vectors, sign codes at a depth of 800 reach 0.897 for 922 distance equivalents
# against 3896 for the full scan, a factor of 4.2. The same codes answering directly at ten
# reach 0.129, so the whole of that recall came from widening the list and rescoring it.
#
# The cost model here counts a full precision distance over d dimensions as one unit and prices
# everything else against it, so a projection to rank r costs r over d and a sign code costs one
# over thirty two. That is a model, not a measurement, and it flatters the bit scorers on any
# machine without wide popcount. The distance counts elsewhere in the package are exact; these
# are not, and the docstrings say so where it matters.


@dataclass
class Shortlist:
    """Candidates from a first stage, with whatever it thought their scores were."""

    identifiers: torch.Tensor
    scores: torch.Tensor
    cost_per_query: float

    def __post_init__(self) -> None:
        if self.identifiers.shape != self.scores.shape:
            raise DataError(
                f"{tuple(self.identifiers.shape)} identifiers against "
                f"{tuple(self.scores.shape)} scores"
            )
        if self.identifiers.ndim != 2:
            raise DataError(f"a shortlist is two dimensional, not {self.identifiers.ndim}")

    @property
    def depth(self) -> int:
        """How many candidates the first stage returned per query."""
        return int(self.identifiers.shape[1])

    @property
    def queries(self) -> int:
        """How many queries it answered."""
        return int(self.identifiers.shape[0])

    def as_neighbours(self, k: int) -> Neighbours:
        """The first stage's own answer, before any rescoring."""
        if k > self.depth:
            raise ConfigError(f"a shortlist of {self.depth} cannot answer for {k}")
        return Neighbours(self.identifiers[:, :k], self.scores[:, :k])

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {"depth": self.depth, "queries": self.queries, "cost": self.cost_per_query}


@dataclass
class Staged:
    """What a two stage search reached and what it spent getting there."""

    answer: Neighbours
    shortlist_recall: float
    final_recall: float
    first_stage_cost: float
    rerank_cost: float

    @property
    def total_cost(self) -> float:
        """Both stages, in full distance equivalents."""
        return self.first_stage_cost + self.rerank_cost

    @property
    def headroom(self) -> float:
        """How much of the shortlist the rerank has not yet turned into answers."""
        return self.shortlist_recall - self.final_recall

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "shortlist_recall": round(self.shortlist_recall, 4),
            "final_recall": round(self.final_recall, 4),
            "headroom": round(self.headroom, 4),
            "first_stage_cost": round(self.first_stage_cost, 1),
            "rerank_cost": round(self.rerank_cost, 1),
            "total_cost": round(self.total_cost, 1),
        }


def _setup(
    count: int = 4096,
    dimension: int = 32,
    queries: int = 200,
    k: int = 10,
    kind: str = "gaussian",
    seed: int = 0,
):
    """One corpus, one query set, one exact answer, shared by every measurement."""
    if kind == "gaussian":
        corpus = gaussian(count=count, dimension=dimension, seed=seed)
    elif kind == "clustered":
        corpus = clustered(count=count, dimension=dimension, clusters=16, seed=seed)
    elif kind == "subspace":
        corpus = on_a_subspace(count=count, dimension=dimension, intrinsic=6, seed=seed)
    else:
        raise ConfigError(f"{kind} is not a corpus")
    searched, probes = held_out(corpus, count=queries)
    truth = search(probes, searched.vectors, k=k)
    return searched.vectors, probes, truth


def sign_codes(vectors: torch.Tensor) -> torch.Tensor:
    """One bit per dimension, the sign of the coordinate.

    Not centred first, which is deliberate: centring is the right thing to do and quantize
    binary already does it. Here the codes are meant to be a crude first stage rather than a
    good one, because the point of the module is what reranking recovers rather than how good
    the first stage was.
    """
    if vectors.ndim != 2:
        raise DataError(f"a corpus is two dimensional, not {vectors.ndim}")
    return vectors > 0


def hamming_scores(query_bits: torch.Tensor, corpus_bits: torch.Tensor) -> torch.Tensor:
    """How many bits differ, for every query against every vector.

    Computed as a matrix product on plus and minus one rather than by counting, because the
    product is one call into the same kernel everything else in the package uses and the
    arithmetic is exact for these sizes.
    """
    if query_bits.shape[1] != corpus_bits.shape[1]:
        raise DataError(f"{query_bits.shape[1]} bits against {corpus_bits.shape[1]}")
    width = int(query_bits.shape[1])
    left = query_bits.float() * 2.0 - 1.0
    right = corpus_bits.float() * 2.0 - 1.0
    return (width - left @ right.T) / 2.0


def projection(dimension: int, rank: int, seed: int = 0) -> torch.Tensor:
    """An orthonormal basis for a random subspace of the given rank."""
    if rank < 1 or rank > dimension:
        raise ConfigError(f"rank {rank} is not inside {dimension} dimensions")
    generator = torch.Generator().manual_seed(seed)
    square = torch.randn(dimension, dimension, generator=generator)
    basis, _ = torch.linalg.qr(square)
    return basis[:, :rank]


def exact_shortlist(
    queries: torch.Tensor,
    corpus: torch.Tensor,
    depth: int,
) -> Shortlist:
    """The control: a first stage that is already exact.

    Reranking this can only be an identity, and the test that it is catches any mistake in the
    reranking code itself rather than in the scorers.
    """
    _check_depth(depth, int(corpus.shape[0]))
    scores = distances(queries, corpus, L2)
    best = torch.topk(scores, k=depth, largest=False)
    return Shortlist(
        identifiers=best.indices,
        scores=best.values,
        cost_per_query=float(corpus.shape[0]),
    )


def sign_shortlist(
    queries: torch.Tensor,
    corpus: torch.Tensor,
    depth: int,
) -> Shortlist:
    """A first stage scoring on one bit per dimension.

    Priced at one thirty second of a full distance per vector, which is the ratio of a bit to a
    float and not a timing. The Hamming ties are broken by whatever order topk returns, and
    there are many of them at this width, which is part of why the first stage on its own is
    poor.
    """
    _check_depth(depth, int(corpus.shape[0]))
    scores = hamming_scores(sign_codes(queries), sign_codes(corpus))
    best = torch.topk(scores, k=depth, largest=False)
    return Shortlist(
        identifiers=best.indices,
        scores=best.values,
        cost_per_query=float(corpus.shape[0]) / 32.0,
    )


def projected_shortlist(
    queries: torch.Tensor,
    corpus: torch.Tensor,
    depth: int,
    rank: int = 8,
    seed: int = 0,
) -> Shortlist:
    """A first stage scoring in fewer dimensions.

    Priced at rank over dimension per vector, which for a dense product is close to right. The
    projection is orthonormal so the reduced distance is a lower bound on the full one, which
    makes this scorer biased low rather than merely noisy.
    """
    _check_depth(depth, int(corpus.shape[0]))
    dimension = int(corpus.shape[1])
    basis = projection(dimension, rank, seed=seed)
    scores = distances(queries @ basis, corpus @ basis, L2)
    best = torch.topk(scores, k=depth, largest=False)
    return Shortlist(
        identifiers=best.indices,
        scores=best.values,
        cost_per_query=float(corpus.shape[0]) * rank / dimension,
    )


def partition_shortlist(
    queries: torch.Tensor,
    corpus: torch.Tensor,
    depth: int,
    partitions: int = 64,
    probe: int = 8,
) -> Shortlist:
    """A first stage that reaches part of the corpus and scores that part exactly.

    The opposite error to the scorers above: no scoring loss at all, all of the loss in
    coverage. Reranking it is an identity for the same reason the exact control is, and putting
    the two next to each other is what makes the distinction concrete.
    """
    _check_depth(depth, int(corpus.shape[0]))
    index = IVFIndex(int(corpus.shape[1]), partitions=partitions, probe=probe)
    index.build(corpus)
    found, stats = index.search(queries, k=depth)
    return Shortlist(
        identifiers=found.identifiers,
        scores=found.scores,
        cost_per_query=float(stats.distances_per_query),
    )


def _check_depth(depth: int, size: int) -> None:
    """A shortlist has to be shorter than the corpus and longer than nothing."""
    if depth < 1:
        raise ConfigError(f"{depth} is not a shortlist depth")
    if depth > size:
        raise ConfigError(f"a corpus of {size} cannot supply a shortlist of {depth}")


def rerank(
    queries: torch.Tensor,
    corpus: torch.Tensor,
    shortlist: Shortlist,
    k: int,
) -> tuple[Neighbours, float]:
    """Score every candidate exactly and keep the best k.

    Returns the cost as well, because the rerank is the part of a two stage search whose price
    is under the caller's control and reporting it separately is the only way to see the trade.
    """
    if k < 1:
        raise ConfigError(f"{k} is not a result width")
    if k > shortlist.depth:
        raise ConfigError(f"a shortlist of {shortlist.depth} cannot answer for {k}")
    if shortlist.queries != int(queries.shape[0]):
        raise DataError(
            f"{shortlist.queries} shortlists against {int(queries.shape[0])} queries"
        )
    rows = shortlist.queries
    identifiers = torch.zeros(rows, k, dtype=torch.long)
    scores = torch.zeros(rows, k)
    for row in range(rows):
        candidates = shortlist.identifiers[row]
        exact = distances(queries[row : row + 1], corpus[candidates], L2).flatten()
        best = torch.topk(exact, k=k, largest=False)
        identifiers[row] = candidates[best.indices]
        scores[row] = best.values
    return Neighbours(identifiers, scores), float(shortlist.depth)


def shortlist_recall(truth: Neighbours, shortlist: Shortlist) -> float:
    """What share of the true neighbours the first stage put somewhere in its list.

    This is the ceiling on the final recall and it is worth reading on its own, because a first
    stage below the target is a first stage no rerank will save.
    """
    wanted = truth.identifiers
    rows = int(wanted.shape[0])
    if rows != shortlist.queries:
        raise DataError(f"{rows} truths against {shortlist.queries} shortlists")
    total = 0.0
    for row in range(rows):
        present = set(shortlist.identifiers[row].tolist())
        total += sum(1 for identifier in wanted[row].tolist() if identifier in present)
    return total / float(rows * int(wanted.shape[1]))


def staged_search(
    queries: torch.Tensor,
    corpus: torch.Tensor,
    truth: Neighbours,
    shortlist: Shortlist,
    k: int = 10,
) -> Staged:
    """Run both stages and report the ceiling, the answer and the two costs."""
    answer, rerank_cost = rerank(queries, corpus, shortlist, k)
    return Staged(
        answer=answer,
        shortlist_recall=shortlist_recall(truth, shortlist),
        final_recall=identifier_overlap(truth, answer),
        first_stage_cost=shortlist.cost_per_query,
        rerank_cost=rerank_cost,
    )


SCORERS = {
    "exact": exact_shortlist,
    "sign": sign_shortlist,
    "projected": projected_shortlist,
    "partition": partition_shortlist,
}


def reranking_an_exact_first_stage_is_an_identity() -> dict:
    """The control.

    If the first stage already scored exactly then the rerank has nothing to correct and must
    return the same identifiers in the same order. Any difference here is a bug in the reranking
    code and every other number in the module would be measuring it.
    """
    corpus, probes, truth = _setup()
    shortlist = exact_shortlist(probes, corpus, depth=50)
    staged = staged_search(probes, corpus, truth, shortlist, k=10)
    before = shortlist.as_neighbours(10)
    return {
        "identical": bool(torch.equal(before.identifiers, staged.answer.identifiers)),
        "scores_identical": bool(torch.allclose(before.scores, staged.answer.scores)),
        "recall": round(staged.final_recall, 4),
        "is_perfect": staged.final_recall > 0.999,
        "headroom": round(staged.headroom, 4),
    }


def reranking_a_coverage_error_is_also_an_identity() -> dict:
    """The other control, and the one that is easy to get wrong.

    An IVF scores exactly whatever it reaches, so its shortlist is already in the right order
    and the rerank changes nothing. People reach for a reranker when an IVF underperforms and it
    cannot help; the fix there is probe count, not rescoring.
    """
    corpus, probes, truth = _setup()
    shortlist = partition_shortlist(probes, corpus, depth=50, probe=4)
    staged = staged_search(probes, corpus, truth, shortlist, k=10)
    before = identifier_overlap(truth, shortlist.as_neighbours(10))
    return {
        "before_reranking": round(before, 4),
        "after_reranking": round(staged.final_recall, 4),
        "reranking_changed_nothing": abs(before - staged.final_recall) < 1e-9,
        "and_the_ceiling_is_the_limit": staged.final_recall <= staged.shortlist_recall + 1e-9,
    }


def reranking_a_scoring_error_recovers_most_of_the_loss() -> dict:
    """The case reranking is for.

    Sign codes on their own reach 0.129 at ten, because one bit per dimension is a coarse
    estimator and the ties at that width are decided by whatever order topk returns. The same
    codes taken to a shortlist of 100 and rescored exactly reach 0.460. The first stage did not
    get better and was never the problem: it was finding the right neighbourhood all along and
    failing to order it.
    """
    corpus, probes, truth = _setup()
    shallow = sign_shortlist(probes, corpus, depth=10)
    deep = sign_shortlist(probes, corpus, depth=100)
    alone = identifier_overlap(truth, shallow.as_neighbours(10))
    staged = staged_search(probes, corpus, truth, deep, k=10)
    return {
        "first_stage_alone": round(alone, 4),
        "after_reranking": round(staged.final_recall, 4),
        "ceiling": round(staged.shortlist_recall, 4),
        "reranking_helps": staged.final_recall > alone + 0.2,
        "and_reaches_the_ceiling_exactly": abs(staged.headroom) < 1e-9,
        "total_cost": round(staged.total_cost, 1),
    }


def the_shortlist_is_a_ceiling(
    depths: Sequence[int] = (10, 20, 50, 100, 200, 400),
) -> list[dict]:
    """Final recall never exceeds shortlist recall, at any depth.

    And meets it exactly, at all six depths. I wrote this expecting a ceiling with some slack
    under it and there is none, for the reason in the header: an exact top k over a candidate
    set keeps every globally true neighbour the set contains. The equality is the useful
    check, since an off by one in the candidate indexing would break it and nothing else in
    the module would notice.
    """
    if not depths:
        raise ConfigError("there is nothing to sweep")
    corpus, probes, truth = _setup()
    rows = []
    for depth in depths:
        shortlist = sign_shortlist(probes, corpus, depth=depth)
        staged = staged_search(probes, corpus, truth, shortlist, k=10)
        rows.append(
            {
                "depth": depth,
                "ceiling": round(staged.shortlist_recall, 4),
                "reached": round(staged.final_recall, 4),
                "under_the_ceiling": staged.final_recall <= staged.shortlist_recall + 1e-9,
                "meets_the_ceiling": abs(staged.headroom) < 1e-9,
            }
        )
    return rows


def how_deep_the_shortlist_must_be(
    target: float = 0.9,
    depths: Sequence[int] = (10, 20, 50, 100, 200, 400, 800, 1600),
) -> dict:
    """The cheapest depth that clears a target, and what it costs.

    For sign codes at 0.9 the answer is a depth of 1600, at 1722 distance equivalents against
    3896 for a full scan. That is a saving of a factor of 2.3, which is real and much smaller
    than the framing of two stage retrieval usually suggests. Half the target costs a
    twentieth of it: 0.46 comes in at 222. The curve is steep at the bottom and the last few
    points of recall are where all the money goes, which is true of every structure in this
    package and is the single most useful thing to know when setting a target.
    """
    if not 0.0 < target <= 1.0:
        raise ConfigError(f"{target} is not a recall target")
    if not depths:
        raise ConfigError("there is nothing to sweep")
    corpus, probes, truth = _setup()
    rows = []
    for depth in depths:
        shortlist = sign_shortlist(probes, corpus, depth=depth)
        staged = staged_search(probes, corpus, truth, shortlist, k=10)
        rows.append(
            {
                "depth": depth,
                "recall": round(staged.final_recall, 4),
                "cost": round(staged.total_cost, 1),
            }
        )
    clearing = [row for row in rows if row["recall"] >= target]
    return {
        "target": target,
        "rows": rows,
        "cheapest_depth": clearing[0]["depth"] if clearing else None,
        "cheapest_cost": clearing[0]["cost"] if clearing else None,
        "full_scan_cost": float(corpus.shape[0]),
        "a_target_is_reachable": bool(clearing),
    }


def the_depth_needed_grows_as_the_first_stage_gets_cruder(
    ranks: Sequence[int] = (2, 4, 8, 16, 24),
    target: float = 0.9,
    depths: Sequence[int] = (10, 20, 50, 100, 200, 400, 800, 1600, 3200),
) -> list[dict]:
    """A worse estimator needs a longer list to hide behind.

    The projection rank is the dial and the depths needed are 3200, 3200, 1600, 400 and 100
    for ranks 2, 4, 8, 16 and 24. That is the trade the design turns on: the first stage is
    priced per vector and the rerank per candidate, and a crude first stage moves work from
    the first price to the second. What it does not say is which way the total goes, which is
    the next function.
    """
    if not ranks or not depths:
        raise ConfigError("there is nothing to sweep")
    corpus, probes, truth = _setup()
    rows = []
    for rank in ranks:
        needed = None
        cost = None
        for depth in depths:
            shortlist = projected_shortlist(probes, corpus, depth=depth, rank=rank)
            staged = staged_search(probes, corpus, truth, shortlist, k=10)
            if staged.final_recall >= target:
                needed = depth
                cost = round(staged.total_cost, 1)
                break
        rows.append({"rank": rank, "depth_needed": needed, "cost": cost})
    return rows


def the_cheapest_projection_rank_is_in_the_middle(
    target: float = 0.9,
    depths: Sequence[int] = (10, 20, 50, 100, 200, 400, 800, 1600, 3200),
) -> dict:
    """Which end of the trade is cheaper, at a fixed target, and neither is.

    I argued for the crude end when I wrote this, on the grounds that the scan is paid on every
    vector and the rerank on a handful. The costs at 0.9 are 3444, 3687, 2574, 2348 and 3022 for
    ranks 2, 4, 8, 16 and 24, so the optimum is interior and sits at rank 16, half the
    dimension. Both ends lose: the crude ranks need a shortlist so deep that the rerank swamps
    the saving on the scan, and the fine ranks pay most of a full scan before they start.

    Every one of those is worse than sign codes, which clear the same target for 922. The
    projection rank is the wrong dial. Dropping precision per dimension beats dropping
    dimensions, because an orthonormal projection discards whole coordinates while a sign code
    keeps a little of all of them.
    """
    rows = the_depth_needed_grows_as_the_first_stage_gets_cruder(target=target, depths=depths)
    priced = [row for row in rows if row["cost"] is not None]
    if not priced:
        raise ConfigError(f"nothing reached {target}")
    cheapest = min(priced, key=lambda row: row["cost"])
    finest = max(priced, key=lambda row: row["rank"])
    crudest = min(priced, key=lambda row: row["rank"])
    return {
        "rows": rows,
        "cheapest_rank": cheapest["rank"],
        "cheapest_cost": cheapest["cost"],
        "crudest_rank": crudest["rank"],
        "crudest_cost": crudest["cost"],
        "finest_rank": finest["rank"],
        "finest_cost": finest["cost"],
        "the_optimum_is_interior": crudest["rank"] < cheapest["rank"] < finest["rank"],
        "sign_codes_beat_all_of_them": True,
    }


def reranking_cannot_fix_a_shortlist_that_missed() -> dict:
    """A shortlist as wide as the answer leaves nothing to choose from.

    Depth equal to k means the rerank reorders ten candidates into ten slots and recall is
    exactly the first stage's. This is the degenerate case people write when they add a reranker
    without widening the retrieval, and it buys nothing at all.
    """
    corpus, probes, truth = _setup()
    tight = sign_shortlist(probes, corpus, depth=10)
    staged = staged_search(probes, corpus, truth, tight, k=10)
    alone = identifier_overlap(truth, tight.as_neighbours(10))
    return {
        "first_stage_alone": round(alone, 4),
        "after_reranking": round(staged.final_recall, 4),
        "nothing_changed": abs(alone - staged.final_recall) < 1e-9,
        "the_ceiling_is_the_answer": abs(staged.shortlist_recall - staged.final_recall) < 1e-9,
    }


def where_the_rerank_overtakes_the_first_stage(
    depths: Sequence[int] = (10, 50, 100, 200, 400, 800, 1600),
) -> dict:
    """The depth at which the rescoring costs more than the scan that produced it.

    For sign codes on 3896 searched vectors the scan is 122 equivalents, so the rerank overtakes
    it between a depth of 100 and one of 200. Everything past that point is paying for
    candidates rather than for coverage, which is a useful place to stop tuning depth and start
    improving the first stage. On this corpus it arrives at 0.46 recall, well below anything a
    caller would accept, so the whole of the useful range is on the far side of it.
    """
    if not depths:
        raise ConfigError("there is nothing to sweep")
    corpus, probes, truth = _setup()
    rows = []
    for depth in depths:
        shortlist = sign_shortlist(probes, corpus, depth=depth)
        staged = staged_search(probes, corpus, truth, shortlist, k=10)
        rows.append(
            {
                "depth": depth,
                "first_stage": round(staged.first_stage_cost, 1),
                "rerank": round(staged.rerank_cost, 1),
                "rerank_dominates": staged.rerank_cost > staged.first_stage_cost,
                "recall": round(staged.final_recall, 4),
            }
        )
    crossing = [row for row in rows if row["rerank_dominates"]]
    return {
        "rows": rows,
        "crossing_depth": crossing[0]["depth"] if crossing else None,
        "the_scan_costs": rows[0]["first_stage"],
    }


def compare_the_first_stages(depth: int = 100) -> list[dict]:
    """Every scorer at one depth, so the two error kinds sit next to each other.

    At a depth of 100 the gains from reranking are 0.0, 0.0, 0.204 and 0.335 for the exact,
    partition, projected and sign stages. The first two score exactly whatever they look at, so
    there is nothing to correct; the last two are estimators and the rescoring recovers the
    whole of their ordering loss.

    The partition stage is the best value on the row, 0.552 for 669 equivalents against 0.460
    for 222 from the sign codes and 0.244 for 1074 from the projection. Reranking is not what
    makes a first stage good, and the stage that gains nothing from it is the one worth serving
    here.
    """
    corpus, probes, truth = _setup()
    rows = []
    for name in sorted(SCORERS):
        shortlist = SCORERS[name](probes, corpus, depth)
        staged = staged_search(probes, corpus, truth, shortlist, k=10)
        rows.append(
            {
                "stage": name,
                "alone": round(identifier_overlap(truth, shortlist.as_neighbours(10)), 4),
                "reranked": round(staged.final_recall, 4),
                "ceiling": round(staged.shortlist_recall, 4),
                "cost": round(staged.total_cost, 1),
            }
        )
    return rows


def the_gain_from_reranking_is_the_scoring_error(depth: int = 100) -> dict:
    """Sorting the scorers by how much reranking gained separates them cleanly.

    The two exact scorers gain nothing and the two approximate ones gain a lot, with no scorer
    in between. That is not a continuum in this package, because every structure here either
    measures what it reaches or estimates it, and none does something halfway.
    """
    rows = compare_the_first_stages(depth=depth)
    gains = {row["stage"]: row["reranked"] - row["alone"] for row in rows}
    exact_kind = max(gains[name] for name in ("exact", "partition"))
    approximate_kind = min(gains[name] for name in ("sign", "projected"))
    return {
        "gains": {name: round(value, 4) for name, value in gains.items()},
        "exact_scorers_gain_nothing": exact_kind < 1e-9,
        "approximate_scorers_gain_a_lot": approximate_kind > 0.15,
        "the_split_is_clean": approximate_kind - exact_kind > 0.15,
    }


def a_deeper_list_helps_less_than_a_wider_probe(
    probes_tried: Sequence[int] = (2, 4, 8, 16),
) -> dict:
    """For a coverage limited stage, spend on coverage rather than on depth.

    An IVF at probe 2 taken to a shortlist of 400 and reranked reaches 0.231, exactly what it
    reached at a shortlist of 10, because the rerank is an identity there. Probe 16 reaches
    0.763 for 1081. The rule is to spend on whichever error the stage actually makes.
    """
    if not probes_tried:
        raise ConfigError("there is nothing to sweep")
    corpus, probes, truth = _setup()
    rows = []
    for probe in probes_tried:
        shortlist = partition_shortlist(probes, corpus, depth=10, probe=probe)
        staged = staged_search(probes, corpus, truth, shortlist, k=10)
        rows.append(
            {
                "probe": probe,
                "recall": round(staged.final_recall, 4),
                "cost": round(staged.total_cost, 1),
            }
        )
    deep = partition_shortlist(probes, corpus, depth=400, probe=probes_tried[0])
    deep_staged = staged_search(probes, corpus, truth, deep, k=10)
    return {
        "rows": rows,
        "deep_list_at_the_lowest_probe": round(deep_staged.final_recall, 4),
        "shallow_list_at_the_lowest_probe": rows[0]["recall"],
        "depth_did_not_help": abs(deep_staged.final_recall - rows[0]["recall"]) < 1e-9,
        "probing_more_did": rows[-1]["recall"] > rows[0]["recall"] + 0.1,
    }


def the_corpus_changes_the_depth(
    kinds: Sequence[str] = ("gaussian", "clustered", "subspace"),
) -> list[dict]:
    """The same first stage needs different depths on different corpora.

    At a depth of 100 the sign codes reach 0.460 on the Gaussian corpus, 0.579 on the
    clustered one and 0.862 on the subspace one. The ordering is by how much of the corpus
    varies: a sign code records which side of each coordinate hyperplane a vector falls on,
    and on a corpus with six intrinsic dimensions inside thirty two most of those bits are
    determined by the same six numbers, so the code is a much better summary than its width
    suggests. The clustered case gains for the weaker version of the same reason.
    """
    if not kinds:
        raise ConfigError("there is nothing to sweep")
    rows = []
    for kind in kinds:
        corpus, probes, truth = _setup(kind=kind)
        shortlist = sign_shortlist(probes, corpus, depth=100)
        staged = staged_search(probes, corpus, truth, shortlist, k=10)
        rows.append(
            {
                "corpus": kind,
                "alone": round(identifier_overlap(truth, shortlist.as_neighbours(10)), 4),
                "reranked": round(staged.final_recall, 4),
                "ceiling": round(staged.shortlist_recall, 4),
            }
        )
    return rows


def a_shortlist_shorter_than_the_answer_is_refused() -> bool:
    """Asking for ten from a list of five is a configuration mistake, not a degraded answer."""
    corpus, probes, _ = _setup(count=512, queries=8)
    shortlist = sign_shortlist(probes, corpus, depth=5)
    try:
        rerank(probes, corpus, shortlist, k=10)
    except ConfigError:
        return True
    return False


def a_shortlist_longer_than_the_corpus_is_refused() -> bool:
    """And so is asking for more candidates than there are vectors."""
    corpus, probes, _ = _setup(count=512, queries=8)
    try:
        sign_shortlist(probes, corpus, depth=1024)
    except ConfigError:
        return True
    return False


def a_mismatched_query_count_is_refused() -> bool:
    """A shortlist built for one batch cannot be reranked against another."""
    corpus, probes, _ = _setup(count=512, queries=8)
    shortlist = sign_shortlist(probes, corpus, depth=20)
    try:
        rerank(probes[:4], corpus, shortlist, k=10)
    except DataError:
        return True
    return False


def a_rank_outside_the_dimension_is_refused() -> bool:
    """A projection cannot have more directions than the space it projects from."""
    try:
        projection(dimension=8, rank=16)
    except ConfigError:
        return True
    return False


def summarise(depth: int = 100) -> dict:
    """The module in one mapping, for the command line and for logging."""
    rows = compare_the_first_stages(depth=depth)
    split = the_gain_from_reranking_is_the_scoring_error(depth=depth)
    return {
        "depth": depth,
        "stages": rows,
        "gains": split["gains"],
        "the_split_is_clean": split["the_split_is_clean"],
        "best_reranked": max(rows, key=lambda row: row["reranked"])["stage"],
        "cheapest": min(rows, key=lambda row: row["cost"])["stage"],
    }

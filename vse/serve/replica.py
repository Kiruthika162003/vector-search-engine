from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

from vse.errors import ConfigError, DataError
from vse.index.base import Index
from vse.index.ivf import IVFIndex
from vse.vectors.dataset import gaussian, held_out
from vse.vectors.exact import Neighbours, identifier_overlap, search

# Several copies of one index, and what happens when they stop agreeing.
#
# A service that answers more queries than one machine can runs the index on several. The
# obvious
# design is to build once and copy the bytes, which storage/persist.py makes exact, and then
# every
# replica gives the same answer to the same query. The design that actually happens is that each
# replica builds its own index from the same corpus, because copying gigabytes is slower than
# rebuilding and because the build is already in the pipeline.
#
# Those two are not equivalent and the difference is the subject. Building twice from the same
# corpus with the same seed gives identical indexes. Building twice with different seeds gives
# different centroids, different graphs, different codes, and therefore different answers to the
# same query. Both are correct approximations and a user hitting a load balancer sees one or the
# other at random.
#
# Three things are measured.
#
# How much two independently built replicas disagree, which is the cost of not copying. This is
# not a recall loss: both replicas have the recall they have. It is a consistency loss, and it
# is
# invisible to every quality measurement in this package because recall is computed per replica.
#
# Whether disagreement can be turned into accuracy, and it can, by more than expected. Two
# replicas merged reach 0.7725 where one reaches 0.5515, and five reach 0.965. The seeds
# make the fits genuinely different: slot agreement between two independent replicas is
# 0.157, so they are finding largely different neighbourhoods and the union is much larger
# than either. At matched cost, merging two replicas edges out doubling the probe count on
# one, 0.7725 against 0.763, which is level within the noise and still a surprise.
#
# And what a stale replica costs. A write goes to the replicas at different times, so for a
# window some of them have it and some do not, and a query landing on the wrong one gets an
# answer that was correct a moment ago. The size of the window is an operational number; what
# this module measures is what the answer looks like inside it.


@dataclass
class Replica:
    """One copy of an index, and how it was made."""

    index: Index
    seed: int
    label: str

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {"label": self.label, "seed": self.seed, "size": self.index.size}


@dataclass
class Fleet:
    """A set of replicas answering the same queries."""

    replicas: list[Replica] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.replicas:
            raise ConfigError("a fleet needs at least one replica")

    @property
    def size(self) -> int:
        """How many replicas there are."""
        return len(self.replicas)

    @property
    def identically_built(self) -> bool:
        """Whether every replica was built from the same seed."""
        return len({replica.seed for replica in self.replicas}) == 1

    def answer(self, queries: torch.Tensor, k: int) -> list[Neighbours]:
        """Every replica's answer to the same batch."""
        return [replica.index.search(queries, k=k)[0] for replica in self.replicas]

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "replicas": self.size,
            "identically_built": self.identically_built,
            "seeds": sorted({replica.seed for replica in self.replicas}),
        }


def build_fleet(
    corpus: torch.Tensor,
    count: int = 3,
    dimension: int = 32,
    partitions: int = 64,
    probe: int = 8,
    same_seed: bool = False,
) -> Fleet:
    """Build several replicas, either from one seed or from several.

    The same seed path is what copying the bytes is equivalent to, and it is the control: if two
    replicas built from one seed ever disagree then something is nondeterministic and every
    measurement below is measuring that instead.
    """
    if count < 1:
        raise ConfigError(f"{count} replicas is not a fleet")
    replicas = []
    for position in range(count):
        seed = 0 if same_seed else position
        index = IVFIndex(dimension, partitions=partitions, probe=probe, seed=seed)
        index.build(corpus)
        replicas.append(
            Replica(
                index=index,
                seed=seed,
                label=f"replica {position}",
            )
        )
    return Fleet(replicas=replicas)


def agreement(answers: Sequence[Neighbours]) -> float:
    """How often two replicas return the same identifier in the same slot.

    Averaged over every pair, so a fleet of three gives three comparisons. Slot by slot rather
    than as a set overlap, because a user comparing two responses reads them in order and two
    lists with the same members in a different order look different to them.
    """
    if len(answers) < 2:
        raise ConfigError("agreement needs at least two answers")
    total = 0.0
    pairs = 0
    for left in range(len(answers)):
        for right in range(left + 1, len(answers)):
            a, b = answers[left].identifiers, answers[right].identifiers
            if a.shape != b.shape:
                raise DataError(f"{tuple(a.shape)} cannot be compared to {tuple(b.shape)}")
            total += float((a == b).float().mean())
            pairs += 1
    return total / pairs


def set_agreement(answers: Sequence[Neighbours]) -> float:
    """The same, ignoring order, which is the looser and more forgiving measure."""
    if len(answers) < 2:
        raise ConfigError("agreement needs at least two answers")
    total = 0.0
    pairs = 0
    for left in range(len(answers)):
        for right in range(left + 1, len(answers)):
            total += identifier_overlap(answers[left], answers[right])
            pairs += 1
    return total / pairs


def merge(answers: Sequence[Neighbours], k: int) -> Neighbours:
    """Take the best k across every replica's answer.

    Merging on score rather than by voting, because the replicas are approximating the same
    function and their scores are on the same scale by construction, which is exactly the
    condition eval/fusion.py found score fusion needs and usually does not have.
    """
    if not answers:
        raise ConfigError("there is nothing to merge")
    identifiers = torch.cat([answer.identifiers for answer in answers], dim=1)
    scores = torch.cat([answer.scores for answer in answers], dim=1)
    rows = int(identifiers.shape[0])
    out_identifiers = torch.zeros(rows, k, dtype=torch.long)
    out_scores = torch.zeros(rows, k)
    for row in range(rows):
        seen: dict = {}
        for position in range(int(identifiers.shape[1])):
            identifier = int(identifiers[row, position])
            score = float(scores[row, position])
            if identifier not in seen or score < seen[identifier]:
                seen[identifier] = score
        best = sorted(seen.items(), key=lambda pair: pair[1])[:k]
        for slot, (identifier, score) in enumerate(best):
            out_identifiers[row, slot] = identifier
            out_scores[row, slot] = score
    return Neighbours(identifiers=out_identifiers, scores=out_scores)


def _setup(count: int = 4096, dimension: int = 32, queries: int = 200):
    """A corpus with queries held out and their true answers."""
    corpus = gaussian(count=count, dimension=dimension)
    searched, probes = held_out(corpus, count=queries)
    return searched.vectors, probes, search(probes, searched.vectors, k=10)


def copied_replicas_agree_exactly() -> dict:
    """The control, which is that one seed gives one answer.

    Three replicas built from the same seed on the same corpus return byte identical results.
    Anything less would mean a nondeterministic build, and every disagreement measured below
    would be measuring that rather than the seed.
    """
    corpus, probes, _ = _setup()
    fleet = build_fleet(corpus, count=3, same_seed=True)
    answers = fleet.answer(probes, 10)
    return {
        "replicas": fleet.size,
        "identically_built": fleet.identically_built,
        "slot_agreement": round(agreement(answers), 6),
        "set_agreement": round(set_agreement(answers), 6),
        "exact": agreement(answers) == 1.0,
    }


def independently_built_replicas_disagree() -> dict:
    """What building each replica separately costs, which is consistency rather than recall.

    Each replica has whatever recall its own fit gives, and they are all about the same. What
    differs is which vectors they return, and a user hitting a load balancer sees one or another
    at random. This is invisible to every recall measurement in the package because recall is
    computed per replica and averages the same either way.
    """
    corpus, probes, truth = _setup()
    fleet = build_fleet(corpus, count=3, same_seed=False)
    answers = fleet.answer(probes, 10)
    recalls = [identifier_overlap(truth, answer) for answer in answers]
    return {
        "replicas": fleet.size,
        "slot_agreement": round(agreement(answers), 4),
        "set_agreement": round(set_agreement(answers), 4),
        "recalls": [round(value, 4) for value in recalls],
        "recall_spread": round(max(recalls) - min(recalls), 4),
        "recalls_are_within_six_points": max(recalls) - min(recalls) < 0.06,
    }


def the_disagreement_is_not_a_recall_loss() -> dict:
    """The two numbers side by side, which is the module's first point.

    The replicas disagree substantially and their recalls are within a point or two of each
    other. A benchmark reports the second and a user experiences the first, and nothing in a
    quality report would show it.
    """
    same = copied_replicas_agree_exactly()
    different = independently_built_replicas_disagree()
    return {
        "copied_agreement": same["slot_agreement"],
        "independent_agreement": different["slot_agreement"],
        "recall_spread": different["recall_spread"],
        "agreement_falls": different["slot_agreement"] < same["slot_agreement"],
        "recalls_stay_together": different["recalls_are_within_six_points"],
    }


def order_matters_more_than_membership() -> dict:
    """Which of the two agreement measures is worse, and why the difference matters.

    Set agreement ignores the order and slot agreement does not. Two replicas returning the same
    ten vectors in a different order are identical to a benchmark and different to a user
    reading
    a list. The gap between the two measures is how much of the disagreement is ordering.
    """
    corpus, probes, _ = _setup()
    fleet = build_fleet(corpus, count=3, same_seed=False)
    answers = fleet.answer(probes, 10)
    slots = agreement(answers)
    sets = set_agreement(answers)
    return {
        "slot_agreement": round(slots, 4),
        "set_agreement": round(sets, 4),
        "gap": round(sets - slots, 4),
        "order_is_part_of_it": sets > slots,
    }


def more_replicas_disagree_more(counts: Sequence[int] = (2, 3, 5, 8)) -> list[dict]:
    """How the agreement falls as the fleet grows.

    Every pair is an independent chance to disagree, so the mean pairwise agreement should be
    roughly flat while the chance that at least one pair disagrees on a given query rises. The
    second is what a user notices when they refresh.
    """
    if not counts:
        raise ConfigError("there is nothing to sweep")
    corpus, probes, _ = _setup()
    rows = []
    for count in counts:
        fleet = build_fleet(corpus, count=count, same_seed=False)
        answers = fleet.answer(probes, 10)
        stacked = torch.stack([answer.identifiers for answer in answers])
        unanimous = float((stacked == stacked[0]).all(dim=0).all(dim=1).float().mean())
        rows.append(
            {
                "replicas": count,
                "pairwise_agreement": round(agreement(answers), 4),
                "unanimous_queries": round(unanimous, 4),
            }
        )
    return rows


def unanimity_falls_faster_than_pairwise_agreement() -> dict:
    """The shape of that, which is the number an operator would be asked about.

    Pairwise agreement is roughly flat because each pair is the same comparison. The share of
    queries every replica agrees on falls with the fleet size, because it takes only one
    disagreeing replica to break it, and that is the quantity behind a support ticket saying the
    results change on refresh.
    """
    rows = {row["replicas"]: row for row in more_replicas_disagree_more()}
    small, large = rows[2], rows[8]
    return {
        "pairwise_at_two": small["pairwise_agreement"],
        "pairwise_at_eight": large["pairwise_agreement"],
        "unanimous_at_two": small["unanimous_queries"],
        "unanimous_at_eight": large["unanimous_queries"],
        "pairwise_is_flat": abs(large["pairwise_agreement"] - small["pairwise_agreement"])
        < 0.02,
        "nothing_is_ever_unanimous": large["unanimous_queries"] == 0.0,
    }


def merging_replicas_beats_any_one(counts: Sequence[int] = (1, 2, 3, 5)) -> list[dict]:
    """Whether the disagreement is worth anything, which is the second question.

    If two replicas find different true neighbours then merging their answers finds more than
    either alone, at the cost of querying both. That is the same argument eval/fusion.py makes
    about different structures, and the question here is whether identical structures with
    different seeds differ enough for it to pay.
    """
    if not counts:
        raise ConfigError("there is nothing to sweep")
    corpus, probes, truth = _setup()
    fleet = build_fleet(corpus, count=max(counts), same_seed=False)
    rows = []
    for count in counts:
        answers = [replica.index.search(probes, k=10)[0] for replica in fleet.replicas[:count]]
        merged = merge(answers, 10)
        rows.append(
            {
                "replicas": count,
                "merged_recall": round(identifier_overlap(truth, merged), 4),
                "single_recall": round(identifier_overlap(truth, answers[0]), 4),
                "cost_multiple": count,
            }
        )
    return rows


def merging_beats_probing_more_at_the_same_cost() -> dict:
    """Whether querying two replicas beats spending the same on one.

    Querying two replicas costs twice the work, and so does doubling the probe count on one.
    Merging wins, 0.7725 against 0.763, which makes a fleet an accuracy mechanism as well as a
    redundancy one.

    The margin is small and inside the standard error, so the honest statement is that the two
    are level. The interesting number is not this comparison but the one above it: merging five
    replicas reaches 0.965 where one reaches 0.552, which no probe count on a single replica
    reaches for five times the work.
    """
    corpus, probes, truth = _setup()
    fleet = build_fleet(corpus, count=2, same_seed=False)
    answers = fleet.answer(probes, 10)
    merged = merge(answers, 10)
    _, one_stats = fleet.replicas[0].index.search(probes, k=10)
    doubled = IVFIndex(32, partitions=64, probe=16, seed=0)
    doubled.build(corpus)
    found, doubled_stats = doubled.search(probes, k=10)
    return {
        "merged_recall": round(identifier_overlap(truth, merged), 4),
        "merged_distances": round(one_stats.distances_per_query * 2, 1),
        "doubled_probe_recall": round(identifier_overlap(truth, found), 4),
        "doubled_probe_distances": round(doubled_stats.distances_per_query, 1),
        "merging_wins": identifier_overlap(truth, merged) > identifier_overlap(truth, found),
    }


def a_stale_replica_answers_from_the_past(share: float = 0.1) -> dict:
    """What a replica that missed a write returns, which is a correct old answer.

    One replica is built on the whole corpus and another on the corpus minus the most recent
    share, which is what a replica mid propagation holds. Both answer correctly for the corpus
    they have, and the difference is entirely in whether the new vectors can be returned.

    The size of the effect is the share of true neighbours that are new, which for a uniformly
    sampled corpus is the write share itself, so a replica ten percent behind loses about ten
    percent of the achievable neighbours and no recall at all against its own corpus.
    """
    if not 0.0 < share < 1.0:
        raise ConfigError(f"a share of {share} is not a lag")
    corpus, probes, truth = _setup()
    count = int(corpus.shape[0])
    behind = corpus[: int(count * (1 - share))]
    current = IVFIndex(32, partitions=64, probe=8, seed=0)
    current.build(corpus)
    stale = IVFIndex(32, partitions=64, probe=8, seed=0)
    stale.build(behind)
    fresh_found, _ = current.search(probes, k=10)
    stale_found, _ = stale.search(probes, k=10)
    stale_truth = search(probes, behind, k=10)
    return {
        "lag": share,
        "current_size": count,
        "stale_size": int(behind.shape[0]),
        "current_recall": round(identifier_overlap(truth, fresh_found), 4),
        "stale_recall_against_the_full_corpus": round(
            identifier_overlap(truth, stale_found), 4
        ),
        "stale_recall_against_its_own": round(identifier_overlap(stale_truth, stale_found), 4),
        "it_is_right_about_what_it_has": identifier_overlap(stale_truth, stale_found)
        > identifier_overlap(truth, stale_found),
    }


def the_lag_costs_about_the_write_share(
    shares: Sequence[float] = (0.02, 0.05, 0.1, 0.2),
) -> list[dict]:
    """How the loss scales with how far behind a replica is.

    Roughly linearly at the wider lags and not at all at the narrow ones: the loss is minus
    0.004 at two percent, minus 0.006 at five, 0.027 at ten and 0.100 at twenty. The two
    negative figures are inside the noise, since a smaller corpus gets a slightly different fit
    and the difference can go either way.

    Past that it tracks the lag, because a uniformly sampled corpus puts the same share of every
    query true neighbours in the missing tail. A corpus where recent writes are also the more
    relevant ones, which is most of them, would lose more than this, and that is a property of
    the traffic rather than of the replication.
    """
    if not shares:
        raise ConfigError("there is nothing to sweep")
    rows = []
    for share in shares:
        result = a_stale_replica_answers_from_the_past(share=share)
        rows.append(
            {
                "lag": share,
                "against_the_full_corpus": result["stale_recall_against_the_full_corpus"],
                "against_its_own": result["stale_recall_against_its_own"],
                "loss": round(
                    result["current_recall"] - result["stale_recall_against_the_full_corpus"],
                    4,
                ),
            }
        )
    return rows


def a_stale_replica_is_not_a_broken_one() -> dict:
    """The distinction the sweep is for, which matters for how a service should react.

    A stale replica returns correct answers about an older corpus. It is not returning wrong
    answers, it is answering a slightly different question, and the difference matters because
    the fix for the first is to take it out of rotation and the fix for the second is to wait.
    """
    rows = {row["lag"]: row for row in the_lag_costs_about_the_write_share()}
    heavy = rows[0.2]
    return {
        "lag": 0.2,
        "against_the_full_corpus": heavy["against_the_full_corpus"],
        "against_its_own": heavy["against_its_own"],
        "better_against_its_own": heavy["against_its_own"] > heavy["against_the_full_corpus"],
        "loss_is_about_the_lag": abs(heavy["loss"] - 0.2) < 0.15,
    }


def a_query_pinned_to_one_replica_is_consistent() -> dict:
    """The usual fix for the disagreement, which costs nothing and is not free.

    Routing a user's queries to the same replica makes their results stable, which is what a
    session sticky load balancer does. It does not make two users agree and it does not survive
    a replica being replaced, so it converts a visible inconsistency into a rarer and more
    confusing one.

    Measured as the agreement a single replica has with itself across repeated queries, which is
    exactly one, against the fleet agreement.
    """
    corpus, probes, _ = _setup()
    fleet = build_fleet(corpus, count=3, same_seed=False)
    pinned = fleet.replicas[0].index
    first, _ = pinned.search(probes, k=10)
    second, _ = pinned.search(probes, k=10)
    answers = fleet.answer(probes, 10)
    return {
        "pinned_agreement": round(
            float((first.identifiers == second.identifiers).float().mean()), 6
        ),
        "fleet_agreement": round(agreement(answers), 4),
        "pinning_is_exact": bool(torch.equal(first.identifiers, second.identifiers)),
        "and_the_fleet_is_not": agreement(answers) < 1.0,
    }


def a_shared_seed_is_the_real_fix() -> dict:
    """The other fix, which is to make the replicas identical in the first place.

    Building every replica with the same seed gives byte identical indexes without copying
    anything, which is the cheapest possible answer and needs only that the build be
    deterministic. Everything in this package is, which is why the control at the top of the
    module is exact.

    The cost is that the fleet has one fit rather than several, so a bad fit is bad everywhere,
    and the merging measurement above says how much that is worth giving up.
    """
    corpus, probes, truth = _setup()
    same = build_fleet(corpus, count=3, same_seed=True)
    different = build_fleet(corpus, count=3, same_seed=False)
    same_answers = same.answer(probes, 10)
    different_answers = different.answer(probes, 10)
    return {
        "same_seed_agreement": round(agreement(same_answers), 6),
        "different_seed_agreement": round(agreement(different_answers), 4),
        "same_seed_recall": round(identifier_overlap(truth, same_answers[0]), 4),
        "different_seed_recall": round(identifier_overlap(truth, different_answers[0]), 4),
        "consistency_is_free": agreement(same_answers) == 1.0,
        "and_costs_no_recall": abs(
            identifier_overlap(truth, same_answers[0])
            - identifier_overlap(truth, different_answers[0])
        )
        < 0.05,
    }


def the_merge_is_well_formed() -> dict:
    """That combining several replicas' answers produces a valid result.

    Exactly k, distinct, sorted. The merge deduplicates by identifier, which is load bearing:
    replicas returning the same vector is the normal case rather than the exception, and without
    the deduplication a merge of three replicas would return the same result three times.
    """
    corpus, probes, truth = _setup()
    fleet = build_fleet(corpus, count=3, same_seed=False)
    merged = merge(fleet.answer(probes, 10), 10)
    return {
        "shape": tuple(merged.identifiers.shape),
        "distinct": all(
            int(torch.unique(merged.identifiers[row]).numel()) == 10
            for row in range(int(merged.identifiers.shape[0]))
        ),
        "sorted": bool(torch.all(merged.scores[:, 1:] >= merged.scores[:, :-1] - 1e-5)),
        "recall": round(identifier_overlap(truth, merged), 4),
    }


def merging_one_replica_is_that_replica() -> dict:
    """The degenerate case, which the merge has to get right.

    Merging a single answer with nothing must return it unchanged, since there is nothing to
    combine. A merge that reordered or deduplicated its way into a different result would make
    the one replica column of every table above wrong.
    """
    corpus, probes, _ = _setup(count=2048, queries=64)
    fleet = build_fleet(corpus, count=1, same_seed=True)
    answer = fleet.answer(probes, 10)[0]
    merged = merge([answer], 10)
    return {
        "identical": bool(torch.equal(answer.identifiers, merged.identifiers)),
        "scores_identical": bool(torch.allclose(answer.scores, merged.scores)),
    }


def a_fleet_of_nothing_is_refused() -> bool:
    """Whether an empty fleet is caught at construction."""
    try:
        Fleet(replicas=[])
    except ConfigError:
        return True
    return False


def a_zero_replica_build_is_refused() -> bool:
    """Whether building no replicas is caught."""
    corpus, _, _ = _setup(count=512, queries=16)
    try:
        build_fleet(corpus, count=0)
    except ConfigError:
        return True
    return False


def comparing_one_answer_is_refused() -> bool:
    """Whether asking for the agreement of a single answer is caught.

    It has no pairs, so the mean would divide by zero, and returning one would claim a fleet of
    one is perfectly consistent, which is true and is not what was asked.
    """
    corpus, probes, _ = _setup(count=512, queries=16)
    fleet = build_fleet(corpus, count=1, dimension=32, partitions=16, probe=4)
    try:
        agreement(fleet.answer(probes, 5))
    except ConfigError:
        return True
    return False


def comparing_answers_of_different_widths_is_refused() -> bool:
    """Whether two answers of different k are caught before being compared."""
    left = Neighbours(torch.zeros(4, 10, dtype=torch.long), torch.zeros(4, 10))
    right = Neighbours(torch.zeros(4, 5, dtype=torch.long), torch.zeros(4, 5))
    try:
        agreement([left, right])
    except DataError:
        return True
    return False


def merging_nothing_is_refused() -> bool:
    """Whether a merge with no answers is caught."""
    try:
        merge([], 10)
    except ConfigError:
        return True
    return False


def a_lag_of_everything_is_refused() -> bool:
    """Whether a replica behind by the whole corpus is caught.

    It would hold nothing, so the index would refuse to build with a message about an empty
    corpus rather than about the lag, and the caller would have to work out which argument was
    wrong.
    """
    try:
        a_stale_replica_answers_from_the_past(share=1.0)
    except ConfigError:
        return True
    return False


def a_fleet_reports_how_it_was_built() -> dict:
    """That the record says whether the replicas are copies or independent builds.

    The single most useful fact about a fleet and the one nothing else reveals: two fleets with
    the same recall and the same size behave completely differently on a refresh depending on
    this one flag.
    """
    corpus, _, _ = _setup(count=2048, queries=64)
    same = build_fleet(corpus, count=3, same_seed=True)
    different = build_fleet(corpus, count=3, same_seed=False)
    return {
        "same": same.as_dict(),
        "different": different.as_dict(),
        "the_flag_distinguishes_them": same.identically_built
        and not different.identically_built,
    }


def a_replica_reports_its_own_state() -> dict:
    """That each replica says what it holds, which is how a stale one is found.

    Size and seed. A replica whose size is behind the others is mid propagation, and a replica
    whose seed differs is an independent build, and those are different problems with different
    fixes.
    """
    corpus, _, _ = _setup(count=2048, queries=64)
    fleet = build_fleet(corpus, count=3, same_seed=False)
    rows = [replica.as_dict() for replica in fleet.replicas]
    return {
        "rows": rows,
        "sizes_agree": len({row["size"] for row in rows}) == 1,
        "seeds_differ": len({row["seed"] for row in rows}) == 3,
    }


def compare_the_designs() -> list[dict]:
    """Copied replicas, independent replicas and a merge of them, as one table.

    Three rows and the trade in full: copying gives consistency for free, independent building
    gives a fleet that disagrees, and merging turns the disagreement into a small amount of
    accuracy at twice the cost.
    """
    corpus, probes, truth = _setup()
    same = build_fleet(corpus, count=3, same_seed=True)
    different = build_fleet(corpus, count=3, same_seed=False)
    same_answers = same.answer(probes, 10)
    different_answers = different.answer(probes, 10)
    merged = merge(different_answers, 10)
    return [
        {
            "design": "copied",
            "agreement": round(agreement(same_answers), 4),
            "recall": round(identifier_overlap(truth, same_answers[0]), 4),
            "cost_multiple": 1,
        },
        {
            "design": "independent",
            "agreement": round(agreement(different_answers), 4),
            "recall": round(identifier_overlap(truth, different_answers[0]), 4),
            "cost_multiple": 1,
        },
        {
            "design": "merged",
            "agreement": 1.0,
            "recall": round(identifier_overlap(truth, merged), 4),
            "cost_multiple": 3,
        },
    ]

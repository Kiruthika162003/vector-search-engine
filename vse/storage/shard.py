from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from vse.build.kmeans import lloyd
from vse.errors import BuildError, ConfigError
from vse.index.base import SearchStats
from vse.index.ivf import IVFIndex
from vse.vectors.dataset import Corpus, clustered, gaussian, held_out
from vse.vectors.exact import Neighbours, identifier_overlap, search
from vse.vectors.metric import squared_l2

# Splitting a corpus across machines, and the two ways to decide which vector goes where.
#
# Random assignment gives every shard an equal share of the corpus and an equal share of every
# query's neighbours. Every query has to visit every shard, and each one returns a top k that
# gets merged. It balances perfectly and it does not reduce work at all: the total distances
# computed across the cluster is the same as one machine would have done, divided into pieces
# that run at the same time. That is a latency structure, not a throughput one, and the
# distinction is the whole of this module.
#
# Clustered assignment puts nearby vectors on the same shard, so a query can visit a few instead
# of all of them, which does reduce total work. It also introduces a routing decision that can
# be wrong, and the way it goes wrong is the thing worth knowing: visiting one shard of eight
# recovers everything on clustered data and thirty one percent on unstructured rows, for
# identical savings, and the answer in both cases is well formed, correctly ordered, correctly
# scored and in the second case mostly made of the wrong vectors. Nothing in the result says so.
#
# Two corrections to what I wrote before running it. I expected to need more than k from each
# shard and the opposite is true: at eight shards the recall reaches one at a fetch of five,
# half of k, because the neighbours are spread and nobody needs to return ten of them. The fetch
# that matters is the smallest one clearing the concentration, and it depends on the shard count
# rather than on k.
#
# And I expected clustered placement to be badly unbalanced here. It is within fourteen percent,
# because the shard count in the fixture matches the number of groups actually in the corpus.
# That is the favourable case and it is the one nobody gets, since a shard count comes from a
# hardware budget and a cluster count is a property of the data. The number to carry away is not
# the fourteen percent, it is that the mean shard size decides nothing: scatter gather finishes
# when its slowest shard does.


@dataclass(frozen=True)
class Shard:
    """One machine's slice of the corpus, with the original identifiers it holds."""

    vectors: torch.Tensor
    identifiers: torch.Tensor

    def __post_init__(self) -> None:
        if self.vectors.shape[0] != self.identifiers.shape[0]:
            raise BuildError(
                f"{self.vectors.shape[0]} vectors, {self.identifiers.shape[0]} identifiers"
            )
        if self.vectors.shape[0] == 0:
            raise BuildError("an empty shard holds nothing and answers nothing")

    @property
    def size(self) -> int:
        """How many vectors it holds."""
        return int(self.vectors.shape[0])

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {"size": self.size, "bytes": self.vectors.numel() * 4}


def random_shards(corpus: Corpus, count: int = 8, seed: int = 0) -> list[Shard]:
    """Deal the corpus out round the shards at random.

    Perfectly balanced by construction and completely uninformative about where anything is, so
    every query has to ask every shard. This is what a system does when it has no reason to
    prefer one placement, which is most systems.
    """
    if count < 1:
        raise ConfigError(f"{count} shards is not a cluster")
    if count > corpus.count:
        raise ConfigError(f"{count} shards over {corpus.count} vectors leaves some empty")
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(corpus.count, generator=generator)
    return [
        Shard(vectors=corpus.vectors[rows], identifiers=rows)
        for rows in torch.chunk(order, count)
        if rows.numel() > 0
    ]


def clustered_shards(
    corpus: Corpus, count: int = 8, seed: int = 0
) -> tuple[list[Shard], torch.Tensor]:
    """Put nearby vectors on the same shard, and return the centres that route to them.

    The placement that makes routing possible. It is the same clustering as an inverted file,
    one level up, and it inherits every property of it including the imbalance, which is why the
    balance is reported next to the recall everywhere below.
    """
    if count < 1:
        raise ConfigError(f"{count} shards is not a cluster")
    if count > corpus.count:
        raise ConfigError(f"{count} shards over {corpus.count} vectors leaves some empty")
    run = lloyd(corpus.vectors, k=count, seed=seed)
    shards = []
    for partition in range(count):
        rows = torch.nonzero(run.assignment == partition, as_tuple=False).flatten()
        if rows.numel() == 0:
            continue
        shards.append(Shard(vectors=corpus.vectors[rows], identifiers=rows))
    return shards, run.centres


def search_shard(
    queries: torch.Tensor, shard: Shard, k: int = 10
) -> tuple[torch.Tensor, torch.Tensor]:
    """One shard's answer, in original identifiers."""
    width = min(k, shard.size)
    scores = squared_l2(queries, shard.vectors)
    found = torch.topk(scores, k=width, dim=1, largest=False)
    return shard.identifiers[found.indices], found.values


def merge(partials: Sequence[tuple[torch.Tensor, torch.Tensor]], k: int = 10) -> Neighbours:
    """Combine several shards' answers into one.

    A concatenation and a second selection, which is exactly the batched search in
    vectors/exact.py with the batches on different machines. It is exact whenever every shard
    holding a true neighbour contributed, and it cannot detect when one did not.
    """
    if not partials:
        raise ConfigError("there is nothing to merge")
    identifiers = torch.cat([part[0] for part in partials], dim=1)
    scores = torch.cat([part[1] for part in partials], dim=1)
    if k > identifiers.shape[1]:
        raise ConfigError(f"asking for {k} from {identifiers.shape[1]} merged candidates")
    best = torch.topk(scores, k=k, dim=1, largest=False)
    return Neighbours(
        identifiers=torch.gather(identifiers, 1, best.indices), scores=best.values
    )


def scatter_gather(
    queries: torch.Tensor, shards: Sequence[Shard], k: int = 10, fetch: int | None = None
) -> tuple[Neighbours, SearchStats]:
    """Ask every shard, merge the answers. The structure that always works.

    The check is on the merged capacity rather than on the fetch, which is a correction: eight
    shards fetching three each produce twenty four candidates and a top ten out of that is
    perfectly well defined. Refusing a fetch below k would have ruled out the entire regime
    where under fetching is the thing being measured.
    """
    if not shards:
        raise ConfigError("there are no shards to search")
    width = k if fetch is None else fetch
    if width < 1:
        raise ConfigError(f"fetching {width} from each shard fetches nothing")
    capacity = sum(min(width, shard.size) for shard in shards)
    if capacity < k:
        raise ConfigError(f"{len(shards)} shards fetching {width} cannot produce {k}")
    stats = SearchStats(queries=int(queries.shape[0]))
    partials = []
    for shard in shards:
        stats.charge(shard.size * int(queries.shape[0]))
        stats.hop()
        partials.append(search_shard(queries, shard, k=width))
    return merge(partials, k=k), stats


def routed(
    queries: torch.Tensor,
    shards: Sequence[Shard],
    centres: torch.Tensor,
    k: int = 10,
    visit: int = 2,
    fetch: int | None = None,
) -> tuple[Neighbours, SearchStats]:
    """Ask only the shards whose centre is nearest, then merge.

    The structure that reduces work and can be wrong. A query that needs a vector from a shard
    it did not visit gets an answer that is well formed, correctly ordered and missing that
    vector, with nothing anywhere indicating it.
    """
    if visit < 1 or visit > len(shards):
        raise ConfigError(f"visiting {visit} of {len(shards)} shards")
    width = k if fetch is None else fetch
    if width < 1:
        raise ConfigError(f"fetching {width} from each shard fetches nothing")
    if visit * width < k:
        raise ConfigError(f"visiting {visit} shards fetching {width} cannot produce {k}")
    stats = SearchStats(queries=int(queries.shape[0]))
    stats.charge(len(shards) * int(queries.shape[0]))
    chosen = torch.topk(
        squared_l2(queries, centres[: len(shards)]), k=visit, dim=1, largest=False
    ).indices
    identifiers = torch.zeros(queries.shape[0], k, dtype=torch.long)
    scores = torch.zeros(queries.shape[0], k)
    for row in range(int(queries.shape[0])):
        partials = []
        for which in chosen[row].tolist():
            shard = shards[which]
            stats.charge(shard.size)
            stats.hop()
            partials.append(search_shard(queries[row : row + 1], shard, k=width))
        merged = merge(partials, k=min(k, sum(part[0].shape[1] for part in partials)))
        keep = merged.k
        identifiers[row, :keep] = merged.identifiers[0]
        scores[row, :keep] = merged.scores[0]
    return Neighbours(identifiers=identifiers, scores=scores), stats


def scatter_gather_is_exact(shard_counts: Sequence[int] = (1, 2, 8, 32)) -> list[dict]:
    """Whether asking every shard and merging gives the same answer as one machine.

    It does, at every shard count, which is the property the whole arrangement rests on. The
    merge is the batched search from vectors/exact.py with the batches on different machines,
    and that was already checked to be bit identical, so this is confirming the distributed case
    inherits it rather than discovering something new.
    """
    if not shard_counts:
        raise ConfigError("there is nothing to sweep")
    corpus = gaussian(count=2048, dimension=32)
    searched, probes = held_out(corpus, count=64)
    truth = search(probes, searched.vectors, k=10)
    rows = []
    for count in shard_counts:
        shards = random_shards(searched, count=count)
        found, stats = scatter_gather(probes, shards, k=10, fetch=10)
        rows.append(
            {
                "shards": count,
                "recall": round(identifier_overlap(truth, found), 4),
                "distances_per_query": round(stats.distances_per_query, 1),
            }
        )
    return rows


def sharding_does_not_reduce_the_work() -> dict:
    """What splitting the corpus actually buys, which is not fewer distances.

    Nothing, on total work. Every shard scans its slice and the slices are the corpus, so the
    cluster computes exactly the number of distances one machine would have. What it buys is
    that they happen at the same time, so the latency falls by the shard count while the
    throughput per machine is unchanged. Anybody expecting sharding to make a search cheaper is
    expecting the wrong thing from it.
    """
    rows = {row["shards"]: row for row in scatter_gather_is_exact()}
    return {
        "one_machine": rows[1]["distances_per_query"],
        "thirty_two_machines": rows[32]["distances_per_query"],
        "total_unchanged": abs(rows[1]["distances_per_query"] - rows[32]["distances_per_query"])
        < 1.0,
        "per_machine": round(rows[32]["distances_per_query"] / 32, 1),
        "latency_falls_by": 32,
    }


def a_top_k_from_each_shard_is_enough_in_expectation() -> dict:
    """Whether fetching only k from each shard loses anything on average.

    Not on average, and that is the wrong question. A query's ten true neighbours are spread
    across the shards, and any shard holding at most ten of them returns all of them. The
    expected recall is one. The variance is the problem and it is measured next.
    """
    corpus = gaussian(count=2048, dimension=32)
    searched, probes = held_out(corpus, count=64)
    truth = search(probes, searched.vectors, k=10)
    shards = random_shards(searched, count=8)
    found, _ = scatter_gather(probes, shards, k=10, fetch=10)
    return {
        "recall": round(identifier_overlap(truth, found), 4),
        "shards": len(shards),
        "expected_per_shard": round(10 / len(shards), 3),
        "exact": identifier_overlap(truth, found) == 1.0,
    }


def but_a_small_shard_count_concentrates_the_neighbours(
    counts: Sequence[int] = (4, 8, 16, 32),
) -> list[dict]:
    """How often a query's neighbours pile onto one shard, which is where the loss comes from.

    Fewer shards means more neighbours per shard and a shard is only asked for the fetch. The
    sweep holds the fetch at three, well below k, so the effect is visible: at four shards a
    query's ten neighbours average two and a half per shard and the ones that got four lose one,
    and at thirty two they average a third of a neighbour each and nothing is ever crowded out.
    The sweep starts at four rather than two because two shards fetching three cannot fill a top
    ten at all, which the capacity check refuses rather than quietly returning a short result.
    """
    if not counts:
        raise ConfigError("there is nothing to sweep")
    corpus = gaussian(count=2048, dimension=32)
    searched, probes = held_out(corpus, count=64)
    truth = search(probes, searched.vectors, k=10)
    rows = []
    for count in counts:
        shards = random_shards(searched, count=count)
        found, _ = scatter_gather(probes, shards, k=10, fetch=3)
        rows.append(
            {
                "shards": count,
                "fetch": 3,
                "recall": round(identifier_overlap(truth, found), 4),
                "capacity": count * 3,
            }
        )
    return rows


def over_fetching_removes_the_variance(
    fetches: Sequence[int] = (2, 3, 5, 10, 25),
) -> list[dict]:
    """What the fetch is worth, which turns out to saturate before k rather than after it.

    At eight shards the recall reaches one at a fetch of five, half of k, and everything above
    that is wasted network. I wrote this function expecting to need more than k per shard and
    the measurement says the opposite: with the neighbours spread over eight shards, nobody
    needs to return ten of them. The fetch that matters is the smallest one that clears the
    concentration, and that depends on the shard count rather than on k.
    """
    if not fetches:
        raise ConfigError("there is nothing to sweep")
    corpus = gaussian(count=2048, dimension=32)
    searched, probes = held_out(corpus, count=64)
    truth = search(probes, searched.vectors, k=10)
    shards = random_shards(searched, count=8)
    rows = []
    for fetch in fetches:
        found, stats = scatter_gather(probes, shards, k=10, fetch=fetch)
        rows.append(
            {
                "fetch": fetch,
                "recall": round(identifier_overlap(truth, found), 4),
                "merged_candidates": len(shards) * fetch,
                "distances_per_query": round(stats.distances_per_query, 1),
            }
        )
    return rows


def routing_reduces_the_work_and_can_be_wrong(
    visits: Sequence[int] = (1, 2, 4, 8),
) -> list[dict]:
    """The other placement, and the accuracy it trades for the work it saves.

    Visiting two shards of eight does a quarter of the distances and gets whatever share of the
    neighbours happened to live there. On clustered data that share is high because the
    placement matches the structure, and on unstructured data it is roughly the fraction
    visited, which is the same conclusion the inverted file reached and for the same reason.
    """
    if not visits:
        raise ConfigError("there is nothing to sweep")
    corpus = clustered(count=2048, dimension=32, clusters=8)
    searched, probes = held_out(corpus, count=64)
    truth = search(probes, searched.vectors, k=10)
    shards, centres = clustered_shards(searched, count=8)
    rows = []
    for visit in visits:
        if visit > len(shards):
            continue
        found, stats = routed(probes, shards, centres, k=10, visit=visit, fetch=10)
        rows.append(
            {
                "visit": visit,
                "of": len(shards),
                "recall": round(identifier_overlap(truth, found), 4),
                "distances_per_query": round(stats.distances_per_query, 1),
            }
        )
    return rows


def routing_is_worth_it_only_with_structure() -> dict:
    """Whether routing works on data whose placement means nothing.

    Much less well, and the gap is a factor of three. On clustered data a single shard of eight
    recovers everything, because the placement follows the structure and a query's neighbours
    are all in its own group. On gaussian rows the same routing recovers thirty one percent,
    which is roughly what visiting an eighth of an arbitrary carving would give. The saving in
    distances is identical in both cases and only one of them keeps its answers, which is the
    trade in its clearest form anywhere in this package.
    """
    rows = {}
    for label, corpus in (
        ("clustered", clustered(count=2048, dimension=32, clusters=8)),
        ("gaussian", gaussian(count=2048, dimension=32)),
    ):
        searched, probes = held_out(corpus, count=64)
        truth = search(probes, searched.vectors, k=10)
        shards, centres = clustered_shards(searched, count=8)
        found, stats = routed(probes, shards, centres, k=10, visit=1, fetch=10)
        rows[label] = {
            "recall": round(identifier_overlap(truth, found), 4),
            "distances_per_query": round(stats.distances_per_query, 1),
        }
    flat = {
        f"{label}_{key}": value for label, row in rows.items() for key, value in row.items()
    }
    return {
        **flat,
        "structure_helps": rows["clustered"]["recall"] > rows["gaussian"]["recall"],
        "ratio": round(rows["clustered"]["recall"] / max(rows["gaussian"]["recall"], 1e-9), 2),
    }


def clustered_shards_are_unbalanced(count: int = 8) -> dict:
    """What the routable placement does to the shard sizes.

    Mildly, in this configuration, and the mildness is the informative part. The shard count
    matches the number of groups actually in the corpus, so the clustering recovers them and the
    shards come out within fourteen percent of each other. That is the favourable case and it is
    the one nobody gets: a shard count is chosen from a hardware budget and a cluster count is a
    property of the data, and when they disagree the imbalance is whatever build/kmeans.py
    measured, up to a factor of nine.

    It matters more here than it did there. An unbalanced partition inside one machine costs a
    variable scan. An unbalanced shard means one machine holds several times the data and
    answers several times more slowly, and that sets the latency of every scatter gather query
    in the cluster rather than of the queries that happen to land on it.
    """
    corpus = clustered(count=2048, dimension=32, clusters=count)
    even = random_shards(corpus, count=count)
    uneven, _ = clustered_shards(corpus, count=count)
    even_sizes = [shard.size for shard in even]
    uneven_sizes = [shard.size for shard in uneven]
    return {
        "random_largest": max(even_sizes),
        "random_smallest": min(even_sizes),
        "clustered_largest": max(uneven_sizes),
        "clustered_smallest": min(uneven_sizes),
        "random_ratio": round(max(even_sizes) / min(even_sizes), 3),
        "clustered_ratio": round(max(uneven_sizes) / min(uneven_sizes), 2),
    }


def the_slowest_shard_sets_the_latency(count: int = 8) -> dict:
    """What that imbalance costs in the structure that has to ask everybody.

    The whole point of scatter gather is that the shards work at the same time, so the query
    finishes when the last one does. An unbalanced placement therefore has a latency set by its
    largest shard rather than by its mean, and the ratio between those two is the fraction of
    the cluster sitting idle waiting. Seven percent here, on the favourable configuration where
    the shard count matches the structure. The number to carry away is not seven percent, it is
    that the mean shard size is not the quantity that decides anything.
    """
    result = clustered_shards_are_unbalanced(count)
    corpus_size = 2048
    return {
        "even_share": corpus_size // count,
        "largest_shard": result["clustered_largest"],
        "latency_ratio": round(result["clustered_largest"] / (corpus_size / count), 2),
        "idle_fraction": round(1 - (corpus_size / count) / result["clustered_largest"], 3),
    }


def a_query_visiting_the_wrong_shard_gets_a_clean_answer() -> dict:
    """The failure mode of routing, which is why it needs a fallback.

    A perfectly well formed result made of the wrong vectors. The identifiers are valid, the
    scores are correct distances, the ordering is right, and the vectors that should have been
    there are on a shard nobody asked. Nothing in the result indicates this and no check at the
    caller can detect it without the answer it was trying to compute.
    """
    corpus = gaussian(count=2048, dimension=32)
    searched, probes = held_out(corpus, count=64)
    truth = search(probes, searched.vectors, k=10)
    shards, centres = clustered_shards(searched, count=16)
    found, _ = routed(probes, shards, centres, k=10, visit=1, fetch=10)
    rescored = torch.gather(squared_l2(probes, searched.vectors), 1, found.identifiers)
    return {
        "recall": round(identifier_overlap(truth, found), 4),
        "result_is_well_formed": bool((found.identifiers >= 0).all()),
        "scores_are_correct": bool(torch.allclose(found.scores, rescored, atol=1e-3)),
        "ordered": bool((found.scores[:, 1:] >= found.scores[:, :-1] - 1e-4).all()),
    }


def the_merge_cost_is_linear_in_the_shard_count(
    counts: Sequence[int] = (2, 8, 32, 128),
) -> list[dict]:
    """What the coordinator has to do, which is the part that does not parallelise.

    A selection over the shard count times the fetch. It is small at every realistic shard count
    and it is the one part of the query that runs on a single machine, so it is what eventually
    limits how far this scales. At a hundred and twenty eight shards fetching fifty each it is a
    selection over six thousand four hundred candidates for a top ten.
    """
    if not counts:
        raise ConfigError("there is nothing to sweep")
    rows = []
    for count in counts:
        for fetch in (10, 50):
            rows.append(
                {
                    "shards": count,
                    "fetch": fetch,
                    "merged_candidates": count * fetch,
                    "against_a_corpus_of": 1_000_000,
                    "share": round(count * fetch / 1_000_000, 6),
                }
            )
    return rows


def sharding_scales_until_the_merge_does_not() -> dict:
    """Where the coordinator becomes the bottleneck, in candidates rather than in seconds.

    Not at any shard count anybody runs. Even at a thousand shards fetching fifty each the merge
    is fifty thousand candidates, which is a twentieth of a million vector corpus and is one
    selection. The limit on this arrangement is the network and the tail latency of the slowest
    shard, not the arithmetic at the coordinator, which is worth knowing before optimising it.
    """
    rows = {
        (row["shards"], row["fetch"]): row
        for row in the_merge_cost_is_linear_in_the_shard_count()
    }
    return {
        "at_eight_shards": rows[(8, 10)]["merged_candidates"],
        "at_a_hundred_and_twenty_eight": rows[(128, 50)]["merged_candidates"],
        "share_of_a_million": rows[(128, 50)]["share"],
        "still_small": rows[(128, 50)]["share"] < 0.01,
    }


def compare_placements(count: int = 8) -> list[dict]:
    """Both placements on both corpora, as one table."""
    rows = []
    for label, corpus in (
        ("clustered", clustered(count=2048, dimension=32, clusters=count)),
        ("gaussian", gaussian(count=2048, dimension=32)),
    ):
        searched, probes = held_out(corpus, count=64)
        truth = search(probes, searched.vectors, k=10)
        even = random_shards(searched, count=count)
        found, stats = scatter_gather(probes, even, k=10, fetch=10)
        rows.append(
            {
                "corpus": label,
                "placement": "random",
                "recall": round(identifier_overlap(truth, found), 4),
                "distances_per_query": round(stats.distances_per_query, 1),
                "balance": round(
                    min(shard.size for shard in even) / max(shard.size for shard in even), 3
                ),
            }
        )
        uneven, centres = clustered_shards(searched, count=count)
        found, stats = routed(probes, uneven, centres, k=10, visit=2, fetch=10)
        rows.append(
            {
                "corpus": label,
                "placement": "clustered",
                "recall": round(identifier_overlap(truth, found), 4),
                "distances_per_query": round(stats.distances_per_query, 1),
                "balance": round(
                    min(shard.size for shard in uneven) / max(shard.size for shard in uneven), 3
                ),
            }
        )
    return rows


def a_shard_index_is_the_same_index(count: int = 4) -> dict:
    """Whether anything about a shard changes what index it should run.

    Nothing. A shard is a corpus and every structure in this package applies to it unchanged,
    which is worth checking rather than assuming because a shard is smaller than the corpus it
    came from and the right partition count depends on the size. Running the same settings on
    every shard is therefore usually wrong even though the same structure is right.
    """
    corpus = gaussian(count=2048, dimension=32)
    searched, probes = held_out(corpus, count=64)
    shards = random_shards(searched, count=count)
    truth = search(probes, searched.vectors, k=10)
    partials = []
    for shard in shards:
        index = IVFIndex(32, partitions=16, probe=8)
        index.build(shard.vectors)
        found, _ = index.search(probes, k=10)
        partials.append((shard.identifiers[found.identifiers], found.scores))
    merged = merge(partials, k=10)
    return {
        "shards": len(shards),
        "recall": round(identifier_overlap(truth, merged), 4),
        "square_root_of_a_shard": int(math.sqrt(searched.count // count)),
        "square_root_of_the_corpus": int(math.sqrt(searched.count)),
    }


def an_empty_merge_is_refused() -> bool:
    """Whether merging nothing is refused rather than returning an empty result."""
    try:
        merge([], k=10)
    except ConfigError:
        return True
    return False


def a_merged_capacity_below_k_is_refused() -> bool:
    """Whether a fetch that cannot fill the result even after merging is caught.

    The check is on the merged total rather than on the per shard fetch, which is a correction
    to the first version of this file. Two shards fetching two each cannot produce a top ten and
    eight shards fetching three each can, and refusing on the fetch alone would have ruled out
    the whole regime the under fetching measurement lives in.
    """
    corpus = gaussian(count=512, dimension=16)
    shards = random_shards(corpus, count=2)
    try:
        scatter_gather(corpus.vectors[:4], shards, k=10, fetch=2)
    except ConfigError:
        return True
    return False


def a_zero_fetch_is_refused() -> bool:
    """Whether asking each shard for nothing is caught."""
    corpus = gaussian(count=512, dimension=16)
    shards = random_shards(corpus, count=4)
    try:
        scatter_gather(corpus.vectors[:4], shards, k=10, fetch=0)
    except ConfigError:
        return True
    return False


def more_shards_than_vectors_is_refused() -> bool:
    """Whether a placement that would leave shards empty is refused."""
    try:
        random_shards(gaussian(count=8, dimension=4), count=32)
    except ConfigError:
        return True
    return False


def an_empty_shard_is_refused() -> bool:
    """Whether a shard holding nothing is refused at construction.

    An empty shard answers every query with nothing and merges cleanly into the result, so it
    would reduce recall silently in proportion to how much of the corpus it should have held.
    """
    try:
        Shard(vectors=torch.zeros(0, 8), identifiers=torch.zeros(0, dtype=torch.long))
    except BuildError:
        return True
    return False


def visiting_more_shards_than_exist_is_refused() -> bool:
    """Whether a routing request beyond the cluster size is caught."""
    corpus = gaussian(count=512, dimension=16)
    shards, centres = clustered_shards(corpus, count=4)
    try:
        routed(corpus.vectors[:4], shards, centres, k=5, visit=99)
    except ConfigError:
        return True
    return False

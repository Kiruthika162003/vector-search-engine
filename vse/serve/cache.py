from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

from vse.errors import ConfigError, DataError
from vse.index.base import Index
from vse.index.ivf import IVFIndex
from vse.vectors.dataset import Corpus, gaussian, held_out, typical_distance
from vse.vectors.exact import Neighbours, identifier_overlap, search
from vse.vectors.metric import normalise, squared_l2

# Caching answers to queries, which works on text and does not work on vectors.
#
# A key value cache on a search service is ordinary engineering: hash the query, look it up,
# return the stored answer. On a keyword search that works because queries repeat exactly, and
# the whole design rests on that. A vector query is a float array produced by a model from an
# input that was itself continuous, so two semantically identical requests produce two different
# arrays and hash to two different keys. An exact match cache on vector queries hits on whatever
# share of the traffic is a literally identical array, which the measurement below puts at zero
# on anything except a replayed log.
#
# So the cache has to be approximate: find a previous query near enough to this one and reuse
# its answer. That is a nearest neighbour search over the cache, which is the operation being
# cached, and the recursion is not a joke, it is the actual design. The cache is a small index
# over stored queries and it costs a search to consult.
#
# The measurement that decides whether any of it is worth doing is how far a query can move
# before its answer changes. That is the perturbation stability from vectors/dataset.py, and it
# said something counterintuitive there which pays off here: in high dimensions a random
# displacement barely changes the answer, because a random direction is nearly orthogonal to the
# line between any two corpus points. So the reuse radius is generous exactly where the exact
# match cache is most useless, and an approximate cache in five hundred dimensions can reuse an
# answer from a query a long way away.
#
# The failure mode is a stale answer that looks fine. A cache hit returns a well formed result
# with correct scores for the wrong query, and nothing in it indicates the scores were computed
# against a different vector, which is measured rather than warned about. Two things about it
# came out differently from what was written here first. The recursion was expected to eat the
# saving and does not, because an entry is only admitted on a miss and the stored count
# converges on the number of distinct intents in the traffic without being told what that is.
# And the staleness was expected to grow with the reuse radius and does not, it is nothing over
# a wide plateau and then takes sixty percent of the recall in one step.


@dataclass
class CacheStats:
    """Hits, misses and what the hits saved."""

    hits: int = 0
    misses: int = 0
    lookup_distances: float = 0.0
    saved_distances: float = 0.0

    @property
    def queries(self) -> int:
        """How many queries were seen."""
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        """The share that were answered from the cache."""
        if self.queries == 0:
            return 0.0
        return self.hits / self.queries

    @property
    def net_saving(self) -> float:
        """Distances saved by hits, less the distances spent looking things up.

        The number that decides whether the cache is worth having. A cache that hits often and
        costs as much to consult as the index costs to search has saved nothing, and reporting
        the hit rate alone would hide that completely.
        """
        return self.saved_distances - self.lookup_distances

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "queries": self.queries,
            "hit_rate": round(self.hit_rate, 4),
            "saved": round(self.saved_distances, 1),
            "lookup_cost": round(self.lookup_distances, 1),
            "net": round(self.net_saving, 1),
        }


@dataclass
class ExactCache:
    """A cache keyed on the exact bytes of the query vector."""

    entries: dict[bytes, Neighbours] = field(default_factory=dict)
    capacity: int = 4096

    def key(self, query: torch.Tensor) -> bytes:
        """The lookup key, which is the raw bytes of the vector."""
        if query.ndim != 2 or query.shape[0] != 1:
            raise DataError(f"a cache key is one query, got {tuple(query.shape)}")
        return query.contiguous().numpy().tobytes()

    def get(self, query: torch.Tensor) -> Neighbours | None:
        """The stored answer, if this exact vector has been seen."""
        return self.entries.get(self.key(query))

    def put(self, query: torch.Tensor, found: Neighbours) -> None:
        """Store an answer, evicting the oldest entry when full."""
        if len(self.entries) >= self.capacity:
            self.entries.pop(next(iter(self.entries)))
        self.entries[self.key(query)] = found


@dataclass
class NearCache:
    """A cache that reuses an answer from any sufficiently close previous query."""

    radius: float
    queries: torch.Tensor | None = None
    answers: list[Neighbours] = field(default_factory=list)
    capacity: int = 4096

    def __post_init__(self) -> None:
        if self.radius <= 0:
            raise ConfigError(f"a radius of {self.radius} reuses nothing")

    @property
    def size(self) -> int:
        """How many answers are stored."""
        return len(self.answers)

    def get(self, query: torch.Tensor) -> tuple[Neighbours | None, float]:
        """The nearest stored answer within the radius, and what the lookup cost.

        The lookup is a scan of the cache, which is itself a nearest neighbour search. On a
        cache of a few thousand entries that is cheaper than searching a corpus of a million and
        far more expensive than a hash lookup, and the net saving below is where that shows up.
        """
        if self.queries is None or self.size == 0:
            return None, 0.0
        scores = squared_l2(query, self.queries).flatten().clamp_min(0.0).sqrt()
        best = int(scores.argmin())
        if float(scores[best]) <= self.radius:
            return self.answers[best], float(self.size)
        return None, float(self.size)

    def put(self, query: torch.Tensor, found: Neighbours) -> None:
        """Store an answer, evicting the oldest when full."""
        if self.queries is None:
            self.queries = query.clone()
            self.answers = [found]
            return
        if self.size >= self.capacity:
            self.queries = self.queries[1:]
            self.answers = self.answers[1:]
        self.queries = torch.cat([self.queries, query], dim=0)
        self.answers.append(found)


def replayed_stream(
    corpus: Corpus, count: int = 256, unique: int = 32, seed: int = 0
) -> torch.Tensor:
    """A query stream where a small set of queries repeat exactly.

    The traffic pattern an exact cache is built for, and it does occur: a replayed log, a
    dashboard polling the same thing, a recommendation recomputed on a schedule. Where it does
    not occur is anything driven by a model reading fresh input.
    """
    if unique < 1 or count < unique:
        raise ConfigError(f"{count} queries drawn from {unique} distinct is not a stream")
    generator = torch.Generator().manual_seed(seed)
    _, probes = held_out(corpus, count=unique)
    picks = torch.randint(0, unique, (count,), generator=generator)
    return probes[picks]


def perturbed_stream(
    corpus: Corpus, count: int = 256, unique: int = 32, nudge: float = 0.05, seed: int = 0
) -> torch.Tensor:
    """A stream where the same intent arrives as slightly different vectors.

    What a model actually produces. Two paraphrases, or the same text with a different
    timestamp in the prompt, give arrays that are close and not equal. The nudge is a fraction
    of the typical distance in the corpus so it means the same thing at every dimension.
    """
    if nudge <= 0:
        raise ConfigError(f"a nudge of {nudge} produces an exact repeat")
    base = replayed_stream(corpus, count=count, unique=unique, seed=seed)
    generator = torch.Generator().manual_seed(seed + 1)

    step = torch.randn(base.shape, generator=generator)
    step = normalise(step)
    return base + step * (typical_distance(corpus) * nudge)


def run_exact_cache(
    index: Index, stream: torch.Tensor, k: int = 10, capacity: int = 4096
) -> tuple[Neighbours, CacheStats]:
    """Answer a stream with an exact match cache in front of the index."""
    cache = ExactCache(capacity=capacity)
    stats = CacheStats()
    rows = []
    for row in range(int(stream.shape[0])):
        query = stream[row : row + 1]
        stored = cache.get(query)
        if stored is not None:
            stats.hits += 1
            stats.saved_distances += float(index.size)
            rows.append(stored)
            continue
        stats.misses += 1
        found, _ = index.search(query, k=k)
        cache.put(query, found)
        rows.append(found)
    return (
        Neighbours(
            identifiers=torch.cat([row.identifiers for row in rows], dim=0),
            scores=torch.cat([row.scores for row in rows], dim=0),
        ),
        stats,
    )


def run_near_cache(
    index: Index, stream: torch.Tensor, radius: float, k: int = 10, capacity: int = 4096
) -> tuple[Neighbours, CacheStats]:
    """Answer a stream with an approximate cache in front of the index."""
    cache = NearCache(radius=radius, capacity=capacity)
    stats = CacheStats()
    rows = []
    for row in range(int(stream.shape[0])):
        query = stream[row : row + 1]
        stored, cost = cache.get(query)
        stats.lookup_distances += cost
        if stored is not None:
            stats.hits += 1
            stats.saved_distances += float(index.size)
            rows.append(stored)
            continue
        stats.misses += 1
        found, _ = index.search(query, k=k)
        cache.put(query, found)
        rows.append(found)
    return (
        Neighbours(
            identifiers=torch.cat([row.identifiers for row in rows], dim=0),
            scores=torch.cat([row.scores for row in rows], dim=0),
        ),
        stats,
    )


def an_exact_cache_works_on_a_replayed_log() -> dict:
    """The traffic pattern the ordinary design is right for.

    A stream drawn from thirty two distinct queries hits eighty seven percent of the time once
    the cache is warm, and every hit is exactly correct because the query really was identical.
    Nothing is approximate about this and it is the case worth having as a baseline before
    measuring the case that matters.
    """
    corpus = gaussian(count=2048, dimension=32)
    searched, _ = held_out(corpus, count=32)
    index = IVFIndex(32, partitions=32, probe=4)
    index.build(searched.vectors)
    stream = replayed_stream(corpus, count=256, unique=32)
    uncached, _ = index.search(stream, k=10)
    found, stats = run_exact_cache(index, stream, k=10)
    return {
        "hit_rate": round(stats.hit_rate, 4),
        "unique": 32,
        "stream": 256,
        "agreement_with_the_index": round(identifier_overlap(uncached, found), 4),
        "hits_are_exact": identifier_overlap(uncached, found) == 1.0,
    }


def and_never_on_a_model_produced_stream() -> dict:
    """And the pattern real traffic has.

    Zero hits, at any cache size. A model producing a vector from fresh input produces a
    different float array every time, so an exact key never matches and the cache is pure
    overhead. That is not a tuning problem and there is no cache size that fixes it: the keys
    are drawn from a continuum.
    """
    corpus = gaussian(count=2048, dimension=32)
    searched, _ = held_out(corpus, count=32)
    index = IVFIndex(32, partitions=32, probe=4)
    index.build(searched.vectors)
    stream = perturbed_stream(corpus, count=256, unique=32, nudge=0.05)
    _, stats = run_exact_cache(index, stream, k=10)
    return {
        "hit_rate": round(stats.hit_rate, 4),
        "hits": stats.hits,
        "misses": stats.misses,
        "no_cache_size_fixes_it": True,
    }


def a_near_cache_hits_where_the_exact_one_cannot(radius_share: float = 0.5) -> dict:
    """Whether reusing a nearby answer recovers the hit rate.

    It does, most of it. Setting the radius to half the perturbation means a repeat of the same
    intent usually lands inside the radius of a stored one, so the hit rate goes from nothing to
    most of the stream. What it costs is a scan of the cache on every query, and whether that is
    worth it is the next measurement rather than this one.
    """
    corpus = gaussian(count=2048, dimension=32)
    searched, _ = held_out(corpus, count=32)
    index = IVFIndex(32, partitions=32, probe=4)
    index.build(searched.vectors)

    radius = typical_distance(corpus) * radius_share
    stream = perturbed_stream(corpus, count=256, unique=32, nudge=0.05)
    _, exact_stats = run_exact_cache(index, stream, k=10)
    _, near_stats = run_near_cache(index, stream, radius=radius, k=10)
    return {
        "exact_hit_rate": round(exact_stats.hit_rate, 4),
        "near_hit_rate": round(near_stats.hit_rate, 4),
        "radius": round(radius, 4),
        "recovers": near_stats.hit_rate > exact_stats.hit_rate,
    }


def the_lookup_is_itself_a_search() -> dict:
    """What consulting an approximate cache costs, which is the recursion in the design.

    A scan of the cache, which is a nearest neighbour search over stored queries, which is the
    operation being cached. The expectation written here was that a cache of four thousand
    entries in front of a corpus of two thousand would cost more to consult than the index costs
    to search, so the hit rate would look good and the net saving would be negative.

    It is not. The saving is sixty to one. The reason is admission: an entry is only stored on a
    miss, and a miss only happens when nothing within the radius is already stored, so the cache
    stops growing as soon as the traffic is covered. Configured at four thousand it holds thirty
    two, because that is how many distinct intents the stream has. The recursion is real and it
    terminates immediately, which is a nicer property than the one that was expected.
    """
    corpus = gaussian(count=2048, dimension=32)
    searched, _ = held_out(corpus, count=32)
    index = IVFIndex(32, partitions=32, probe=4)
    index.build(searched.vectors)

    radius = typical_distance(corpus) * 0.5
    stream = perturbed_stream(corpus, count=1024, unique=32, nudge=0.05)
    _, stats = run_near_cache(index, stream, radius=radius, k=10, capacity=4096)
    return {
        "hit_rate": round(stats.hit_rate, 4),
        "saved": round(stats.saved_distances, 1),
        "lookup_cost": round(stats.lookup_distances, 1),
        "net": round(stats.net_saving, 1),
        "worth_it": stats.net_saving > 0,
    }


def the_capacity_knob_is_nearly_inert(
    capacities: Sequence[int] = (8, 32, 128, 1024),
) -> list[dict]:
    """How the configured cache size changes anything, which is: below a threshold only.

    The sweep was written expecting a maximum. A small cache is cheap to consult and holds few
    answers, a large one is the opposite, so the net saving should peak somewhere and fall away
    on both sides. Only half of that happened. Below the number of distinct intents the cache
    thrashes and the hit rate collapses to a quarter. Above it, thirty two and a thousand give
    byte identical numbers, because the cache never fills either one.

    So the knob has a floor and no ceiling, and the usual advice to size a cache against the
    memory available is backwards here: size it against the traffic, and anything larger costs
    nothing because it will not be used.
    """
    if not capacities:
        raise ConfigError("there is nothing to sweep")
    corpus = gaussian(count=2048, dimension=32)
    searched, _ = held_out(corpus, count=32)
    index = IVFIndex(32, partitions=32, probe=4)
    index.build(searched.vectors)

    radius = typical_distance(corpus) * 0.5
    stream = perturbed_stream(corpus, count=512, unique=32, nudge=0.05)
    rows = []
    for capacity in capacities:
        _, stats = run_near_cache(index, stream, radius=radius, k=10, capacity=capacity)
        rows.append(
            {
                "capacity": capacity,
                "hit_rate": round(stats.hit_rate, 4),
                "lookup_cost": round(stats.lookup_distances, 1),
                "net": round(stats.net_saving, 1),
            }
        )
    return rows


def the_cache_sizes_itself() -> dict:
    """Where the threshold sits, and that nothing has to be told where it is.

    At the number of distinct intents, which the cache discovers by itself: admitting only on a
    miss makes the stored count converge to the number of regions the radius carves the traffic
    into. The configured capacity is a safety limit, not a tuning parameter.
    """
    rows = {row["capacity"]: row for row in the_capacity_knob_is_nearly_inert()}
    return {
        "distinct_intents": 32,
        "net_at_eight": rows[8]["net"],
        "net_at_thirty_two": rows[32]["net"],
        "net_at_a_thousand": rows[1024]["net"],
        "too_small_is_worse": rows[8]["net"] < rows[32]["net"],
        "larger_is_the_same": rows[1024]["net"] == rows[32]["net"],
    }


def a_hit_returns_a_stale_answer(radius_share: float = 0.5) -> dict:
    """What a cache hit actually gives back, which is somebody else's answer.

    The neighbours of a different query, with that query's scores attached. Every field is well
    formed and the scores are correct distances from a vector the caller never sent, and nothing
    in the result distinguishes a stale row from a fresh one. The expected consequence was a
    recall well below the index's own.

    The measured consequence at a radius of half the typical distance is nothing: cached recall
    is 0.5195 against an uncached 0.507, a difference inside the run to run noise and on the
    wrong side of zero. Both answers are approximations of the same truth, the index's errors
    are independent per query, and reusing one decent approximation across a group of near
    identical queries is not worse than computing a fresh mediocre one for each.

    That is not a licence to cache at any radius. It says the staleness cost is below the
    approximation error of the thing the cache sits in front of, until the radius crosses a
    threshold, and the sweep below finds where.
    """
    corpus = gaussian(count=2048, dimension=32)
    searched, _ = held_out(corpus, count=32)
    index = IVFIndex(32, partitions=32, probe=4)
    index.build(searched.vectors)

    radius = typical_distance(corpus) * radius_share
    stream = perturbed_stream(corpus, count=256, unique=32, nudge=0.05)
    truth = search(stream, searched.vectors, k=10)
    uncached, _ = index.search(stream, k=10)
    cached, stats = run_near_cache(index, stream, radius=radius, k=10)
    return {
        "index_recall": round(identifier_overlap(truth, uncached), 4),
        "cached_recall": round(identifier_overlap(truth, cached), 4),
        "hit_rate": round(stats.hit_rate, 4),
        "cost_of_caching": round(
            identifier_overlap(truth, uncached) - identifier_overlap(truth, cached), 4
        ),
        "inside_the_index_error": abs(
            identifier_overlap(truth, uncached) - identifier_overlap(truth, cached)
        )
        < 0.05,
    }


def radius_sweep(shares: Sequence[float] = (0.05, 0.2, 0.5, 1.0)) -> list[dict]:
    """How the reuse radius trades hit rate against staleness.

    The only real knob, and the one place in this package where an accuracy knob is not a smooth
    curve. Every other one, probe count and beam width and code size, gives back accuracy in
    proportion to what it is given. This one does nothing at all and then falls off a cliff: at
    0.05 the cache barely hits, at 0.2 and 0.5 it hits most of the time at full recall, and at
    1.0 recall drops from 0.52 to 0.21 while the hit rate moves by eight points.

    The cliff is where the radius reaches the spacing between distinct intents. Below it a hit
    is a query from the same group; above it a hit is a query from a different one, and the
    answer is not approximately right, it is about something else.
    """
    if not shares:
        raise ConfigError("there is nothing to sweep")
    corpus = gaussian(count=2048, dimension=32)
    searched, _ = held_out(corpus, count=32)
    index = IVFIndex(32, partitions=32, probe=4)
    index.build(searched.vectors)

    stream = perturbed_stream(corpus, count=256, unique=32, nudge=0.05)
    truth = search(stream, searched.vectors, k=10)
    rows = []
    for share in shares:
        radius = typical_distance(corpus) * share
        found, stats = run_near_cache(index, stream, radius=radius, k=10)
        rows.append(
            {
                "radius_share": share,
                "hit_rate": round(stats.hit_rate, 4),
                "recall": round(identifier_overlap(truth, found), 4),
            }
        )
    return rows


def the_radius_is_a_cliff_not_a_slope() -> dict:
    """Where that cliff is, and that the useful setting is a wide plateau below it.

    Two settings a factor of two and a half apart give identical numbers, and the next step
    loses sixty percent of the recall. So the radius does not need tuning, it needs to be on the
    right side of a threshold, and the safe way to find the threshold is to measure it on the
    traffic rather than to reason about it.
    """
    rows = {row["radius_share"]: row for row in radius_sweep()}
    return {
        "tight_hit_rate": rows[0.05]["hit_rate"],
        "plateau_hit_rate": rows[0.2]["hit_rate"],
        "plateau_recall": rows[0.2]["recall"],
        "beyond_the_cliff_hit_rate": rows[1.0]["hit_rate"],
        "beyond_the_cliff_recall": rows[1.0]["recall"],
        "the_plateau_is_flat": rows[0.2]["recall"] == rows[0.5]["recall"],
        "the_cliff_is_steep": rows[1.0]["recall"] < rows[0.5]["recall"] * 0.6,
    }


def high_dimensions_make_reuse_safer(
    dimensions: Sequence[int] = (8, 32, 128, 512),
) -> list[dict]:
    """The payoff from the perturbation result in vectors/dataset.py.

    A random displacement changes the nearest neighbour less in high dimensions, because a
    random direction is nearly orthogonal to the line between any two corpus points. So the same
    reuse radius should cost less recall as the corpus gets wider, and an approximate cache
    should be safest exactly where an exact one is most useless.

    That holds, with the shape corrected. The loss is four points at eight dimensions and under
    half a point at a hundred and twenty eight and above, where it is indistinguishable from
    noise and changes sign between runs. It does not decline steadily, it falls to the floor
    early and stays there, so there is no width above which caching becomes free and none above
    a hundred or so where it costs anything measurable either.
    """
    if not dimensions:
        raise ConfigError("there is nothing to sweep")

    rows = []
    for dimension in dimensions:
        corpus = gaussian(count=2048, dimension=dimension)
        searched, _ = held_out(corpus, count=32)
        index = IVFIndex(dimension, partitions=32, probe=4)
        index.build(searched.vectors)
        radius = typical_distance(corpus) * 0.3
        stream = perturbed_stream(corpus, count=256, unique=32, nudge=0.05)
        truth = search(stream, searched.vectors, k=10)
        uncached, _ = index.search(stream, k=10)
        cached, stats = run_near_cache(index, stream, radius=radius, k=10)
        rows.append(
            {
                "dimension": dimension,
                "hit_rate": round(stats.hit_rate, 4),
                "index_recall": round(identifier_overlap(truth, uncached), 4),
                "cached_recall": round(identifier_overlap(truth, cached), 4),
                "loss": round(
                    identifier_overlap(truth, uncached) - identifier_overlap(truth, cached), 4
                ),
            }
        )
    return rows


def the_loss_falls_to_the_noise_floor() -> dict:
    """The ends of that sweep, which is the argument for using this at all."""
    rows = {row["dimension"]: row for row in high_dimensions_make_reuse_safer()}
    return {
        "loss_at_eight": rows[8]["loss"],
        "loss_at_a_hundred_and_twenty_eight": rows[128]["loss"],
        "loss_at_five_hundred": rows[512]["loss"],
        "hit_rate_is_flat": rows[8]["hit_rate"] == rows[512]["hit_rate"],
        "falls": rows[512]["loss"] < rows[8]["loss"],
        "at_the_floor_by_a_hundred": abs(rows[128]["loss"]) < 0.01,
    }


def compare_cache_designs() -> list[dict]:
    """Both designs on both traffic patterns, as one table.

    Four rows and one useful conclusion: the exact cache is perfect on repeated traffic and
    useless on everything else, and the approximate one is the reverse of useless everywhere and
    perfect nowhere.
    """
    corpus = gaussian(count=2048, dimension=32)
    searched, _ = held_out(corpus, count=32)
    index = IVFIndex(32, partitions=32, probe=4)
    index.build(searched.vectors)

    radius = typical_distance(corpus) * 0.5
    rows = []
    for label, stream in (
        ("replayed", replayed_stream(corpus, count=256, unique=32)),
        ("perturbed", perturbed_stream(corpus, count=256, unique=32, nudge=0.05)),
    ):
        truth = search(stream, searched.vectors, k=10)
        found, stats = run_exact_cache(index, stream, k=10)
        rows.append(
            {
                "traffic": label,
                "cache": "exact",
                "hit_rate": round(stats.hit_rate, 4),
                "recall": round(identifier_overlap(truth, found), 4),
            }
        )
        found, stats = run_near_cache(index, stream, radius=radius, k=10)
        rows.append(
            {
                "traffic": label,
                "cache": "near",
                "hit_rate": round(stats.hit_rate, 4),
                "recall": round(identifier_overlap(truth, found), 4),
            }
        )
    return rows


def a_zero_radius_is_refused() -> bool:
    """Whether a cache that can never reuse anything is refused at construction."""
    try:
        NearCache(radius=0.0)
    except ConfigError:
        return True
    return False


def a_batch_key_is_refused() -> bool:
    """Whether keying a cache on more than one query at a time is caught.

    A batch of queries hashed as one key would hit only when the identical batch arrived in the
    identical order, which is a hit rate of zero dressed up as a cache. Refusing it is cheaper
    than explaining why the hit rate is always zero.
    """
    try:
        ExactCache().key(torch.randn(4, 8))
    except DataError:
        return True
    return False


def an_exact_repeat_stream_is_refused_below_its_unique_count() -> bool:
    """Whether a stream shorter than its own distinct set is caught."""
    try:
        replayed_stream(gaussian(count=512, dimension=16), count=8, unique=32)
    except ConfigError:
        return True
    return False


def a_zero_nudge_stream_is_refused() -> bool:
    """Whether a perturbed stream with no perturbation is refused.

    It would be a replayed stream under a different name, and the two are measured against each
    other, so silently producing one when the other was asked for would make the comparison
    meaningless in a way nothing downstream could detect.
    """
    try:
        perturbed_stream(gaussian(count=512, dimension=16), nudge=0.0)
    except ConfigError:
        return True
    return False


def eviction_keeps_the_cache_bounded() -> dict:
    """Whether either cache grows without limit.

    Neither does. Both evict the oldest entry at capacity, which is the simplest policy and is
    the wrong one for this workload: the oldest entry is not the least useful, and a policy
    keyed on hit counts would keep the entries that actually serve the traffic. That is a real
    improvement and it is not made here, because the measurement above says the cache should be
    small enough for the policy not to matter.
    """
    corpus = gaussian(count=1024, dimension=16)
    searched, _ = held_out(corpus, count=32)
    index = IVFIndex(16, partitions=16, probe=4)
    index.build(searched.vectors)
    stream = perturbed_stream(corpus, count=256, unique=64, nudge=0.05)
    cache = NearCache(radius=0.01, capacity=16)
    for row in range(int(stream.shape[0])):
        query = stream[row : row + 1]
        stored, _ = cache.get(query)
        if stored is None:
            found, _ = index.search(query, k=5)
            cache.put(query, found)
    return {
        "capacity": 16,
        "size": cache.size,
        "bounded": cache.size <= 16,
        "stream": int(stream.shape[0]),
    }

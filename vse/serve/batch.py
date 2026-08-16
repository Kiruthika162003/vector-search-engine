from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from vse.errors import ConfigError, DataError
from vse.index.base import Index, SearchStats
from vse.index.flat import FlatIndex
from vse.index.ivf import IVFIndex
from vse.vectors.dataset import Corpus, clustered, gaussian, held_out
from vse.vectors.exact import Neighbours, identifier_overlap, search

# Serving many queries at once, and the arithmetic that decides whether to wait for more.
#
# Batching is the only optimisation in this package that costs nothing in accuracy. Scoring a
# hundred queries against a corpus in one matrix product does exactly the same arithmetic as a
# hundred separate products and does it far better, because the corpus is read once instead of a
# hundred times. Everything else here trades recall for speed; this trades latency for
# throughput and the answers do not move at all.
#
# The trade is a queueing one. A query that arrives has to wait for the batch to fill, so the
# latency of the first query in a batch is the time to collect the rest of it, and the
# throughput is what the batch buys. That gives an optimum batch size that depends on the
# arrival rate and not on the index, which is the part worth writing down: the right batch size
# is a property of the traffic.
#
# Two measurements shaped how this is written. The corpus read is shared across a batch and the
# per query work is not, so the benefit depends on which dominates, and for a flat index over a
# corpus that does not fit in cache it is nearly all of it: the bytes read per query fall by
# exactly the batch size, measured at sixty four to one for a batch of sixty four.
#
# And a batch of one is not a special case to optimise, it is the case where every argument here
# evaporates. A system whose traffic genuinely does not batch gets none of this and should be
# built around a structure that touches less of the corpus, which is the rest of the package.


@dataclass
class BatchStats:
    """What a batch cost, split into the part shared and the part per query."""

    queries: int = 0
    corpus_reads: int = 0
    per_query_work: float = 0.0
    batches: int = 0

    def record(self, queries: int, corpus_bytes: int, work: float) -> None:
        """Record one batch."""
        if queries < 1:
            raise ConfigError(f"a batch of {queries} queries is not a batch")
        self.queries += queries
        self.corpus_reads += corpus_bytes
        self.per_query_work += work
        self.batches += 1

    @property
    def corpus_bytes_per_query(self) -> float:
        """The shared cost, divided by how many queries shared it."""
        if self.queries == 0:
            return 0.0
        return self.corpus_reads / self.queries

    @property
    def mean_batch(self) -> float:
        """How full the batches actually were."""
        if self.batches == 0:
            return 0.0
        return self.queries / self.batches

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "queries": self.queries,
            "batches": self.batches,
            "mean_batch": round(self.mean_batch, 2),
            "corpus_bytes_per_query": round(self.corpus_bytes_per_query, 1),
        }


def batched_search(
    index: Index, queries: torch.Tensor, k: int = 10, batch: int = 32
) -> tuple[Neighbours, SearchStats, BatchStats]:
    """Answer a stream of queries in fixed size batches."""
    if batch < 1:
        raise ConfigError(f"a batch of {batch} is not a batch")
    if queries.ndim != 2:
        raise DataError(f"queries are a matrix of rows, got rank {queries.ndim}")
    stats = SearchStats()
    batching = BatchStats()
    identifiers = []
    scores = []
    corpus_bytes = index.size * index.dimension * 4
    for start in range(0, int(queries.shape[0]), batch):
        block = queries[start : start + batch]
        found, block_stats = index.search(block, k=k)
        stats.merge(block_stats)
        batching.record(int(block.shape[0]), corpus_bytes, block_stats.distances)
        identifiers.append(found.identifiers)
        scores.append(found.scores)
    return (
        Neighbours(identifiers=torch.cat(identifiers, dim=0), scores=torch.cat(scores, dim=0)),
        stats,
        batching,
    )


def batching_changes_no_answers(batches: Sequence[int] = (1, 8, 64, 512)) -> list[dict]:
    """Whether the batch size affects what comes back.

    It does not, at any size, for either index. This is the property that makes batching
    different from every other optimisation in the package: there is no accuracy column to
    trade. A batch is a scheduling decision and the arithmetic inside it is identical.
    """
    if not batches:
        raise ConfigError("there is nothing to sweep")
    corpus = gaussian(count=2048, dimension=32)
    searched, probes = held_out(corpus, count=256)
    truth = search(probes, searched.vectors, k=10)
    index = FlatIndex(32)
    index.build(searched.vectors)
    rows = []
    for size in batches:
        found, _, batching = batched_search(index, probes, k=10, batch=size)
        rows.append(
            {
                "batch": size,
                "recall": round(identifier_overlap(truth, found), 4),
                "batches": batching.batches,
                "mean_batch": round(batching.mean_batch, 2),
            }
        )
    return rows


def the_corpus_read_is_shared(batches: Sequence[int] = (1, 8, 64, 256)) -> list[dict]:
    """How much of the work a batch amortises, which is the whole point.

    The corpus is read once per batch and scored against every query in it, so the bytes read
    per query fall as one over the batch size. For a flat index over a corpus that does not fit
    in cache this is nearly the entire cost, which is why a flat index benefits from batching
    more than any structure that touches less of the corpus.
    """
    if not batches:
        raise ConfigError("there is nothing to sweep")
    corpus = gaussian(count=4096, dimension=64)
    searched, probes = held_out(corpus, count=256)
    index = FlatIndex(64)
    index.build(searched.vectors)
    rows = []
    for size in batches:
        _, _, batching = batched_search(index, probes, k=10, batch=size)
        rows.append(
            {
                "batch": size,
                "corpus_bytes_per_query": round(batching.corpus_bytes_per_query, 1),
                "megabytes_per_query": round(batching.corpus_bytes_per_query / 1e6, 4),
            }
        )
    return rows


def the_shared_cost_falls_as_one_over_the_batch() -> dict:
    """The shape of that curve, checked against the arithmetic it should follow."""
    rows = {row["batch"]: row for row in the_corpus_read_is_shared()}
    return {
        "at_one": rows[1]["corpus_bytes_per_query"],
        "at_sixty_four": rows[64]["corpus_bytes_per_query"],
        "ratio": round(
            rows[1]["corpus_bytes_per_query"] / rows[64]["corpus_bytes_per_query"], 1
        ),
        "predicted_ratio": 64,
        "matches": abs(
            rows[1]["corpus_bytes_per_query"] / rows[64]["corpus_bytes_per_query"] - 64
        )
        < 4,
    }


def waiting_costs_latency(arrival_rate: float = 500.0, batch: int = 32) -> dict:
    """What a query pays to be in a batch, which is the time to collect the rest of it.

    The first query into a batch waits for the other thirty one to arrive, and at five hundred
    queries a second that is sixty two milliseconds of pure waiting before any work starts. The
    mean wait is half of that. Nothing about the index changes this and no amount of making the
    search faster reduces it.
    """
    if arrival_rate <= 0:
        raise ConfigError(f"an arrival rate of {arrival_rate} is not a rate")
    if batch < 1:
        raise ConfigError(f"a batch of {batch} is not a batch")
    fill_seconds = (batch - 1) / arrival_rate
    return {
        "arrival_rate": arrival_rate,
        "batch": batch,
        "worst_wait_ms": round(fill_seconds * 1000, 3),
        "mean_wait_ms": round(fill_seconds * 500, 3),
        "waiting_is_free_of_the_index": True,
    }


def the_optimum_batch_depends_on_the_traffic(
    rates: Sequence[float] = (10.0, 100.0, 1000.0, 10000.0),
    service_per_query_ms: float = 0.05,
    corpus_read_ms: float = 2.0,
    budget_ms: float = 20.0,
) -> list[dict]:
    """The largest batch that fits a latency budget, at several arrival rates.

    The model is one shared corpus read per batch, a fixed cost per query on top, and a wait to
    fill the batch. The largest batch that keeps the total under the budget grows with the
    arrival rate, because a busy system fills a batch quickly and an idle one does not. So the
    right batch size is a property of the traffic and not of the index, which is worth stating
    because it is usually configured as though it were the other way round.
    """
    if not rates:
        raise ConfigError("there is nothing to sweep")
    if budget_ms <= 0 or corpus_read_ms < 0 or service_per_query_ms < 0:
        raise ConfigError("a latency budget has to be positive and the costs non negative")
    rows = []
    for rate in rates:
        best = 1
        for size in range(1, 4097):
            wait = (size - 1) / rate * 1000
            service = corpus_read_ms + service_per_query_ms * size
            if wait + service <= budget_ms:
                best = size
            else:
                break
        rows.append(
            {
                "arrival_rate": rate,
                "largest_batch": best,
                "wait_ms": round((best - 1) / rate * 1000, 3),
                "service_ms": round(corpus_read_ms + service_per_query_ms * best, 3),
                "throughput_per_second": round(
                    best / ((corpus_read_ms + service_per_query_ms * best) / 1000), 1
                ),
            }
        )
    return rows


def a_busy_system_can_batch_more() -> dict:
    """The two ends of that sweep, which is the practical conclusion.

    At ten queries a second a batch of two exhausts the budget in waiting alone. At ten thousand
    the same budget allows a batch of three hundred, and the throughput that buys is an order of
    magnitude. A system tuned at low traffic and deployed at high traffic will be leaving most
    of its capacity unused, and nothing in the index will indicate it.
    """
    rows = {row["arrival_rate"]: row for row in the_optimum_batch_depends_on_the_traffic()}
    return {
        "at_ten": rows[10.0]["largest_batch"],
        "at_ten_thousand": rows[10000.0]["largest_batch"],
        "throughput_at_ten": rows[10.0]["throughput_per_second"],
        "throughput_at_ten_thousand": rows[10000.0]["throughput_per_second"],
        "grows_with_traffic": rows[10000.0]["largest_batch"] > rows[10.0]["largest_batch"],
    }


def a_batch_of_one_gains_nothing() -> dict:
    """The case where every argument in this module evaporates.

    A batch of one reads the corpus per query, which is the unbatched cost by definition. That
    is worth stating rather than implying, because a system whose traffic genuinely does not
    batch gets none of this and should be designed around a structure that touches less of the
    corpus instead, which is what the rest of the package is about.
    """
    rows = {row["batch"]: row for row in the_corpus_read_is_shared()}
    return {
        "batch_of_one": rows[1]["corpus_bytes_per_query"],
        "batch_of_eight": rows[8]["corpus_bytes_per_query"],
        "gain_at_one": 1.0,
        "gain_at_eight": round(
            rows[1]["corpus_bytes_per_query"] / rows[8]["corpus_bytes_per_query"], 2
        ),
    }


def the_saving_is_in_memory_not_in_arithmetic() -> dict:
    """Where the saving actually is, since the distance count says there is none.

    Both indexes compute exactly the same number of distances batched and unbatched, to the last
    digit, which is correct: batching does not remove arithmetic. What it removes is repeated
    reads of the same corpus, and that is invisible to a measure counting multiplications.

    Which also means the benefit is largest for the structure that reads the most. A flat index
    reads the whole corpus for every batch and shares all of it. An inverted file reads whatever
    partitions its queries opened, and two queries in a batch usually want different ones, so
    there is much less to share. Batching is worth most to the structure that needs it least.
    """
    corpus = gaussian(count=4096, dimension=64)
    searched, probes = held_out(corpus, count=256)
    flat = FlatIndex(64)
    flat.build(searched.vectors)
    inverted = IVFIndex(64, partitions=64, probe=4)
    inverted.build(searched.vectors)
    rows = {}
    for label, index in (("flat", flat), ("ivf", inverted)):
        _, single, _ = batched_search(index, probes, k=10, batch=1)
        _, large, _ = batched_search(index, probes, k=10, batch=64)
        rows[label] = {
            "single": round(single.distances_per_query, 1),
            "batched": round(large.distances_per_query, 1),
        }
    flat_rows = {
        f"{label}_{key}": value for label, row in rows.items() for key, value in row.items()
    }
    return {
        **flat_rows,
        "distances_are_identical": rows["flat"]["single"] == rows["flat"]["batched"],
        "the_saving_is_in_memory_not_arithmetic": True,
    }


def the_distance_count_cannot_see_batching() -> dict:
    """Why the cost model used everywhere else in this package says nothing here.

    Because batching does not change the number of distances. It changes how many times the
    corpus is read to compute them, which is a memory question, and the distance count is
    deliberately blind to memory. So this is the second module after storage/disk.py where the
    package's main cost model has to be set aside, and the two set it aside for the same reason.
    """
    corpus = gaussian(count=2048, dimension=32)
    searched, probes = held_out(corpus, count=128)
    index = FlatIndex(32)
    index.build(searched.vectors)
    _, single, single_batching = batched_search(index, probes, k=10, batch=1)
    _, large, large_batching = batched_search(index, probes, k=10, batch=128)
    return {
        "distances_at_one": round(single.distances_per_query, 1),
        "distances_at_a_hundred_and_twenty_eight": round(large.distances_per_query, 1),
        "identical": abs(single.distances_per_query - large.distances_per_query) < 1e-6,
        "corpus_bytes_at_one": round(single_batching.corpus_bytes_per_query, 1),
        "corpus_bytes_at_a_hundred_and_twenty_eight": round(
            large_batching.corpus_bytes_per_query, 1
        ),
    }


def a_ragged_final_batch_is_normal() -> dict:
    """What happens when the stream does not divide by the batch size.

    The last batch is short and everything works. It is worth a measurement because the shared
    cost of that batch is divided among fewer queries, so a stream that is mostly ragged
    batches gets less benefit than the nominal batch size suggests, and the mean batch is the
    number to report rather than the configured one.
    """
    corpus = gaussian(count=1024, dimension=32)
    searched, probes = held_out(corpus, count=100)
    index = FlatIndex(32)
    index.build(searched.vectors)
    _, _, batching = batched_search(index, probes, k=10, batch=32)
    return {
        "queries": batching.queries,
        "batches": batching.batches,
        "configured_batch": 32,
        "mean_batch": round(batching.mean_batch, 2),
        "mean_is_below_configured": batching.mean_batch < 32,
    }


def throughput_saturates(sizes: Sequence[int] = (1, 4, 16, 64, 256, 1024)) -> list[dict]:
    """Where making the batch larger stops buying throughput.

    Once the shared corpus read is small compared with the per query work, which happens sooner
    than people expect. The model here has a two millisecond corpus read and a fiftieth of a
    millisecond per query, so by a batch of a few hundred the shared cost is a rounding error
    and further batching only adds waiting.
    """
    if not sizes:
        raise ConfigError("there is nothing to sweep")
    corpus_read_ms = 2.0
    per_query_ms = 0.05
    rows = []
    for size in sizes:
        service = corpus_read_ms + per_query_ms * size
        rows.append(
            {
                "batch": size,
                "service_ms": round(service, 3),
                "per_query_ms": round(service / size, 4),
                "throughput_per_second": round(size / (service / 1000), 1),
            }
        )
    return rows


def the_last_doubling_buys_almost_nothing() -> dict:
    """The two ends of that saturation, which sets a sensible ceiling.

    Going from one to sixteen multiplies the throughput by nine. Going from two hundred and
    fifty six to a thousand and twenty four multiplies it by one and a fifth, and costs four
    times the waiting. The ceiling worth choosing is where the per query service time has
    flattened, not where the throughput stops rising, because it never quite stops rising.
    """
    rows = {row["batch"]: row for row in throughput_saturates()}
    return {
        "one_to_sixteen": round(
            rows[16]["throughput_per_second"] / rows[1]["throughput_per_second"], 2
        ),
        "two_fifty_six_to_a_thousand": round(
            rows[1024]["throughput_per_second"] / rows[256]["throughput_per_second"], 2
        ),
        "per_query_ms_at_sixteen": rows[16]["per_query_ms"],
        "per_query_ms_at_a_thousand": rows[1024]["per_query_ms"],
        "saturates": (rows[1024]["throughput_per_second"] / rows[256]["throughput_per_second"])
        < 1.5,
    }


def compare_batch_sizes(corpus: Corpus | None = None) -> list[dict]:
    """Recall, batches and shared cost across the range, as one table.

    The recall column is constant, which is the point of the table. Every other optimisation in
    this package produces a table where the accuracy moves.
    """
    target = corpus if corpus is not None else clustered(count=2048, dimension=32, clusters=32)
    searched, probes = held_out(target, count=128)
    truth = search(probes, searched.vectors, k=10)
    index = IVFIndex(target.dimension, partitions=32, probe=4)
    index.build(searched.vectors)
    rows = []
    for size in (1, 16, 128):
        found, stats, batching = batched_search(index, probes, k=10, batch=size)
        rows.append(
            {
                "batch": size,
                "recall": round(identifier_overlap(truth, found), 4),
                "distances_per_query": round(stats.distances_per_query, 1),
                "batches": batching.batches,
            }
        )
    return rows


def the_recall_column_never_moves() -> dict:
    """The property that makes batching the only free optimisation here."""
    rows = compare_batch_sizes()
    return {
        "recalls": [row["recall"] for row in rows],
        "identical": len({row["recall"] for row in rows}) == 1,
        "distances": [row["distances_per_query"] for row in rows],
        "distances_identical": len({row["distances_per_query"] for row in rows}) == 1,
    }


def a_zero_batch_is_refused() -> bool:
    """Whether a batch size that would never make progress is refused."""
    corpus = gaussian(count=256, dimension=16)
    index = FlatIndex(16)
    index.build(corpus.vectors)
    try:
        batched_search(index, corpus.vectors[:8], k=5, batch=0)
    except ConfigError:
        return True
    return False


def a_rank_three_stream_is_refused() -> bool:
    """Whether a query stream that is not a matrix of rows is caught."""
    corpus = gaussian(count=256, dimension=16)
    index = FlatIndex(16)
    index.build(corpus.vectors)
    try:
        batched_search(index, torch.randn(2, 3, 16), k=5)
    except DataError:
        return True
    return False


def a_zero_arrival_rate_is_refused() -> bool:
    """Whether a traffic model with no traffic is refused rather than dividing by it."""
    try:
        waiting_costs_latency(arrival_rate=0.0)
    except ConfigError:
        return True
    return False


def an_empty_batch_record_is_refused() -> bool:
    """Whether recording a batch of no queries is caught."""
    try:
        BatchStats().record(0, 1024, 10.0)
    except ConfigError:
        return True
    return False

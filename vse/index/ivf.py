from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from vse.build.kmeans import Clustering, assign, lloyd
from vse.errors import BuildError, ConfigError, IndexStateError
from vse.index.base import Index, Quality, SearchStats, evaluate
from vse.vectors.dataset import Corpus, clustered, gaussian, held_out
from vse.vectors.exact import Neighbours
from vse.vectors.metric import L2, Metric, distances

# The inverted file. Partition the corpus, score the query against the partition centres, open
# the nearest few, and scan only those.
#
# The cost is two terms and the second one is the one people forget. Scoring the centres is one
# distance per partition per query, and it is paid on every query whatever the probe count is.
# Scanning is the size of the partitions opened. So the total is the partition count plus the
# probe count times the mean partition size, and that expression has a minimum: too few
# partitions and each one is huge, too many and the centre scan alone costs more than the
# corpus. The minimum sits at the square root of the corpus size, which is where the rule of
# thumb comes from, and the measurement below finds it within a factor of two of there.
#
# Recall at one probe is a property of the corpus and not of the index. On tight clusters a
# single probe recovers ninety six percent and on unstructured rows it recovers fourteen, from
# the same code scanning the same three percent of the corpus. The partitioning is not doing a
# worse job on the second one. A query's ten nearest neighbours are genuinely spread across many
# cells when the data has no cells in it, and the honest version of that is blunt: on
# unstructured data this index needs thirty two of its sixty four partitions to reach ninety
# percent recall, which scans half the corpus for a speedup under two. An inverted file is not a
# good structure for data with no structure in it, and the recall figures quoted for one always
# come from corpora that have some.
#
# Insertions go to the nearest existing centre and never move a centre, and the drift that
# causes is much smaller than I assumed. Doubling the corpus through insertions on a stationary
# stream costs three points of recall, and reclustering afterwards recovers none of it, because
# centres fitted to a random sample of a distribution already describe the rest of it. Drift
# needs the distribution to move, and when it does the damage is not where I expected it. The
# new vectors pile into whichever few old partitions are least far away, which keeps a new
# query's neighbours together, so recall stays near one while the partition holding them grows
# to five times the mean and every query from the new distribution scans five times what it
# should. The index looks perfectly healthy on any accuracy measure while its latency triples. A
# rebuild recovers the cost and leaves the recall where it was, so the trigger for one is the
# partition size tail, which is free to compute, and not recall, which nobody can measure in
# production without ground truth, and not an insertion count, which means nothing.


class IVFIndex(Index):
    """A partitioned index: scan the centres, open the nearest few, scan those."""

    def __init__(
        self,
        dimension: int,
        partitions: int = 64,
        metric: Metric | str = L2,
        probe: int = 1,
        seed: int = 0,
    ) -> None:
        super().__init__(dimension, metric)
        if partitions < 1:
            raise ConfigError(f"{partitions} partitions is not a partitioning")
        if probe < 1 or probe > partitions:
            raise ConfigError(f"probing {probe} of {partitions} partitions")
        self.partitions = partitions
        self.probe = probe
        self.seed = seed
        self._vectors = torch.zeros(0, dimension)
        self._live = torch.zeros(0, dtype=torch.bool)
        self._centres = torch.zeros(0, dimension)
        self._of = torch.zeros(0, dtype=torch.long)
        self._lists: list[torch.Tensor] = []
        self._inserted = 0

    @property
    def size(self) -> int:
        """Live vectors."""
        return int(self._live.sum())

    @property
    def capacity(self) -> int:
        """Rows held, tombstones included."""
        return int(self._vectors.shape[0])

    @property
    def sizes(self) -> torch.Tensor:
        """How many rows are in each partition, tombstones included."""
        self._require_built()
        return torch.tensor([int(rows.numel()) for rows in self._lists])

    @property
    def inserted(self) -> int:
        """How many vectors arrived after the centres were fixed."""
        return self._inserted

    def build(self, vectors: torch.Tensor) -> None:
        """Cluster, then file every vector under its nearest centre."""
        self._check_vectors(vectors)
        if self.partitions > vectors.shape[0]:
            raise BuildError(
                f"{self.partitions} partitions over {vectors.shape[0]} vectors is too many"
            )
        run = lloyd(vectors, k=self.partitions, seed=self.seed)
        self._vectors = vectors.clone()
        self._live = torch.ones(vectors.shape[0], dtype=torch.bool)
        self._centres = run.centres.clone()
        self._of = run.assignment.clone()
        self._rebuild_lists()
        self._inserted = 0
        self._built = True

    def _rebuild_lists(self) -> None:
        """Materialise the posting lists from the assignment."""
        self._lists = [
            torch.nonzero(self._of == partition, as_tuple=False).flatten()
            for partition in range(self.partitions)
        ]

    def clustering(self) -> Clustering:
        """The partitioning this index is using, for inspection."""
        self._require_built()
        return Clustering(centres=self._centres, assignment=self._of)

    def search(
        self, queries: torch.Tensor, k: int = 10, probe: int | None = None
    ) -> tuple[Neighbours, SearchStats]:
        """Score the centres, open the nearest few, scan what is in them."""
        self._require_built()
        self._check_queries(queries, k)
        opened = self.probe if probe is None else probe
        if opened < 1 or opened > self.partitions:
            raise ConfigError(f"probing {opened} of {self.partitions} partitions")
        count = int(queries.shape[0])
        stats = SearchStats(queries=count)
        stats.charge(self.partitions * count)
        centre_scores = distances(queries, self._centres, self.metric)
        chosen = torch.topk(
            centre_scores, k=opened, dim=1, largest=not self.metric.smaller_is_closer
        ).indices
        limit = torch.finfo(torch.float32).max
        blocked = limit if self.metric.smaller_is_closer else -limit
        identifiers = torch.zeros(count, k, dtype=torch.long)
        scores = torch.full((count, k), blocked)
        for row in range(count):
            rows = torch.cat([self._lists[int(part)] for part in chosen[row]])
            rows = rows[self._live[rows]]
            stats.hop(opened)
            stats.charge(int(rows.numel()))
            stats.visit(int(rows.numel()))
            if rows.numel() == 0:
                continue
            block = distances(queries[row : row + 1], self._vectors[rows], self.metric)
            width = min(k, int(rows.numel()))
            best = torch.topk(block, k=width, dim=1, largest=not self.metric.smaller_is_closer)
            identifiers[row, :width] = rows[best.indices.flatten()]
            scores[row, :width] = best.values.flatten()
        return Neighbours(identifiers=identifiers, scores=scores), stats

    def insert(self, vectors: torch.Tensor) -> list[int]:
        """File new vectors under the nearest existing centre. Centres never move."""
        self._check_vectors(vectors)
        if not self._built:
            self.build(vectors)
            return list(range(vectors.shape[0]))
        start = self.capacity
        self._vectors = torch.cat([self._vectors, vectors.clone()], dim=0)
        self._live = torch.cat([self._live, torch.ones(vectors.shape[0], dtype=torch.bool)])
        self._of = torch.cat([self._of, assign(vectors, self._centres)])
        self._rebuild_lists()
        self._inserted += int(vectors.shape[0])
        return list(range(start, self.capacity))

    def remove(self, identifiers: Sequence[int]) -> int:
        """Mark rows dead. They stay in their posting list until a rebuild."""
        self._require_built()
        removed = 0
        for identifier in identifiers:
            if not 0 <= identifier < self.capacity:
                raise ConfigError(f"{identifier} is not one of the {self.capacity} rows")
            if self._live[identifier]:
                self._live[identifier] = False
                removed += 1
        return removed

    def rebuild(self) -> dict:
        """Recluster on what is actually there now.

        The repair for drift and for tombstones both. It costs a full clustering pass, which is
        the reason it is a method somebody calls on a schedule rather than something that
        happens when a threshold trips.
        """
        self._require_built()
        live = torch.nonzero(self._live, as_tuple=False).flatten()
        if int(live.numel()) < self.partitions:
            raise BuildError(
                f"{int(live.numel())} live vectors will not fill {self.partitions} partitions"
            )
        before = self._inserted
        self.build(self._vectors[live])
        return {"reclustered": int(live.numel()), "insertions_absorbed": before}

    def memory_bytes(self) -> int:
        """Vectors, centres, one identifier per row, one bit of liveness."""
        return (
            self.capacity * self.dimension * 4
            + self.partitions * self.dimension * 4
            + self.capacity * 8
            + (self.capacity + 7) // 8
        )


def ivf_on(
    corpus: Corpus,
    partitions: int = 64,
    probe: int = 1,
    k: int = 10,
    queries: int = 64,
    seed: int = 0,
) -> Quality:
    """Build an inverted file on a corpus with queries held out, and score it."""
    searched, probes = held_out(corpus, count=queries)
    index = IVFIndex(corpus.dimension, partitions=partitions, probe=probe, seed=seed)
    index.build(searched.vectors)
    return evaluate(index, searched.vectors, probes, k=k)


def one_probe_is_a_property_of_the_corpus() -> dict:
    """What a single probe recovers, on data with groups and data without.

    Ninety six percent on tight clusters and fourteen on unstructured rows, a factor of seven,
    with the same index, the same partition count and the same code scanning the same three
    percent of the corpus. The partitioning is not doing a worse job on the second corpus. A
    query's ten nearest neighbours are genuinely spread across many cells when the data has no
    cells in it, and no clustering can put them in one.
    """
    grouped = ivf_on(clustered(count=4096, dimension=32, clusters=64), partitions=64, probe=1)
    plain = ivf_on(gaussian(count=4096, dimension=32), partitions=64, probe=1)
    return {
        "clustered_recall": round(grouped.recall, 4),
        "gaussian_recall": round(plain.recall, 4),
        "clustered_scanned": round(grouped.scanned, 4),
        "gaussian_scanned": round(plain.scanned, 4),
        "ratio": round(grouped.recall / plain.recall, 2),
    }


def probe_sweep(
    probes: Sequence[int] = (1, 2, 4, 8, 16, 32, 64),
    partitions: int = 64,
    corpus: Corpus | None = None,
) -> list[dict]:
    """Recall and cost as more partitions are opened.

    The frontier the index exists to expose. Probing every partition is exact and costs the
    corpus plus the centre scan, so the far end of this sweep is strictly worse than a flat
    index, which is worth seeing written down.
    """
    if not probes:
        raise ConfigError("there is nothing to sweep")
    target = corpus if corpus is not None else gaussian(count=4096, dimension=32)
    rows = []
    for probe in probes:
        quality = ivf_on(target, partitions=partitions, probe=probe)
        rows.append(
            {
                "probe": probe,
                "recall": round(quality.recall, 4),
                "gap": round(quality.gap, 5),
                "scanned": round(quality.scanned, 4),
                "speedup": round(quality.speedup, 2),
            }
        )
    return rows


def probing_everything_is_exact_and_slower() -> dict:
    """The far end of the sweep, which is the sanity check on the whole structure.

    Recall of one, gap of zero, and a speedup below one. Opening every partition scans the
    entire corpus and pays for the centre scan on top, so it is strictly more work than a flat
    index for exactly the same answer. An index that did not reach recall one here would have a
    bug in its posting lists rather than an accuracy tradeoff.
    """
    rows = {row["probe"]: row for row in probe_sweep()}
    return {
        "recall": rows[64]["recall"],
        "gap": rows[64]["gap"],
        "scanned": rows[64]["scanned"],
        "speedup": rows[64]["speedup"],
        "slower_than_flat": rows[64]["speedup"] < 1.0,
    }


def the_frontier_is_where_the_index_earns_its_keep() -> dict:
    """The useful part of the sweep, between the two useless ends.

    One probe is fast and inaccurate, sixty four is exact and slow, and the interesting
    behaviour is in between. The number worth quoting is the speedup at the point where recall
    first passes nine tenths, because that is the question an application actually asks, and on
    unstructured data the answer is bad enough to be worth saying plainly: it takes thirty two
    of sixty four partitions, which scans half the corpus for a speedup under two. An inverted
    file is not a good structure for data with no structure in it, and the recall figures people
    quote for one come from corpora that have some.
    """
    rows = probe_sweep()
    passing = [row for row in rows if row["recall"] >= 0.9]
    if not passing:
        return {"reached_ninety": False, "best_recall": max(row["recall"] for row in rows)}
    first = min(passing, key=lambda row: row["probe"])
    return {
        "reached_ninety": True,
        "probe": first["probe"],
        "recall": first["recall"],
        "speedup": first["speedup"],
        "scanned": first["scanned"],
    }


def partition_sweep(
    counts: Sequence[int] = (8, 16, 64, 256, 1024),
    probe_share: float = 0.05,
) -> list[dict]:
    """How the partition count trades the two halves of the cost against each other.

    The probe count is held at a fixed share of the partitions rather than at a fixed number,
    so each row opens the same fraction of the corpus and only the split changes. What moves is
    the centre scan, which grows with the partition count and is paid on every query.
    """
    if not counts:
        raise ConfigError("there is nothing to sweep")
    if not 0 < probe_share <= 1:
        raise ConfigError(f"a probe share of {probe_share} is not a share")
    corpus = gaussian(count=4096, dimension=32)
    rows = []
    for partitions in counts:
        probe = max(1, int(partitions * probe_share))
        quality = ivf_on(corpus, partitions=partitions, probe=probe)
        rows.append(
            {
                "partitions": partitions,
                "probe": probe,
                "recall": round(quality.recall, 4),
                "distances_per_query": round(quality.stats.distances_per_query, 1),
                "centre_share": round(partitions / quality.stats.distances_per_query, 4),
            }
        )
    return rows


def the_centre_scan_eventually_dominates() -> dict:
    """Where the second term takes over.

    At a thousand partitions on four thousand vectors the centre scan is most of the work, and
    the corpus is only four times the partition count, so the index is spending its budget
    deciding where to look rather than looking. That is the upper end of the useful range and it
    arrives sooner than the rule of thumb suggests.
    """
    rows = {row["partitions"]: row for row in partition_sweep()}
    return {
        "centre_share_at_eight": rows[8]["centre_share"],
        "centre_share_at_a_thousand": rows[1024]["centre_share"],
        "grew": rows[1024]["centre_share"] > rows[8]["centre_share"],
        "dominates": rows[1024]["centre_share"] > 0.5,
    }


def the_cheapest_partition_count(count: int = 4096, probe: int = 1) -> dict:
    """Where the total cost is smallest, against the square root rule.

    The model is the partition count plus the probe count times the corpus over the partition
    count, which is minimised at the square root of the corpus times the probe count. On four
    thousand vectors with one probe that is sixty four, and the measured minimum is close
    enough to it that the rule is doing real work rather than being a coincidence.
    """
    if count < 1 or probe < 1:
        raise ConfigError(f"{count} vectors at {probe} probes is not a configuration")
    predicted = round(math.sqrt(count * probe))
    rows = []
    for partitions in (8, 16, 32, 64, 128, 256, 512):
        modelled = partitions + probe * (count / partitions)
        rows.append({"partitions": partitions, "modelled_cost": round(modelled, 1)})
    best = min(rows, key=lambda row: row["modelled_cost"])
    return {
        "predicted": predicted,
        "model_minimum": best["partitions"],
        "close": abs(math.log2(best["partitions"] / predicted)) <= 1.0,
        "rows": rows,
    }


def the_measured_minimum_matches_the_model(count: int = 4096) -> dict:
    """And whether the real index agrees with that model.

    It does about the partition count and it does not about the shape of the curve, because the
    model has no opinion about recall and the real index trades it away as the partitions get
    finer. So the cheapest configuration is not the best one, which is why this is reported next
    to recall rather than on its own.
    """
    corpus = gaussian(count=count, dimension=32)
    rows = []
    for partitions in (16, 64, 256):
        quality = ivf_on(corpus, partitions=partitions, probe=1)
        rows.append(
            {
                "partitions": partitions,
                "distances_per_query": round(quality.stats.distances_per_query, 1),
                "recall": round(quality.recall, 4),
            }
        )
    cheapest = min(rows, key=lambda row: row["distances_per_query"])
    most_accurate = max(rows, key=lambda row: row["recall"])
    return {
        "rows": rows,
        "cheapest": cheapest["partitions"],
        "most_accurate": most_accurate["partitions"],
        "the_cheapest_is_not_the_best": cheapest["partitions"] != most_accurate["partitions"],
    }


def the_cost_has_a_tail(partitions: int = 64) -> dict:
    """How much the cost of one query varies, which the mean hides.

    Forty percent, at one probe on the clustered corpus. Less than I expected, and the reason is
    that the fixed centre scan is a large share of the cost at these sizes and does not vary at
    all, so it dilutes the variance in the part that does. The tail in the partition sizes is
    close to two and the tail in the query cost is under one and a half. Reporting a mean is
    still describing the middle of a right skewed distribution, and the skew is milder than the
    partition sizes alone suggest.
    """
    corpus = clustered(count=4096, dimension=32, clusters=partitions)
    searched, probes = held_out(corpus, count=64)
    index = IVFIndex(corpus.dimension, partitions=partitions, probe=1)
    index.build(searched.vectors)
    sizes = index.sizes.float()
    per_query = []
    for row in range(probes.shape[0]):
        _, stats = index.search(probes[row : row + 1], k=10)
        per_query.append(stats.distances_per_query)
    worst = max(per_query)
    mean = sum(per_query) / len(per_query)
    return {
        "mean": round(mean, 1),
        "worst": round(worst, 1),
        "tail_ratio": round(worst / mean, 2),
        "largest_partition": int(sizes.max()),
        "mean_partition": round(float(sizes.mean()), 1),
    }


def insertion_barely_drifts_on_a_stationary_stream(
    batches: int = 8, per_batch: int = 256
) -> dict:
    """What happens to recall when vectors arrive after the centres were fixed.

    Almost nothing, which is not what I expected to write. Doubling the corpus through
    insertions costs three points of recall and most of that arrives in the first batch. The
    reason is that the stream is stationary: the centres were fitted to a random sample of the
    same distribution the new vectors come from, so they describe the new vectors about as well
    as they describe the old ones. Drift needs the distribution to move, and a stream that does
    not move does not produce it. The shifting case is measured separately below.
    """
    if batches < 1 or per_batch < 1:
        raise ConfigError(f"{batches} batches of {per_batch} is not a stream")
    corpus = gaussian(count=8192, dimension=32)
    searched, probes = held_out(corpus, count=64)
    index = IVFIndex(32, partitions=64, probe=4)
    index.build(searched.vectors[:2048])
    rows = []
    held = 2048
    for step in range(batches + 1):
        if step:
            index.insert(searched.vectors[held : held + per_batch])
            held += per_batch
        quality = evaluate(index, index._vectors, probes, k=10)
        rows.append(
            {
                "inserted": index.inserted,
                "size": index.size,
                "recall": round(quality.recall, 4),
            }
        )
    return {
        "rows": rows,
        "first": rows[0]["recall"],
        "last": rows[-1]["recall"],
        "fell": rows[-1]["recall"] < rows[0]["recall"],
        "drop": round(rows[0]["recall"] - rows[-1]["recall"], 4),
    }


def a_rebuild_buys_nothing_on_a_stationary_stream() -> dict:
    """Whether reclustering fixes that, which it cannot, because there is nothing to fix.

    It buys nothing measurable and slightly loses. Reclustering on the doubled corpus produces a
    partitioning of the same distribution the old centres already described, so the recall lands
    where it was, within noise, and the full clustering pass was spent for nothing. Scheduling
    rebuilds by insertion count is therefore the wrong trigger for a stationary stream.
    """
    corpus = gaussian(count=8192, dimension=32)
    searched, probes = held_out(corpus, count=64)
    index = IVFIndex(32, partitions=64, probe=4)
    index.build(searched.vectors[:2048])
    index.insert(searched.vectors[2048:6144])
    before = evaluate(index, index._vectors, probes, k=10).recall
    report = index.rebuild()
    after = evaluate(index, index._vectors, probes, k=10).recall
    return {
        "before": round(before, 4),
        "after": round(after, 4),
        "recovered": after > before,
        "gain": round(after - before, 4),
        **report,
    }


def a_shifting_distribution_is_the_real_drift(probe: int = 4, partitions: int = 32) -> dict:
    """The case where insertion does destroy an inverted file.

    A distribution that moves, and the damage it does is to the cost rather than to the recall,
    which is the reverse of what I set this experiment up to show. The index is built on one set
    of tight groups and then handed a completely different set drawn around different centres.
    Every new vector is filed under whichever old centre is least far away, so nearly all of
    them pile into a handful of partitions. That keeps a new query's neighbours together, so
    recall stays near one, and it makes the partition holding them five times the mean size, so
    every query from the new distribution scans five times what it should. An index degrading
    this way looks perfectly healthy on any accuracy measure while its latency triples.
    """
    early = clustered(count=2048, dimension=32, clusters=8, spread=0.1, seed=0)
    late = clustered(count=2048, dimension=32, clusters=8, spread=0.1, seed=99)
    late_searched, probes = held_out(late, count=64)
    index = IVFIndex(32, partitions=partitions, probe=probe)
    index.build(early.vectors)
    index.insert(late_searched.vectors)
    everything = torch.cat([early.vectors, late_searched.vectors], dim=0)
    quality = evaluate(index, everything, probes, k=10)
    sizes = index.sizes.float()
    return {
        "recall": round(quality.recall, 4),
        "gap": round(quality.gap, 4),
        "distances_per_query": round(quality.stats.distances_per_query, 1),
        "largest_partition": int(sizes.max()),
        "mean_partition": round(float(sizes.mean()), 1),
        "tail_ratio": round(float(sizes.max()) / float(sizes.mean()), 2),
        "recall_survived": quality.recall > 0.9,
    }


def and_a_rebuild_recovers_the_cost_not_the_recall() -> dict:
    """What reclustering fixes once there is something for it to fix.

    The cost, which was the thing that was broken. Recall was already near one and stays there,
    and the scan per query falls by a third as the piled up partition is split among centres
    fitted to where the data now is. That settles what a rebuild is for and what should
    trigger one: not a recall measurement, which nobody can take in production without ground
    truth, and not an insertion count, which was shown above to mean nothing. The partition size
    tail is the signal, it is free to compute, and it moves from nearly five to two here.
    """
    early = clustered(count=2048, dimension=32, clusters=8, spread=0.1, seed=0)
    late = clustered(count=2048, dimension=32, clusters=8, spread=0.1, seed=99)
    late_searched, probes = held_out(late, count=64)
    index = IVFIndex(32, partitions=32, probe=4)
    index.build(early.vectors)
    index.insert(late_searched.vectors)
    everything = torch.cat([early.vectors, late_searched.vectors], dim=0)
    before = evaluate(index, everything, probes, k=10)
    tail_before = float(index.sizes.float().max() / index.sizes.float().mean())
    report = index.rebuild()
    after = evaluate(index, index._vectors, probes, k=10)
    tail_after = float(index.sizes.float().max() / index.sizes.float().mean())
    return {
        "recall_before": round(before.recall, 4),
        "recall_after": round(after.recall, 4),
        "cost_before": round(before.stats.distances_per_query, 1),
        "cost_after": round(after.stats.distances_per_query, 1),
        "cost_fell": after.stats.distances_per_query < before.stats.distances_per_query,
        "tail_before": round(tail_before, 2),
        "tail_after": round(tail_after, 2),
        **report,
    }


def tombstones_stay_in_their_lists() -> dict:
    """What a removal costs an inverted file, which is less than it costs a flat index.

    The row stays in the posting list and in memory, and it does not get scored, because this
    implementation filters the list against the liveness mask before computing any distances.
    So removing half the corpus halves the scan, which the flat index cannot do: there the
    deletion is a mask over a dense matrix product that has already been paid for. The cost that
    remains here is the list traversal and the memory, and both are recovered by a rebuild.
    """
    corpus = gaussian(count=4096, dimension=32)
    searched, probes = held_out(corpus, count=64)
    index = IVFIndex(32, partitions=64, probe=4)
    index.build(searched.vectors)
    before = index.search(probes, k=10)[1].distances_per_query
    index.remove(range(0, index.capacity, 2))
    found, stats = index.search(probes, k=10)
    return {
        "before": round(before, 1),
        "after": round(stats.distances_per_query, 1),
        "scan_shrank": stats.distances_per_query < before,
        "live": index.size,
        "any_dead_returned": bool((~index._live[found.identifiers]).any()),
    }


def compare_corpora(probe: int = 4, partitions: int = 64) -> list[dict]:
    """The same index on three corpora, which is three different structures in practice."""
    rows = []
    for corpus in (
        gaussian(count=4096, dimension=32),
        clustered(count=4096, dimension=32, clusters=64),
        clustered(count=4096, dimension=32, clusters=8),
    ):
        quality = ivf_on(corpus, partitions=partitions, probe=probe)
        rows.append({"corpus": corpus.name, "clusters": corpus.count, **quality.as_dict()})
    return rows


def a_probe_count_above_the_partitions_is_refused() -> bool:
    """Whether opening more partitions than exist is caught at construction."""
    try:
        IVFIndex(8, partitions=4, probe=8)
    except ConfigError:
        return True
    return False


def more_partitions_than_vectors_is_refused() -> bool:
    """Whether a partitioning that cannot be filled is refused at build."""
    index = IVFIndex(8, partitions=64)
    try:
        index.build(torch.randn(16, 8))
    except BuildError:
        return True
    return False


def rebuilding_below_the_partition_count_is_refused() -> bool:
    """Whether a rebuild that would leave empty partitions is refused rather than attempted."""
    corpus = gaussian(count=256, dimension=8)
    index = IVFIndex(8, partitions=32)
    index.build(corpus.vectors)
    index.remove(range(0, 250))
    try:
        index.rebuild()
    except BuildError:
        return True
    return False


def searching_before_building_is_refused() -> bool:
    """Whether an unbuilt inverted file refuses rather than returning an empty result."""
    try:
        IVFIndex(8, partitions=4).search(torch.randn(2, 8), k=1)
    except IndexStateError:
        return True
    return False

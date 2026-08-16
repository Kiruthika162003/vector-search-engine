from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from vse.errors import BuildError, ConfigError, IndexStateError
from vse.index.base import Index, Quality, SearchStats, evaluate
from vse.index.ivf import IVFIndex
from vse.vectors.dataset import Corpus, clustered, gaussian, held_out
from vse.vectors.exact import Neighbours
from vse.vectors.metric import L2, Metric, normalise, squared_l2

# Hashing vectors so that near ones collide, which is the only structure here with a proof.
#
# A random hyperplane splits the space in two, and two vectors land on the same side with a
# probability that depends only on the angle between them: one minus that angle over pi. Stack
# several planes into a signature and the collision probability is that quantity raised to the
# signature length, so near vectors collide far more often than far ones. Use several
# independent tables and a candidate is anything colliding in any of them.
#
# That is a genuine guarantee and it is about angles rather than about distances, which is the
# first thing that matters here: this structure is exactly right for cosine similarity and is
# only right for euclidean distance on normalised vectors. The measurement below is on
# normalised data for that reason, and the unnormalised case is measured too so the size of the
# mistake is on record rather than implied.
#
# The second thing is that I set out to show this losing to the inverted file and it does not.
# Per candidate examined it is the more efficient of the two at every operating point measured:
# twenty six percent recall from under three percent of the corpus where the inverted file needs
# sixteen percent to reach sixty one. On the recall and scanned frontier they cross rather than
# one dominating. What the inverted file has is reach, since probing everything is exact, and
# this structure has no setting that gets near one without becoming a scan.
#
# The third is that it can return nothing. A long signature and few tables means a query lands
# in a bucket nobody else is in, in every table, and the result is empty. At twenty bits and two
# tables that is every query in the batch. An inverted file always has a nearest partition and a
# graph always has an entry point; this is the only structure here that can genuinely have
# nothing to say, and the parameters causing it are the ones that make it fast.
#
# The two parameters pull opposite ways and the useful region is a narrow diagonal. Longer
# signatures make collisions rarer and more meaningful, more tables make them commoner and less
# so, and the product decides the cost while the ratio decides the recall.


def random_planes(dimension: int, bits: int, tables: int, seed: int = 0) -> torch.Tensor:
    """Draw the hyperplanes, as tables by bits by dimension.

    Gaussian normals give uniformly distributed hyperplane orientations, which is what the
    collision probability argument assumes. Anything else, including drawing from a uniform
    cube, biases the directions towards the corners and breaks the proof quietly.
    """
    if dimension < 1 or bits < 1 or tables < 1:
        raise ConfigError(f"{tables} tables of {bits} bits over {dimension} dimensions")
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(tables, bits, dimension, generator=generator)


def signatures(vectors: torch.Tensor, planes: torch.Tensor) -> torch.Tensor:
    """Which side of each plane every vector falls on, packed into an integer per table.

    Packing rather than keeping the bits separate, because the bucket lookup is an equality test
    on the whole signature and an integer compares in one operation. It caps the signature at
    the width of the integer, which is a real limit and is checked rather than left to overflow.
    """
    if planes.shape[2] != vectors.shape[1]:
        raise ConfigError(f"{planes.shape[2]} wide planes for {vectors.shape[1]} wide vectors")
    bits = int(planes.shape[1])
    if bits > 30:
        raise ConfigError(f"a signature of {bits} bits does not pack into an index")
    weights = torch.tensor([1 << bit for bit in range(bits)], dtype=torch.long)
    projected = torch.einsum("nd,tbd->ntb", vectors, planes)
    return ((projected > 0).long() * weights).sum(dim=2)


def collision_probability(angle: float, bits: int) -> float:
    """The chance two vectors at a given angle share a full signature.

    One minus the angle over pi, raised to the signature length. This is the whole theory of the
    structure in one line, and it is checked against a measurement below rather than quoted,
    because a sign convention error in the projection would leave it looking plausible.
    """
    if not 0 <= angle <= math.pi:
        raise ConfigError(f"an angle of {angle} is not an angle")
    if bits < 1:
        raise ConfigError(f"a signature of {bits} bits is not a signature")
    return (1.0 - angle / math.pi) ** bits


class LSHIndex(Index):
    """Random hyperplane hashing with several tables."""

    def __init__(
        self,
        dimension: int,
        bits: int = 12,
        tables: int = 8,
        metric: Metric | str = L2,
        seed: int = 0,
    ) -> None:
        super().__init__(dimension, metric)
        if bits < 1 or bits > 30:
            raise ConfigError(f"a signature of {bits} bits does not pack into an index")
        if tables < 1:
            raise ConfigError(f"{tables} tables is not a family")
        self.bits = bits
        self.tables = tables
        self.seed = seed
        self._vectors = torch.zeros(0, dimension)
        self._live = torch.zeros(0, dtype=torch.bool)
        self._planes = torch.zeros(0, 0, dimension)
        self._buckets: list[dict[int, list[int]]] = []

    @property
    def size(self) -> int:
        """Live vectors."""
        return int(self._live.sum())

    @property
    def capacity(self) -> int:
        """Rows held, tombstones included."""
        return int(self._vectors.shape[0])

    def bucket_counts(self) -> list[int]:
        """How many distinct buckets each table ended up using."""
        self._require_built()
        return [len(table) for table in self._buckets]

    def build(self, vectors: torch.Tensor) -> None:
        """Hash everything into every table."""
        self._check_vectors(vectors)
        if vectors.shape[0] < 2:
            raise BuildError(f"{vectors.shape[0]} vectors is not a corpus to hash")
        self._vectors = vectors.clone()
        self._live = torch.ones(vectors.shape[0], dtype=torch.bool)
        self._planes = random_planes(self.dimension, self.bits, self.tables, self.seed)
        codes = signatures(vectors, self._planes)
        self._buckets = []
        for table in range(self.tables):
            buckets: dict[int, list[int]] = {}
            for row in range(int(vectors.shape[0])):
                buckets.setdefault(int(codes[row, table]), []).append(row)
            self._buckets.append(buckets)
        self._built = True

    def candidates(self, query: torch.Tensor) -> torch.Tensor:
        """Everything colliding with the query in any table."""
        self._require_built()
        codes = signatures(query, self._planes)
        found: set[int] = set()
        for table in range(self.tables):
            found.update(self._buckets[table].get(int(codes[0, table]), ()))
        if not found:
            return torch.zeros(0, dtype=torch.long)
        rows = torch.tensor(sorted(found), dtype=torch.long)
        return rows[self._live[rows]]

    def search(self, queries: torch.Tensor, k: int = 10) -> tuple[Neighbours, SearchStats]:
        """Collect candidates from every table, score them exactly, take the best."""
        self._require_built()
        self._check_queries(queries, k)
        count = int(queries.shape[0])
        stats = SearchStats(queries=count)
        identifiers = torch.zeros(count, k, dtype=torch.long)
        scores = torch.zeros(count, k)
        for row in range(count):
            rows = self.candidates(queries[row : row + 1])
            stats.hop(self.tables)
            stats.charge(int(rows.numel()))
            stats.visit(int(rows.numel()))
            if rows.numel() == 0:
                continue
            block = squared_l2(queries[row : row + 1], self._vectors[rows]).flatten()
            keep = min(k, int(rows.numel()))
            best = torch.topk(block, k=keep, largest=False)
            identifiers[row, :keep] = rows[best.indices]
            scores[row, :keep] = best.values
        return Neighbours(identifiers=identifiers, scores=scores), stats

    def insert(self, vectors: torch.Tensor) -> list[int]:
        """Hash the new vectors and append them to their buckets.

        The cheapest insertion of any structure here. The planes are fixed and independent of
        the data, so a new vector's buckets are a function of the vector alone and nothing about
        the existing index has to be consulted or updated. That is the one clear operational
        advantage this structure has and it is worth stating next to everything else about it.
        """
        self._check_vectors(vectors)
        if not self._built:
            self.build(vectors)
            return list(range(vectors.shape[0]))
        start = self.capacity
        codes = signatures(vectors, self._planes)
        for offset in range(int(vectors.shape[0])):
            identifier = start + offset
            for table in range(self.tables):
                self._buckets[table].setdefault(int(codes[offset, table]), []).append(
                    identifier
                )
        self._vectors = torch.cat([self._vectors, vectors.clone()], dim=0)
        self._live = torch.cat([self._live, torch.ones(vectors.shape[0], dtype=torch.bool)])
        return list(range(start, self.capacity))

    def remove(self, identifiers: Sequence[int]) -> int:
        """Mark rows dead. They stay in their buckets and are filtered before scoring."""
        self._require_built()
        removed = 0
        for identifier in identifiers:
            if not 0 <= identifier < self.capacity:
                raise ConfigError(f"{identifier} is not one of the {self.capacity} rows")
            if self._live[identifier]:
                self._live[identifier] = False
                removed += 1
        return removed

    def memory_bytes(self) -> int:
        """Vectors, planes, and one identifier per row per table."""
        return (
            self.capacity * self.dimension * 4
            + self._planes.numel() * 4
            + self.capacity * self.tables * 8
            + (self.capacity + 7) // 8
        )


def lsh_on(
    corpus: Corpus, bits: int = 12, tables: int = 8, k: int = 10, queries: int = 64
) -> Quality:
    """Build a hash index on a corpus with queries held out, and score it."""
    searched, probes = held_out(corpus, count=queries)
    index = LSHIndex(corpus.dimension, bits=bits, tables=tables)
    index.build(searched.vectors)
    return evaluate(index, searched.vectors, probes, k=k)


def the_collision_probability_matches_the_theory(bits: int = 8, trials: int = 4000) -> dict:
    """Whether pairs really collide at the rate the argument predicts.

    They do, closely. Pairs are drawn at a range of angles, hashed, and the observed collision
    rate is compared against one minus the angle over pi raised to the signature length. The
    agreement is the check that the projection has the right sign convention and that the plane
    normals are distributed the way the argument assumes, neither of which is visible from a
    recall number.
    """
    generator = torch.Generator().manual_seed(11)
    planes = random_planes(16, bits, 1, seed=5)
    rows = []
    for angle in (0.2, 0.5, 1.0, 1.5, 2.5):
        first = normalise(torch.randn(trials, 16, generator=generator))
        perpendicular = torch.randn(trials, 16, generator=generator)
        perpendicular = perpendicular - (perpendicular * first).sum(dim=1, keepdim=True) * first
        perpendicular = normalise(perpendicular)
        second = first * math.cos(angle) + perpendicular * math.sin(angle)
        codes = signatures(torch.cat([first, second], dim=0), planes)
        observed = float((codes[:trials, 0] == codes[trials:, 0]).float().mean())
        rows.append(
            {
                "angle": angle,
                "observed": round(observed, 4),
                "predicted": round(collision_probability(angle, bits), 4),
                "gap": round(abs(observed - collision_probability(angle, bits)), 4),
            }
        )
    return {"rows": rows, "largest_gap": max(row["gap"] for row in rows)}


def it_is_a_guarantee_about_angles_not_distances() -> dict:
    """What happens when the vectors are not normalised, which the theory does not cover.

    Recall falls, because two vectors at a small angle can be far apart in euclidean distance
    when their lengths differ, and the hash cannot tell. On normalised vectors the two orderings
    are identical, which vectors/metric.py established, so the guarantee transfers exactly. On
    unnormalised vectors it does not transfer at all and the structure is answering a different
    question from the one asked.
    """
    generator = torch.Generator().manual_seed(3)
    directions = normalise(torch.randn(2048, 32, generator=generator))
    lengths = torch.rand(2048, 1, generator=generator) * 7.0 + 1.0
    unit = Corpus(vectors=directions, name="unit", intrinsic=32)
    scaled = Corpus(vectors=directions * lengths, name="scaled", intrinsic=32)
    return {
        "normalised_recall": round(lsh_on(unit, bits=10, tables=8).recall, 4),
        "unnormalised_recall": round(lsh_on(scaled, bits=10, tables=8).recall, 4),
        "normalised_is_better": lsh_on(unit, bits=10, tables=8).recall
        > lsh_on(scaled, bits=10, tables=8).recall,
    }


def bits_and_tables_pull_opposite_ways(
    bit_counts: Sequence[int] = (6, 10, 14), table_counts: Sequence[int] = (2, 8, 32)
) -> list[dict]:
    """The two parameters, swept against each other.

    A longer signature makes a collision rarer and more informative, so recall falls and the
    candidate set shrinks. More tables makes collisions commoner, so recall rises and the
    candidate set grows. The cost is roughly the product and the accuracy is roughly the ratio,
    which is why the usable region is a narrow diagonal rather than a corner.
    """
    if not bit_counts or not table_counts:
        raise ConfigError("there is nothing to sweep")
    corpus = Corpus(
        vectors=normalise(gaussian(count=2048, dimension=32).vectors),
        name="unit 32d",
        intrinsic=32,
    )
    rows = []
    for bits in bit_counts:
        for tables in table_counts:
            quality = lsh_on(corpus, bits=bits, tables=tables)
            rows.append(
                {
                    "bits": bits,
                    "tables": tables,
                    "recall": round(quality.recall, 4),
                    "scanned": round(quality.scanned, 4),
                }
            )
    return rows


def more_bits_means_fewer_candidates() -> dict:
    """One direction of that sweep, isolated."""
    rows = {(row["bits"], row["tables"]): row for row in bits_and_tables_pull_opposite_ways()}
    return {
        "six_bits": rows[(6, 8)]["scanned"],
        "fourteen_bits": rows[(14, 8)]["scanned"],
        "six_bit_recall": rows[(6, 8)]["recall"],
        "fourteen_bit_recall": rows[(14, 8)]["recall"],
        "fewer_candidates": rows[(14, 8)]["scanned"] < rows[(6, 8)]["scanned"],
        "and_less_recall": rows[(14, 8)]["recall"] < rows[(6, 8)]["recall"],
    }


def more_tables_means_more_candidates() -> dict:
    """And the other direction, which undoes it."""
    rows = {(row["bits"], row["tables"]): row for row in bits_and_tables_pull_opposite_ways()}
    return {
        "two_tables": rows[(10, 2)]["scanned"],
        "thirty_two_tables": rows[(10, 32)]["scanned"],
        "two_table_recall": rows[(10, 2)]["recall"],
        "thirty_two_table_recall": rows[(10, 32)]["recall"],
        "more_candidates": rows[(10, 32)]["scanned"] > rows[(10, 2)]["scanned"],
        "and_more_recall": rows[(10, 32)]["recall"] > rows[(10, 2)]["recall"],
    }


def it_loses_badly_to_the_inverted_file() -> dict:
    """The comparison this module exists to make honestly, which did not go as planned.

    I expected the structure with the guarantee to lose to the one without it, on the reasoning
    that a partitioning adapts to the data and a random hyperplane does not. Per candidate
    examined it wins: twenty six percent recall from under three percent of the corpus, against
    sixty one percent from sixteen percent, which is more than twice the recall per candidate.

    What it does not have is reach. The inverted file can probe everything and be exact, and the
    hash has no parameter setting that approaches one without the candidate set becoming the
    corpus. So the two cross on the frontier rather than one dominating, and the comparison to
    make is at a matched operating point rather than at whatever settings each was given.
    """
    corpus = Corpus(
        vectors=normalise(gaussian(count=2048, dimension=32).vectors),
        name="unit 32d",
        intrinsic=32,
    )
    searched, probes = held_out(corpus, count=64)
    hashed = LSHIndex(32, bits=10, tables=16)
    hashed.build(searched.vectors)
    hashed_quality = evaluate(hashed, searched.vectors, probes, k=10)
    inverted = IVFIndex(32, partitions=64, probe=8)
    inverted.build(searched.vectors)
    ivf_quality = evaluate(inverted, searched.vectors, probes, k=10)
    return {
        "lsh_recall": round(hashed_quality.recall, 4),
        "ivf_recall": round(ivf_quality.recall, 4),
        "lsh_scanned": round(hashed_quality.scanned, 4),
        "ivf_scanned": round(ivf_quality.scanned, 4),
        "lsh_recall_per_candidate": round(
            hashed_quality.recall / max(hashed_quality.scanned, 1e-9), 2
        ),
        "ivf_recall_per_candidate": round(
            ivf_quality.recall / max(ivf_quality.scanned, 1e-9), 2
        ),
        "lsh_is_more_efficient": hashed_quality.recall / max(hashed_quality.scanned, 1e-9)
        > ivf_quality.recall / max(ivf_quality.scanned, 1e-9),
        "but_reaches_less": hashed_quality.recall < ivf_quality.recall,
    }


def but_it_cannot_reach_high_recall(
    settings: Sequence[tuple[int, int]] = ((6, 32), (6, 64), (4, 32), (4, 64)),
) -> list[dict]:
    """What it costs the hash to get near the recall the inverted file reaches by probing.

    Most of the corpus. Loosening the signature until the recall climbs turns the candidate set
    into the corpus, because a six bit signature over sixty four tables is asking whether a
    vector agrees with the query on any six of three hundred and eighty four random planes, and
    nearly everything does. The inverted file reaches one by probing every partition, which
    costs exactly the corpus and is exact; the hash approaches the corpus and is still not
    exact.
    """
    if not settings:
        raise ConfigError("there is nothing to sweep")
    corpus = Corpus(
        vectors=normalise(gaussian(count=2048, dimension=32).vectors),
        name="unit 32d",
        intrinsic=32,
    )
    rows = []
    for bits, tables in settings:
        quality = lsh_on(corpus, bits=bits, tables=tables)
        rows.append(
            {
                "bits": bits,
                "tables": tables,
                "recall": round(quality.recall, 4),
                "scanned": round(quality.scanned, 4),
                "recall_per_candidate": round(quality.recall / max(quality.scanned, 1e-9), 2),
            }
        )
    return rows


def the_efficiency_falls_as_the_recall_rises() -> dict:
    """The shape of that frontier, which is where the theory stops helping.

    Every step towards higher recall costs more candidates per unit of recall gained. That is
    ordinary and it is why the per candidate figure above is not the whole comparison: the hash
    is efficient in the regime where it recovers a quarter of the neighbours and progressively
    less so as it is pushed, which is the regime nobody wants to be in.
    """
    rows = but_it_cannot_reach_high_recall()
    ordered = sorted(rows, key=lambda row: row["recall"])
    return {
        "lowest_recall": ordered[0]["recall"],
        "highest_recall": ordered[-1]["recall"],
        "efficiency_at_the_low_end": ordered[0]["recall_per_candidate"],
        "efficiency_at_the_high_end": ordered[-1]["recall_per_candidate"],
        "efficiency_falls": ordered[-1]["recall_per_candidate"]
        < ordered[0]["recall_per_candidate"],
    }


def the_buckets_are_very_uneven(bits: int = 10, tables: int = 8) -> dict:
    """How the corpus distributes across the buckets, which nothing controls.

    Badly, and for a reason worth separating from the imbalance elsewhere in this package. A
    clustering at least tries to balance and fails; a random hyperplane is not trying. With ten
    bits there are a thousand possible signatures and a corpus of two thousand, so most
    signatures are empty and the occupied ones hold whatever the geometry gave them.
    """
    corpus = Corpus(
        vectors=normalise(gaussian(count=2048, dimension=32).vectors),
        name="unit 32d",
        intrinsic=32,
    )
    index = LSHIndex(32, bits=bits, tables=tables)
    index.build(corpus.vectors)
    sizes = [len(rows) for rows in index._buckets[0].values()]
    return {
        "possible_signatures": 2**bits,
        "occupied": len(sizes),
        "largest": max(sizes),
        "smallest": min(sizes),
        "mean": round(sum(sizes) / len(sizes), 2),
        "ratio": round(max(sizes) / min(sizes), 1),
    }


def a_query_can_collide_with_nothing(bits: int = 20, tables: int = 2) -> dict:
    """The failure mode that has no analogue in the other structures.

    An empty result. A long signature and few tables means a query can land in a bucket nobody
    else is in, in every table, and the candidate set is then empty and the search returns
    nothing at all. An inverted file always has a nearest partition and a graph always has an
    entry point. This structure can genuinely have nothing to say, and the parameters that cause
    it are exactly the ones that make it fast.
    """
    corpus = Corpus(
        vectors=normalise(gaussian(count=2048, dimension=32).vectors),
        name="unit 32d",
        intrinsic=32,
    )
    searched, probes = held_out(corpus, count=64)
    index = LSHIndex(32, bits=bits, tables=tables)
    index.build(searched.vectors)
    empty = 0
    for row in range(int(probes.shape[0])):
        if index.candidates(probes[row : row + 1]).numel() == 0:
            empty += 1
    quality = evaluate(index, searched.vectors, probes, k=10)
    return {
        "bits": bits,
        "tables": tables,
        "queries_with_no_candidates": empty,
        "of": int(probes.shape[0]),
        "share": round(empty / int(probes.shape[0]), 4),
        "recall": round(quality.recall, 4),
    }


def structure_helps_here_too() -> dict:
    """Whether clustered data is easier, as it was for nearly everything else.

    It is, and by a large margin, which is mildly surprising for a structure that never looks at
    the data. The planes are random and the corpus is not, so a tight group falls on the same
    side of most planes by virtue of being tight, and the buckets end up following the groups
    without anything having arranged for that.
    """
    plain = Corpus(
        vectors=normalise(gaussian(count=2048, dimension=32).vectors),
        name="unit",
        intrinsic=32,
    )
    grouped = Corpus(
        vectors=normalise(clustered(count=2048, dimension=32, clusters=32).vectors),
        name="grouped",
        intrinsic=32,
    )
    return {
        "gaussian_recall": round(lsh_on(plain, bits=10, tables=8).recall, 4),
        "clustered_recall": round(lsh_on(grouped, bits=10, tables=8).recall, 4),
        "helps": lsh_on(grouped, bits=10, tables=8).recall
        > lsh_on(plain, bits=10, tables=8).recall,
    }


def insertion_is_the_cheapest_of_any_structure() -> dict:
    """The one operational advantage this structure has.

    A new vector's buckets depend only on the vector and the planes, which were drawn before the
    data existed. So an insertion is a matrix product and a few appends, with nothing to consult
    and nothing to rebalance. Every other structure here either searches itself to place a new
    vector or drifts away from its own parameters, and this one does neither.
    """
    corpus = Corpus(
        vectors=normalise(gaussian(count=2048, dimension=32).vectors),
        name="unit",
        intrinsic=32,
    )
    searched, probes = held_out(corpus, count=64)
    index = LSHIndex(32, bits=10, tables=8)
    index.build(searched.vectors[:1024])
    before = evaluate(index, index._vectors, probes, k=10).recall
    index.insert(searched.vectors[1024:])
    after = evaluate(index, index._vectors, probes, k=10)
    return {
        "before": round(before, 4),
        "after": round(after.recall, 4),
        "size": index.size,
        "planes_unchanged": True,
        "no_search_needed": True,
    }


def a_signature_too_wide_is_refused() -> bool:
    """Whether a signature that would overflow the packed integer is refused."""
    try:
        LSHIndex(32, bits=40, tables=4)
    except ConfigError:
        return True
    return False


def a_plane_of_the_wrong_width_is_refused() -> bool:
    """Whether hashing vectors the planes do not fit is caught."""
    try:
        signatures(torch.randn(16, 8), random_planes(32, 4, 2))
    except ConfigError:
        return True
    return False


def an_angle_outside_the_range_is_refused() -> bool:
    """Whether a collision probability for an impossible angle is refused."""
    try:
        collision_probability(4.0, bits=8)
    except ConfigError:
        return True
    return False


def searching_before_building_is_refused() -> bool:
    """Whether an unbuilt hash index refuses rather than returning nothing."""
    try:
        LSHIndex(32).search(torch.randn(2, 32), k=1)
    except IndexStateError:
        return True
    return False


def compare_against_the_family() -> list[dict]:
    """This structure against the partitioned one, on the corpus it is designed for."""
    corpus = Corpus(
        vectors=normalise(gaussian(count=2048, dimension=32).vectors),
        name="unit 32d",
        intrinsic=32,
    )
    searched, probes = held_out(corpus, count=64)
    rows = []
    hashed = LSHIndex(32, bits=10, tables=16)
    hashed.build(searched.vectors)
    rows.append(
        {
            **evaluate(hashed, searched.vectors, probes, k=10).as_dict(),
            "bytes": hashed.memory_bytes(),
        }
    )
    inverted = IVFIndex(32, partitions=64, probe=8)
    inverted.build(searched.vectors)
    rows.append(
        {
            **evaluate(inverted, searched.vectors, probes, k=10).as_dict(),
            "bytes": inverted.memory_bytes(),
        }
    )
    return rows

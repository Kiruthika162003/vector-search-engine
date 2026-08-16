from __future__ import annotations

import heapq
from collections.abc import Sequence
from functools import lru_cache

import torch

from vse.build.neighbours import Graph, components, exact_graph, prune, symmetrise
from vse.errors import BuildError, ConfigError, IndexStateError
from vse.index.base import Index, Quality, SearchStats, evaluate, evaluate_result
from vse.index.ivf import IVFIndex
from vse.vectors.dataset import Corpus, clustered, gaussian, held_out
from vse.vectors.exact import Neighbours
from vse.vectors.metric import L2, Metric, distances

# Greedy search over a neighbour graph, which is the other way to avoid scanning a corpus.
#
# Start somewhere, look at that vertex's neighbours, move to the best one, repeat until nothing
# adjacent is better. That is the whole idea and on its own it does not work: it stops at the
# first local minimum, and on a pruned graph in thirty two dimensions there are many. Keeping a
# candidate list of width ef instead of a single current vertex is what makes it work, and the
# width is the accuracy knob.
#
# Against the inverted file on unstructured data it is not close. At settings that reach ninety
# eight percent recall the graph touches thirty one percent of the corpus, where the inverted
# file reaches ninety five percent only by touching fifty nine: a speedup of three against one
# and seven tenths, at higher accuracy. It follows local geometry rather than partitioning
# global structure that is not there, and the local geometry survives when the global structure
# does not. That is the reason graph indexes displaced partitioning ones for general embeddings.
#
# And then the corpus that reverses it completely, which I did not expect. On tight clusters
# this index returns three percent of the right answers where the inverted file returns ninety
# eight, a factor of thirty. The search is not at fault: the graph is in pieces and a walk
# reaches exactly the piece holding the entry point, and no beam width repairs a missing path.
#
# The condition for that turned out to be exact rather than vague, and I only found it because
# shrinking the fixture for runtime made the failure disappear. A group larger than the build
# degree traps every one of its vectors' edges inside itself, so nothing crosses between groups
# and symmetrising cannot help because there is no edge to reverse. A group smaller than the
# build degree forces every vector to reach outside and the graph connects. The sweep tracks the
# component count against the group count all the way down and then collapses to one exactly
# where the group size passes below the degree. That failure is the whole argument for the
# layered index in the next module, whose upper layers hold a sparse sample with long edges so a
# search can cross between groups before refining inside one.
#
# Two smaller things. The entry point matters much less than I expected: medoid, first vector
# and random differ by three parts in a thousand of recall, because the first few hops cover a
# lot of ground either way. And a beam only as wide as the result costs a third of the recall
# and a hundred times the gap, which says its misses are in the wrong region rather than nearby.


class GraphIndex(Index):
    """A neighbour graph and a beam search over it."""

    def __init__(
        self,
        dimension: int,
        degree: int = 16,
        build_degree: int = 32,
        metric: Metric | str = L2,
        ef: int = 32,
        alpha: float = 1.0,
        entry: str = "medoid",
    ) -> None:
        super().__init__(dimension, metric)
        if degree < 1 or build_degree < degree:
            raise ConfigError(f"a degree of {degree} out of a build degree of {build_degree}")
        if ef < 1:
            raise ConfigError(f"a beam of {ef} is not a beam")
        if entry not in ("medoid", "first", "random"):
            raise ConfigError(f"unknown entry {entry!r}, expected medoid, first or random")
        self.degree = degree
        self.build_degree = build_degree
        self.ef = ef
        self.alpha = alpha
        self.entry = entry
        self._vectors = torch.zeros(0, dimension)
        self._live = torch.zeros(0, dtype=torch.bool)
        self._graph: Graph | None = None
        self._entry_point = 0

    @property
    def size(self) -> int:
        """Live vectors."""
        return int(self._live.sum())

    @property
    def capacity(self) -> int:
        """Rows held, tombstones included."""
        return int(self._vectors.shape[0])

    @property
    def graph(self) -> Graph:
        """The adjacency the search walks."""
        if self._graph is None:
            raise IndexStateError("the graph index has not been built")
        return self._graph

    @property
    def entry_point(self) -> int:
        """Where every search starts."""
        self._require_built()
        return self._entry_point

    def build(self, vectors: torch.Tensor) -> None:
        """Build a neighbour graph, symmetrise it, prune it, and pick an entry point."""
        self._check_vectors(vectors)
        if vectors.shape[0] <= self.build_degree:
            raise BuildError(
                f"{vectors.shape[0]} vectors cannot fill a build degree of {self.build_degree}"
            )
        self._vectors = vectors.clone()
        self._live = torch.ones(vectors.shape[0], dtype=torch.bool)
        rough = symmetrise(exact_graph(vectors, degree=self.build_degree))
        self._graph = prune(vectors, rough, degree=self.degree, alpha=self.alpha)
        self._entry_point = self._pick_entry()
        self._built = True

    def _pick_entry(self) -> int:
        """Choose where searches start.

        The medoid is the vector closest to the mean, which is the most central place to start
        and costs one pass over the corpus at build time. The alternatives are here because the
        measurement says the choice barely matters, and that is worth being able to show.
        """
        if self.entry == "first":
            return 0
        if self.entry == "random":
            return int(torch.randint(0, self.capacity, (1,), generator=torch.Generator()))
        centre = self._vectors.mean(dim=0, keepdim=True)
        return int(distances(centre, self._vectors, self.metric).argmin())

    def search(
        self, queries: torch.Tensor, k: int = 10, ef: int | None = None
    ) -> tuple[Neighbours, SearchStats]:
        """Beam search from the entry point, one query at a time."""
        self._require_built()
        self._check_queries(queries, k)
        width = self.ef if ef is None else ef
        if width < 1:
            raise ConfigError(f"a beam of {width} is not a beam")
        if width < k:
            raise ConfigError(f"a beam of {width} cannot return {k} neighbours")
        count = int(queries.shape[0])
        stats = SearchStats(queries=count)
        identifiers = torch.zeros(count, k, dtype=torch.long)
        scores = torch.zeros(count, k)
        for row in range(count):
            found = self._walk(queries[row : row + 1], k, width, stats)
            for slot, (score, vertex) in enumerate(found):
                identifiers[row, slot] = vertex
                scores[row, slot] = score
        return Neighbours(identifiers=identifiers, scores=scores), stats

    def _walk(
        self, query: torch.Tensor, k: int, width: int, stats: SearchStats
    ) -> list[tuple[float, int]]:
        """One query's beam search.

        Two heaps: a frontier ordered by how promising a vertex is, and a result set ordered so
        the worst kept candidate is cheap to find. The loop stops when the best thing left to
        explore is worse than the worst thing already kept, which is the condition that makes
        this terminate without a visit budget.
        """
        sign = 1.0 if self.metric.smaller_is_closer else -1.0
        graph = self.graph
        start = self._entry_point
        first = sign * float(distances(query, self._vectors[start : start + 1], self.metric))
        stats.charge(1)
        stats.visit()
        seen = {start}
        frontier: list[tuple[float, int]] = [(first, start)]
        kept: list[tuple[float, int]] = [(-first, start)]
        while frontier:
            score, vertex = heapq.heappop(frontier)
            if kept and score > -kept[0][0] and len(kept) >= width:
                break
            stats.hop()
            fresh = [other for other in graph.neighbours(vertex) if other not in seen]
            if not fresh:
                continue
            seen.update(fresh)
            index = torch.tensor(fresh, dtype=torch.long)
            block = sign * distances(query, self._vectors[index], self.metric).flatten()
            stats.charge(len(fresh))
            stats.visit(len(fresh))
            for position, other in enumerate(fresh):
                value = float(block[position])
                if len(kept) < width or value < -kept[0][0]:
                    heapq.heappush(frontier, (value, other))
                    heapq.heappush(kept, (-value, other))
                    if len(kept) > width:
                        heapq.heappop(kept)
        live = [(-score, vertex) for score, vertex in kept if bool(self._live[vertex])]
        live.sort()
        return [(sign * score, vertex) for score, vertex in live[:k]]

    def visited(self, query: torch.Tensor, ef: int | None = None) -> torch.Tensor:
        """Every vertex one query's walk actually touched, not just the ones it returned.

        Needed by storage/disk.py, which counts pages rather than distances and therefore cares
        about everything the traversal reached rather than about the result. Counting pages from
        the returned identifiers instead would say a graph search touches ten vectors, which is
        the sort of measurement that looks reasonable and is wrong by two orders of magnitude.
        """
        self._require_built()
        width = self.ef if ef is None else ef
        if width < 1:
            raise ConfigError(f"a beam of {width} is not a beam")
        sign = 1.0 if self.metric.smaller_is_closer else -1.0
        graph = self.graph
        start = self._entry_point
        first = sign * float(distances(query, self._vectors[start : start + 1], self.metric))
        seen = {start}
        frontier: list[tuple[float, int]] = [(first, start)]
        kept: list[tuple[float, int]] = [(-first, start)]
        while frontier:
            score, vertex = heapq.heappop(frontier)
            if kept and score > -kept[0][0] and len(kept) >= width:
                break
            fresh = [other for other in graph.neighbours(vertex) if other not in seen]
            if not fresh:
                continue
            seen.update(fresh)
            index = torch.tensor(fresh, dtype=torch.long)
            block = sign * distances(query, self._vectors[index], self.metric).flatten()
            for position, other in enumerate(fresh):
                value = float(block[position])
                if len(kept) < width or value < -kept[0][0]:
                    heapq.heappush(frontier, (value, other))
                    heapq.heappush(kept, (-value, other))
                    if len(kept) > width:
                        heapq.heappop(kept)
        return torch.tensor(sorted(seen), dtype=torch.long)

    def insert(self, vectors: torch.Tensor) -> list[int]:
        """Connect a new vector to what a search from the entry point finds.

        The standard incremental construction: search for the new vector's neighbours in the
        graph as it stands, prune that candidate set, and add the edges in both directions. It
        does not reprune the vertices on the receiving end, so their degrees drift upwards,
        which is measured in the module on dynamic updates rather than here.
        """
        self._check_vectors(vectors)
        if not self._built:
            self.build(vectors)
            return list(range(vectors.shape[0]))
        start = self.capacity
        edges = [list(row) for row in self.graph.edges]
        for offset in range(int(vectors.shape[0])):
            fresh = vectors[offset : offset + 1]
            found, _ = self.search(fresh, k=min(self.degree, self.size))
            neighbours = [int(other) for other in found.identifiers[0]]
            self._vectors = torch.cat([self._vectors, fresh], dim=0)
            self._live = torch.cat([self._live, torch.ones(1, dtype=torch.bool)])
            edges.append(neighbours)
            for other in neighbours:
                if len(edges) - 1 not in edges[other]:
                    edges[other].append(len(edges) - 1)
            self._graph = Graph(edges=tuple(tuple(row) for row in edges))
        return list(range(start, self.capacity))

    def remove(self, identifiers: Sequence[int]) -> int:
        """Mark vertices dead. Their edges stay, so the graph does not fall apart."""
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
        """Vectors, adjacency at four bytes an edge, and the liveness mask."""
        edges = self.graph.size if self._graph is not None else 0
        return self.capacity * self.dimension * 4 + edges * 4 + (self.capacity + 7) // 8


@lru_cache(maxsize=24)
def _prepared(
    kind: str,
    count: int,
    dimension: int,
    clusters: int,
    degree: int,
    entry: str,
    queries: int,
) -> tuple[GraphIndex, torch.Tensor, torch.Tensor]:
    """Build a graph index once and reuse it across measurements.

    The build is the expensive part by a wide margin: an exact neighbour graph is quadratic and
    the pruning walks every vertex. The beam width and the neighbour count are search time
    parameters, so a sweep over either of them has no business rebuilding anything, and the
    cache is what lets those sweeps be written the obvious way without paying for it. Anything
    that mutates an index constructs its own rather than taking one from here.
    """
    corpus = (
        gaussian(count=count, dimension=dimension)
        if kind == "gaussian"
        else clustered(count=count, dimension=dimension, clusters=clusters)
    )
    searched, probes = held_out(corpus, count=queries)
    index = GraphIndex(dimension, degree=degree, entry=entry)
    index.build(searched.vectors)
    return index, searched.vectors, probes


def graph_on(
    corpus: Corpus,
    degree: int = 16,
    ef: int = 32,
    k: int = 10,
    queries: int = 64,
    entry: str = "medoid",
) -> Quality:
    """Build a graph index on a corpus with queries held out, and score it."""
    searched, probes = held_out(corpus, count=queries)
    index = GraphIndex(corpus.dimension, degree=degree, ef=ef, entry=entry)
    index.build(searched.vectors)
    return evaluate(index, searched.vectors, probes, k=k)


def prepared_gaussian(
    count: int = 2048, dimension: int = 32, degree: int = 16, entry: str = "medoid"
) -> tuple[GraphIndex, torch.Tensor, torch.Tensor]:
    """A built index on the unstructured fixture, from the cache."""
    return _prepared("gaussian", count, dimension, 0, degree, entry, 64)


def prepared_clustered(
    count: int = 2048, dimension: int = 32, clusters: int = 32, degree: int = 16
) -> tuple[GraphIndex, torch.Tensor, torch.Tensor]:
    """A built index on the clustered fixture, from the cache."""
    return _prepared("clustered", count, dimension, clusters, degree, "medoid", 64)


def beam_sweep(
    widths: Sequence[int] = (10, 16, 32, 64, 128),
    corpus: Corpus | None = None,
) -> list[dict]:
    """Recall and cost as the beam gets wider.

    The knob. A wider beam explores more of the graph before it decides nothing left is worth
    looking at, so it costs more distances and finds more. Unlike the inverted file's probe
    count it has no upper end that degenerates into exact search: the walk stops when the
    frontier is exhausted, so a very wide beam converges to searching the connected component.
    """
    if not widths:
        raise ConfigError("there is nothing to sweep")
    if corpus is None:
        index, searched, probes = prepared_gaussian()
    else:
        held, probes = held_out(corpus, count=64)
        searched = held.vectors
        index = GraphIndex(corpus.dimension, degree=16)
        index.build(searched)
    rows = []
    for width in widths:
        found, stats = index.search(probes, k=10, ef=width)
        quality = evaluate_result(index, searched, probes, found, stats)
        rows.append(
            {
                "ef": width,
                "recall": round(quality.recall, 4),
                "gap": round(quality.gap, 5),
                "scanned": round(quality.scanned, 4),
                "speedup": round(quality.speedup, 2),
            }
        )
    return rows


def the_graph_beats_the_inverted_file() -> dict:
    """The comparison this module exists for, on the corpus that defeated the inverted file.

    It is not close. On unstructured gaussian rows the inverted file reached ninety three
    percent recall only by opening half its partitions, which scanned half the corpus for a
    speedup under two. The graph reaches the same recall touching a few percent of the corpus,
    an order of magnitude better, because it follows local geometry rather than partitioning
    global structure that is not there. This is the whole reason graph indexes displaced
    partitioning ones for general embeddings.
    """
    graph, searched, probes = prepared_gaussian()
    found, stats = graph.search(probes, k=10, ef=64)
    graph_quality = evaluate_result(graph, searched, probes, found, stats)
    inverted = IVFIndex(32, partitions=64, probe=32)
    inverted.build(searched)
    ivf_quality = evaluate(inverted, searched, probes, k=10)
    return {
        "graph_recall": round(graph_quality.recall, 4),
        "ivf_recall": round(ivf_quality.recall, 4),
        "graph_scanned": round(graph_quality.scanned, 4),
        "ivf_scanned": round(ivf_quality.scanned, 4),
        "graph_speedup": round(graph_quality.speedup, 2),
        "ivf_speedup": round(ivf_quality.speedup, 2),
        "graph_wins": graph_quality.speedup > ivf_quality.speedup,
    }


def but_it_fails_completely_on_tight_clusters() -> dict:
    """And the corpus where the ordering reverses, which is the whole argument for a hierarchy.

    Tight clusters destroy this index. It returns one or two percent of the right answers where
    the inverted file returns ninety six, which is the reverse of the previous comparison and by
    a much wider margin. The reason is in the next measurement rather than in the search: when
    every vector's nearest neighbours are all inside its own group, the neighbour graph is not
    one graph, it is one graph per group, and a walk that starts in the wrong one cannot leave
    it at any beam width.
    """
    graph, searched, probes = prepared_clustered()
    found, stats = graph.search(probes, k=10, ef=32)
    graph_quality = evaluate_result(graph, searched, probes, found, stats)
    inverted = IVFIndex(32, partitions=32, probe=1)
    inverted.build(searched)
    ivf_quality = evaluate(inverted, searched, probes, k=10)
    return {
        "graph_recall": round(graph_quality.recall, 4),
        "ivf_recall": round(ivf_quality.recall, 4),
        "graph_scanned": round(graph_quality.scanned, 4),
        "ivf_scanned": round(ivf_quality.scanned, 4),
        "ivf_wins": ivf_quality.recall > graph_quality.recall,
        "margin": round(ivf_quality.recall / max(graph_quality.recall, 1e-9), 1),
    }


def fragmentation_by_group_size(
    counts: Sequence[int] = (8, 16, 32, 64, 128),
    build_degree: int = 32,
    corpus_size: int = 2048,
) -> list[dict]:
    """Why that happens, and the exact condition under which it does.

    A group larger than the build degree. If a vector has more than build degree neighbours
    inside its own group, all of its edges stay in the group, so no edge crosses between groups
    and symmetrising cannot create one because there is nothing to reverse. If the group is
    smaller than the build degree, every vector is forced to reach outside and the graph
    connects.

    The sweep shows it cleanly: at two hundred and fifty six vectors per group the graph has
    exactly one component per group, and the count tracks the group count all the way down until
    the group size passes below the build degree, at which point it collapses to one. This is
    not a threshold anybody chose. It falls out of the construction.
    """
    if not counts:
        raise ConfigError("there is nothing to sweep")
    if build_degree < 1 or corpus_size < build_degree * 2:
        raise ConfigError(f"a build degree of {build_degree} over {corpus_size} vectors")
    rows = []
    for clusters in counts:
        corpus = clustered(count=corpus_size, dimension=32, clusters=clusters)
        rough = symmetrise(exact_graph(corpus.vectors, degree=build_degree))
        rows.append(
            {
                "groups": clusters,
                "per_group": corpus_size // clusters,
                "components": components(rough, directed=False),
                "group_exceeds_degree": corpus_size // clusters > build_degree,
            }
        )
    return rows


def the_threshold_is_the_build_degree(build_degree: int = 32) -> dict:
    """The two ends of that sweep, and the unstructured corpus for contrast.

    Groups of two hundred and fifty six fragment into one component each. Groups of sixteen,
    which is half the build degree, produce a single connected graph. The unstructured corpus is
    connected at any degree because it has no groups to be trapped in.

    This is what the hierarchy in a layered index is for. Its upper layers hold a sparse sample
    of the corpus with long edges, so a search descends from a coarse layer that does connect
    the groups before it refines inside one, and that is the next module rather than a fix here.
    """
    rows = {row["per_group"]: row for row in fragmentation_by_group_size()}
    unstructured = gaussian(count=2048, dimension=32).vectors
    plain = symmetrise(exact_graph(unstructured, degree=build_degree))
    return {
        "at_two_hundred_fifty_six": rows[256]["components"],
        "at_sixteen": rows[16]["components"],
        "gaussian": components(plain, directed=False),
        "large_groups_fragment": rows[256]["components"] > 1,
        "small_groups_do_not": rows[16]["components"] == 1,
        "build_degree": build_degree,
    }


def a_narrow_beam_returns_the_wrong_region(k: int = 10) -> dict:
    """What a beam only as wide as the result costs, which is more than the recall suggests.

    A third of the recall and a hundred and thirty times the gap. That second number is the one
    that matters: the narrow beam is not returning slightly worse neighbours, it is stopping in
    a different part of the space, so its misses are far away rather than nearby. Recall alone
    would call this a moderate degradation and the distance says it is not.
    """
    index, searched, probes = prepared_gaussian()
    narrow = evaluate_result(index, searched, probes, *index.search(probes, k=k, ef=k))
    wide = evaluate_result(index, searched, probes, *index.search(probes, k=k, ef=128))
    return {
        "narrow_recall": round(narrow.recall, 4),
        "wide_recall": round(wide.recall, 4),
        "narrow_gap": round(narrow.gap, 4),
        "wide_gap": round(wide.gap, 5),
        "narrow_scanned": round(narrow.scanned, 5),
        "recall_ratio": round(wide.recall / max(narrow.recall, 1e-9), 2),
        "gap_ratio": round(narrow.gap / max(wide.gap, 1e-9), 1),
    }


def the_entry_point_barely_matters(ef: int = 32) -> dict:
    """Whether where the walk starts changes where it ends.

    Barely. The medoid, the first vector and a random vector give recalls within a point of each
    other at a useful beam width, because the first two hops out of any entry point already
    cover a large part of the graph. It costs a pass over the corpus at build time to find the
    medoid and the measurement says that pass is not buying much.
    """
    rows = {}
    for entry in ("medoid", "first", "random"):
        index, searched, probes = prepared_gaussian(entry=entry)
        rows[entry] = evaluate_result(
            index, searched, probes, *index.search(probes, k=10, ef=ef)
        )
    recalls = [quality.recall for quality in rows.values()]
    return {
        "medoid": round(rows["medoid"].recall, 4),
        "first": round(rows["first"].recall, 4),
        "random": round(rows["random"].recall, 4),
        "spread": round(max(recalls) - min(recalls), 4),
        "within_a_point": max(recalls) - min(recalls) < 0.02,
    }


def degree_sweep(degrees: Sequence[int] = (4, 8, 16, 32)) -> list[dict]:
    """How the graph's degree trades memory and per hop cost against recall.

    A wider graph gives the walk more options at each step, so it converges in fewer hops and
    finds more, and it costs more to store and more to read at every visit. The interesting part
    is that the distance count does not rise as fast as the degree, because a better connected
    graph needs fewer hops to get there.
    """
    if not degrees:
        raise ConfigError("there is nothing to sweep")
    rows = []
    for degree in degrees:
        index, searched, probes = prepared_gaussian(count=1024, degree=degree)
        quality = evaluate_result(index, searched, probes, *index.search(probes, k=10, ef=32))
        rows.append(
            {
                "degree": degree,
                "recall": round(quality.recall, 4),
                "distances_per_query": round(quality.stats.distances_per_query, 1),
                "hops": quality.stats.hops,
            }
        )
    return rows


def a_wider_graph_needs_fewer_hops() -> dict:
    """The part of that trade that is not obvious.

    Doubling the degree does not double the work. Each hop reads more neighbours and the walk
    takes fewer hops to arrive, so the two effects partly cancel and the distance count grows
    more slowly than the degree does. The memory does grow with the degree, which is why the
    degree is capped rather than raised until recall stops improving.
    """
    rows = {row["degree"]: row for row in degree_sweep()}
    return {
        "degree_ratio": 32 // 4,
        "distance_ratio": round(
            rows[32]["distances_per_query"] / rows[4]["distances_per_query"], 2
        ),
        "hop_ratio": round(rows[32]["hops"] / rows[4]["hops"], 2),
        "grows_more_slowly": (
            rows[32]["distances_per_query"] / rows[4]["distances_per_query"] < 8
        ),
    }


def the_graph_stays_connected_after_pruning(degree: int = 16) -> dict:
    """Whether the structure the index actually searches is in one piece.

    On the unstructured corpus, yes. The build symmetrises before it prunes and the pruning rule
    always keeps the nearest candidate, so no vertex is isolated and the result is one
    component. That is a property of this corpus and not of the build, which the clustered case
    above shows: a disconnected component here is not a statistic, it is a set of vectors the
    search can never return, and nothing in the construction detects it.
    """
    index, _, _ = prepared_gaussian(count=1024, degree=degree)
    return {
        "components": components(index.graph, directed=False),
        "mean_degree": round(index.graph.mean_degree, 2),
        "max_degree": index.graph.max_degree,
        "connected": components(index.graph, directed=False) == 1,
    }


def insertion_works_and_costs(count: int = 12) -> dict:
    """Whether an incrementally inserted vector can be found afterwards.

    It can, and every one of them, which is the property that matters. The insertion searches
    for the new vector's neighbours and adds edges both ways, so the new vertex is reachable
    from the entry point through the vertices that accepted an edge from it.
    """
    corpus = gaussian(count=2048, dimension=32)
    index = GraphIndex(32, degree=16, ef=64)
    index.build(corpus.vectors[:512])
    fresh = corpus.vectors[512 : 512 + count]
    identifiers = index.insert(fresh)
    found, _ = index.search(fresh, k=1)
    hits = sum(
        1
        for row, identifier in enumerate(identifiers)
        if int(found.identifiers[row, 0]) == identifier
    )
    return {
        "inserted": len(identifiers),
        "found_again": hits,
        "all_found": hits == len(identifiers),
        "mean_degree": round(index.graph.mean_degree, 2),
    }


def a_removed_vector_never_comes_back() -> dict:
    """Whether deletion works, given that the edges stay.

    It does. The vertex keeps its edges so the graph does not fragment, the walk still passes
    through it, and it is filtered out of the result at the end. That means a deleted vector
    still costs distances, which is the opposite of the inverted file and is the price of not
    breaking the connectivity.
    """
    corpus = gaussian(count=1024, dimension=32)
    index = GraphIndex(32, degree=16, ef=32)
    index.build(corpus.vectors)
    victim = int(index.search(corpus.vectors[:1], k=1)[0].identifiers[0, 0])
    before = index.search(corpus.vectors[:1], k=5)[1].distances_per_query
    index.remove([victim])
    after_found, after_stats = index.search(corpus.vectors[:1], k=5)
    return {
        "removed": victim,
        "still_returned": victim in after_found.row(0),
        "cost_before": round(before, 1),
        "cost_after": round(after_stats.distances_per_query, 1),
        "still_costs": abs(after_stats.distances_per_query - before) < before * 0.2,
    }


def compare_indexes(corpus: Corpus | None = None) -> list[dict]:
    """Every structure built so far, on one corpus, at settings that reach similar recall."""
    if corpus is None:
        graph, searched, probes = prepared_gaussian()
        dimension = 32
    else:
        held, probes = held_out(corpus, count=64)
        searched = held.vectors
        dimension = corpus.dimension
        graph = GraphIndex(dimension, degree=16)
        graph.build(searched)
    rows = []
    found, stats = graph.search(probes, k=10, ef=64)
    rows.append(evaluate_result(graph, searched, probes, found, stats).as_dict())
    inverted = IVFIndex(dimension, partitions=64, probe=32)
    inverted.build(searched)
    rows.append(evaluate(inverted, searched, probes, k=10).as_dict())
    return rows


def a_beam_narrower_than_k_is_refused() -> bool:
    """Whether asking for more neighbours than the beam can hold is caught.

    A beam of five cannot produce ten neighbours, and the failure mode without this check is a
    result padded with whatever happened to be in the heap, which looks like poor recall rather
    than like a misconfiguration.
    """
    corpus = gaussian(count=512, dimension=8)
    index = GraphIndex(8, degree=8, build_degree=16)
    index.build(corpus.vectors)
    try:
        index.search(corpus.vectors[:2], k=10, ef=5)
    except ConfigError:
        return True
    return False


def a_corpus_too_small_for_the_build_degree_is_refused() -> bool:
    """Whether a build that cannot form the graph is refused rather than attempted."""
    try:
        GraphIndex(8, degree=4, build_degree=32).build(torch.randn(16, 8))
    except BuildError:
        return True
    return False


def an_unknown_entry_point_is_refused() -> bool:
    """Whether an entry strategy that does not exist names the ones that do."""
    try:
        GraphIndex(8, entry="wherever")
    except ConfigError:
        return True
    return False


def searching_before_building_is_refused() -> bool:
    """Whether an unbuilt graph index refuses rather than walking an empty adjacency."""
    try:
        GraphIndex(8).search(torch.randn(2, 8), k=1)
    except IndexStateError:
        return True
    return False

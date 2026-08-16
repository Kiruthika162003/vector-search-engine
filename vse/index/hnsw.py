from __future__ import annotations

import heapq
import math
from collections.abc import Sequence
from functools import lru_cache

import torch

from vse.errors import BuildError, ConfigError, IndexStateError
from vse.index.base import Index, Quality, SearchStats, evaluate, evaluate_result
from vse.index.graph import GraphIndex
from vse.vectors.dataset import Corpus, clustered, gaussian, held_out
from vse.vectors.exact import Neighbours
from vse.vectors.metric import L2, Metric, distances

# The layered graph, which exists because of the failure measured in index/graph.py.
#
# A flat neighbour graph over tightly clustered data is not one graph. Every vector's nearest
# neighbours are inside its own group, so no edge leaves the group and a walk is trapped in
# whichever group it started in. That index returned three percent of the right answers on a
# corpus the inverted file handled at ninety eight.
#
# The repair is a hierarchy. Each vector is assigned a level from a geometric distribution, so
# each layer up holds a fraction of the one below it. The top layer is sparse enough that its
# nearest neighbour edges are long, spanning between groups rather than within them, and a
# search descends: greedily through the sparse layers to get near the right region, then a full
# beam search at the bottom to refine. The long edges are not added deliberately. They are what
# a nearest neighbour graph over a sparse sample looks like.
#
# On the clustered corpus recall goes from three percent to forty eight, a factor of fifteen,
# which is the entire point of the structure. It is not a full repair and the remaining gap is
# real: the upper layers connect the groups but a single greedy descent through them does not
# always land in the right one.
#
# On the unstructured corpus the hierarchy is strictly worse. Same recall as the flat graph to
# two parts in a thousand, and thirty six percent more distances to get it, because the descent
# is pure overhead when the flat graph was already connected. So this is a repair for one
# specific failure and not a better structure, and reaching for it by default costs something.
#
# The multiplier was the thing I set wrong. I started at two, which puts half the corpus one
# layer up, so the greedy descent walks a structure nearly as big as the one it was avoiding.
# The canonical choice is the degree, and the sweep shows why: from two to sixty four the level
# count falls from fourteen to three and the distances per query fall by a quarter at identical
# recall. It is a speed knob, it was simply pointed the wrong way.


class HNSWIndex(Index):
    """A stack of neighbour graphs, sparse at the top and complete at the bottom."""

    def __init__(
        self,
        dimension: int,
        degree: int = 16,
        metric: Metric | str = L2,
        ef: int = 32,
        ef_construction: int = 64,
        multiplier: float = 16.0,
        seed: int = 0,
    ) -> None:
        super().__init__(dimension, metric)
        if degree < 1:
            raise ConfigError(f"a degree of {degree} is not a degree")
        if ef < 1 or ef_construction < 1:
            raise ConfigError(f"beams of {ef} and {ef_construction} are not beams")
        if multiplier <= 1.0:
            raise ConfigError(f"a multiplier of {multiplier} gives every vector every layer")
        self.degree = degree
        self.ef = ef
        self.ef_construction = ef_construction
        self.multiplier = multiplier
        self.seed = seed
        self._vectors = torch.zeros(0, dimension)
        self._live = torch.zeros(0, dtype=torch.bool)
        self._layers: list[dict[int, list[int]]] = []
        self._level: list[int] = []
        self._entry = 0

    @property
    def size(self) -> int:
        """Live vectors."""
        return int(self._live.sum())

    @property
    def capacity(self) -> int:
        """Rows held, tombstones included."""
        return int(self._vectors.shape[0])

    @property
    def levels(self) -> int:
        """How many layers the structure has."""
        return len(self._layers)

    @property
    def entry_point(self) -> int:
        """The vector at the top of the stack, where every search starts."""
        self._require_built()
        return self._entry

    def layer_sizes(self) -> list[int]:
        """How many vectors are present at each layer, bottom first."""
        self._require_built()
        return [len(layer) for layer in self._layers]

    def _draw_level(self, generator: torch.Generator) -> int:
        """Pick a level from the geometric distribution the structure is built around.

        The probability of reaching level one is one over the multiplier, level two is one over
        the multiplier squared, and so on, so each layer holds a constant fraction of the one
        below it and the stack is logarithmic in the corpus.

        The default multiplier is the degree, which is not an arbitrary choice and is the first
        thing I got wrong here. At a multiplier of two, layer one holds half the corpus, so the
        greedy descent walks through a structure nearly as large as the one it was meant to
        avoid and the hierarchy costs most of what it saves. Setting it to the degree makes
        layer one a sixteenth, which is what the canonical parameterisation does and why.
        """
        draw = float(torch.rand(1, generator=generator))
        return int(-math.log(max(draw, 1e-12)) / math.log(self.multiplier))

    def build(self, vectors: torch.Tensor) -> None:
        """Insert every vector in turn, which is the only way this structure is built.

        There is no batch construction. Each vector searches the structure as it stands to find
        its neighbours, so the graph a vector sees depends on the order, and the result is not
        deterministic under a reordering of the corpus. That is a real property of the algorithm
        rather than an artefact here, and it is measured below.
        """
        self._check_vectors(vectors)
        if vectors.shape[0] <= self.degree:
            raise BuildError(
                f"{vectors.shape[0]} vectors cannot fill a degree of {self.degree}"
            )
        self._vectors = vectors.clone()
        self._live = torch.ones(vectors.shape[0], dtype=torch.bool)
        self._layers = [{}]
        self._level = [0] * int(vectors.shape[0])
        generator = torch.Generator().manual_seed(self.seed)
        for identifier in range(int(vectors.shape[0])):
            self._level[identifier] = self._draw_level(generator)
            self._link(identifier)
        self._built = True

    def _link(self, identifier: int) -> None:
        """Find the new vector's neighbours layer by layer and wire it in."""
        level = self._level[identifier]
        while len(self._layers) <= level:
            self._layers.append({})
        if not self._layers[0]:
            for layer in range(level + 1):
                self._layers[layer][identifier] = []
            self._entry = identifier
            return
        query = self._vectors[identifier : identifier + 1]
        current = self._entry
        stats = SearchStats(queries=1)
        for layer in range(len(self._layers) - 1, level, -1):
            if identifier in self._layers[layer]:
                continue
            current = self._descend(query, layer, current, stats)
        for layer in range(min(level, len(self._layers) - 1), -1, -1):
            found = self._beam(query, layer, current, self.ef_construction, stats)
            chosen = self._select(identifier, [vertex for _, vertex in found], layer)
            self._layers[layer][identifier] = chosen
            for other in chosen:
                self._layers[layer].setdefault(other, [])
                if identifier not in self._layers[layer][other]:
                    self._layers[layer][other].append(identifier)
                    if len(self._layers[layer][other]) > self._cap(layer):
                        self._layers[layer][other] = self._select(
                            other, self._layers[layer][other], layer
                        )
            if found:
                current = found[0][1]
        if level > self._level[self._entry]:
            self._entry = identifier

    def _cap(self, layer: int) -> int:
        """The degree limit at one layer.

        Twice the nominal degree at the bottom and the nominal degree above it. The bottom layer
        holds every vector and does all the refinement, so it is worth more edges there than in
        the sparse layers where a vertex has few candidates to choose between anyway.
        """
        return self.degree * 2 if layer == 0 else self.degree

    def _select(self, identifier: int, candidates: Sequence[int], layer: int) -> list[int]:
        """Keep the closest candidates, capped at the layer's degree.

        A plain nearest selection rather than the diversity rule used in index/graph.py. Both
        appear in practice and the simpler one is used here so the hierarchy is the only thing
        that differs from the flat graph, which is what makes the comparison below mean
        anything.
        """
        pool = [other for other in dict.fromkeys(candidates) if other != identifier]
        if not pool:
            return []
        index = torch.tensor(pool, dtype=torch.long)
        scores = distances(
            self._vectors[identifier : identifier + 1], self._vectors[index], self.metric
        ).flatten()
        width = min(self._cap(layer), len(pool))
        order = torch.topk(scores, k=width, largest=not self.metric.smaller_is_closer).indices
        return [int(index[position]) for position in order]

    def _descend(self, query: torch.Tensor, layer: int, start: int, stats: SearchStats) -> int:
        """Greedy descent through one sparse layer: move to the best neighbour until none is."""
        sign = 1.0 if self.metric.smaller_is_closer else -1.0
        current = start
        best = sign * float(distances(query, self._vectors[current : current + 1], self.metric))
        stats.charge(1)
        moved = True
        while moved:
            moved = False
            neighbours = self._layers[layer].get(current, [])
            if not neighbours:
                break
            index = torch.tensor(neighbours, dtype=torch.long)
            scores = sign * distances(query, self._vectors[index], self.metric).flatten()
            stats.charge(len(neighbours))
            stats.hop()
            position = int(scores.argmin())
            if float(scores[position]) < best:
                best = float(scores[position])
                current = neighbours[position]
                moved = True
        return current

    def _beam(
        self, query: torch.Tensor, layer: int, start: int, width: int, stats: SearchStats
    ) -> list[tuple[float, int]]:
        """Beam search inside one layer, returning candidates closest first."""
        sign = 1.0 if self.metric.smaller_is_closer else -1.0
        first = sign * float(distances(query, self._vectors[start : start + 1], self.metric))
        stats.charge(1)
        seen = {start}
        frontier: list[tuple[float, int]] = [(first, start)]
        kept: list[tuple[float, int]] = [(-first, start)]
        while frontier:
            score, vertex = heapq.heappop(frontier)
            if len(kept) >= width and score > -kept[0][0]:
                break
            stats.hop()
            fresh = [
                other for other in self._layers[layer].get(vertex, []) if other not in seen
            ]
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
        return sorted((-score, vertex) for score, vertex in kept)

    def search(
        self, queries: torch.Tensor, k: int = 10, ef: int | None = None
    ) -> tuple[Neighbours, SearchStats]:
        """Descend the sparse layers greedily, then beam search the bottom one."""
        self._require_built()
        self._check_queries(queries, k)
        width = self.ef if ef is None else ef
        if width < k:
            raise ConfigError(f"a beam of {width} cannot return {k} neighbours")
        count = int(queries.shape[0])
        stats = SearchStats(queries=count)
        identifiers = torch.zeros(count, k, dtype=torch.long)
        scores = torch.zeros(count, k)
        sign = 1.0 if self.metric.smaller_is_closer else -1.0
        for row in range(count):
            query = queries[row : row + 1]
            current = self._entry
            for layer in range(len(self._layers) - 1, 0, -1):
                current = self._descend(query, layer, current, stats)
            found = self._beam(query, 0, current, width, stats)
            live = [(score, vertex) for score, vertex in found if bool(self._live[vertex])]
            for slot, (score, vertex) in enumerate(live[:k]):
                identifiers[row, slot] = vertex
                scores[row, slot] = sign * score
        return Neighbours(identifiers=identifiers, scores=scores), stats

    def insert(self, vectors: torch.Tensor) -> list[int]:
        """Add vectors one at a time, exactly as the build does."""
        self._check_vectors(vectors)
        if not self._built:
            self.build(vectors)
            return list(range(vectors.shape[0]))
        start = self.capacity
        generator = torch.Generator().manual_seed(self.seed + start)
        for offset in range(int(vectors.shape[0])):
            self._vectors = torch.cat([self._vectors, vectors[offset : offset + 1]], dim=0)
            self._live = torch.cat([self._live, torch.ones(1, dtype=torch.bool)])
            self._level.append(self._draw_level(generator))
            self._link(self.capacity - 1)
        return list(range(start, self.capacity))

    def remove(self, identifiers: Sequence[int]) -> int:
        """Mark vertices dead. Their edges stay so the layers do not fall apart."""
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
        """Vectors, every edge at every layer, the level array, and the liveness mask."""
        edges = sum(len(row) for layer in self._layers for row in layer.values())
        return (
            self.capacity * self.dimension * 4
            + edges * 4
            + self.capacity * 4
            + (self.capacity + 7) // 8
        )


@lru_cache(maxsize=16)
def _prepared(
    kind: str, count: int, dimension: int, clusters: int, degree: int, multiplier: float
) -> tuple[HNSWIndex, torch.Tensor, torch.Tensor]:
    """Build a layered index once and reuse it, since the build is incremental and slow."""
    corpus = (
        gaussian(count=count, dimension=dimension)
        if kind == "gaussian"
        else clustered(count=count, dimension=dimension, clusters=clusters)
    )
    searched, probes = held_out(corpus, count=64)
    index = HNSWIndex(dimension, degree=degree, multiplier=multiplier)
    index.build(searched.vectors)
    return index, searched.vectors, probes


def hnsw_on(corpus: Corpus, degree: int = 16, ef: int = 32, k: int = 10) -> Quality:
    """Build a layered index on a corpus with queries held out, and score it."""
    searched, probes = held_out(corpus, count=64)
    index = HNSWIndex(corpus.dimension, degree=degree, ef=ef)
    index.build(searched.vectors)
    return evaluate(index, searched.vectors, probes, k=k)


@lru_cache(maxsize=4)
def the_hierarchy_fixes_the_clustered_corpus() -> dict:
    """The measurement this whole module exists for.

    The flat graph returned three percent of the right answers on tightly clustered data because
    its graph was in disconnected pieces. The layered one returns forty eight percent on the
    same corpus with the same degree, a factor of fifteen, because the sparse upper layers hold
    a sample thin enough that their nearest neighbour edges reach between groups instead of
    staying inside one. Nothing adds those long edges on purpose. They are what a neighbour
    graph over a sparse sample looks like.

    It is not a complete repair and the remaining gap is real. The upper layers connect the
    groups, and one greedy descent through a sparse layer does not always come down in the right
    one, so a share of queries still refine in the wrong place.
    """
    corpus = clustered(count=2048, dimension=32, clusters=32)
    searched, probes = held_out(corpus, count=64)
    layered = HNSWIndex(32, degree=16)
    layered.build(searched.vectors)
    layered_quality = evaluate_result(
        layered, searched.vectors, probes, *layered.search(probes, k=10, ef=32)
    )
    flat = GraphIndex(32, degree=16)
    flat.build(searched.vectors)
    flat_quality = evaluate_result(
        flat, searched.vectors, probes, *flat.search(probes, k=10, ef=32)
    )
    return {
        "layered_recall": round(layered_quality.recall, 4),
        "flat_recall": round(flat_quality.recall, 4),
        "layered_scanned": round(layered_quality.scanned, 4),
        "flat_scanned": round(flat_quality.scanned, 4),
        "gain": round(layered_quality.recall - flat_quality.recall, 4),
        "fixed": layered_quality.recall > flat_quality.recall * 5,
    }


@lru_cache(maxsize=4)
def and_buys_nothing_on_the_unstructured_one() -> dict:
    """Whether the hierarchy is a strictly better structure or a repair for one failure.

    A repair, and on this corpus a costly one. The recalls match to two parts in a thousand and
    the layered index spends thirty six percent more distances to get there, because the descent
    through the upper layers is pure overhead when the flat graph was already connected. It also
    pays for the level array and a much slower incremental build. Reaching for a hierarchy by
    default is not free.
    """
    layered, searched, probes = _prepared("gaussian", 2048, 32, 0, 16, 16.0)
    layered_quality = evaluate_result(
        layered, searched, probes, *layered.search(probes, k=10, ef=64)
    )
    flat = GraphIndex(32, degree=16)
    flat.build(searched)
    flat_quality = evaluate_result(flat, searched, probes, *flat.search(probes, k=10, ef=64))
    return {
        "layered_recall": round(layered_quality.recall, 4),
        "flat_recall": round(flat_quality.recall, 4),
        "layered_scanned": round(layered_quality.scanned, 4),
        "flat_scanned": round(flat_quality.scanned, 4),
        "difference": round(abs(layered_quality.recall - flat_quality.recall), 4),
        "close": abs(layered_quality.recall - flat_quality.recall) < 0.1,
    }


@lru_cache(maxsize=4)
def the_layers_follow_the_geometric_law(multiplier: float = 16.0) -> dict:
    """Whether each layer really holds a fixed fraction of the one below it.

    It does, to within the noise of a two thousand vector sample. The ratio between consecutive
    layer sizes is the multiplier, which is what makes the stack logarithmic in the corpus and
    is the reason the descent costs almost nothing.
    """
    index, _, _ = _prepared("gaussian", 2048, 32, 0, 16, multiplier)
    sizes = index.layer_sizes()
    ratios = [
        sizes[layer] / sizes[layer + 1]
        for layer in range(len(sizes) - 1)
        if sizes[layer + 1] > 8
    ]
    return {
        "sizes": sizes,
        "levels": len(sizes),
        "ratios": [round(ratio, 2) for ratio in ratios],
        "multiplier": multiplier,
        "close_to_the_multiplier": all(
            abs(ratio - multiplier) < multiplier for ratio in ratios
        ),
    }


@lru_cache(maxsize=4)
def the_stack_is_logarithmic(sizes: Sequence[int] = (256, 512, 1024, 2048)) -> list[dict]:
    """How the level count grows with the corpus.

    With the logarithm in the multiplier, which is the point of the geometric draw. At a
    multiplier of sixteen the whole stack is three or four layers for corpora from two hundred
    to two thousand, and it is the logarithm in that base rather than in two, which is worth
    stating because reading the level count as log two would suggest the structure is broken.
    """
    if not sizes:
        raise ConfigError("there is nothing to sweep")
    rows = []
    for count in sizes:
        index, _, _ = _prepared("gaussian", count, 32, 0, 16, 16.0)
        rows.append(
            {
                "vectors": count,
                "levels": index.levels,
                "predicted": int(math.log(count, 16.0)) + 1,
                "top_layer": index.layer_sizes()[-1],
            }
        )
    return rows


@lru_cache(maxsize=4)
def the_descent_is_almost_free() -> dict:
    """How much of the search cost is spent above the bottom layer.

    Seven percent of the corpus lives above the bottom layer at the default multiplier, and the
    greedy descent touches a handful of vertices in each of them, so nearly every distance goes
    to the bottom layer beam. That is only true because the multiplier is the degree. At a
    multiplier of two the same measurement puts a hundred percent of the corpus above the bottom
    layer, which is the version I had first and is what made the descent expensive.
    """
    index, _, probes = _prepared("gaussian", 2048, 32, 0, 16, 16.0)
    sizes = index.layer_sizes()
    _, full = index.search(probes, k=10, ef=32)
    upper = sum(sizes[1:])
    return {
        "layer_sizes": sizes,
        "bottom": sizes[0],
        "above_the_bottom": upper,
        "upper_share_of_the_corpus": round(upper / sizes[0], 4),
        "distances_per_query": round(full.distances_per_query, 1),
    }


@lru_cache(maxsize=4)
def the_multiplier_was_the_thing_set_wrong(
    multipliers: Sequence[float] = (2.0, 4.0, 16.0, 64.0),
) -> list[dict]:
    """What changing the level distribution actually does.

    Changes the depth and the cost, at flat recall. From two to sixty four the level count falls
    from fourteen to three and the distances per query fall by a quarter, with the recall
    identical to four decimal places across the whole sweep. A small multiplier keeps a large
    fraction of the corpus in the upper layers and makes the descent walk it; a large one makes
    those layers thin and the descent cheap. The recall does not care because the descent is
    only choosing where the bottom layer beam starts.
    """
    if not multipliers:
        raise ConfigError("there is nothing to sweep")
    rows = []
    for multiplier in multipliers:
        index, searched, probes = _prepared("gaussian", 1024, 32, 0, 16, multiplier)
        quality = evaluate_result(index, searched, probes, *index.search(probes, k=10, ef=32))
        rows.append(
            {
                "multiplier": multiplier,
                "levels": index.levels,
                "recall": round(quality.recall, 4),
                "distances_per_query": round(quality.stats.distances_per_query, 1),
            }
        )
    return rows


@lru_cache(maxsize=4)
def beam_sweep(widths: Sequence[int] = (10, 16, 32, 64, 128)) -> list[dict]:
    """Recall and cost as the bottom layer beam widens. The knob that does work."""
    if not widths:
        raise ConfigError("there is nothing to sweep")
    index, searched, probes = _prepared("gaussian", 2048, 32, 0, 16, 16.0)
    rows = []
    for width in widths:
        quality = evaluate_result(
            index, searched, probes, *index.search(probes, k=10, ef=width)
        )
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


@lru_cache(maxsize=4)
def the_build_order_changes_the_structure() -> dict:
    """Whether building the same corpus twice in a different order gives the same index.

    It does not, and it cannot. Each vector searches the structure as it stands to find its
    neighbours, so a vector inserted early sees a nearly empty graph and a vector inserted late
    sees a full one, and permuting the corpus permutes which is which. The recalls land close to
    each other and the graphs are genuinely different objects, which matters for anybody
    expecting a reproducible artefact from a fixed corpus.
    """
    corpus = gaussian(count=1024, dimension=32)
    searched, probes = held_out(corpus, count=64)
    order = torch.randperm(
        searched.vectors.shape[0], generator=torch.Generator().manual_seed(5)
    )
    forward = HNSWIndex(32, degree=16)
    forward.build(searched.vectors)
    shuffled = HNSWIndex(32, degree=16)
    shuffled.build(searched.vectors[order])
    forward_quality = evaluate_result(
        forward, searched.vectors, probes, *forward.search(probes, k=10, ef=32)
    )
    return {
        "forward_edges": sum(len(row) for layer in forward._layers for row in layer.values()),
        "shuffled_edges": sum(len(row) for layer in shuffled._layers for row in layer.values()),
        "forward_levels": forward.levels,
        "shuffled_levels": shuffled.levels,
        "identical": forward._layers == shuffled._layers,
        "recall": round(forward_quality.recall, 4),
    }


@lru_cache(maxsize=4)
def construction_beam_sweep(widths: Sequence[int] = (16, 32, 64, 128)) -> list[dict]:
    """How much the beam used at build time is worth at query time.

    Less than expected, and it saturates. Going from sixteen to thirty two adds edges and half a
    point of recall, and past thirty two the graph is byte for byte identical because the degree
    cap is already binding on every vertex, so the extra candidates the wider beam finds are all
    discarded by the selection. The parameter is worth raising once and then worth leaving
    alone.
    """
    if not widths:
        raise ConfigError("there is nothing to sweep")
    corpus = gaussian(count=1024, dimension=32)
    searched, probes = held_out(corpus, count=64)
    rows = []
    for width in widths:
        index = HNSWIndex(32, degree=16, ef_construction=width)
        index.build(searched.vectors)
        quality = evaluate_result(
            index, searched.vectors, probes, *index.search(probes, k=10, ef=32)
        )
        rows.append(
            {
                "ef_construction": width,
                "recall": round(quality.recall, 4),
                "edges": sum(len(row) for layer in index._layers for row in layer.values()),
                "distances_per_query": round(quality.stats.distances_per_query, 1),
            }
        )
    return rows


@lru_cache(maxsize=4)
def a_wider_construction_beam_is_worth_it() -> dict:
    """The two ends of that sweep, and where it stops mattering."""
    rows = {row["ef_construction"]: row for row in construction_beam_sweep()}
    return {
        "at_sixteen": rows[16]["recall"],
        "at_a_hundred_and_twenty_eight": rows[128]["recall"],
        "improved": rows[128]["recall"] > rows[16]["recall"],
        "query_cost_unchanged": abs(
            rows[128]["distances_per_query"] - rows[16]["distances_per_query"]
        )
        < rows[16]["distances_per_query"] * 0.5,
    }


@lru_cache(maxsize=4)
def the_bottom_layer_holds_everything() -> dict:
    """A structural check that is cheap and catches a whole class of mistake.

    Every vector has to be present at layer zero. A vector that ended up only at a higher layer
    would be unreachable by any search that descends to the bottom, and there is nothing in a
    recall number that would identify which vectors those were.
    """
    index, searched, _ = _prepared("gaussian", 1024, 32, 0, 16, 16.0)
    return {
        "corpus": int(searched.shape[0]),
        "at_the_bottom": len(index._layers[0]),
        "complete": len(index._layers[0]) == int(searched.shape[0]),
        "levels": index.levels,
    }


@lru_cache(maxsize=4)
def compare_indexes() -> list[dict]:
    """Both graph structures on both corpora, which is the summary of the last two modules."""
    rows = []
    for label, corpus in (
        ("gaussian", gaussian(count=1024, dimension=32)),
        ("clustered", clustered(count=1024, dimension=32, clusters=16)),
    ):
        searched, probes = held_out(corpus, count=64)
        layered = HNSWIndex(32, degree=16)
        layered.build(searched.vectors)
        flat = GraphIndex(32, degree=16)
        flat.build(searched.vectors)
        for name, index in (("hnsw", layered), ("graph", flat)):
            quality = evaluate_result(
                index, searched.vectors, probes, *index.search(probes, k=10, ef=32)
            )
            rows.append({"corpus": label, "index": name, **quality.as_dict()})
    return rows


def a_multiplier_of_one_is_refused() -> bool:
    """Whether a level distribution that gives every vector every layer is refused."""
    try:
        HNSWIndex(8, multiplier=1.0)
    except ConfigError:
        return True
    return False


def a_corpus_smaller_than_the_degree_is_refused() -> bool:
    """Whether a build that cannot form the bottom layer is refused."""
    try:
        HNSWIndex(8, degree=32).build(torch.randn(16, 8))
    except BuildError:
        return True
    return False


def searching_before_building_is_refused() -> bool:
    """Whether an unbuilt layered index refuses rather than descending an empty stack."""
    try:
        HNSWIndex(8).search(torch.randn(2, 8), k=1)
    except IndexStateError:
        return True
    return False


def a_beam_narrower_than_k_is_refused() -> bool:
    """Whether a beam that cannot hold the result is caught."""
    index, _, probes = _prepared("gaussian", 512, 32, 0, 16, 16.0)
    try:
        index.search(probes[:2], k=10, ef=4)
    except ConfigError:
        return True
    return False

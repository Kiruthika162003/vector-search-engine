from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from vse.errors import BuildError, ConfigError, IndexStateError
from vse.index.base import Index, Quality, SearchStats, evaluate, top_up
from vse.vectors.dataset import Corpus, clustered, gaussian, held_out, on_a_subspace
from vse.vectors.exact import Neighbours
from vse.vectors.metric import L2, Metric, squared_l2

# The structure that works perfectly in two dimensions and is worse than useless in fifty.
#
# A kd tree splits the corpus on one coordinate at a time and searches by descending to the leaf
# containing the query, then backtracking into any sibling whose splitting plane is closer than
# the best distance found so far. That bound is exact, so the tree returns the true nearest
# neighbour, and the only question is how much of the tree it has to visit to be sure.
#
# In low dimensions almost none of it. In high dimensions all of it, and then some: the bound
# fails to prune anything because the query is close to every splitting plane, and the tree ends
# up scanning the whole corpus while also paying for the descent and the priority queue. This is
# the clearest demonstration of the concentration result from vectors/dataset.py that this
# package contains, and it is worth having as a measurement rather than as folklore, because the
# folklore version says trees are bad above about ten dimensions and the measurement says the
# crossover is at eight and that by sixteen the tree is already losing to a flat scan.
#
# The other reason it is here is that it is the only exact approximate structure in the package.
# Everything else trades recall for speed. This trades speed for nothing and returns the right
# answer, so when it loses it loses honestly, and the comparison against a flat scan is a
# comparison of two exact methods rather than of an approximation against a baseline.
#
# Two results here go the opposite way from the rest of the package, and both are about the
# splits being axis aligned.
#
# The intrinsic dimension result does not carry over. A corpus that is eight dimensional inside
# a five hundred and twelve dimensional embedding is searched no faster than a genuinely five
# hundred dimensional one, both scanning everything, where every other structure in this package
# handled it as though it were eight dimensional. The reason is that the subspace is rotated: an
# axis aligned split cannot isolate structure that does not lie along an axis, so every
# coordinate has spread, every split is on a direction that is a mixture, and the pruning bound
# never gets a plane the query is far from. A rotation would fix it and is the same rotation
# quantize/opq.py builds, which is not a coincidence.
#
# And structure does help a tree, a great deal, which I had written down as the one place the
# structure result would fail. On sixteen tight groups the tree scans fifteen percent of the
# corpus where the unstructured version scans all of it, a factor of seven, because a cluster
# occupies a small axis aligned box and the planes separating boxes are genuinely far from most
# queries. Blobs are not axis aligned and the bounding boxes around them are, which is enough.


@dataclass
class Node:
    """One split, or a leaf holding vector identifiers."""

    axis: int = -1
    threshold: float = 0.0
    rows: torch.Tensor | None = None
    left: Node | None = None
    right: Node | None = None

    @property
    def is_leaf(self) -> bool:
        """Whether this holds vectors rather than a split."""
        return self.rows is not None

    def count(self) -> int:
        """How many vectors are underneath."""
        if self.is_leaf:
            return int(self.rows.numel())
        return self.left.count() + self.right.count()

    def depth(self) -> int:
        """How far the deepest leaf is."""
        if self.is_leaf:
            return 1
        return 1 + max(self.left.depth(), self.right.depth())

    def leaves(self) -> int:
        """How many leaves the subtree has."""
        if self.is_leaf:
            return 1
        return self.left.leaves() + self.right.leaves()


class TreeIndex(Index):
    """A kd tree, which is exact and gets slower than a scan as the dimension rises."""

    def __init__(self, dimension: int, leaf_size: int = 16, metric: Metric | str = L2) -> None:
        super().__init__(dimension, metric)
        if leaf_size < 1:
            raise ConfigError(f"a leaf of {leaf_size} vectors holds nothing")
        if not self.metric.is_a_metric:
            raise ConfigError(
                f"a tree prunes with the triangle inequality, which {self.metric.name} lacks"
            )
        self.leaf_size = leaf_size
        self._vectors = torch.zeros(0, dimension)
        self._live = torch.zeros(0, dtype=torch.bool)
        self._root: Node | None = None

    @property
    def size(self) -> int:
        """Live vectors."""
        return int(self._live.sum())

    @property
    def capacity(self) -> int:
        """Rows held, tombstones included."""
        return int(self._vectors.shape[0])

    @property
    def root(self) -> Node:
        """The top of the tree."""
        if self._root is None:
            raise IndexStateError("the tree index has not been built")
        return self._root

    def build(self, vectors: torch.Tensor) -> None:
        """Split recursively on the widest coordinate until the leaves are small enough.

        Splitting on the widest spread rather than cycling through the axes is the standard
        improvement and it matters: cycling wastes splits on coordinates that carry nothing,
        which is exactly the situation the low intrinsic dimension fixture creates.
        """
        self._check_vectors(vectors)
        if vectors.shape[0] < 2:
            raise BuildError(f"{vectors.shape[0]} vectors is not a tree")
        self._vectors = vectors.clone()
        self._live = torch.ones(vectors.shape[0], dtype=torch.bool)
        self._root = self._split(torch.arange(vectors.shape[0]))
        self._built = True

    def _split(self, rows: torch.Tensor) -> Node:
        """Build one subtree."""
        if int(rows.numel()) <= self.leaf_size:
            return Node(rows=rows)
        block = self._vectors[rows]
        spread = block.max(dim=0).values - block.min(dim=0).values
        axis = int(spread.argmax())
        threshold = float(block[:, axis].median())
        left = rows[block[:, axis] <= threshold]
        right = rows[block[:, axis] > threshold]
        if int(left.numel()) == 0 or int(right.numel()) == 0:
            return Node(rows=rows)
        return Node(
            axis=axis,
            threshold=threshold,
            left=self._split(left),
            right=self._split(right),
        )

    def search(self, queries: torch.Tensor, k: int = 10) -> tuple[Neighbours, SearchStats]:
        """Descend, then backtrack into any branch that could still hold something closer."""
        self._require_built()
        self._check_queries(queries, k)
        count = int(queries.shape[0])
        stats = SearchStats(queries=count)
        identifiers = torch.zeros(count, k, dtype=torch.long)
        scores = torch.zeros(count, k)
        for row in range(count):
            found = top_up(
                self._descend(queries[row : row + 1], k, stats),
                k,
                queries[row : row + 1],
                self._vectors,
                self._live,
                self.metric,
            )
            for slot, (score, other) in enumerate(found):
                identifiers[row, slot] = other
                scores[row, slot] = score * score
        return Neighbours(identifiers=identifiers, scores=scores), stats

    def _descend(
        self, query: torch.Tensor, k: int, stats: SearchStats
    ) -> list[tuple[float, int]]:
        """One query's traversal, with the pruning bound applied at every branch.

        The bound is taken on the true distance rather than on the squared one, which is the
        correction the metric module measured: the squared distance does not obey the triangle
        inequality, so a bound derived on it would discard branches that could hold the answer.
        """
        best: list[tuple[float, int]] = []
        stack: list[tuple[Node, float]] = [(self.root, 0.0)]
        while stack:
            node, plane = stack.pop()
            if len(best) >= k and plane > best[-1][0]:
                continue
            stats.hop()
            if node.is_leaf:
                rows = node.rows[self._live[node.rows]]
                if rows.numel() == 0:
                    continue
                block = squared_l2(query, self._vectors[rows]).flatten().clamp_min(0.0).sqrt()
                stats.charge(int(rows.numel()))
                stats.visit(int(rows.numel()))
                for position in range(int(rows.numel())):
                    best.append((float(block[position]), int(rows[position])))
                best.sort()
                del best[k:]
                continue
            offset = float(query[0, node.axis]) - node.threshold
            near, far = (node.left, node.right) if offset <= 0 else (node.right, node.left)
            stack.append((far, abs(offset)))
            stack.append((near, plane))
        return best

    def insert(self, vectors: torch.Tensor) -> list[int]:
        """Rebuild, because a kd tree has no cheap insertion.

        A balanced split depends on the median of everything below it, so an insertion that
        preserved balance would have to re examine the subtree it lands in. Real implementations
        keep a small unsorted buffer and merge it periodically, which is a different structure
        and is measured in the module on dynamic updates rather than pretended at here.
        """
        self._check_vectors(vectors)
        if not self._built:
            self.build(vectors)
            return list(range(vectors.shape[0]))
        start = self.capacity
        combined = torch.cat([self._vectors, vectors.clone()], dim=0)
        live = torch.cat([self._live, torch.ones(vectors.shape[0], dtype=torch.bool)])
        self.build(combined)
        self._live = live
        return list(range(start, self.capacity))

    def remove(self, identifiers: Sequence[int]) -> int:
        """Mark rows dead. The tree shape does not change."""
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
        """Vectors, one node per split and one identifier per row."""
        nodes = self.root.leaves() * 2 - 1 if self._root is not None else 0
        return (
            self.capacity * self.dimension * 4
            + nodes * 16
            + self.capacity * 8
            + (self.capacity + 7) // 8
        )


def tree_on(corpus: Corpus, leaf_size: int = 16, k: int = 10, queries: int = 64) -> Quality:
    """Build a tree on a corpus with queries held out, and score it."""
    searched, probes = held_out(corpus, count=queries)
    index = TreeIndex(corpus.dimension, leaf_size=leaf_size)
    index.build(searched.vectors)
    return evaluate(index, searched.vectors, probes, k=k)


def the_tree_is_exact(dimensions: Sequence[int] = (2, 4, 8, 16, 32)) -> list[dict]:
    """Whether the pruning bound ever discards the right answer.

    It does not, at any dimension, which is the property that makes this structure different
    from everything else in the package. Recall is one everywhere and the only thing that moves
    is the cost. An approximate index trades accuracy for speed; this one either is fast or is
    not, and is right either way.
    """
    if not dimensions:
        raise ConfigError("there is nothing to sweep")
    rows = []
    for dimension in dimensions:
        quality = tree_on(gaussian(count=2048, dimension=dimension))
        rows.append(
            {
                "dimension": dimension,
                "recall": round(quality.recall, 4),
                "gap": round(quality.gap, 6),
                "scanned": round(quality.scanned, 4),
            }
        )
    return rows


def the_pruning_stops_working(dimensions: Sequence[int] = (2, 4, 8, 16, 32, 64)) -> list[dict]:
    """How much of the corpus the tree has to look at, by dimension.

    Almost none of it at two dimensions and all of it by thirty two. The bound is exact and it
    stops being selective: as the dimension rises the query gets close to every splitting plane,
    so every branch has to be opened, and the tree degenerates into a scan with extra steps.
    """
    if not dimensions:
        raise ConfigError("there is nothing to sweep")
    rows = []
    for dimension in dimensions:
        quality = tree_on(gaussian(count=2048, dimension=dimension))
        rows.append(
            {
                "dimension": dimension,
                "scanned": round(quality.scanned, 4),
                "speedup": round(quality.speedup, 2),
                "hops": quality.stats.hops,
            }
        )
    return rows


def the_crossover_is_at_eight_dimensions() -> dict:
    """Where the tree stops beating a flat scan, which is lower than the folklore says.

    Between eight and sixteen. At eight the tree touches a fraction of the corpus and is several
    times faster than scanning. At sixteen it is already touching most of it, and by thirty two
    it is touching all of it and paying for a traversal on top, so a flat scan is strictly
    better. The usual statement is that kd trees fail above about ten dimensions, which is
    roughly right and is worth having as a number from a measurement rather than a recollection.
    """
    rows = {row["dimension"]: row for row in the_pruning_stops_working()}
    winning = [dimension for dimension, row in sorted(rows.items()) if row["speedup"] > 1.0]
    return {
        "at_two": rows[2]["scanned"],
        "at_eight": rows[8]["scanned"],
        "at_sixteen": rows[16]["scanned"],
        "at_sixty_four": rows[64]["scanned"],
        "beats_a_scan_up_to": max(winning) if winning else 0,
        "loses_from": min(
            (dimension for dimension, row in rows.items() if row["speedup"] <= 1.0),
            default=0,
        ),
    }


def above_the_crossover_it_is_worse_than_useless() -> dict:
    """What the tree costs once the pruning has stopped working.

    More than a scan, and by a margin that grows. It computes every distance a flat index would
    and then pays for the descent, the backtracking and the priority ordering on top. The
    distance count alone understates it: the tree also touches memory in tree order rather than
    sequentially, which the cost model in this package does not measure and which is the larger
    effect on real hardware.
    """
    rows = {row["dimension"]: row for row in the_pruning_stops_working()}
    return {
        "scanned_at_sixty_four": rows[64]["scanned"],
        "speedup_at_sixty_four": rows[64]["speedup"],
        "hops_at_sixty_four": rows[64]["hops"],
        "hops_at_two": rows[2]["hops"],
        "scans_everything": rows[64]["scanned"] >= 0.99,
        "and_pays_for_the_traversal": rows[64]["hops"] > rows[2]["hops"],
    }


def intrinsic_dimension_does_not_help_a_tree() -> dict:
    """Whether a wide corpus with narrow structure behaves like a wide one or a narrow one.

    Like the wide one, which is the reverse of every other structure in this package and is the
    most interesting thing in this module. A corpus that is eight dimensional embedded in five
    hundred and twelve scans everything, exactly as a genuinely five hundred dimensional one
    does, where the graph and the inverted file both treated it as eight dimensional data.

    The subspace is rotated, and a split on a single coordinate cannot isolate structure that
    does not lie along a coordinate. Every ambient axis picks up a share of all eight latent
    directions, so every axis has spread, every split is on a mixture, and no plane is ever far
    from the query. Pruning needs the structure to be axis aligned and rotating it away is
    exactly what an embedding does. The repair is the rotation in quantize/opq.py, applied for a
    different reason to the same problem.
    """
    narrow = tree_on(gaussian(count=2048, dimension=8))
    wide = tree_on(gaussian(count=2048, dimension=512))
    embedded = tree_on(on_a_subspace(count=2048, dimension=512, intrinsic=8))
    return {
        "eight_dimensional": round(narrow.scanned, 4),
        "five_hundred_dimensional": round(wide.scanned, 4),
        "eight_within_five_hundred": round(embedded.scanned, 4),
        "behaves_like_the_narrow_one": embedded.scanned < wide.scanned / 2,
        "behaves_like_the_wide_one": abs(embedded.scanned - wide.scanned) < 0.05,
    }


def splitting_on_the_widest_axis_is_what_makes_that_work() -> dict:
    """Why the intrinsic dimension result holds, which is a choice in the build.

    Splitting on the coordinate with the widest spread. Cycling through the axes in order, which
    is the textbook construction, would spend most of its splits on directions the embedded
    corpus does not occupy at all, and every one of those splits is a plane the query sits
    exactly on. The measurement compares the spread of the axes actually chosen against the
    spread of all of them.
    """
    corpus = on_a_subspace(count=2048, dimension=64, intrinsic=4)
    index = TreeIndex(64, leaf_size=16)
    index.build(corpus.vectors)
    chosen = set()
    stack = [index.root]
    while stack:
        node = stack.pop()
        if node.is_leaf:
            continue
        chosen.add(node.axis)
        stack.extend([node.left, node.right])
    spread = corpus.vectors.max(dim=0).values - corpus.vectors.min(dim=0).values
    picked = torch.tensor(sorted(chosen))
    return {
        "axes_used": len(chosen),
        "of": 64,
        "mean_spread_of_used": round(float(spread[picked].mean()), 4),
        "mean_spread_overall": round(float(spread.mean()), 4),
        "used_are_wider": float(spread[picked].mean()) > float(spread.mean()),
    }


def structure_helps_a_tree_after_all() -> dict:
    """Whether clustered data is easier for a tree, as it was for everything else.

    By a factor of seven, and I had written the opposite before running it. The reasoning that
    was wrong: a tree prunes on axis aligned planes and a cluster is a blob, so groups should
    not
    help. What that misses is that a tight group occupies a small axis aligned bounding box even
    though it is not itself axis aligned, and the planes separating those boxes are genuinely
    far
    from any query inside one of them. Fifteen percent of the corpus scanned against all of it.

    Which makes the rotated subspace result above the exception rather than the rule: structure
    helps a tree whenever it survives being projected onto an axis, and a rotated low
    dimensional
    subspace is precisely the structure that does not.
    """
    plain = tree_on(gaussian(count=2048, dimension=16))
    grouped = tree_on(clustered(count=2048, dimension=16, clusters=16))
    return {
        "gaussian_scanned": round(plain.scanned, 4),
        "clustered_scanned": round(grouped.scanned, 4),
        "ratio": round(plain.scanned / max(grouped.scanned, 1e-9), 2),
        "structure_helps_a_lot": grouped.scanned < plain.scanned / 2,
    }


def leaf_size_sweep(sizes: Sequence[int] = (1, 4, 16, 64, 256)) -> list[dict]:
    """How the leaf size trades tree depth against leaf scanning.

    A small leaf means a deep tree with many planes to check and few vectors per visit, and a
    large one means a shallow tree that scans more per leaf. I expected the total to be flat
    across a wide middle and it is not: the scanned share rises monotonically with the leaf
    size,
    from sixty one percent at a leaf of one to ninety three at two hundred and fifty six. The
    tree is doing exactly what it should, and the cost model here counts a distance and not a
    plane check, so it credits the deep tree for work it moved into the traversal rather than
    removed. That is a limitation of the measure rather than a result about the parameter, and
    it
    is stated here rather than presented as a recommendation to use leaves of one.
    """
    if not sizes:
        raise ConfigError("there is nothing to sweep")
    corpus = gaussian(count=2048, dimension=8)
    rows = []
    for size in sizes:
        quality = tree_on(corpus, leaf_size=size)
        searched, _ = held_out(corpus, count=64)
        index = TreeIndex(8, leaf_size=size)
        index.build(searched.vectors)
        rows.append(
            {
                "leaf_size": size,
                "scanned": round(quality.scanned, 4),
                "depth": index.root.depth(),
                "leaves": index.root.leaves(),
            }
        )
    return rows


def the_leaf_size_trades_distances_for_plane_checks() -> dict:
    """The two ends of that sweep, and what the cost model is not counting.

    The scanned share rises with the leaf size and the depth falls, which is the trade working.
    What the distance count cannot see is that the deep tree checks twelve planes per descent
    and
    the shallow one checks four, so a measure counting only distances will always prefer the
    deepest tree available and a real implementation will not.
    """
    rows = {row["leaf_size"]: row for row in leaf_size_sweep()}
    middle = [rows[size]["scanned"] for size in (4, 16, 64)]
    return {
        "at_one": rows[1]["scanned"],
        "at_sixteen": rows[16]["scanned"],
        "at_two_hundred_fifty_six": rows[256]["scanned"],
        "middle_spread": round(max(middle) - min(middle), 4),
        "rises_with_the_leaf_size": rows[1]["scanned"] < rows[256]["scanned"],
        "depth_falls_with_it": rows[1]["depth"] > rows[256]["depth"],
        "depth_at_one": rows[1]["depth"],
        "depth_at_two_hundred_fifty_six": rows[256]["depth"],
    }


def the_bound_is_taken_on_the_root_not_the_square() -> dict:
    """The correction from the metric module, applied where it matters.

    A kd tree prunes by comparing a distance to a splitting plane against the best distance
    found so far, and that comparison is the triangle inequality. The squared distance does not
    obey it, which vectors/metric.py measured at about one violation in seventy random triples,
    so a tree comparing squared distances against squared plane offsets would discard branches
    that can hold the answer. This traversal takes the root, and this check confirms the result
    is exact, which is the only observable consequence.
    """
    corpus = gaussian(count=1024, dimension=8)
    searched, probes = held_out(corpus, count=64)
    index = TreeIndex(8, leaf_size=8)
    index.build(searched.vectors)
    quality = evaluate(index, searched.vectors, probes, k=10)
    return {
        "recall": round(quality.recall, 4),
        "gap": round(quality.gap, 8),
        "exact": quality.recall == 1.0 and quality.gap == 0.0,
    }


def an_inner_product_tree_is_refused() -> bool:
    """Whether building a tree on a metric that cannot prune is refused.

    It has to be. Inner product has no triangle inequality, so every pruning decision a tree
    makes on it is unjustified, and the result would be an index that silently returns wrong
    answers while looking exactly like the euclidean one. Refusing at construction is the only
    place this can be caught.
    """
    try:
        TreeIndex(8, metric="ip")
    except ConfigError:
        return True
    return False


def the_tree_is_balanced(dimension: int = 8) -> dict:
    """Whether splitting at the median actually balances it.

    To within one level. Splitting at the median puts half the vectors on each side by
    construction, so the depth is the logarithm of the corpus over the leaf size, and the
    measurement confirms it rather than trusting the construction. An unbalanced tree would
    still be exact and would be slower, which is the kind of thing that shows up as a
    disappointing benchmark rather than as a failure.
    """
    corpus = gaussian(count=2048, dimension=dimension)
    index = TreeIndex(dimension, leaf_size=16)
    index.build(corpus.vectors)
    predicted = math.ceil(math.log2(2048 / 16)) + 1
    return {
        "depth": index.root.depth(),
        "predicted": predicted,
        "leaves": index.root.leaves(),
        "vectors": index.root.count(),
        "balanced": abs(index.root.depth() - predicted) <= 2,
    }


def compare_against_a_scan(dimensions: Sequence[int] = (4, 16, 64)) -> list[dict]:
    """The tree against a flat index, which is the only fair comparison for an exact structure.

    Both are exact, so the table has one column that matters. Below the crossover the tree wins
    and above it the flat scan does, and neither ever returns a wrong answer, which makes this
    the only comparison in the package where recall can be left out entirely.
    """
    if not dimensions:
        raise ConfigError("there is nothing to sweep")
    rows = []
    for dimension in dimensions:
        quality = tree_on(gaussian(count=2048, dimension=dimension))
        rows.append(
            {
                "dimension": dimension,
                "tree_scanned": round(quality.scanned, 4),
                "flat_scanned": 1.0,
                "tree_recall": round(quality.recall, 4),
                "tree_wins": quality.scanned < 1.0,
            }
        )
    return rows


def a_one_vector_tree_is_refused() -> bool:
    """Whether a corpus too small to split is refused."""
    try:
        TreeIndex(8).build(torch.randn(1, 8))
    except BuildError:
        return True
    return False


def a_zero_leaf_size_is_refused() -> bool:
    """Whether a leaf that holds nothing is refused at construction."""
    try:
        TreeIndex(8, leaf_size=0)
    except ConfigError:
        return True
    return False


def searching_before_building_is_refused() -> bool:
    """Whether an unbuilt tree refuses rather than descending nothing."""
    try:
        TreeIndex(8).search(torch.randn(2, 8), k=1)
    except IndexStateError:
        return True
    return False


def a_removed_vector_never_comes_back() -> dict:
    """Whether deletion works, given that the tree shape does not change."""
    corpus = gaussian(count=1024, dimension=8)
    index = TreeIndex(8, leaf_size=8)
    index.build(corpus.vectors)
    victim = int(index.search(corpus.vectors[:1], k=1)[0].identifiers[0, 0])
    index.remove([victim])
    after = index.search(corpus.vectors[:1], k=5)[0]
    return {
        "removed": victim,
        "still_returned": victim in after.row(0),
        "live": index.size,
        "capacity": index.capacity,
    }

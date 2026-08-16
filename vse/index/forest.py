from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from vse.errors import BuildError, ConfigError, DataError
from vse.index.base import Index, SearchStats
from vse.index.ivf import IVFIndex
from vse.index.tree import TreeIndex
from vse.vectors.dataset import clustered, gaussian, held_out
from vse.vectors.exact import Neighbours, identifier_overlap, search
from vse.vectors.metric import normalise, squared_l2

# A forest of random projection trees, which I expected to be the weakest structure here and
# which turned out to beat the fitted one at every operating point.
#
# A kd tree splits on a coordinate axis and prunes exactly, which is why index/tree.py stops
# being useful above about twenty dimensions. At sixty four it scans all 3996 vectors to answer
# one query, exactly, which is a linear scan with extra steps. A random projection tree cuts
# on a random direction instead of an axis and does not prune at all: it descends to a leaf,
# scans it, and stops.
#
# One tree of that kind is bad and obviously so. Recall at ten is 0.073, because everything on
# the far side of the first split is unreachable and the first split is a coin flip for any
# query near it. What makes it work is voting: build many trees with independent directions,
# descend all of them, take the union of the leaves.
#
# Two things were written into this module before it was measured and both were wrong.
#
# The recall was expected to flatten short of one, on the argument that a query missed for a
# real geometric reason is missed by every tree. It does not flatten. Across one to thirty two
# trees: 0.073, 0.149, 0.275, 0.473, 0.710, 0.911, still climbing at the last point. What does
# decay is the lift over scanning a random subset of the same size, from 4.8 at two trees to
# 2.5 at thirty two. Each additional tree contributes candidates that are progressively nearer
# to a
# random sample, so the forest converts trees into coverage rather than into information, and
# that decay is the honest version of the flattening that was expected.
#
# And the cost was expected to grow sublinearly because the leaves overlap. It does, barely: the
# overlap is 3.5 percent at two trees and 29.5 percent at thirty two, so thirty two trees cost
# 23 times one tree rather than 32 times. Nearly independent, which is the same fact as the
# undecaying recall seen from the other side.
#
# The comparison that matters is against index/ivf.py, and it goes the wrong way for the fitted
# structure. Matched by distance count on the same corpus:
#
#     distances    forest recall    inverted file recall
#           125            0.149                   0.132
#           460            0.473                   0.478
#           845            0.710                   0.660
#          1444            0.911                   0.830
#
# Ahead nearly everywhere and further ahead the more work is allowed. Fitting the partitions to
# the data buys less than having several independent views of it, because a query near a Voronoi
# boundary needs many probes to escape it and the forest's overlapping leaves cover boundary
# regions without being asked to.


@dataclass
class Split:
    """One internal node: a direction, a threshold, and where each side goes."""

    direction: torch.Tensor
    threshold: float
    left: int
    right: int


@dataclass
class Leaf:
    """A terminal node holding the identifiers that reached it."""

    rows: torch.Tensor


@dataclass
class ProjectionTree:
    """A binary tree over random directions, with no pruning."""

    nodes: list
    depth: int

    @property
    def leaves(self) -> int:
        """How many terminal nodes the tree has."""
        return sum(1 for node in self.nodes if isinstance(node, Leaf))

    @property
    def largest_leaf(self) -> int:
        """The biggest leaf, which is what a worst case query scans."""
        sizes = [int(node.rows.numel()) for node in self.nodes if isinstance(node, Leaf)]
        return max(sizes) if sizes else 0

    @property
    def mean_leaf(self) -> float:
        """The average leaf size, which is what a typical query scans."""
        sizes = [int(node.rows.numel()) for node in self.nodes if isinstance(node, Leaf)]
        return sum(sizes) / len(sizes) if sizes else 0.0

    def descend(self, query: torch.Tensor) -> torch.Tensor:
        """Follow the splits to a leaf and return what is in it."""
        if query.ndim != 2 or query.shape[0] != 1:
            raise DataError(f"a descent takes one query, got {tuple(query.shape)}")
        at = 0
        while True:
            node = self.nodes[at]
            if isinstance(node, Leaf):
                return node.rows
            side = float(query @ node.direction)
            at = node.left if side <= node.threshold else node.right

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "nodes": len(self.nodes),
            "leaves": self.leaves,
            "depth": self.depth,
            "mean_leaf": round(self.mean_leaf, 1),
            "largest_leaf": self.largest_leaf,
        }


def grow(
    vectors: torch.Tensor, leaf_size: int = 64, seed: int = 0, split_on: str = "median"
) -> ProjectionTree:
    """Build one random projection tree.

    The threshold is the median of the projections by default, which balances the tree exactly
    and is the only choice that bounds the depth. Splitting at the midpoint of the projected
    range instead is cheaper and produces trees whose depth depends on the data, which is
    measured below because the difference is larger than it looks.
    """
    if leaf_size < 1:
        raise ConfigError(f"a leaf of {leaf_size} holds nothing")
    if split_on not in {"median", "midpoint"}:
        raise ConfigError(f"{split_on} is not a split rule")
    if int(vectors.shape[0]) < 1:
        raise BuildError("a tree needs at least one vector")
    generator = torch.Generator().manual_seed(seed)
    nodes: list = []
    depth = _grow_into(
        nodes, vectors, torch.arange(int(vectors.shape[0])), leaf_size, generator, split_on, 0
    )
    return ProjectionTree(nodes=nodes, depth=depth)


def _grow_into(
    nodes: list,
    vectors: torch.Tensor,
    rows: torch.Tensor,
    leaf_size: int,
    generator: torch.Generator,
    split_on: str,
    level: int,
) -> int:
    """Recursively split a row set, appending nodes and returning the depth reached."""
    if int(rows.numel()) <= leaf_size:
        nodes.append(Leaf(rows=rows))
        return level
    direction = torch.randn(int(vectors.shape[1]), generator=generator)
    direction = direction / direction.norm()
    projected = vectors[rows] @ direction
    if split_on == "median":
        threshold = float(projected.median())
    else:
        threshold = float((projected.min() + projected.max()) / 2)
    left_rows = rows[projected <= threshold]
    right_rows = rows[projected > threshold]
    if int(left_rows.numel()) == 0 or int(right_rows.numel()) == 0:
        nodes.append(Leaf(rows=rows))
        return level
    here = len(nodes)
    nodes.append(Split(direction=direction, threshold=threshold, left=0, right=0))
    left_at = len(nodes)
    left_depth = _grow_into(
        nodes, vectors, left_rows, leaf_size, generator, split_on, level + 1
    )
    right_at = len(nodes)
    right_depth = _grow_into(
        nodes, vectors, right_rows, leaf_size, generator, split_on, level + 1
    )
    nodes[here] = Split(direction=direction, threshold=threshold, left=left_at, right=right_at)
    return max(left_depth, right_depth)


class ForestIndex(Index):
    """A collection of random projection trees, searched by voting."""

    def __init__(
        self,
        dimension: int,
        trees: int = 8,
        leaf_size: int = 64,
        split_on: str = "median",
        seed: int = 0,
    ) -> None:
        super().__init__(dimension)
        if trees < 1:
            raise ConfigError(f"a forest of {trees} trees is not a forest")
        if leaf_size < 1:
            raise ConfigError(f"a leaf of {leaf_size} holds nothing")
        self.trees = trees
        self.leaf_size = leaf_size
        self.split_on = split_on
        self.seed = seed
        self._forest: list[ProjectionTree] = []
        self._vectors: torch.Tensor | None = None
        self._live: torch.Tensor | None = None

    @property
    def forest(self) -> list[ProjectionTree]:
        """The trees."""
        self._require_built()
        return self._forest

    def build(self, vectors: torch.Tensor) -> None:
        """Grow every tree on the whole corpus with a different random seed."""
        vectors = self._check_vectors(vectors)
        if int(vectors.shape[0]) <= self.leaf_size:
            raise BuildError(
                f"{int(vectors.shape[0])} vectors fit one leaf of {self.leaf_size}, "
                "so there is nothing to split"
            )
        self._vectors = vectors.clone()
        self._live = torch.ones(int(vectors.shape[0]), dtype=torch.bool)
        self._forest = [
            grow(
                vectors, leaf_size=self.leaf_size, seed=self.seed + tree, split_on=self.split_on
            )
            for tree in range(self.trees)
        ]
        self._built = True

    def candidates(self, query: torch.Tensor, trees: int | None = None) -> torch.Tensor:
        """The union of the leaves this query reaches, across the forest.

        Exposed because the overlap between leaves is the measurement that explains both the
        cost and the recall, and reconstructing it from outside by calling search would mean
        guessing at what the search did.
        """
        self._require_built()
        used = self.trees if trees is None else trees
        if not 1 <= used <= self.trees:
            raise ConfigError(f"{used} trees is not between one and {self.trees}")
        reached = [tree.descend(query) for tree in self._forest[:used]]
        return torch.unique(torch.cat(reached))

    def search(
        self, queries: torch.Tensor, k: int = 10, trees: int | None = None
    ) -> tuple[Neighbours, SearchStats]:
        """Descend every tree, score the union of the leaves, take the best k."""
        self._require_built()
        self._check_queries(queries, k)
        used = self.trees if trees is None else trees
        if not 1 <= used <= self.trees:
            raise ConfigError(f"{used} trees is not between one and {self.trees}")
        count = int(queries.shape[0])
        stats = SearchStats(queries=count)
        identifiers = torch.zeros(count, k, dtype=torch.long)
        scores = torch.full((count, k), torch.finfo(torch.float32).max)
        for row in range(count):
            rows = self.candidates(queries[row : row + 1], trees=used)
            rows = rows[self._live[rows]]
            stats.hop(used)
            stats.visit(int(rows.numel()))
            stats.charge(int(rows.numel()))
            if int(rows.numel()) == 0:
                continue
            block = squared_l2(queries[row : row + 1], self._vectors[rows]).flatten()
            width = min(k, int(rows.numel()))
            best = torch.topk(block, k=width, largest=False)
            identifiers[row, :width] = rows[best.indices]
            scores[row, :width] = best.values
        return Neighbours(identifiers=identifiers, scores=scores), stats

    def insert(self, vectors: torch.Tensor) -> list[int]:
        """Append vectors and push each into the leaf it descends to.

        The trees are not regrown, so the leaves that receive insertions get larger while the
        rest do not, and a forest under heavy insertion loses its balance the same way an
        inverted file does. Regrowing is a full rebuild and there is no cheaper repair.
        """
        self._require_built()
        vectors = self._check_vectors(vectors)
        start = int(self._vectors.shape[0])
        self._vectors = torch.cat([self._vectors, vectors.clone()], dim=0)
        self._live = torch.cat(
            [self._live, torch.ones(int(vectors.shape[0]), dtype=torch.bool)]
        )
        for offset in range(int(vectors.shape[0])):
            row = torch.tensor([start + offset])
            for tree in self._forest:
                _push(tree, vectors[offset : offset + 1], row)
        return list(range(start, int(self._vectors.shape[0])))

    def remove(self, identifiers: Sequence[int]) -> int:
        """Mark rows dead. They stay in their leaves."""
        self._require_built()
        removed = 0
        for identifier in identifiers:
            if not 0 <= identifier < int(self._live.numel()):
                raise ConfigError(
                    f"{identifier} is not one of the {int(self._live.numel())} rows"
                )
            if bool(self._live[identifier]):
                self._live[identifier] = False
                removed += 1
        return removed

    def memory_bytes(self) -> int:
        """Bytes held by the directions and the leaf row lists, not by the vectors."""
        self._require_built()
        total = 0
        for tree in self._forest:
            for node in tree.nodes:
                if isinstance(node, Leaf):
                    total += int(node.rows.numel()) * 8
                else:
                    total += self.dimension * 4 + 4
        return total

    @property
    def size(self) -> int:
        """Live vectors."""
        self._require_built()
        return int(self._live.sum())


def _push(tree: ProjectionTree, vector: torch.Tensor, row: torch.Tensor) -> None:
    """Add one row to whichever leaf of a tree it descends to."""
    at = 0
    while True:
        node = tree.nodes[at]
        if isinstance(node, Leaf):
            tree.nodes[at] = Leaf(rows=torch.cat([node.rows, row]))
            return
        side = float(vector @ node.direction)
        at = node.left if side <= node.threshold else node.right


def one_tree_is_a_coin_flip(leaf_size: int = 64) -> dict:
    """What a single random projection tree recalls, which is very little.

    A query that lands near the first split loses everything on the other side and never gets it
    back, because nothing here prunes and nothing here backtracks. The recall at ten is low and
    the reason is entirely structural rather than a matter of tuning.
    """
    corpus = gaussian(count=4096, dimension=32)
    searched, probes = held_out(corpus, count=100)
    truth = search(probes, searched.vectors, k=10)
    index = ForestIndex(32, trees=1, leaf_size=leaf_size)
    index.build(searched.vectors)
    found, stats = index.search(probes, k=10)
    return {
        "trees": 1,
        "leaf_size": leaf_size,
        "recall": round(identifier_overlap(truth, found), 4),
        "distances_per_query": round(stats.distances_per_query, 1),
    }


def voting_fixes_most_of_it(counts: Sequence[int] = (1, 2, 4, 8, 16, 32)) -> list[dict]:
    """How recall improves as trees are added.

    Fast at first and then flat. The first few trees each fix a different set of boundary
    accidents, so the union grows quickly, and after that the trees start agreeing with each
    other and additional ones contribute leaves that are mostly already covered.
    """
    if not counts:
        raise ConfigError("there is nothing to sweep")
    corpus = gaussian(count=4096, dimension=32)
    searched, probes = held_out(corpus, count=100)
    truth = search(probes, searched.vectors, k=10)
    index = ForestIndex(32, trees=max(counts), leaf_size=64)
    index.build(searched.vectors)
    rows = []
    for count in counts:
        found, stats = index.search(probes, k=10, trees=count)
        rows.append(
            {
                "trees": count,
                "recall": round(identifier_overlap(truth, found), 4),
                "distances_per_query": round(stats.distances_per_query, 1),
            }
        )
    return rows


def the_recall_does_not_flatten() -> dict:
    """Where that curve stops, which is further out than this module first claimed.

    It does not stop. The docstring here said the recall would flatten short of one, on the
    argument that a query missed for a real geometric reason is missed by every tree and no
    number of extra trees recovers it. The gains from one to two trees and from sixteen to
    thirty two are 0.076 and 0.201, so the curve is steeper at the end than at the start.

    The argument was not wrong about the mechanism, it was wrong about the scale. There are
    queries no tree finds, and at thirty two trees the forest has not yet run out of the ones it
    can find.
    """
    rows = {row["trees"]: row for row in voting_fixes_most_of_it()}
    early = rows[2]["recall"] - rows[1]["recall"]
    late = rows[32]["recall"] - rows[16]["recall"]
    return {
        "recall_at_one": rows[1]["recall"],
        "recall_at_eight": rows[8]["recall"],
        "recall_at_thirty_two": rows[32]["recall"],
        "gain_from_one_to_two": round(early, 4),
        "gain_from_sixteen_to_thirty_two": round(late, 4),
        "still_climbing": late > early,
        "short_of_one": rows[32]["recall"] < 0.99,
    }


def the_lift_over_a_random_subset_decays(corpus_size: int = 3996) -> list[dict]:
    """What does decay as trees are added, which is the thing worth reporting.

    A forest scanning n candidates should be compared against scanning n candidates chosen at
    random, since that is what the structure has to beat to have earned anything. At one tree it
    beats chance by 4.67 times, at two by 4.82, and then it falls away: 4.55, 4.10, 3.36, 2.52.

    So the trees are not adding information, they are adding coverage, and coverage is what a
    linear scan has for free. A structure whose advantage over random sampling declines as it is
    scaled is a structure with a ceiling, and this measurement finds the ceiling without having
    to run out to it.
    """
    if corpus_size < 1:
        raise ConfigError(f"{corpus_size} is not a corpus")
    rows = []
    for row in voting_fixes_most_of_it():
        chance = row["distances_per_query"] / corpus_size
        rows.append(
            {
                "trees": row["trees"],
                "recall": row["recall"],
                "share_scanned": round(chance, 4),
                "lift": round(row["recall"] / chance, 2) if chance > 0 else None,
            }
        )
    return rows


def the_lift_falls_as_the_forest_grows() -> dict:
    """The two ends of that, which is the module's real conclusion about scaling."""
    rows = {row["trees"]: row for row in the_lift_over_a_random_subset_decays()}
    return {
        "lift_at_two": rows[2]["lift"],
        "lift_at_thirty_two": rows[32]["lift"],
        "share_scanned_at_thirty_two": rows[32]["share_scanned"],
        "recall_at_thirty_two": rows[32]["recall"],
        "falls": rows[32]["lift"] < rows[2]["lift"],
        "always_beats_chance": all(row["lift"] > 1.0 for row in rows.values()),
    }


def the_leaves_overlap(counts: Sequence[int] = (1, 2, 4, 8, 16, 32)) -> list[dict]:
    """How much of what each new tree contributes was already there.

    Less than expected. Two trees over the same corpus were supposed to put many of the same
    vectors in a leaf together, because the corpus has real structure and both trees are cutting
    it. The measured overlap is 3.5 percent at two trees, rising to 29.5 percent at thirty two.

    So the trees really are close to independent, at least while the forest is small, and the
    cost is close to linear in the number of them. The sublinearity is real and it is a
    correction of a fifth rather than the order of magnitude the docstring first implied.
    """
    if not counts:
        raise ConfigError("there is nothing to sweep")
    corpus = gaussian(count=4096, dimension=32)
    searched, probes = held_out(corpus, count=100)
    index = ForestIndex(32, trees=max(counts), leaf_size=64)
    index.build(searched.vectors)
    rows = []
    for count in counts:
        sizes = [
            int(index.candidates(probes[row : row + 1], trees=count).numel())
            for row in range(int(probes.shape[0]))
        ]
        mean = sum(sizes) / len(sizes)
        rows.append(
            {
                "trees": count,
                "candidates": round(mean, 1),
                "if_disjoint": count * 64,
                "overlap_share": round(1.0 - mean / (count * 64), 4),
            }
        )
    return rows


def the_cost_grows_sublinearly() -> dict:
    """The consequence of that overlap, which is the forest's efficiency argument."""
    rows = {row["trees"]: row for row in the_leaves_overlap()}
    growth = rows[32]["candidates"] / rows[1]["candidates"]
    return {
        "candidates_at_one": rows[1]["candidates"],
        "candidates_at_thirty_two": rows[32]["candidates"],
        "growth": round(growth, 2),
        "trees_grew_by": 32,
        "sublinear": growth < 32,
        "overlap_at_thirty_two": rows[32]["overlap_share"],
    }


def the_overlap_and_the_decaying_lift_are_one_thing() -> dict:
    """That the two findings above are the same finding.

    A new tree adds candidates the others did not have, which is why the recall keeps climbing,
    and those candidates are increasingly ones the query has no particular reason to be near,
    which is why the lift over chance decays. The overlap rises from 3.5 percent to 29.5 as the
    lift falls from 4.8 to 2.5, and both are measuring how much of the corpus the forest has
    already looked at.

    Which means the decay is not a defect to engineer around. A forest whose trees never
    overlapped would keep its lift and would cost strictly linearly, and in the limit that
    structure has a name: it is a linear scan.
    """
    lift = {row["trees"]: row for row in the_lift_over_a_random_subset_decays()}
    overlap = {row["trees"]: row for row in the_leaves_overlap()}
    return {
        "overlap_at_two": overlap[2]["overlap_share"],
        "overlap_at_thirty_two": overlap[32]["overlap_share"],
        "lift_at_two": lift[2]["lift"],
        "lift_at_thirty_two": lift[32]["lift"],
        "overlap_rises": overlap[32]["overlap_share"] > overlap[2]["overlap_share"],
        "lift_falls": lift[32]["lift"] < lift[2]["lift"],
        "they_move_opposite_ways": overlap[32]["overlap_share"] > overlap[2]["overlap_share"]
        and lift[32]["lift"] < lift[2]["lift"],
    }


def a_bigger_leaf_buys_recall_directly(
    sizes: Sequence[int] = (16, 64, 256, 1024),
) -> list[dict]:
    """The other knob, which is the leaf size.

    A bigger leaf scans more per tree and is more likely to hold the true neighbour, so it buys
    recall in the most direct way available: by doing more work. The interesting question is
    which knob is more efficient, more trees or bigger leaves, and it is answered below rather
    than assumed.
    """
    if not sizes:
        raise ConfigError("there is nothing to sweep")
    corpus = gaussian(count=4096, dimension=32)
    searched, probes = held_out(corpus, count=100)
    truth = search(probes, searched.vectors, k=10)
    rows = []
    for size in sizes:
        index = ForestIndex(32, trees=8, leaf_size=size)
        index.build(searched.vectors)
        found, stats = index.search(probes, k=10)
        rows.append(
            {
                "leaf_size": size,
                "recall": round(identifier_overlap(truth, found), 4),
                "distances_per_query": round(stats.distances_per_query, 1),
                "mean_leaf": round(index.forest[0].mean_leaf, 1),
                "depth": index.forest[0].depth,
            }
        )
    return rows


def trees_and_leaves_are_not_interchangeable() -> dict:
    """Which of the two knobs buys more recall per distance computed.

    Trees, and not by a small margin. A tree adds a leaf that overlaps the others, so the extra
    candidates are concentrated where the query already is; a bigger leaf adds vectors that were
    simply next in the split order, which are further away on average. Both cost distances and
    one of them spends them better.

    So the tuning advice is to add trees until the memory for the directions matters, and only
    then to grow the leaves.
    """
    by_trees = {row["trees"]: row for row in voting_fixes_most_of_it()}
    by_leaves = {row["leaf_size"]: row for row in a_bigger_leaf_buys_recall_directly()}
    tree_efficiency = by_trees[8]["recall"] / by_trees[8]["distances_per_query"]
    leaf_efficiency = by_leaves[1024]["recall"] / by_leaves[1024]["distances_per_query"]
    return {
        "eight_trees_recall": by_trees[8]["recall"],
        "eight_trees_cost": by_trees[8]["distances_per_query"],
        "big_leaf_recall": by_leaves[1024]["recall"],
        "big_leaf_cost": by_leaves[1024]["distances_per_query"],
        "recall_per_distance_from_trees": round(tree_efficiency, 6),
        "recall_per_distance_from_leaves": round(leaf_efficiency, 6),
        "trees_are_better": tree_efficiency > leaf_efficiency,
    }


def a_median_split_bounds_the_depth() -> dict:
    """Why the threshold is the median rather than the midpoint of the range.

    Because the median splits the rows in half by construction, so the depth is the logarithm of
    the corpus size. On a clustered corpus of 4096 the median rule gives depth 6 and 64 leaves
    of mean size 62.4; the midpoint rule gives depth 12 and 97 leaves of mean size 41.2.

    The largest leaf is the same for both, at 63 and 64, because the recursion stops at the leaf
    size either way and that caps it. What the midpoint rule costs is depth and evenness: twice
    as many comparisons per descent, and leaves a third smaller on average, so a query scans
    less and finds less. Both push the same way and neither shows up in the largest leaf,
    which is what this docstring first claimed to measure.
    """
    corpus = clustered(count=4096, dimension=32, clusters=8)
    searched, _ = held_out(corpus, count=100)
    rows = {}
    for rule in ("median", "midpoint"):
        tree = grow(searched.vectors, leaf_size=64, split_on=rule)
        rows[rule] = {
            "depth": tree.depth,
            "leaves": tree.leaves,
            "mean_leaf": round(tree.mean_leaf, 1),
            "largest_leaf": tree.largest_leaf,
        }
    return {
        "median_depth": rows["median"]["depth"],
        "midpoint_depth": rows["midpoint"]["depth"],
        "median_largest_leaf": rows["median"]["largest_leaf"],
        "midpoint_largest_leaf": rows["midpoint"]["largest_leaf"],
        "median_is_shallower": rows["median"]["depth"] <= rows["midpoint"]["depth"],
        "median_is_more_even": rows["median"]["largest_leaf"]
        <= rows["midpoint"]["largest_leaf"],
    }


def a_random_direction_beats_an_axis(dimension: int = 64) -> dict:
    """The comparison with the kd tree in index/tree.py, which is the reason this exists.

    Not measured, as it turns out, because the two structures are not doing the same job. The
    kd tree in index/tree.py prunes exactly and therefore returns exact answers, and at sixty
    four dimensions it does that by scanning all 3996 vectors: recall 1.0 at full cost, which is
    a linear scan reached by a more expensive route.

    The forest at one tree gets 0.048 for 62.4 distances. Those are not two settings of one
    trade off, they are an exact method and an approximate one, and the honest statement is that
    the kd tree stops offering a trade off at all above about twenty dimensions while the forest
    still has one. That is a real difference and it is not the split rule, it is the pruning.
    """
    corpus = gaussian(count=4096, dimension=dimension)
    searched, probes = held_out(corpus, count=100)
    truth = search(probes, searched.vectors, k=10)

    forest = ForestIndex(dimension, trees=1, leaf_size=64)
    forest.build(searched.vectors)
    forest_found, forest_stats = forest.search(probes, k=10)

    kd = TreeIndex(dimension, leaf_size=64)
    kd.build(searched.vectors)
    kd_found, kd_stats = kd.search(probes, k=10)

    return {
        "dimension": dimension,
        "projection_recall": round(identifier_overlap(truth, forest_found), 4),
        "projection_distances": round(forest_stats.distances_per_query, 1),
        "kd_recall": round(identifier_overlap(truth, kd_found), 4),
        "kd_distances": round(kd_stats.distances_per_query, 1),
        "kd_scans_nearly_everything": kd_stats.distances_per_query
        > int(searched.vectors.shape[0]) * 0.5,
    }


def the_forest_beats_the_inverted_file(
    trees: Sequence[int] = (1, 2, 4, 8, 16, 32),
    probes: Sequence[int] = (1, 2, 4, 8, 16, 32),
) -> list[dict]:
    """Where a forest sits against the structure it most resembles.

    Ahead of it, which is not what this module expected to find. Both work by scanning a subset
    chosen without guarantees; the inverted file chooses by a fitted k-means and the forest by
    random cuts, and the fitted one was supposed to win because it looked at the data first.

    Interpolating both curves to matched distance counts, the forest is ahead at 125, at 845 and
    at 1444, and level at 460. It pulls further ahead the more work is allowed, which is the
    opposite of the shape a structure with a ceiling should have.

    The mechanism is boundaries. A query near a Voronoi boundary needs several probes before it
    reaches the partition holding its neighbours, and every probe costs a whole partition. A
    forest's trees cut in independent directions, so a query near one tree's boundary is in
    another tree's interior, and the coverage of boundary regions comes for free rather than
    being bought a partition at a time.
    """
    if not trees or not probes:
        raise ConfigError("there is nothing to sweep")
    corpus = gaussian(count=4096, dimension=32)
    searched, queries = held_out(corpus, count=100)
    truth = search(queries, searched.vectors, k=10)

    forest = ForestIndex(32, trees=max(trees), leaf_size=64)
    forest.build(searched.vectors)
    partitioned = IVFIndex(32, partitions=64, probe=1)
    partitioned.build(searched.vectors)

    rows = []
    for count in trees:
        found, stats = forest.search(queries, k=10, trees=count)
        rows.append(
            {
                "index": "forest",
                "setting": count,
                "recall": round(identifier_overlap(truth, found), 4),
                "distances": round(stats.distances_per_query, 1),
            }
        )
    for probe in probes:
        partitioned.probe = probe
        found, stats = partitioned.search(queries, k=10)
        rows.append(
            {
                "index": "ivf",
                "setting": probe,
                "recall": round(identifier_overlap(truth, found), 4),
                "distances": round(stats.distances_per_query, 1),
            }
        )
    return rows


def _interpolate(rows: list[dict], at: float) -> float:
    """The recall a curve reaches at a given distance count, linearly between its points."""
    points = sorted((row["distances"], row["recall"]) for row in rows)
    if at <= points[0][0]:
        return points[0][1]
    if at >= points[-1][0]:
        return points[-1][1]
    for index in range(1, len(points)):
        left, right = points[index - 1], points[index]
        if left[0] <= at <= right[0]:
            span = right[0] - left[0]
            if span == 0:
                return right[1]
            return left[1] + (right[1] - left[1]) * (at - left[0]) / span
    return points[-1][1]


def the_forest_is_ahead_at_matched_cost(
    budgets: Sequence[float] = (125.0, 460.0, 845.0, 1444.0),
) -> list[dict]:
    """Both curves read off at the same distance counts, which is the only fair comparison.

    Comparing at matched settings would compare a probe count against a tree count, and those
    are not the same quantity, which is the point autotune.py makes at length. Comparing at
    matched cost asks the only question a deployment has: for this much work, which structure
    returns more of the right answers.
    """
    if not budgets:
        raise ConfigError("there is nothing to sweep")
    rows = the_forest_beats_the_inverted_file()
    forest = [row for row in rows if row["index"] == "forest"]
    partitioned = [row for row in rows if row["index"] == "ivf"]
    return [
        {
            "distances": budget,
            "forest_recall": round(_interpolate(forest, budget), 4),
            "ivf_recall": round(_interpolate(partitioned, budget), 4),
            "forest_ahead": _interpolate(forest, budget) > _interpolate(partitioned, budget),
        }
        for budget in budgets
    ]


def the_gap_widens_with_the_budget() -> dict:
    """How that comparison changes as more work is allowed, which is the useful part."""
    rows = {row["distances"]: row for row in the_forest_is_ahead_at_matched_cost()}
    small = rows[125.0]["forest_recall"] - rows[125.0]["ivf_recall"]
    large = rows[1444.0]["forest_recall"] - rows[1444.0]["ivf_recall"]
    return {
        "gap_at_a_hundred_and_twenty_five": round(small, 4),
        "gap_at_fourteen_hundred": round(large, 4),
        "forest_ahead_at_both": small > 0 and large > 0,
        "widens": large > small,
    }


def a_clustered_corpus_suits_the_forest_better() -> dict:
    """Whether the corpus shape changes the answer, which it does in the forest's favour.

    A clustered corpus has real gaps, and a random cut through a gap separates two things that
    were genuinely separate, so a leaf on a clustered corpus is much more likely to hold a
    query's true neighbours than a leaf on a gaussian one. The forest's recall at the same
    settings is far higher, and the reason is that random cuts are a good way to find structure
    that is actually there.
    """
    rows = {}
    for label, corpus in (
        ("gaussian", gaussian(count=4096, dimension=32)),
        ("clustered", clustered(count=4096, dimension=32, clusters=16)),
    ):
        searched, probes = held_out(corpus, count=100)
        truth = search(probes, searched.vectors, k=10)
        index = ForestIndex(32, trees=8, leaf_size=64)
        index.build(searched.vectors)
        found, stats = index.search(probes, k=10)
        rows[label] = {
            "recall": round(identifier_overlap(truth, found), 4),
            "distances": round(stats.distances_per_query, 1),
        }
    return {
        "gaussian_recall": rows["gaussian"]["recall"],
        "clustered_recall": rows["clustered"]["recall"],
        "gaussian_distances": rows["gaussian"]["distances"],
        "clustered_distances": rows["clustered"]["distances"],
        "structure_helps": rows["clustered"]["recall"] > rows["gaussian"]["recall"],
    }


def normalising_does_not_change_a_projection_tree() -> dict:
    """Whether the corpus needs normalising, which for this structure it does not.

    A random projection tree cuts on a signed projection, which is a linear function of the
    vector, so scaling a vector moves it along the projection and can move it across a
    threshold. Normalising therefore does change the tree. What it does not change is the
    ordering the leaves are scanned by, since the scan is an exact L2 comparison over whatever
    the leaf holds.

    The measured difference is small and it is not zero, which is the honest answer to a
    question whose obvious answer is that it should be zero.
    """
    corpus = gaussian(count=4096, dimension=32)
    rows = {}
    for label, vectors in (
        ("raw", corpus.vectors),
        ("normalised", normalise(corpus.vectors)),
    ):
        searched, probes = vectors[:3996], vectors[3996:]
        truth = search(probes, searched, k=10)
        index = ForestIndex(32, trees=8, leaf_size=64)
        index.build(searched)
        found, _ = index.search(probes, k=10)
        rows[label] = round(identifier_overlap(truth, found), 4)
    return {
        "raw": rows["raw"],
        "normalised": rows["normalised"],
        "difference": round(abs(rows["raw"] - rows["normalised"]), 4),
        "small": abs(rows["raw"] - rows["normalised"]) < 0.15,
    }


def a_forest_of_no_trees_is_refused() -> bool:
    """Whether an empty forest is caught at construction."""
    try:
        ForestIndex(8, trees=0)
    except ConfigError:
        return True
    return False


def a_leaf_of_nothing_is_refused() -> bool:
    """Whether a leaf size of zero is caught."""
    try:
        ForestIndex(8, leaf_size=0)
    except ConfigError:
        return True
    return False


def a_corpus_that_fits_one_leaf_is_refused() -> bool:
    """Whether building a tree that would never split is caught.

    It has to be. A forest over a corpus smaller than its leaf size is a linear scan wearing a
    tree's interface, and it would score perfect recall at full cost while reporting itself as
    an approximate index, which is the most misleading thing a benchmark in this package could
    produce.
    """
    try:
        index = ForestIndex(8, trees=4, leaf_size=1024)
        index.build(torch.randn(512, 8))
    except BuildError:
        return True
    return False


def an_unknown_split_rule_is_refused() -> bool:
    """Whether a split rule nobody implemented is caught at growth time."""
    try:
        grow(torch.randn(256, 8), split_on="mean")
    except ConfigError:
        return True
    return False


def a_batch_descent_is_refused() -> bool:
    """Whether descending with more than one query at a time is caught.

    The descent is inherently one query at a time, since different queries take different
    branches, and accepting a batch would either silently use the first row or return a shape
    nobody expects.
    """
    tree = grow(torch.randn(512, 8), leaf_size=32)
    try:
        tree.descend(torch.randn(4, 8))
    except DataError:
        return True
    return False


def asking_for_more_trees_than_exist_is_refused() -> bool:
    """Whether searching a forest with more trees than it has is caught."""
    corpus = gaussian(count=512, dimension=8)
    index = ForestIndex(8, trees=4, leaf_size=32)
    index.build(corpus.vectors)
    try:
        index.search(corpus.vectors[:2], k=5, trees=100)
    except ConfigError:
        return True
    return False


def every_vector_reaches_exactly_one_leaf() -> dict:
    """A correctness check on the growth, which is easy to get wrong and hard to notice.

    Every row lands in exactly one leaf and every row lands in some leaf. A split that dropped
    the boundary rows would lose a few vectors per node, which on a corpus of four thousand and
    a leaf of sixty four is a few dozen vectors that can never be found, and the recall would
    look slightly low rather than wrong.
    """
    corpus = gaussian(count=2048, dimension=16)
    tree = grow(corpus.vectors, leaf_size=32)
    seen = torch.cat([node.rows for node in tree.nodes if isinstance(node, Leaf)])
    return {
        "corpus": 2048,
        "rows_in_leaves": int(seen.numel()),
        "distinct_rows": int(torch.unique(seen).numel()),
        "every_row_once": int(seen.numel()) == 2048 and int(torch.unique(seen).numel()) == 2048,
    }


def a_descent_lands_where_the_growth_put_it() -> dict:
    """That searching a tree for a corpus vector finds the leaf it was built into.

    The other half of the same check. Growth assigns rows to leaves by comparing projections to
    thresholds; descent finds a leaf by comparing projections to thresholds. If the two used
    different comparisons, one with a strict inequality and one without, the vectors exactly on
    a threshold would be filed one way and looked up the other.
    """
    corpus = gaussian(count=2048, dimension=16)
    tree = grow(corpus.vectors, leaf_size=32)
    misses = 0
    for row in range(0, 2048, 16):
        reached = tree.descend(corpus.vectors[row : row + 1])
        if row not in reached.tolist():
            misses += 1
    return {
        "checked": 128,
        "misses": misses,
        "consistent": misses == 0,
    }


def removal_and_insertion_work() -> dict:
    """That the write path keeps the forest searchable.

    Insertion pushes a row into the leaf it descends to in every tree, which keeps the invariant
    that a descent finds what is there without regrowing anything. Removal marks the row dead
    and leaves it in place, which is the same tombstone the graph uses and has the same cost:
    the leaf still holds it and the scan still walks past it.
    """
    corpus = gaussian(count=2048, dimension=16)
    searched, probes = held_out(corpus, count=32)
    index = ForestIndex(16, trees=4, leaf_size=64)
    index.build(searched.vectors[:1000])
    before = index.size
    index.insert(searched.vectors[1000:1500])
    grown = index.size
    found, _ = index.search(probes, k=5)
    index.remove([0, 1, 2])
    return {
        "built": before,
        "after_insert": grown,
        "after_remove": index.size,
        "insert_worked": grown == before + 500,
        "remove_worked": index.size == grown - 3,
        "still_searchable": int(found.identifiers.shape[0]) == int(probes.shape[0]),
    }

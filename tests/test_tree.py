from __future__ import annotations

import pytest

from vse.errors import ConfigError, IndexStateError
from vse.index.tree import (
    TreeIndex,
    a_one_vector_tree_is_refused,
    a_removed_vector_never_comes_back,
    a_zero_leaf_size_is_refused,
    above_the_crossover_it_is_worse_than_useless,
    an_inner_product_tree_is_refused,
    compare_against_a_scan,
    intrinsic_dimension_does_not_help_a_tree,
    leaf_size_sweep,
    searching_before_building_is_refused,
    splitting_on_the_widest_axis_is_what_makes_that_work,
    structure_helps_a_tree_after_all,
    the_bound_is_taken_on_the_root_not_the_square,
    the_crossover_is_at_eight_dimensions,
    the_leaf_size_trades_distances_for_plane_checks,
    the_pruning_stops_working,
    the_tree_is_balanced,
    the_tree_is_exact,
    tree_on,
)
from vse.vectors.dataset import gaussian


class TestExactness:
    def test_the_tree_is_exact_at_every_dimension(self):
        assert all(row["recall"] == 1.0 for row in the_tree_is_exact())

    def test_with_no_gap_anywhere(self):
        assert all(row["gap"] == 0.0 for row in the_tree_is_exact())

    def test_the_bound_is_taken_on_the_root(self):
        assert the_bound_is_taken_on_the_root_not_the_square()["exact"]

    def test_an_inner_product_tree_is_refused(self):
        assert an_inner_product_tree_is_refused()

    def test_because_it_has_no_triangle_inequality(self):
        with pytest.raises(ConfigError, match="triangle inequality"):
            TreeIndex(8, metric="cosine")

    def test_an_empty_dimension_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            the_tree_is_exact(dimensions=())


class TestTheCrossover:
    def test_the_pruning_works_at_two_dimensions(self):
        assert the_crossover_is_at_eight_dimensions()["at_two"] < 0.05

    def test_and_has_stopped_by_sixteen(self):
        assert the_crossover_is_at_eight_dimensions()["at_sixteen"] == 1.0

    def test_the_crossover_is_between_eight_and_sixteen(self):
        result = the_crossover_is_at_eight_dimensions()
        assert result["beats_a_scan_up_to"] == 8
        assert result["loses_from"] == 16

    def test_the_scanned_share_rises_with_the_dimension(self):
        rows = [row["scanned"] for row in the_pruning_stops_working()]
        assert rows == sorted(rows)

    def test_and_the_speedup_falls(self):
        rows = [row["speedup"] for row in the_pruning_stops_working()]
        assert rows == sorted(rows, reverse=True)

    def test_above_the_crossover_it_scans_everything(self):
        assert above_the_crossover_it_is_worse_than_useless()["scans_everything"]

    def test_and_pays_for_the_traversal_on_top(self):
        assert above_the_crossover_it_is_worse_than_useless()["and_pays_for_the_traversal"]

    def test_by_sixteen_times_the_hops(self):
        result = above_the_crossover_it_is_worse_than_useless()
        assert result["hops_at_sixty_four"] > result["hops_at_two"] * 10

    def test_an_empty_pruning_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            the_pruning_stops_working(dimensions=())


class TestRotatedStructure:
    def test_intrinsic_dimension_does_not_help_a_tree(self):
        assert not intrinsic_dimension_does_not_help_a_tree()["behaves_like_the_narrow_one"]

    def test_it_behaves_like_the_wide_corpus_instead(self):
        assert intrinsic_dimension_does_not_help_a_tree()["behaves_like_the_wide_one"]

    def test_which_is_the_reverse_of_every_other_structure(self):
        # The graph and the inverted file both handled this corpus as eight dimensional.
        result = intrinsic_dimension_does_not_help_a_tree()
        assert result["eight_within_five_hundred"] > result["eight_dimensional"]

    def test_the_splits_do_pick_the_wider_axes(self):
        assert splitting_on_the_widest_axis_is_what_makes_that_work()["used_are_wider"]

    def test_but_only_a_fraction_of_them_are_used(self):
        result = splitting_on_the_widest_axis_is_what_makes_that_work()
        assert result["axes_used"] < result["of"]


class TestStructure:
    def test_structure_helps_a_tree_after_all(self):
        assert structure_helps_a_tree_after_all()["structure_helps_a_lot"]

    def test_by_a_factor_of_seven(self):
        assert structure_helps_a_tree_after_all()["ratio"] > 4.0

    def test_the_unstructured_corpus_scans_everything(self):
        assert structure_helps_a_tree_after_all()["gaussian_scanned"] == 1.0

    def test_where_the_clustered_one_scans_a_seventh(self):
        assert structure_helps_a_tree_after_all()["clustered_scanned"] < 0.25


class TestLeafSize:
    def test_the_scanned_share_rises_with_the_leaf_size(self):
        assert the_leaf_size_trades_distances_for_plane_checks()["rises_with_the_leaf_size"]

    def test_and_the_depth_falls(self):
        assert the_leaf_size_trades_distances_for_plane_checks()["depth_falls_with_it"]

    def test_which_the_distance_count_cannot_see(self):
        # It credits the deep tree for work it moved into the traversal rather than removed.
        result = the_leaf_size_trades_distances_for_plane_checks()
        assert result["depth_at_one"] > result["depth_at_two_hundred_fifty_six"] * 2

    def test_the_leaf_count_falls_with_the_leaf_size(self):
        rows = [row["leaves"] for row in leaf_size_sweep()]
        assert rows == sorted(rows, reverse=True)

    def test_an_empty_leaf_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            leaf_size_sweep(sizes=())

    def test_a_zero_leaf_size_is_refused(self):
        assert a_zero_leaf_size_is_refused()


class TestStructureOfTheTree:
    def test_the_tree_is_balanced(self):
        assert the_tree_is_balanced()["balanced"]

    def test_the_depth_matches_the_prediction(self):
        result = the_tree_is_balanced()
        assert abs(result["depth"] - result["predicted"]) <= 2

    def test_every_vector_is_in_a_leaf(self):
        assert the_tree_is_balanced()["vectors"] == 2048

    def test_the_leaves_are_a_power_of_two(self):
        result = the_tree_is_balanced()
        assert result["leaves"] & (result["leaves"] - 1) == 0

    def test_no_node_is_both_a_leaf_and_a_split(self):
        corpus = gaussian(count=512, dimension=8)
        index = TreeIndex(8, leaf_size=16)
        index.build(corpus.vectors)
        stack = [index.root]
        while stack:
            node = stack.pop()
            if node.is_leaf:
                assert node.left is None and node.right is None
            else:
                stack.extend([node.left, node.right])

    def test_reading_the_root_before_building_is_refused(self):
        with pytest.raises(IndexStateError, match="has not been built"):
            _ = TreeIndex(8).root


class TestMechanics:
    def test_searching_before_building_is_refused(self):
        assert searching_before_building_is_refused()

    def test_a_one_vector_tree_is_refused(self):
        assert a_one_vector_tree_is_refused()

    def test_a_removed_vector_is_never_returned(self):
        assert not a_removed_vector_never_comes_back()["still_returned"]

    def test_and_the_tree_shape_does_not_change(self):
        result = a_removed_vector_never_comes_back()
        assert result["capacity"] == result["live"] + 1

    def test_removing_a_row_that_does_not_exist_is_refused(self):
        index = TreeIndex(8, leaf_size=8)
        index.build(gaussian(count=128, dimension=8).vectors)
        with pytest.raises(ConfigError, match="not one of the 128"):
            index.remove([999])

    def test_inserting_rebuilds_the_tree(self):
        corpus = gaussian(count=512, dimension=8)
        index = TreeIndex(8, leaf_size=8)
        index.build(corpus.vectors[:256])
        before = index.root.leaves()
        index.insert(corpus.vectors[256:])
        assert index.size == 512
        assert index.root.leaves() > before

    def test_inserting_into_an_unbuilt_index_builds_it(self):
        index = TreeIndex(8, leaf_size=8)
        index.insert(gaussian(count=128, dimension=8).vectors)
        assert index.built and index.size == 128

    def test_three_dimensions_are_compared_against_a_scan(self):
        assert len(compare_against_a_scan()) == 3

    def test_the_tree_wins_at_four_dimensions(self):
        rows = {row["dimension"]: row for row in compare_against_a_scan()}
        assert rows[4]["tree_wins"]

    def test_and_loses_at_sixty_four(self):
        rows = {row["dimension"]: row for row in compare_against_a_scan()}
        assert not rows[64]["tree_wins"]

    def test_while_staying_exact_at_both(self):
        assert all(row["tree_recall"] == 1.0 for row in compare_against_a_scan())

    def test_an_empty_comparison_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            compare_against_a_scan(dimensions=())

    def test_the_index_reports_its_name(self):
        assert TreeIndex(8).name == "tree"

    def test_a_small_tree_still_answers(self):
        assert tree_on(gaussian(count=256, dimension=4), leaf_size=8).recall == 1.0

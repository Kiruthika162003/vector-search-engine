from __future__ import annotations

import pytest
import torch

from vse.errors import BuildError, ConfigError, DataError, IndexStateError
from vse.index.forest import (
    ForestIndex,
    Leaf,
    ProjectionTree,
    a_batch_descent_is_refused,
    a_bigger_leaf_buys_recall_directly,
    a_clustered_corpus_suits_the_forest_better,
    a_corpus_that_fits_one_leaf_is_refused,
    a_descent_lands_where_the_growth_put_it,
    a_forest_of_no_trees_is_refused,
    a_leaf_of_nothing_is_refused,
    a_median_split_bounds_the_depth,
    a_random_direction_beats_an_axis,
    an_unknown_split_rule_is_refused,
    asking_for_more_trees_than_exist_is_refused,
    every_vector_reaches_exactly_one_leaf,
    grow,
    normalising_does_not_change_a_projection_tree,
    one_tree_is_a_coin_flip,
    removal_and_insertion_work,
    the_cost_grows_sublinearly,
    the_forest_beats_the_inverted_file,
    the_forest_is_ahead_at_matched_cost,
    the_gap_widens_with_the_budget,
    the_leaves_overlap,
    the_lift_falls_as_the_forest_grows,
    the_lift_over_a_random_subset_decays,
    the_overlap_and_the_decaying_lift_are_one_thing,
    the_recall_does_not_flatten,
    trees_and_leaves_are_not_interchangeable,
    voting_fixes_most_of_it,
)
from vse.vectors.dataset import gaussian


class TestGrowth:
    def test_every_vector_reaches_exactly_one_leaf(self):
        assert every_vector_reaches_exactly_one_leaf()["every_row_once"]

    def test_the_leaf_count_matches_the_corpus(self):
        assert every_vector_reaches_exactly_one_leaf()["rows_in_leaves"] == 2048

    def test_with_no_duplicates(self):
        result = every_vector_reaches_exactly_one_leaf()
        assert result["distinct_rows"] == result["rows_in_leaves"]

    def test_a_descent_lands_where_the_growth_put_it(self):
        assert a_descent_lands_where_the_growth_put_it()["consistent"]

    def test_with_no_misses(self):
        assert a_descent_lands_where_the_growth_put_it()["misses"] == 0

    def test_a_median_split_halves_the_rows(self):
        tree = grow(torch.randn(1024, 16), leaf_size=64)
        assert tree.leaves == 16

    def test_and_bounds_the_depth(self):
        tree = grow(torch.randn(1024, 16), leaf_size=64)
        assert tree.depth == 4

    def test_a_median_split_is_shallower_than_a_midpoint_one(self):
        assert a_median_split_bounds_the_depth()["median_is_shallower"]

    def test_by_a_factor_of_two(self):
        result = a_median_split_bounds_the_depth()
        assert result["midpoint_depth"] >= result["median_depth"] * 2

    def test_the_largest_leaf_is_capped_either_way(self):
        result = a_median_split_bounds_the_depth()
        assert abs(result["median_largest_leaf"] - result["midpoint_largest_leaf"]) <= 2

    def test_an_unknown_split_rule_is_refused(self):
        assert an_unknown_split_rule_is_refused()

    def test_a_leaf_of_nothing_is_refused_at_growth(self):
        with pytest.raises(ConfigError, match="holds nothing"):
            grow(torch.randn(128, 8), leaf_size=0)

    def test_an_empty_corpus_is_refused(self):
        with pytest.raises(BuildError, match="at least one vector"):
            grow(torch.zeros(0, 8))

    def test_a_tree_serialises(self):
        tree = grow(torch.randn(512, 8), leaf_size=64)
        assert tree.as_dict()["leaves"] == tree.leaves

    def test_an_empty_tree_reports_no_leaf_size(self):
        assert ProjectionTree(nodes=[], depth=0).mean_leaf == 0.0

    def test_and_no_largest_leaf(self):
        assert ProjectionTree(nodes=[], depth=0).largest_leaf == 0

    def test_a_batch_descent_is_refused(self):
        assert a_batch_descent_is_refused()

    def test_a_rank_one_descent_is_refused(self):
        tree = grow(torch.randn(256, 8), leaf_size=32)
        with pytest.raises(DataError, match="one query"):
            tree.descend(torch.randn(8))


class TestVoting:
    def test_one_tree_barely_works(self):
        assert one_tree_is_a_coin_flip()["recall"] < 0.2

    def test_the_recall_rises_with_the_trees(self):
        rows = [row["recall"] for row in voting_fixes_most_of_it()]
        assert rows == sorted(rows)

    def test_and_does_not_flatten(self):
        assert the_recall_does_not_flatten()["still_climbing"]

    def test_but_has_not_reached_one(self):
        assert the_recall_does_not_flatten()["short_of_one"]

    def test_thirty_two_trees_reach_nine_tenths(self):
        assert the_recall_does_not_flatten()["recall_at_thirty_two"] > 0.85

    def test_the_cost_rises_too(self):
        rows = [row["distances_per_query"] for row in voting_fixes_most_of_it()]
        assert rows == sorted(rows)

    def test_an_empty_tree_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            voting_fixes_most_of_it(counts=())


class TestLift:
    def test_the_forest_always_beats_chance(self):
        assert the_lift_falls_as_the_forest_grows()["always_beats_chance"]

    def test_but_the_lift_falls(self):
        assert the_lift_falls_as_the_forest_grows()["falls"]

    def test_from_nearly_five_to_two_and_a_half(self):
        result = the_lift_falls_as_the_forest_grows()
        assert result["lift_at_two"] > 4.0
        assert result["lift_at_thirty_two"] < 3.0

    def test_thirty_two_trees_scan_a_third_of_the_corpus(self):
        assert the_lift_falls_as_the_forest_grows()["share_scanned_at_thirty_two"] > 0.3

    def test_the_lift_is_reported_for_every_tree_count(self):
        assert len(the_lift_over_a_random_subset_decays()) == 6

    def test_a_corpus_of_nothing_is_refused(self):
        with pytest.raises(ConfigError, match="not a corpus"):
            the_lift_over_a_random_subset_decays(corpus_size=0)


class TestOverlap:
    def test_the_leaves_barely_overlap_at_two_trees(self):
        rows = {row["trees"]: row for row in the_leaves_overlap()}
        assert rows[2]["overlap_share"] < 0.1

    def test_and_substantially_at_thirty_two(self):
        rows = {row["trees"]: row for row in the_leaves_overlap()}
        assert rows[32]["overlap_share"] > 0.25

    def test_the_overlap_rises_monotonically(self):
        rows = [row["overlap_share"] for row in the_leaves_overlap()]
        assert rows == sorted(rows)

    def test_the_cost_grows_sublinearly(self):
        assert the_cost_grows_sublinearly()["sublinear"]

    def test_but_only_just(self):
        assert the_cost_grows_sublinearly()["growth"] > 20

    def test_the_overlap_and_the_lift_move_opposite_ways(self):
        assert the_overlap_and_the_decaying_lift_are_one_thing()["they_move_opposite_ways"]

    def test_an_empty_overlap_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            the_leaves_overlap(counts=())


class TestKnobs:
    def test_a_bigger_leaf_buys_recall(self):
        rows = [row["recall"] for row in a_bigger_leaf_buys_recall_directly()]
        assert rows == sorted(rows)

    def test_and_costs_more(self):
        rows = [row["distances_per_query"] for row in a_bigger_leaf_buys_recall_directly()]
        assert rows == sorted(rows)

    def test_a_bigger_leaf_makes_a_shallower_tree(self):
        rows = [row["depth"] for row in a_bigger_leaf_buys_recall_directly()]
        assert rows == sorted(rows, reverse=True)

    def test_trees_buy_more_recall_per_distance(self):
        assert trees_and_leaves_are_not_interchangeable()["trees_are_better"]

    def test_by_a_factor_of_three(self):
        result = trees_and_leaves_are_not_interchangeable()
        assert (
            result["recall_per_distance_from_trees"]
            > result["recall_per_distance_from_leaves"] * 3
        )

    def test_an_empty_leaf_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            a_bigger_leaf_buys_recall_directly(sizes=())


class TestAgainstOtherStructures:
    def test_the_forest_is_ahead_at_most_budgets(self):
        rows = the_forest_is_ahead_at_matched_cost()
        assert sum(1 for row in rows if row["forest_ahead"]) >= 3

    def test_and_the_gap_widens(self):
        assert the_gap_widens_with_the_budget()["widens"]

    def test_ahead_at_the_smallest_budget(self):
        assert the_gap_widens_with_the_budget()["gap_at_a_hundred_and_twenty_five"] > 0

    def test_and_at_the_largest(self):
        assert the_gap_widens_with_the_budget()["gap_at_fourteen_hundred"] > 0.05

    def test_both_curves_are_measured(self):
        rows = the_forest_beats_the_inverted_file()
        assert len([row for row in rows if row["index"] == "forest"]) == 6
        assert len([row for row in rows if row["index"] == "ivf"]) == 6

    def test_an_empty_comparison_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            the_forest_beats_the_inverted_file(trees=())

    def test_an_empty_budget_list_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            the_forest_is_ahead_at_matched_cost(budgets=())

    def test_the_kd_tree_scans_everything_at_sixty_four_dimensions(self):
        assert a_random_direction_beats_an_axis()["kd_scans_nearly_everything"]

    def test_and_is_exact_for_it(self):
        assert a_random_direction_beats_an_axis()["kd_recall"] == 1.0

    def test_where_the_forest_is_cheap_and_approximate(self):
        result = a_random_direction_beats_an_axis()
        assert result["projection_distances"] < result["kd_distances"] / 10

    def test_a_clustered_corpus_suits_the_forest(self):
        assert a_clustered_corpus_suits_the_forest_better()["structure_helps"]

    def test_by_a_lot(self):
        result = a_clustered_corpus_suits_the_forest_better()
        assert result["clustered_recall"] > result["gaussian_recall"] * 1.8

    def test_and_costs_less_too(self):
        result = a_clustered_corpus_suits_the_forest_better()
        assert result["clustered_distances"] < result["gaussian_distances"]

    def test_normalising_barely_changes_it(self):
        assert normalising_does_not_change_a_projection_tree()["small"]


class TestTheIndex:
    def test_it_returns_k_neighbours(self):
        corpus = gaussian(count=1024, dimension=16)
        index = ForestIndex(16, trees=4, leaf_size=32)
        index.build(corpus.vectors)
        found, _ = index.search(corpus.vectors[:8], k=7)
        assert tuple(found.identifiers.shape) == (8, 7)

    def test_it_finds_the_query_itself(self):
        corpus = gaussian(count=1024, dimension=16)
        index = ForestIndex(16, trees=8, leaf_size=64)
        index.build(corpus.vectors)
        found, _ = index.search(corpus.vectors[:1], k=1)
        assert int(found.identifiers[0, 0]) == 0

    def test_a_forest_of_no_trees_is_refused(self):
        assert a_forest_of_no_trees_is_refused()

    def test_a_leaf_of_nothing_is_refused(self):
        assert a_leaf_of_nothing_is_refused()

    def test_a_corpus_that_fits_one_leaf_is_refused(self):
        assert a_corpus_that_fits_one_leaf_is_refused()

    def test_asking_for_more_trees_than_exist_is_refused(self):
        assert asking_for_more_trees_than_exist_is_refused()

    def test_asking_for_no_trees_at_search_time_is_refused(self):
        corpus = gaussian(count=512, dimension=8)
        index = ForestIndex(8, trees=4, leaf_size=32)
        index.build(corpus.vectors)
        with pytest.raises(ConfigError, match="is not between one"):
            index.search(corpus.vectors[:2], k=5, trees=0)

    def test_searching_before_building_is_refused(self):
        with pytest.raises(IndexStateError):
            ForestIndex(8).search(torch.randn(1, 8), k=5)

    def test_the_candidates_are_the_union_of_the_leaves(self):
        corpus = gaussian(count=1024, dimension=16)
        index = ForestIndex(16, trees=4, leaf_size=32)
        index.build(corpus.vectors)
        one = index.candidates(corpus.vectors[:1], trees=1)
        four = index.candidates(corpus.vectors[:1], trees=4)
        assert int(four.numel()) >= int(one.numel())

    def test_and_hold_no_duplicates(self):
        corpus = gaussian(count=1024, dimension=16)
        index = ForestIndex(16, trees=8, leaf_size=32)
        index.build(corpus.vectors)
        rows = index.candidates(corpus.vectors[:1])
        assert int(torch.unique(rows).numel()) == int(rows.numel())

    def test_insertion_and_removal_work(self):
        result = removal_and_insertion_work()
        assert result["insert_worked"]
        assert result["remove_worked"]

    def test_and_leave_it_searchable(self):
        assert removal_and_insertion_work()["still_searchable"]

    def test_removing_a_row_that_is_not_there_is_refused(self):
        corpus = gaussian(count=512, dimension=8)
        index = ForestIndex(8, trees=2, leaf_size=32)
        index.build(corpus.vectors)
        with pytest.raises(ConfigError, match="is not one of"):
            index.remove([9999])

    def test_the_memory_counts_directions_and_rows(self):
        corpus = gaussian(count=1024, dimension=16)
        index = ForestIndex(16, trees=4, leaf_size=64)
        index.build(corpus.vectors)
        assert index.memory_bytes() > 1024 * 4 * 8

    def test_a_leaf_holds_its_rows(self):
        leaf = Leaf(rows=torch.arange(5))
        assert int(leaf.rows.numel()) == 5

from __future__ import annotations

import pytest
import torch

from vse.errors import BuildError, ConfigError, IndexStateError
from vse.index.hnsw import (
    HNSWIndex,
    a_beam_narrower_than_k_is_refused,
    a_corpus_smaller_than_the_degree_is_refused,
    a_multiplier_of_one_is_refused,
    a_wider_construction_beam_is_worth_it,
    and_buys_nothing_on_the_unstructured_one,
    beam_sweep,
    compare_indexes,
    construction_beam_sweep,
    hnsw_on,
    searching_before_building_is_refused,
    the_bottom_layer_holds_everything,
    the_build_order_changes_the_structure,
    the_descent_is_almost_free,
    the_hierarchy_fixes_the_clustered_corpus,
    the_layers_follow_the_geometric_law,
    the_multiplier_was_the_thing_set_wrong,
    the_stack_is_logarithmic,
)
from vse.vectors.dataset import gaussian


class TestTheRepair:
    def test_the_hierarchy_fixes_the_clustered_corpus(self):
        assert the_hierarchy_fixes_the_clustered_corpus()["fixed"]

    def test_by_a_factor_of_fifteen(self):
        result = the_hierarchy_fixes_the_clustered_corpus()
        assert result["layered_recall"] > result["flat_recall"] * 10

    def test_but_it_is_not_a_complete_repair(self):
        # The upper layers connect the groups; one greedy descent still lands in the wrong one.
        assert the_hierarchy_fixes_the_clustered_corpus()["layered_recall"] < 0.9

    def test_the_flat_graph_is_still_near_zero_there(self):
        assert the_hierarchy_fixes_the_clustered_corpus()["flat_recall"] < 0.1

    def test_it_buys_nothing_on_unstructured_data(self):
        assert and_buys_nothing_on_the_unstructured_one()["close"]

    def test_and_costs_more_to_get_there(self):
        result = and_buys_nothing_on_the_unstructured_one()
        assert result["layered_scanned"] > result["flat_scanned"]

    def test_so_it_is_a_repair_and_not_an_improvement(self):
        result = and_buys_nothing_on_the_unstructured_one()
        assert result["difference"] < 0.01

    def test_four_rows_in_the_comparison(self):
        assert len(compare_indexes()) == 4

    def test_the_layered_index_wins_on_the_clustered_corpus(self):
        rows = {(row["corpus"], row["index"]): row for row in compare_indexes()}
        assert rows[("clustered", "hnsw")]["recall"] > rows[("clustered", "graph")]["recall"]

    def test_and_the_flat_one_is_cheaper_on_the_other(self):
        rows = {(row["corpus"], row["index"]): row for row in compare_indexes()}
        assert rows[("gaussian", "graph")]["speedup"] > rows[("gaussian", "hnsw")]["speedup"]


class TestLayers:
    def test_each_layer_is_a_fraction_of_the_one_below(self):
        assert the_layers_follow_the_geometric_law()["close_to_the_multiplier"]

    def test_the_ratios_are_near_the_multiplier(self):
        result = the_layers_follow_the_geometric_law()
        assert all(8 < ratio < 24 for ratio in result["ratios"])

    def test_the_stack_is_shallow_at_the_default_multiplier(self):
        assert the_layers_follow_the_geometric_law()["levels"] < 8

    def test_the_level_count_grows_with_the_logarithm(self):
        rows = [row["levels"] for row in the_stack_is_logarithmic()]
        assert rows == sorted(rows)

    def test_in_the_multiplier_and_not_in_two(self):
        rows = the_stack_is_logarithmic()
        assert all(abs(row["levels"] - row["predicted"]) <= 2 for row in rows)

    def test_the_top_layer_is_tiny(self):
        assert all(row["top_layer"] <= 4 for row in the_stack_is_logarithmic())

    def test_an_empty_size_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            the_stack_is_logarithmic(sizes=())

    def test_the_bottom_layer_holds_every_vector(self):
        assert the_bottom_layer_holds_everything()["complete"]

    def test_a_multiplier_of_one_is_refused(self):
        assert a_multiplier_of_one_is_refused()


class TestTheMultiplier:
    def test_the_upper_layers_are_a_small_share_of_the_corpus(self):
        assert the_descent_is_almost_free()["upper_share_of_the_corpus"] < 0.15

    def test_the_bottom_layer_is_the_whole_corpus(self):
        result = the_descent_is_almost_free()
        assert result["bottom"] > result["above_the_bottom"] * 5

    def test_a_larger_multiplier_makes_a_shallower_stack(self):
        rows = [row["levels"] for row in the_multiplier_was_the_thing_set_wrong()]
        assert rows == sorted(rows, reverse=True)

    def test_and_a_cheaper_search(self):
        rows = [row["distances_per_query"] for row in the_multiplier_was_the_thing_set_wrong()]
        assert rows == sorted(rows, reverse=True)

    def test_at_identical_recall(self):
        rows = {row["recall"] for row in the_multiplier_was_the_thing_set_wrong()}
        assert len(rows) == 1

    def test_the_saving_from_two_to_sixty_four_is_a_quarter(self):
        rows = {row["multiplier"]: row for row in the_multiplier_was_the_thing_set_wrong()}
        ratio = rows[64.0]["distances_per_query"] / rows[2.0]["distances_per_query"]
        assert 0.6 < ratio < 0.9

    def test_a_multiplier_of_two_puts_half_the_corpus_one_layer_up(self):
        index = HNSWIndex(16, degree=8, multiplier=2.0)
        index.build(gaussian(count=512, dimension=16).vectors)
        sizes = index.layer_sizes()
        assert sizes[1] > sizes[0] * 0.4

    def test_where_the_default_puts_a_sixteenth(self):
        index = HNSWIndex(16, degree=8, multiplier=16.0)
        index.build(gaussian(count=512, dimension=16).vectors)
        sizes = index.layer_sizes()
        assert sizes[1] < sizes[0] * 0.15

    def test_an_empty_multiplier_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            the_multiplier_was_the_thing_set_wrong(multipliers=())


class TestSearch:
    def test_recall_rises_with_the_beam(self):
        rows = [row["recall"] for row in beam_sweep()]
        assert rows == sorted(rows)

    def test_and_the_speedup_falls(self):
        rows = [row["speedup"] for row in beam_sweep()]
        assert rows == sorted(rows, reverse=True)

    def test_a_wide_beam_reaches_nearly_everything(self):
        rows = {row["ef"]: row for row in beam_sweep()}
        assert rows[128]["recall"] > 0.97

    def test_an_empty_beam_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            beam_sweep(widths=())

    def test_a_beam_narrower_than_k_is_refused(self):
        assert a_beam_narrower_than_k_is_refused()

    def test_searching_before_building_is_refused(self):
        assert searching_before_building_is_refused()

    def test_the_results_come_back_closest_first(self):
        corpus = gaussian(count=512, dimension=16)
        index = HNSWIndex(16, degree=8, ef=32)
        index.build(corpus.vectors)
        found, _ = index.search(corpus.vectors[:8], k=10)
        assert bool((found.scores[:, 1:] >= found.scores[:, :-1] - 1e-5).all())

    def test_a_vector_finds_itself(self):
        corpus = gaussian(count=512, dimension=16)
        index = HNSWIndex(16, degree=16, ef=64)
        index.build(corpus.vectors)
        found, _ = index.search(corpus.vectors[:16], k=1)
        assert found.identifiers.flatten().tolist() == list(range(16))

    def test_a_small_corpus_still_works(self):
        assert hnsw_on(gaussian(count=512, dimension=16), degree=8, ef=32).recall > 0.5


class TestConstruction:
    def test_a_wider_construction_beam_improves_recall(self):
        assert a_wider_construction_beam_is_worth_it()["improved"]

    def test_without_changing_the_query_cost(self):
        assert a_wider_construction_beam_is_worth_it()["query_cost_unchanged"]

    def test_but_it_saturates(self):
        # Past thirty two the degree cap binds and the extra candidates are all discarded.
        rows = {row["ef_construction"]: row for row in construction_beam_sweep()}
        assert rows[64]["edges"] == rows[128]["edges"]

    def test_and_the_first_step_is_where_the_edges_come_from(self):
        rows = {row["ef_construction"]: row for row in construction_beam_sweep()}
        assert rows[32]["edges"] > rows[16]["edges"]

    def test_an_empty_construction_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            construction_beam_sweep(widths=())

    def test_the_build_order_changes_the_structure(self):
        assert not the_build_order_changes_the_structure()["identical"]

    def test_though_the_recall_is_unaffected(self):
        assert the_build_order_changes_the_structure()["recall"] > 0.9

    def test_the_level_counts_still_agree(self):
        result = the_build_order_changes_the_structure()
        assert result["forward_levels"] == result["shuffled_levels"]

    def test_a_corpus_smaller_than_the_degree_is_refused(self):
        assert a_corpus_smaller_than_the_degree_is_refused()

    def test_a_zero_degree_is_refused(self):
        with pytest.raises(ConfigError, match="not a degree"):
            HNSWIndex(8, degree=0)

    def test_a_zero_beam_is_refused(self):
        with pytest.raises(ConfigError, match="not beams"):
            HNSWIndex(8, ef=0)


class TestUpdates:
    def test_an_inserted_vector_can_be_found(self):
        corpus = gaussian(count=512, dimension=16)
        index = HNSWIndex(16, degree=16, ef=64)
        index.build(corpus.vectors[:256])
        fresh = corpus.vectors[256:264]
        identifiers = index.insert(fresh)
        found, _ = index.search(fresh, k=1)
        assert found.identifiers.flatten().tolist() == identifiers

    def test_inserting_into_an_unbuilt_index_builds_it(self):
        index = HNSWIndex(8, degree=4)
        index.insert(gaussian(count=128, dimension=8).vectors)
        assert index.built and index.size == 128

    def test_a_removed_vector_is_never_returned(self):
        corpus = gaussian(count=512, dimension=16)
        index = HNSWIndex(16, degree=8, ef=32)
        index.build(corpus.vectors)
        victim = int(index.search(corpus.vectors[:1], k=1)[0].identifiers[0, 0])
        index.remove([victim])
        assert victim not in index.search(corpus.vectors[:1], k=5)[0].row(0)

    def test_removing_a_row_that_does_not_exist_is_refused(self):
        index = HNSWIndex(8, degree=4)
        index.build(gaussian(count=128, dimension=8).vectors)
        with pytest.raises(ConfigError, match="not one of the 128"):
            index.remove([999])

    def test_removing_from_an_unbuilt_index_is_refused(self):
        with pytest.raises(IndexStateError, match="has not been built"):
            HNSWIndex(8).remove([0])

    def test_the_memory_covers_every_layer(self):
        corpus = gaussian(count=512, dimension=16)
        index = HNSWIndex(16, degree=8)
        index.build(corpus.vectors)
        assert index.memory_bytes() > 512 * 16 * 4

    def test_the_entry_point_is_at_the_top(self):
        corpus = gaussian(count=512, dimension=16)
        index = HNSWIndex(16, degree=8)
        index.build(corpus.vectors)
        assert index._level[index.entry_point] == max(index._level)

    def test_an_unbuilt_index_has_no_entry_point(self):
        with pytest.raises(IndexStateError, match="has not been built"):
            _ = HNSWIndex(8).entry_point

    def test_a_build_error_names_the_degree(self):
        with pytest.raises(BuildError, match="cannot fill a degree"):
            HNSWIndex(8, degree=64).build(torch.randn(32, 8))

    def test_the_index_reports_its_name(self):
        assert HNSWIndex(8).name == "hnsw"

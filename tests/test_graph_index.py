from __future__ import annotations

import pytest
import torch

from vse.build.neighbours import components
from vse.errors import BuildError, ConfigError, IndexStateError
from vse.index.graph import (
    GraphIndex,
    a_beam_narrower_than_k_is_refused,
    a_corpus_too_small_for_the_build_degree_is_refused,
    a_narrow_beam_returns_the_wrong_region,
    a_removed_vector_never_comes_back,
    a_wider_graph_needs_fewer_hops,
    an_unknown_entry_point_is_refused,
    beam_sweep,
    but_it_fails_completely_on_tight_clusters,
    compare_indexes,
    degree_sweep,
    fragmentation_by_group_size,
    graph_on,
    insertion_works_and_costs,
    searching_before_building_is_refused,
    the_entry_point_barely_matters,
    the_graph_beats_the_inverted_file,
    the_graph_stays_connected_after_pruning,
    the_threshold_is_the_build_degree,
)
from vse.vectors.dataset import clustered, gaussian


class TestBeam:
    def test_recall_rises_with_the_beam(self):
        rows = [row["recall"] for row in beam_sweep()]
        assert rows == sorted(rows)

    def test_and_the_speedup_falls(self):
        rows = [row["speedup"] for row in beam_sweep()]
        assert rows == sorted(rows, reverse=True)

    def test_and_the_gap_falls(self):
        rows = [row["gap"] for row in beam_sweep()]
        assert rows == sorted(rows, reverse=True)

    def test_a_wide_beam_reaches_nearly_perfect_recall(self):
        rows = {row["ef"]: row for row in beam_sweep()}
        assert rows[128]["recall"] > 0.97

    def test_while_still_touching_under_half_the_corpus(self):
        rows = {row["ef"]: row for row in beam_sweep()}
        assert rows[128]["scanned"] < 0.5

    def test_a_narrow_beam_loses_a_third_of_the_recall(self):
        assert a_narrow_beam_returns_the_wrong_region()["recall_ratio"] >= 1.5

    def test_and_a_hundred_times_the_gap(self):
        assert a_narrow_beam_returns_the_wrong_region()["gap_ratio"] > 50

    def test_so_its_misses_are_not_near_misses(self):
        result = a_narrow_beam_returns_the_wrong_region()
        assert result["narrow_gap"] > 1.0

    def test_an_empty_beam_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            beam_sweep(widths=())

    def test_a_beam_narrower_than_k_is_refused(self):
        assert a_beam_narrower_than_k_is_refused()

    def test_a_zero_beam_is_refused(self):
        with pytest.raises(ConfigError, match="not a beam"):
            GraphIndex(8, ef=0)


class TestAgainstTheInvertedFile:
    def test_the_graph_wins_on_unstructured_data(self):
        assert the_graph_beats_the_inverted_file()["graph_wins"]

    def test_at_higher_recall(self):
        result = the_graph_beats_the_inverted_file()
        assert result["graph_recall"] > result["ivf_recall"]

    def test_touching_much_less_of_the_corpus(self):
        result = the_graph_beats_the_inverted_file()
        assert result["graph_scanned"] < result["ivf_scanned"] * 0.6

    def test_but_it_fails_completely_on_tight_clusters(self):
        assert but_it_fails_completely_on_tight_clusters()["ivf_wins"]

    def test_by_a_factor_of_thirty(self):
        assert but_it_fails_completely_on_tight_clusters()["margin"] > 20

    def test_the_graph_recall_there_is_near_zero(self):
        assert but_it_fails_completely_on_tight_clusters()["graph_recall"] < 0.1

    def test_two_indexes_are_compared(self):
        assert len(compare_indexes()) == 2

    def test_the_graph_is_the_faster_of_the_two(self):
        rows = {row["index"]: row for row in compare_indexes()}
        assert rows["graph"]["speedup"] > rows["ivf"]["speedup"]


class TestFragmentation:
    def test_large_groups_fragment_the_graph(self):
        assert the_threshold_is_the_build_degree()["large_groups_fragment"]

    def test_and_small_ones_do_not(self):
        assert the_threshold_is_the_build_degree()["small_groups_do_not"]

    def test_the_unstructured_corpus_is_always_connected(self):
        assert the_threshold_is_the_build_degree()["gaussian"] == 1

    def test_the_component_count_tracks_the_group_count(self):
        # One component per group, exactly, while the groups exceed the build degree.
        rows = [row for row in fragmentation_by_group_size() if row["group_exceeds_degree"]]
        assert all(row["components"] == row["groups"] for row in rows)

    def test_and_collapses_once_the_group_fits_inside_the_degree(self):
        rows = {row["per_group"]: row for row in fragmentation_by_group_size()}
        assert rows[16]["components"] == 1

    def test_the_threshold_is_not_a_number_anybody_chose(self):
        rows = fragmentation_by_group_size()
        worst = max(rows, key=lambda row: row["components"])
        assert worst["per_group"] > the_threshold_is_the_build_degree()["build_degree"]

    def test_an_empty_group_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            fragmentation_by_group_size(counts=())

    def test_a_corpus_too_small_for_the_degree_is_refused(self):
        with pytest.raises(ConfigError, match="build degree"):
            fragmentation_by_group_size(build_degree=64, corpus_size=64)

    def test_the_unstructured_graph_stays_connected_after_pruning(self):
        assert the_graph_stays_connected_after_pruning()["connected"]

    def test_and_the_degree_is_capped(self):
        result = the_graph_stays_connected_after_pruning()
        assert result["max_degree"] <= 16

    def test_the_mean_degree_is_below_the_cap(self):
        # The pruning rule stops early when everything left is covered.
        result = the_graph_stays_connected_after_pruning()
        assert result["mean_degree"] < result["max_degree"]

    def test_a_disconnected_graph_is_not_detected_by_the_build(self):
        # Nothing raises. The index builds happily and answers badly.
        index = GraphIndex(32, degree=16)
        index.build(clustered(count=2048, dimension=32, clusters=32).vectors)
        assert components(index.graph, directed=False) > 1


class TestEntryPoint:
    def test_the_entry_point_barely_matters(self):
        assert the_entry_point_barely_matters()["within_a_point"]

    def test_all_three_strategies_land_together(self):
        assert the_entry_point_barely_matters()["spread"] < 0.02

    def test_an_unknown_entry_point_is_refused(self):
        assert an_unknown_entry_point_is_refused()

    def test_the_first_strategy_starts_at_zero(self):
        index = GraphIndex(8, degree=4, build_degree=8, entry="first")
        index.build(gaussian(count=256, dimension=8).vectors)
        assert index.entry_point == 0

    def test_the_medoid_is_near_the_mean(self):
        corpus = gaussian(count=512, dimension=8)
        index = GraphIndex(8, degree=4, build_degree=8, entry="medoid")
        index.build(corpus.vectors)
        centre = corpus.vectors.mean(dim=0)
        chosen = corpus.vectors[index.entry_point]
        furthest = float(corpus.vectors.pow(2).sum(dim=1).max())
        assert float((chosen - centre).pow(2).sum()) < furthest


class TestDegree:
    def test_recall_rises_with_the_degree(self):
        rows = [row["recall"] for row in degree_sweep()]
        assert rows == sorted(rows)

    def test_and_so_does_the_distance_count(self):
        rows = [row["distances_per_query"] for row in degree_sweep()]
        assert rows == sorted(rows)

    def test_but_more_slowly_than_the_degree(self):
        assert a_wider_graph_needs_fewer_hops()["grows_more_slowly"]

    def test_because_a_wider_graph_needs_fewer_hops(self):
        assert a_wider_graph_needs_fewer_hops()["hop_ratio"] < 1.0

    def test_an_empty_degree_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            degree_sweep(degrees=())

    def test_a_degree_above_the_build_degree_is_refused(self):
        with pytest.raises(ConfigError, match="out of a build degree"):
            GraphIndex(8, degree=32, build_degree=16)


class TestUpdates:
    def test_every_inserted_vector_can_be_found_again(self):
        assert insertion_works_and_costs()["all_found"]

    def test_inserting_into_an_unbuilt_index_builds_it(self):
        index = GraphIndex(8, degree=4, build_degree=8)
        index.insert(gaussian(count=128, dimension=8).vectors)
        assert index.built and index.size == 128

    def test_a_removed_vector_is_never_returned(self):
        assert not a_removed_vector_never_comes_back()["still_returned"]

    def test_but_it_still_costs_distances(self):
        # Its edges stay so the walk still passes through it, unlike the inverted file.
        assert a_removed_vector_never_comes_back()["still_costs"]

    def test_removing_a_row_that_does_not_exist_is_refused(self):
        index = GraphIndex(8, degree=4, build_degree=8)
        index.build(gaussian(count=128, dimension=8).vectors)
        with pytest.raises(ConfigError, match="not one of the 128"):
            index.remove([999])

    def test_removing_from_an_unbuilt_index_is_refused(self):
        with pytest.raises(IndexStateError, match="has not been built"):
            GraphIndex(8).remove([0])


class TestMechanics:
    def test_searching_before_building_is_refused(self):
        assert searching_before_building_is_refused()

    def test_reading_the_graph_before_building_is_refused(self):
        with pytest.raises(IndexStateError, match="has not been built"):
            _ = GraphIndex(8).graph

    def test_a_corpus_too_small_for_the_build_degree_is_refused(self):
        assert a_corpus_too_small_for_the_build_degree_is_refused()

    def test_the_results_come_back_closest_first(self):
        corpus = gaussian(count=1024, dimension=16)
        index = GraphIndex(16, degree=8, build_degree=16, ef=32)
        index.build(corpus.vectors)
        found, _ = index.search(corpus.vectors[:8], k=10)
        assert bool((found.scores[:, 1:] >= found.scores[:, :-1] - 1e-5).all())

    def test_a_vector_finds_itself(self):
        corpus = gaussian(count=1024, dimension=16)
        index = GraphIndex(16, degree=16, build_degree=32, ef=64)
        index.build(corpus.vectors)
        found, _ = index.search(corpus.vectors[:16], k=1)
        assert found.identifiers.flatten().tolist() == list(range(16))

    def test_the_memory_is_the_vectors_plus_the_edges(self):
        corpus = gaussian(count=1024, dimension=16)
        index = GraphIndex(16, degree=8, build_degree=16)
        index.build(corpus.vectors)
        assert index.memory_bytes() > 1024 * 16 * 4

    def test_a_query_of_the_wrong_width_is_refused(self):
        index = GraphIndex(16, degree=8, build_degree=16)
        index.build(gaussian(count=512, dimension=16).vectors)
        with pytest.raises(Exception, match="wide"):
            index.search(torch.randn(2, 8), k=5)

    def test_the_index_reports_its_name(self):
        assert GraphIndex(8).name == "graph"

    def test_a_small_corpus_still_works(self):
        assert graph_on(gaussian(count=512, dimension=16), degree=8, ef=32).recall > 0.5

    def test_a_build_degree_below_the_degree_is_refused(self):
        with pytest.raises(ConfigError, match="out of a build degree"):
            GraphIndex(8, degree=16, build_degree=8)

    def test_an_unbuilt_index_has_no_entry_point(self):
        with pytest.raises(IndexStateError, match="has not been built"):
            _ = GraphIndex(8).entry_point

    def test_a_build_error_names_the_corpus_size(self):
        with pytest.raises(BuildError, match="cannot fill"):
            GraphIndex(8, degree=8, build_degree=64).build(torch.randn(32, 8))

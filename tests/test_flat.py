from __future__ import annotations

import pytest
import torch

from vse.errors import ConfigError, IndexStateError
from vse.index.flat import (
    FlatIndex,
    a_hundred_thousand_vectors_is_not_a_problem,
    a_query_of_the_wrong_width_is_refused,
    a_removed_vector_never_comes_back,
    building_on_an_empty_index_by_inserting,
    compaction_reclaims_the_memory,
    compaction_renumbers_everything,
    compare_dimensions,
    flat_on,
    insertion_is_an_append,
    it_costs_the_whole_corpus,
    it_is_exactly_linear,
    memory_is_the_vectors_plus_a_bit,
    removing_a_row_that_does_not_exist_is_refused,
    removing_the_same_row_twice_counts_once,
    searching_before_building_is_refused,
    the_baseline_is_exact,
    the_cost_does_not_depend_on_k,
    the_cost_grows_with_the_corpus,
    the_distance_count_hides_the_dimension,
    tombstones_still_cost,
)
from vse.vectors.dataset import clustered, gaussian


class TestExactness:
    def test_the_baseline_is_exact(self):
        assert the_baseline_is_exact()["exact"]

    def test_with_perfect_recall(self):
        assert the_baseline_is_exact()["recall"] == 1.0

    def test_and_no_gap(self):
        assert the_baseline_is_exact()["gap"] == 0.0

    def test_it_scans_the_whole_corpus(self):
        assert it_costs_the_whole_corpus()["scanned"] == 1.0

    def test_so_it_has_no_speedup(self):
        assert it_costs_the_whole_corpus()["speedup"] == 1.0

    def test_it_is_exact_on_every_fixture_width(self):
        assert all(row["recall"] == 1.0 for row in compare_dimensions())


class TestCost:
    def test_the_cost_does_not_depend_on_k(self):
        assert the_cost_does_not_depend_on_k()["identical"]

    def test_an_empty_k_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            the_cost_does_not_depend_on_k(values=())

    def test_it_scans_the_corpus_exactly_once_at_every_size(self):
        assert it_is_exactly_linear()["scans_the_corpus_once"]

    def test_the_ratio_between_rows_is_about_two(self):
        assert it_is_exactly_linear()["close_to_two"]

    def test_and_the_offset_is_the_held_out_queries(self):
        assert it_is_exactly_linear()["held_out_explains_the_offset"]

    def test_an_empty_size_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            the_cost_grows_with_the_corpus(sizes=())

    def test_a_hundred_thousand_vectors_is_fifty_megabytes(self):
        assert 50 < a_hundred_thousand_vectors_is_not_a_problem()["megabytes"] < 52

    def test_and_thirteen_million_operations(self):
        result = a_hundred_thousand_vectors_is_not_a_problem()
        assert result["multiply_accumulates"] == 100_000 * 128

    def test_a_corpus_of_nothing_is_refused(self):
        with pytest.raises(ConfigError, match="not a corpus"):
            a_hundred_thousand_vectors_is_not_a_problem(count=0)

    def test_the_distance_count_ignores_the_dimension(self):
        assert the_distance_count_hides_the_dimension()["identical"]

    def test_though_the_arithmetic_is_sixty_four_times_larger(self):
        assert the_distance_count_hides_the_dimension()["arithmetic_ratio"] == 64

    def test_an_empty_dimension_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            compare_dimensions(dimensions=())


class TestDeletion:
    def test_a_removed_vector_is_never_returned(self):
        assert not a_removed_vector_never_comes_back()["still_returned"]

    def test_but_it_still_costs(self):
        assert tombstones_still_cost()["unchanged"]

    def test_even_after_removing_half_the_corpus(self):
        result = tombstones_still_cost()
        assert result["live"] * 2 == result["capacity"]

    def test_and_it_still_takes_memory(self):
        assert compaction_reclaims_the_memory()["removal_freed_nothing"]

    def test_removing_the_same_row_twice_counts_once(self):
        result = removing_the_same_row_twice_counts_once()
        assert (result["first"], result["again"]) == (2, 0)

    def test_removing_a_row_that_does_not_exist_is_refused(self):
        assert removing_a_row_that_does_not_exist_is_refused()

    def test_reading_a_removed_vector_is_refused(self):
        index = FlatIndex(8)
        index.build(gaussian(count=32, dimension=8).vectors)
        index.remove([5])
        with pytest.raises(IndexStateError, match="was removed"):
            index.vector(5)

    def test_reading_a_vector_that_does_not_exist_is_refused(self):
        index = FlatIndex(8)
        index.build(gaussian(count=32, dimension=8).vectors)
        with pytest.raises(ConfigError, match="not one of the 32"):
            index.vector(99)

    def test_removing_from_an_unbuilt_index_is_refused(self):
        with pytest.raises(IndexStateError, match="has not been built"):
            FlatIndex(8).remove([0])


class TestCompaction:
    def test_compaction_reclaims_the_memory(self):
        assert compaction_reclaims_the_memory()["compaction_halved_it"]

    def test_and_removes_the_tombstones(self):
        assert compaction_reclaims_the_memory()["after_compacting"] < 140_000

    def test_but_it_renumbers_everything(self):
        assert compaction_renumbers_everything()["identifiers_changed"]

    def test_so_an_old_identifier_now_points_somewhere_else(self):
        result = compaction_renumbers_everything()
        assert not result["vector_at_the_old_position"]

    def test_while_the_vector_itself_is_still_there(self):
        assert compaction_renumbers_everything()["vector_at_the_new_position"]

    def test_it_reports_what_it_reclaimed(self):
        assert compaction_renumbers_everything()["reclaimed"] == 3

    def test_compacting_an_unbuilt_index_is_refused(self):
        with pytest.raises(IndexStateError, match="has not been built"):
            FlatIndex(8).compact()

    def test_compacting_with_no_tombstones_changes_nothing(self):
        index = FlatIndex(8)
        index.build(gaussian(count=64, dimension=8).vectors)
        assert index.compact()["reclaimed"] == 0


class TestInsertion:
    def test_insertion_is_an_append(self):
        assert insertion_is_an_append()["first_new_identifier"] == 512

    def test_and_never_degrades_the_answer(self):
        assert insertion_is_an_append()["still_exact"]

    def test_it_doubles_the_size(self):
        result = insertion_is_an_append()
        assert result["after"] == result["before"] * 2

    def test_inserting_into_an_unbuilt_index_builds_it(self):
        assert building_on_an_empty_index_by_inserting()["built"]

    def test_and_returns_every_identifier(self):
        result = building_on_an_empty_index_by_inserting()
        assert result["identifiers"] == result["size"]

    def test_inserting_the_wrong_width_is_refused(self):
        index = FlatIndex(8)
        index.build(gaussian(count=16, dimension=8).vectors)
        with pytest.raises(Exception, match="wide"):
            index.insert(torch.randn(4, 16))

    def test_identifiers_continue_past_tombstones(self):
        # A hole does not get reused, because reusing it would hand out an identifier that a
        # caller may still be holding from before the deletion.
        index = FlatIndex(8)
        index.build(gaussian(count=16, dimension=8).vectors)
        index.remove([0, 1])
        assert index.insert(torch.randn(1, 8))[0] == 16


class TestMemory:
    def test_the_overhead_is_one_bit_a_vector(self):
        result = memory_is_the_vectors_plus_a_bit()
        assert result["overhead"] == 4096 // 8

    def test_which_is_a_thousandth_of_the_data(self):
        assert memory_is_the_vectors_plus_a_bit()["overhead_share"] < 0.001

    def test_an_unbuilt_index_reports_no_bytes(self):
        assert FlatIndex(8).as_dict()["bytes"] == 0

    def test_the_capacity_counts_tombstones(self):
        index = FlatIndex(8)
        index.build(gaussian(count=64, dimension=8).vectors)
        index.remove([1, 2, 3])
        assert (index.size, index.capacity, index.tombstones) == (61, 64, 3)

    def test_a_flat_index_on_a_clustered_corpus_is_still_exact(self):
        assert flat_on(clustered(count=512, dimension=16, clusters=8)).recall == 1.0

    def test_searching_before_building_is_refused(self):
        assert searching_before_building_is_refused()

    def test_a_query_of_the_wrong_width_is_refused(self):
        assert a_query_of_the_wrong_width_is_refused()

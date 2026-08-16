from __future__ import annotations

import pytest
import torch

from vse.errors import BuildError, ConfigError, IndexStateError
from vse.index.composite import (
    CompositeIndex,
    a_removed_vector_never_comes_back,
    a_shortlist_shorter_than_k_is_refused,
    a_width_that_does_not_split_is_refused,
    and_much_less_on_unstructured_data,
    compare_the_whole_family,
    composite_on,
    insertion_uses_the_existing_codebooks,
    probe_and_shortlist_are_different_knobs,
    residuals_help_on_structured_data,
    searching_before_building_is_refused,
    the_composite_index_is_the_smallest,
    the_memory_is_the_point,
    the_partition_identifier_is_half_the_index,
    the_probe_count_is_the_ceiling,
    the_rerank_makes_the_codes_almost_irrelevant,
    the_two_memory_numbers_are_separate,
)
from vse.vectors.dataset import clustered, gaussian


class TestResiduals:
    def test_the_residual_helps_on_structured_data(self):
        assert residuals_help_on_structured_data()["helps"]

    def test_by_sixteen_points(self):
        assert residuals_help_on_structured_data()["gain"] > 0.1

    def test_at_the_same_number_of_bytes(self):
        assert residuals_help_on_structured_data()["same_bytes"]

    def test_and_costs_a_little_on_unstructured_data(self):
        assert and_much_less_on_unstructured_data()["gain"] < 0.0

    def test_which_is_a_large_swing_between_corpora(self):
        structured = residuals_help_on_structured_data()["gain"]
        unstructured = and_much_less_on_unstructured_data()["gain"]
        assert structured - unstructured > 0.15

    def test_the_comparison_needs_a_short_shortlist(self):
        # At a shortlist of a hundred the rerank hides the code quality entirely.
        corpus = clustered(count=1024, dimension=64, clusters=16)
        long_list = composite_on(corpus, partitions=16, probe=1, shortlist=100, residual=True)
        short_list = composite_on(corpus, partitions=16, probe=1, shortlist=15, residual=True)
        assert long_list.recall > short_list.recall


class TestKnobs:
    def test_nine_combinations_are_swept(self):
        assert len(probe_and_shortlist_are_different_knobs()) == 9

    def test_the_probe_sets_the_ceiling(self):
        result = the_probe_count_is_the_ceiling()
        assert result["recall_at_sixteen"] > result["recall_at_one_probe"]

    def test_and_the_shortlist_gains_are_similar_at_both(self):
        result = the_probe_count_is_the_ceiling()
        assert (
            abs(
                result["shortlist_gain_at_one_probe"]
                - result["shortlist_gain_at_sixteen_probes"]
            )
            < 0.05
        )

    def test_so_the_two_barely_interact(self):
        rows = {
            (row["probe"], row["shortlist"]): row
            for row in probe_and_shortlist_are_different_knobs()
        }
        assert rows[(4, 100)]["recall"] >= rows[(4, 20)]["recall"]

    def test_a_longer_shortlist_costs_more_distances(self):
        rows = {
            (row["probe"], row["shortlist"]): row
            for row in probe_and_shortlist_are_different_knobs()
        }
        assert rows[(4, 400)]["distances_per_query"] > rows[(4, 20)]["distances_per_query"]

    def test_an_empty_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            probe_and_shortlist_are_different_knobs(probes=())


class TestMemory:
    def test_the_index_is_twenty_times_smaller_than_the_vectors(self):
        assert the_memory_is_the_point()["ratio"] > 15.0

    def test_a_hundred_thousand_vectors_fit_in_a_few_megabytes(self):
        assert the_memory_is_the_point()["index_megabytes"] < 5.0

    def test_where_the_raw_vectors_need_fifty(self):
        assert the_memory_is_the_point()["raw_megabytes"] > 50.0

    def test_the_identifiers_are_the_same_size_as_the_codes(self):
        assert the_partition_identifier_is_half_the_index()["identifiers_match_codes"]

    def test_and_are_six_times_wider_than_they_need_to_be(self):
        result = the_partition_identifier_is_half_the_index()
        assert result["bits_used"] / result["bits_actually_needed"] > 6

    def test_which_wastes_a_third_of_the_index(self):
        assert the_partition_identifier_is_half_the_index()["wasted_share"] > 0.25

    def test_a_corpus_of_nothing_is_refused(self):
        with pytest.raises(ConfigError, match="not a corpus"):
            the_memory_is_the_point(count=0)

    def test_the_two_memory_numbers_are_separate(self):
        result = the_two_memory_numbers_are_separate()
        assert result["with_vectors"] > result["without_vectors"]

    def test_and_the_vectors_dominate_the_larger_one(self):
        result = the_two_memory_numbers_are_separate()
        assert result["ratio"] > 3.0


class TestQuality:
    def test_the_compression_costs_nothing_measurable(self):
        assert the_rerank_makes_the_codes_almost_irrelevant()["compression_costs_little"]

    def test_it_matches_an_uncompressed_partitioned_index(self):
        result = the_rerank_makes_the_codes_almost_irrelevant()
        assert result["composite_recall"] >= result["uncompressed_partition_recall"] - 0.01

    def test_three_indexes_are_compared(self):
        assert len(compare_the_whole_family()) == 3

    def test_the_composite_one_is_the_smallest(self):
        assert the_composite_index_is_the_smallest()["smallest"] == "composite"

    def test_by_a_factor_of_five_against_the_flat_one(self):
        result = the_composite_index_is_the_smallest()
        assert result["flat_bytes"] > result["composite_bytes"] * 4

    def test_and_the_inverted_file_is_larger_than_the_flat_one(self):
        # It stores the vectors plus centres plus posting lists.
        result = the_composite_index_is_the_smallest()
        assert result["ivf_bytes"] > result["flat_bytes"]

    def test_the_composite_one_is_also_the_fastest(self):
        rows = {row["index"]: row for row in compare_the_whole_family()}
        assert rows["composite"]["speedup"] > rows["ivf"]["speedup"]


class TestUpdates:
    def test_insertion_uses_the_existing_codebooks(self):
        assert insertion_uses_the_existing_codebooks()["still_works"]

    def test_and_doubles_the_index(self):
        assert insertion_uses_the_existing_codebooks()["size"] > 1900

    def test_inserting_into_an_unbuilt_index_builds_it(self):
        index = CompositeIndex(32, partitions=8, probe=2, centroids=64)
        index.insert(gaussian(count=512, dimension=32).vectors)
        assert index.built and index.size == 512

    def test_a_removed_vector_is_never_returned(self):
        assert not a_removed_vector_never_comes_back()["still_returned"]

    def test_removing_a_row_that_does_not_exist_is_refused(self):
        index = CompositeIndex(32, partitions=8, probe=2, centroids=64)
        index.build(gaussian(count=512, dimension=32).vectors)
        with pytest.raises(ConfigError, match="not one of the 512"):
            index.remove([9999])

    def test_removing_from_an_unbuilt_index_is_refused(self):
        with pytest.raises(IndexStateError, match="has not been built"):
            CompositeIndex(32, partitions=4, probe=1).remove([0])


class TestMechanics:
    def test_searching_before_building_is_refused(self):
        assert searching_before_building_is_refused()

    def test_a_width_that_does_not_split_is_refused(self):
        assert a_width_that_does_not_split_is_refused()

    def test_a_shortlist_shorter_than_k_is_refused(self):
        assert a_shortlist_shorter_than_k_is_refused()

    def test_a_probe_above_the_partition_count_is_refused(self):
        with pytest.raises(ConfigError, match="probing 16 of 4"):
            CompositeIndex(32, partitions=4, probe=16)

    def test_a_zero_shortlist_is_refused(self):
        with pytest.raises(ConfigError, match="returns nothing"):
            CompositeIndex(32, partitions=4, probe=1, shortlist=0)

    def test_a_corpus_too_small_to_train_is_refused(self):
        with pytest.raises(BuildError, match="cannot train"):
            CompositeIndex(32, partitions=4, probe=1, centroids=256).build(torch.randn(64, 32))

    def test_more_partitions_than_vectors_is_refused(self):
        index = CompositeIndex(32, partitions=128, probe=1, centroids=16)
        with pytest.raises(BuildError, match="partitions over"):
            index.build(torch.randn(64, 32))

    def test_reading_the_codes_before_building_is_refused(self):
        with pytest.raises(IndexStateError, match="has not been built"):
            _ = CompositeIndex(32, partitions=4, probe=1).codes

    def test_the_results_come_back_closest_first(self):
        corpus = clustered(count=1024, dimension=32, clusters=16)
        index = CompositeIndex(32, partitions=16, probe=4, shortlist=50, centroids=64)
        index.build(corpus.vectors)
        found, _ = index.search(corpus.vectors[:8], k=10)
        assert bool((found.scores[:, 1:] >= found.scores[:, :-1] - 1e-4).all())

    def test_the_index_reports_its_name(self):
        assert CompositeIndex(32, partitions=4, probe=1).name == "composite"

    def test_a_probe_override_works(self):
        corpus = clustered(count=1024, dimension=32, clusters=16)
        index = CompositeIndex(32, partitions=16, probe=1, shortlist=50, centroids=64)
        index.build(corpus.vectors)
        cheap = index.search(corpus.vectors[:8], k=5)[1].distances_per_query
        dear = index.search(corpus.vectors[:8], k=5, probe=8)[1].distances_per_query
        assert dear > cheap

    def test_an_override_above_the_partition_count_is_refused(self):
        corpus = clustered(count=1024, dimension=32, clusters=16)
        index = CompositeIndex(32, partitions=16, probe=1, shortlist=50, centroids=64)
        index.build(corpus.vectors)
        with pytest.raises(ConfigError, match="probing 99 of 16"):
            index.search(corpus.vectors[:2], k=5, probe=99)

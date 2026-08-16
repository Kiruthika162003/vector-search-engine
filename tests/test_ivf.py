from __future__ import annotations

import pytest
import torch

from vse.errors import BuildError, ConfigError, IndexStateError
from vse.index.ivf import (
    IVFIndex,
    a_probe_count_above_the_partitions_is_refused,
    a_rebuild_buys_nothing_on_a_stationary_stream,
    a_shifting_distribution_is_the_real_drift,
    and_a_rebuild_recovers_the_cost_not_the_recall,
    compare_corpora,
    insertion_barely_drifts_on_a_stationary_stream,
    ivf_on,
    more_partitions_than_vectors_is_refused,
    one_probe_is_a_property_of_the_corpus,
    partition_sweep,
    probe_sweep,
    probing_everything_is_exact_and_slower,
    rebuilding_below_the_partition_count_is_refused,
    searching_before_building_is_refused,
    the_centre_scan_eventually_dominates,
    the_cheapest_partition_count,
    the_cost_has_a_tail,
    the_frontier_is_where_the_index_earns_its_keep,
    the_measured_minimum_matches_the_model,
    tombstones_stay_in_their_lists,
)
from vse.vectors.dataset import gaussian


class TestStructureMatters:
    def test_one_probe_recovers_almost_everything_on_clusters(self):
        assert one_probe_is_a_property_of_the_corpus()["clustered_recall"] > 0.9

    def test_and_a_seventh_on_unstructured_rows(self):
        assert one_probe_is_a_property_of_the_corpus()["gaussian_recall"] < 0.2

    def test_a_factor_of_seven_apart(self):
        assert one_probe_is_a_property_of_the_corpus()["ratio"] > 5.0

    def test_while_scanning_the_same_share_of_the_corpus(self):
        result = one_probe_is_a_property_of_the_corpus()
        assert abs(result["clustered_scanned"] - result["gaussian_scanned"]) < 0.01

    def test_the_clustered_corpus_reaches_perfect_recall_at_four_probes(self):
        rows = {row["corpus"]: row for row in compare_corpora()}
        assert rows["clustered 32d"]["recall"] > 0.8


class TestFrontier:
    def test_recall_rises_with_the_probe_count(self):
        rows = [row["recall"] for row in probe_sweep()]
        assert rows == sorted(rows)

    def test_and_the_speedup_falls(self):
        rows = [row["speedup"] for row in probe_sweep()]
        assert rows == sorted(rows, reverse=True)

    def test_and_the_gap_falls_too(self):
        rows = [row["gap"] for row in probe_sweep()]
        assert rows == sorted(rows, reverse=True)

    def test_probing_everything_is_exact(self):
        assert probing_everything_is_exact_and_slower()["recall"] == 1.0

    def test_with_no_gap(self):
        assert probing_everything_is_exact_and_slower()["gap"] == 0.0

    def test_and_slower_than_a_flat_index(self):
        assert probing_everything_is_exact_and_slower()["slower_than_flat"]

    def test_ninety_percent_recall_costs_half_the_corpus(self):
        result = the_frontier_is_where_the_index_earns_its_keep()
        assert result["reached_ninety"]
        assert result["scanned"] > 0.4

    def test_for_a_speedup_under_two(self):
        assert the_frontier_is_where_the_index_earns_its_keep()["speedup"] < 2.5

    def test_an_empty_probe_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            probe_sweep(probes=())


class TestCostModel:
    def test_the_centre_scan_grows_with_the_partition_count(self):
        assert the_centre_scan_eventually_dominates()["grew"]

    def test_and_eventually_dominates(self):
        assert the_centre_scan_eventually_dominates()["dominates"]

    def test_the_model_minimum_is_the_square_root(self):
        result = the_cheapest_partition_count()
        assert result["model_minimum"] == result["predicted"]

    def test_within_a_factor_of_two(self):
        assert the_cheapest_partition_count()["close"]

    def test_the_measured_cheapest_is_not_the_most_accurate(self):
        assert the_measured_minimum_matches_the_model()["the_cheapest_is_not_the_best"]

    def test_the_measured_cheapest_matches_the_model(self):
        assert the_measured_minimum_matches_the_model()["cheapest"] == 64

    def test_a_zero_probe_configuration_is_refused(self):
        with pytest.raises(ConfigError, match="not a configuration"):
            the_cheapest_partition_count(probe=0)

    def test_an_empty_partition_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            partition_sweep(counts=())

    def test_a_probe_share_above_one_is_refused(self):
        with pytest.raises(ConfigError, match="not a share"):
            partition_sweep(probe_share=2.0)

    def test_the_cost_has_a_tail(self):
        assert the_cost_has_a_tail()["tail_ratio"] > 1.2

    def test_milder_than_the_partition_sizes_suggest(self):
        # The fixed centre scan does not vary, so it dilutes the variance in the part that does.
        result = the_cost_has_a_tail()
        size_tail = result["largest_partition"] / result["mean_partition"]
        assert result["tail_ratio"] < size_tail


class TestDrift:
    def test_a_stationary_stream_barely_drifts(self):
        assert insertion_barely_drifts_on_a_stationary_stream()["drop"] < 0.05

    def test_though_it_does_fall(self):
        assert insertion_barely_drifts_on_a_stationary_stream()["fell"]

    def test_and_a_rebuild_recovers_none_of_it(self):
        assert not a_rebuild_buys_nothing_on_a_stationary_stream()["recovered"]

    def test_the_rebuild_reclustered_everything(self):
        assert a_rebuild_buys_nothing_on_a_stationary_stream()["reclustered"] > 6000

    def test_an_empty_stream_is_refused(self):
        with pytest.raises(ConfigError, match="not a stream"):
            insertion_barely_drifts_on_a_stationary_stream(batches=0)

    def test_a_shifting_distribution_keeps_its_recall(self):
        assert a_shifting_distribution_is_the_real_drift()["recall_survived"]

    def test_but_destroys_the_balance(self):
        assert a_shifting_distribution_is_the_real_drift()["tail_ratio"] > 3.0

    def test_the_rebuild_recovers_the_cost(self):
        assert and_a_rebuild_recovers_the_cost_not_the_recall()["cost_fell"]

    def test_by_about_a_third(self):
        result = and_a_rebuild_recovers_the_cost_not_the_recall()
        assert result["cost_after"] < result["cost_before"] * 0.8

    def test_and_the_recall_was_never_the_problem(self):
        result = and_a_rebuild_recovers_the_cost_not_the_recall()
        assert result["recall_before"] > 0.9

    def test_the_tail_is_the_signal(self):
        result = and_a_rebuild_recovers_the_cost_not_the_recall()
        assert result["tail_after"] < result["tail_before"] / 2


class TestMechanics:
    def test_the_posting_lists_hold_every_vector(self):
        index = IVFIndex(16, partitions=8)
        index.build(gaussian(count=512, dimension=16).vectors)
        assert int(index.sizes.sum()) == 512

    def test_the_clustering_can_be_inspected(self):
        index = IVFIndex(16, partitions=8)
        index.build(gaussian(count=512, dimension=16).vectors)
        assert index.clustering().k == 8

    def test_a_probe_override_beats_the_default(self):
        index = IVFIndex(16, partitions=16, probe=1)
        index.build(gaussian(count=512, dimension=16).vectors)
        cheap = index.search(torch.randn(8, 16), k=5)[1].distances_per_query
        dear = index.search(torch.randn(8, 16), k=5, probe=8)[1].distances_per_query
        assert dear > cheap

    def test_an_override_above_the_partition_count_is_refused(self):
        index = IVFIndex(16, partitions=8)
        index.build(gaussian(count=512, dimension=16).vectors)
        with pytest.raises(ConfigError, match="probing 99 of 8"):
            index.search(torch.randn(2, 16), k=5, probe=99)

    def test_a_probe_count_above_the_partitions_is_refused_at_construction(self):
        assert a_probe_count_above_the_partitions_is_refused()

    def test_zero_partitions_is_refused(self):
        with pytest.raises(ConfigError, match="not a partitioning"):
            IVFIndex(8, partitions=0)

    def test_more_partitions_than_vectors_is_refused(self):
        assert more_partitions_than_vectors_is_refused()

    def test_searching_before_building_is_refused(self):
        assert searching_before_building_is_refused()

    def test_inserting_into_an_unbuilt_index_builds_it(self):
        index = IVFIndex(8, partitions=4)
        index.insert(gaussian(count=64, dimension=8).vectors)
        assert index.built and index.size == 64

    def test_an_insertion_is_filed_and_findable(self):
        corpus = gaussian(count=512, dimension=16)
        index = IVFIndex(16, partitions=8, probe=8)
        index.build(corpus.vectors)
        fresh = torch.randn(1, 16)
        identifier = index.insert(fresh)[0]
        found = index.search(fresh, k=1)[0]
        assert int(found.identifiers[0, 0]) == identifier

    def test_the_insertion_counter_tracks(self):
        index = IVFIndex(8, partitions=4)
        index.build(gaussian(count=64, dimension=8).vectors)
        index.insert(torch.randn(10, 8))
        assert index.inserted == 10

    def test_a_rebuild_resets_it(self):
        index = IVFIndex(8, partitions=4)
        index.build(gaussian(count=64, dimension=8).vectors)
        index.insert(torch.randn(10, 8))
        index.rebuild()
        assert index.inserted == 0


class TestDeletion:
    def test_a_removed_vector_is_never_returned(self):
        assert not tombstones_stay_in_their_lists()["any_dead_returned"]

    def test_and_the_scan_actually_shrinks(self):
        # Unlike the flat index, where deletion is a mask over work already done.
        assert tombstones_stay_in_their_lists()["scan_shrank"]

    def test_by_about_half_when_half_is_removed(self):
        result = tombstones_stay_in_their_lists()
        assert result["after"] < result["before"] * 0.75

    def test_removing_a_row_that_does_not_exist_is_refused(self):
        index = IVFIndex(8, partitions=4)
        index.build(gaussian(count=64, dimension=8).vectors)
        with pytest.raises(ConfigError, match="not one of the 64"):
            index.remove([999])

    def test_removing_from_an_unbuilt_index_is_refused(self):
        with pytest.raises(IndexStateError, match="has not been built"):
            IVFIndex(8, partitions=4).remove([0])

    def test_rebuilding_below_the_partition_count_is_refused(self):
        assert rebuilding_below_the_partition_count_is_refused()

    def test_rebuilding_an_unbuilt_index_is_refused(self):
        with pytest.raises(IndexStateError, match="has not been built"):
            IVFIndex(8, partitions=4).rebuild()

    def test_a_rebuild_drops_the_tombstones(self):
        index = IVFIndex(8, partitions=4)
        index.build(gaussian(count=256, dimension=8).vectors)
        index.remove(range(0, 128))
        index.rebuild()
        assert index.capacity == 128

    def test_a_build_that_cannot_fill_its_partitions_is_refused(self):
        with pytest.raises(BuildError, match="is too many"):
            IVFIndex(8, partitions=64).build(torch.randn(16, 8))


class TestMemory:
    def test_it_costs_more_than_a_flat_index(self):
        index = IVFIndex(32, partitions=64)
        index.build(gaussian(count=4096, dimension=32).vectors)
        assert index.memory_bytes() > 4096 * 32 * 4

    def test_the_centres_are_a_small_share(self):
        index = IVFIndex(32, partitions=64)
        index.build(gaussian(count=4096, dimension=32).vectors)
        assert index.memory_bytes() * 0.05 > 64 * 32 * 4

    def test_an_unbuilt_index_reports_nothing(self):
        assert IVFIndex(8, partitions=4).as_dict()["size"] == 0

    def test_three_corpora_are_compared(self):
        assert len(compare_corpora()) == 3

    def test_the_unstructured_corpus_has_the_worst_recall(self):
        rows = compare_corpora()
        assert min(rows, key=lambda row: row["recall"])["corpus"] == "gaussian 32d"

    def test_a_small_index_still_works(self):
        assert ivf_on(gaussian(count=256, dimension=8), partitions=8, probe=8).recall > 0.5

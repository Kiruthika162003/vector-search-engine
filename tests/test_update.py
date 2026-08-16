from __future__ import annotations

import pytest
import torch

from vse.errors import ConfigError
from vse.index.graph import GraphIndex
from vse.index.ivf import IVFIndex
from vse.index.update import (
    Churn,
    Degradation,
    a_partitioned_index_survives_deletion_better,
    a_rebuild_after_drift_recovers_it,
    a_rebuild_fixes_something_that_was_never_broken,
    a_removal_share_of_one_is_refused,
    a_split_that_leaves_nothing_to_insert_is_refused,
    a_zero_batch_insert_is_refused,
    an_insert_into_an_unbuilt_index_builds_it,
    clustered_deletion_is_worse_than_uniform,
    compare_structures_under_churn,
    deletion_fragments_the_graph,
    drift_lands_the_new_corpus_on_a_few_centroids,
    insert_in_batches,
    measure,
    queries_in_the_drifted_region_recall_more_and_cost_more,
    remove_in_batches,
    removing_something_that_is_not_there_is_not_an_error,
    split_for_churn,
    the_churn_counter_tracks_writes_not_size,
    the_drifted_region_is_the_expensive_one,
    the_partitions_go_uneven,
    the_spread_is_where_drift_shows_up,
    the_tombstones_are_holding_the_graph_together,
    the_tombstones_are_still_traversed,
)
from vse.vectors.dataset import gaussian


class TestChurnAccounting:
    def test_an_empty_churn_is_zero(self):
        assert Churn().churn == 0.0

    def test_and_has_no_size(self):
        assert Churn().size == 0

    def test_inserts_raise_the_size(self):
        assert Churn(built_size=100, inserted=50).size == 150

    def test_removals_lower_it(self):
        assert Churn(built_size=100, removed=40).size == 60

    def test_the_insert_share_is_against_the_built_corpus(self):
        assert Churn(built_size=200, inserted=50).insert_share == 0.25

    def test_and_so_is_the_removal_share(self):
        assert Churn(built_size=200, removed=50).remove_share == 0.25

    def test_churn_counts_writes_not_net(self):
        assert the_churn_counter_tracks_writes_not_size()["churn_is_one"]

    def test_even_when_the_size_did_not_move(self):
        assert the_churn_counter_tracks_writes_not_size()["size_unchanged"]

    def test_it_serialises(self):
        assert Churn(built_size=100, inserted=10).as_dict()["size"] == 110

    def test_a_degradation_serialises(self):
        row = Degradation(churn=0.5, recall=0.912345, distances=100.0, size=10)
        assert row.as_dict()["recall"] == 0.9123

    def test_and_carries_its_detail(self):
        row = Degradation(churn=0.0, recall=1.0, distances=1.0, size=1, detail={"live": 4})
        assert row.as_dict()["live"] == 4


class TestSplitting:
    def test_a_split_divides_the_corpus(self):
        built, arriving, queries = split_for_churn(gaussian(count=1024, dimension=8), built=512)
        assert int(built.shape[0]) == 512
        assert int(queries.shape[0]) == 100
        assert int(arriving.shape[0]) == 1024 - 512 - 100

    def test_the_pieces_do_not_overlap(self):
        built, arriving, _ = split_for_churn(gaussian(count=1024, dimension=8), built=512)
        assert int(torch.unique(torch.cat([built, arriving]), dim=0).shape[0]) == int(
            built.shape[0] + arriving.shape[0]
        )

    def test_asking_for_more_than_the_corpus_holds_is_refused(self):
        assert a_split_that_leaves_nothing_to_insert_is_refused()

    def test_the_error_names_the_numbers(self):
        with pytest.raises(ConfigError, match="cannot supply"):
            split_for_churn(gaussian(count=200, dimension=8), built=300, queries=50)


class TestDeletionOnAGraph:
    def test_the_cost_does_not_fall(self):
        result = the_tombstones_are_still_traversed()
        assert result["distances_after_removals"] == result["distances_at_build"]

    def test_even_though_the_corpus_shrank(self):
        result = the_tombstones_are_still_traversed()
        assert result["size_after"] < result["size_at_build"] * 0.7

    def test_a_fresh_graph_over_the_survivors_is_cheaper(self):
        assert the_tombstones_are_still_traversed()["rebuilt_is_cheaper"]

    def test_the_component_count_says_it_is_fine(self):
        rows = deletion_fragments_the_graph()
        assert all(row["components_with_tombstones"] == 1 for row in rows)

    def test_and_the_live_vertices_alone_say_otherwise(self):
        rows = {row["removed_share"]: row for row in deletion_fragments_the_graph()}
        assert rows[0.6]["components_over_live_only"] > 1

    def test_the_fragmentation_grows_with_the_deletions(self):
        rows = [row["components_over_live_only"] for row in deletion_fragments_the_graph()]
        assert rows == sorted(rows)

    def test_an_undeleted_graph_is_one_piece_either_way(self):
        rows = {row["removed_share"]: row for row in deletion_fragments_the_graph()}
        assert rows[0.0]["components_over_live_only"] == 1

    def test_the_tombstones_are_holding_it_together(self):
        assert the_tombstones_are_holding_the_graph_together()[
            "connected_only_through_the_dead"
        ]

    def test_by_a_factor_of_more_than_ten(self):
        result = the_tombstones_are_holding_the_graph_together()
        assert result["components_over_live_only"] > 10

    def test_an_empty_deletion_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            deletion_fragments_the_graph(shares=())


class TestDeletionCompared:
    def test_the_partitioned_index_gets_cheaper(self):
        assert a_partitioned_index_survives_deletion_better()["ivf_got_cheaper"]

    def test_where_the_graph_does_not(self):
        assert a_partitioned_index_survives_deletion_better()["graph_did_not"]

    def test_the_graph_cost_is_exactly_unchanged(self):
        rows = {(row["index"], row["write"]): row for row in compare_structures_under_churn()}
        assert rows[("graph", "remove")]["cost_ratio"] == 1.0

    def test_the_partitioned_cost_falls_with_the_corpus(self):
        rows = {(row["index"], row["write"]): row for row in compare_structures_under_churn()}
        assert rows[("ivf", "remove")]["cost_ratio"] < 0.8

    def test_neither_loses_much_recall(self):
        result = a_partitioned_index_survives_deletion_better()
        assert abs(result["ivf_recall_change"]) < 0.1
        assert abs(result["graph_recall_change"]) < 0.1

    def test_four_rows_compare_the_structures(self):
        assert len(compare_structures_under_churn()) == 4

    def test_both_absorb_inserts(self):
        rows = {(row["index"], row["write"]): row for row in compare_structures_under_churn()}
        assert rows[("ivf", "insert")]["recall_change"] > -0.05
        assert rows[("graph", "insert")]["recall_change"] > -0.05

    def test_clustered_deletion_costs_more_recall(self):
        assert clustered_deletion_is_worse_than_uniform()["clustered_is_worse"]

    def test_and_leaves_less_to_scan(self):
        result = clustered_deletion_is_worse_than_uniform()
        assert result["clustered_distances"] < result["uniform_distances"]

    def test_uniform_deletion_leaves_the_recall_intact(self):
        assert clustered_deletion_is_worse_than_uniform()["uniform_recall"] > 0.95


class TestDrift:
    def test_drift_unbalances_the_partitions(self):
        assert the_spread_is_where_drift_shows_up()["grows"]

    def test_and_doubles_the_largest_one(self):
        result = the_spread_is_where_drift_shows_up()
        assert result["largest_at_eight"] > result["largest_without_drift"] * 1.8

    def test_the_spread_rises_with_the_shift(self):
        rows = [row["spread_after"] for row in drift_lands_the_new_corpus_on_a_few_centroids()]
        assert rows == sorted(rows)

    def test_no_shift_leaves_the_spread_alone(self):
        rows = {row["shift"]: row for row in drift_lands_the_new_corpus_on_a_few_centroids()}
        assert abs(rows[0.0]["spread_after"] - rows[0.0]["spread_at_build"]) < 0.02

    def test_the_drifted_region_recalls_more_not_less(self):
        assert the_drifted_region_is_the_expensive_one()["recall_is_higher"]

    def test_and_costs_more(self):
        assert the_drifted_region_is_the_expensive_one()["cost_is_higher"]

    def test_the_gap_opens_with_the_shift(self):
        rows = {
            row["shift"]: row
            for row in queries_in_the_drifted_region_recall_more_and_cost_more()
        }
        near = rows[0.0]["drifted_region_recall"] - rows[0.0]["original_region_recall"]
        far = rows[8.0]["drifted_region_recall"] - rows[8.0]["original_region_recall"]
        assert far > near

    def test_at_no_shift_the_two_regions_are_the_same(self):
        rows = {
            row["shift"]: row
            for row in queries_in_the_drifted_region_recall_more_and_cost_more()
        }
        assert rows[0.0]["drifted_region_recall"] == rows[0.0]["original_region_recall"]

    def test_an_empty_shift_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            drift_lands_the_new_corpus_on_a_few_centroids(shifts=())

    def test_an_empty_region_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            queries_in_the_drifted_region_recall_more_and_cost_more(shifts=())

    def test_a_rebuild_recovers_after_drift(self):
        assert a_rebuild_after_drift_recovers_it()["recovered"]

    def test_by_a_measurable_amount(self):
        assert a_rebuild_after_drift_recovers_it()["gain"] > 0.01


class TestRebuild:
    def test_a_rebuild_halves_the_spread(self):
        result = a_rebuild_fixes_something_that_was_never_broken()
        assert result["spread_after"] < result["spread_before"] * 0.7

    def test_and_lowers_the_cost(self):
        assert a_rebuild_fixes_something_that_was_never_broken()["cost_recovered"]

    def test_and_lifts_the_recall(self):
        result = a_rebuild_fixes_something_that_was_never_broken()
        assert result["recall_after"] > result["recall_before"]

    def test_stationary_inserts_do_not_unbalance_anything(self):
        assert not the_partitions_go_uneven()["grew"]

    def test_the_spread_falls_slightly(self):
        assert the_partitions_go_uneven()["ratio"] < 1.0


class TestWriteMechanics:
    def test_removing_something_twice_is_not_an_error(self):
        assert removing_something_that_is_not_there_is_not_an_error()["idempotent"]

    def test_the_first_removal_counts(self):
        assert removing_something_that_is_not_there_is_not_an_error()["first_removal"] == 3

    def test_an_unbuilt_inverted_file_builds_itself_on_insert(self):
        assert an_insert_into_an_unbuilt_index_builds_it()["ivf_accepted"]

    def test_where_the_graph_refuses(self):
        assert an_insert_into_an_unbuilt_index_builds_it()["graph_refused"]

    def test_the_two_indexes_disagree(self):
        assert an_insert_into_an_unbuilt_index_builds_it()["they_disagree"]

    def test_a_zero_batch_insert_is_refused(self):
        assert a_zero_batch_insert_is_refused()

    def test_a_removal_of_everything_is_refused(self):
        assert a_removal_share_of_one_is_refused()

    def test_a_removal_share_of_zero_is_refused(self):
        built, _, queries = split_for_churn(
            gaussian(count=512, dimension=8), built=256, queries=32
        )
        index = IVFIndex(8, partitions=16, probe=2)
        index.build(built)
        with pytest.raises(ConfigError, match="not a removal"):
            remove_in_batches(index, built, queries, share=0.0)

    def test_more_batches_than_vectors_is_refused(self):
        built, arriving, queries = split_for_churn(
            gaussian(count=512, dimension=8), built=256, queries=32
        )
        index = IVFIndex(8, partitions=16, probe=2)
        index.build(built)
        with pytest.raises(ConfigError, match="do not divide"):
            insert_in_batches(index, built, arriving, queries, batches=500)

    def test_an_insert_schedule_measures_at_every_step(self):
        built, arriving, queries = split_for_churn(
            gaussian(count=1024, dimension=8), built=512, queries=32
        )
        index = IVFIndex(8, partitions=16, probe=2)
        index.build(built)
        rows = insert_in_batches(index, built, arriving, queries, batches=4, k=5)
        assert len(rows) == 5

    def test_and_the_size_grows_at_each_one(self):
        built, arriving, queries = split_for_churn(
            gaussian(count=1024, dimension=8), built=512, queries=32
        )
        index = IVFIndex(8, partitions=16, probe=2)
        index.build(built)
        rows = insert_in_batches(index, built, arriving, queries, batches=4, k=5)
        assert [row.size for row in rows] == sorted(row.size for row in rows)

    def test_a_removal_schedule_shrinks_the_size(self):
        built, _, queries = split_for_churn(
            gaussian(count=1024, dimension=8), built=512, queries=32
        )
        index = IVFIndex(8, partitions=16, probe=2)
        index.build(built)
        rows = remove_in_batches(index, built, queries, batches=4, share=0.4, k=5)
        assert rows[-1].size < rows[0].size

    def test_and_records_the_live_share(self):
        built, _, queries = split_for_churn(
            gaussian(count=1024, dimension=8), built=512, queries=32
        )
        index = IVFIndex(8, partitions=16, probe=2)
        index.build(built)
        rows = remove_in_batches(index, built, queries, batches=4, share=0.4, k=5)
        assert rows[-1].detail["live_share"] < 1.0

    def test_a_measurement_without_a_mask_uses_the_whole_corpus(self):
        corpus = gaussian(count=512, dimension=8)
        index = GraphIndex(8, degree=8, ef=16)
        index.build(corpus.vectors)
        row = measure(index, corpus.vectors, corpus.vectors[:10], Churn(built_size=512), k=5)
        assert row.size == 512

    def test_a_measurement_with_a_mask_uses_the_live_rows(self):
        corpus = gaussian(count=512, dimension=8)
        index = GraphIndex(8, degree=8, ef=16)
        index.build(corpus.vectors)
        alive = torch.ones(512, dtype=torch.bool)
        alive[:100] = False
        row = measure(
            index,
            corpus.vectors,
            corpus.vectors[400:410],
            Churn(built_size=512),
            k=5,
            alive=alive,
        )
        assert row.size == 412

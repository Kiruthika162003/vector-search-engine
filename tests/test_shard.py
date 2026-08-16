from __future__ import annotations

import pytest
import torch

from vse.errors import BuildError, ConfigError
from vse.storage.shard import (
    Shard,
    a_merged_capacity_below_k_is_refused,
    a_query_visiting_the_wrong_shard_gets_a_clean_answer,
    a_shard_index_is_the_same_index,
    a_top_k_from_each_shard_is_enough_in_expectation,
    a_zero_fetch_is_refused,
    an_empty_merge_is_refused,
    an_empty_shard_is_refused,
    but_a_small_shard_count_concentrates_the_neighbours,
    clustered_shards,
    clustered_shards_are_unbalanced,
    compare_placements,
    merge,
    more_shards_than_vectors_is_refused,
    over_fetching_removes_the_variance,
    random_shards,
    routing_is_worth_it_only_with_structure,
    routing_reduces_the_work_and_can_be_wrong,
    scatter_gather,
    scatter_gather_is_exact,
    sharding_does_not_reduce_the_work,
    sharding_scales_until_the_merge_does_not,
    the_merge_cost_is_linear_in_the_shard_count,
    the_slowest_shard_sets_the_latency,
    visiting_more_shards_than_exist_is_refused,
)
from vse.vectors.dataset import gaussian


class TestPlacement:
    def test_random_shards_hold_the_whole_corpus(self):
        corpus = gaussian(count=1024, dimension=16)
        shards = random_shards(corpus, count=8)
        assert sum(shard.size for shard in shards) == 1024

    def test_and_are_perfectly_balanced(self):
        corpus = gaussian(count=1024, dimension=16)
        shards = random_shards(corpus, count=8)
        assert max(shard.size for shard in shards) == min(shard.size for shard in shards)

    def test_every_identifier_appears_exactly_once(self):
        corpus = gaussian(count=512, dimension=16)
        shards = random_shards(corpus, count=4)
        seen = torch.cat([shard.identifiers for shard in shards])
        assert sorted(seen.tolist()) == list(range(512))

    def test_clustered_shards_also_hold_the_whole_corpus(self):
        corpus = gaussian(count=1024, dimension=16)
        shards, _ = clustered_shards(corpus, count=8)
        assert sum(shard.size for shard in shards) == 1024

    def test_but_are_less_balanced(self):
        result = clustered_shards_are_unbalanced()
        assert result["clustered_ratio"] > result["random_ratio"]

    def test_though_only_mildly_when_the_counts_match(self):
        # The shard count matches the number of groups in the fixture, which is the easy case.
        assert clustered_shards_are_unbalanced()["clustered_ratio"] < 1.5

    def test_more_shards_than_vectors_is_refused(self):
        assert more_shards_than_vectors_is_refused()

    def test_zero_shards_is_refused(self):
        with pytest.raises(ConfigError, match="not a cluster"):
            random_shards(gaussian(count=64, dimension=4), count=0)

    def test_an_empty_shard_is_refused(self):
        assert an_empty_shard_is_refused()

    def test_a_mismatched_shard_is_refused(self):
        with pytest.raises(BuildError, match="identifiers"):
            Shard(vectors=torch.randn(8, 4), identifiers=torch.zeros(4, dtype=torch.long))

    def test_it_serialises(self):
        corpus = gaussian(count=512, dimension=16)
        assert random_shards(corpus, count=4)[0].as_dict()["size"] == 128


class TestScatterGather:
    def test_it_is_exact_at_every_shard_count(self):
        assert all(row["recall"] == 1.0 for row in scatter_gather_is_exact())

    def test_the_total_work_does_not_change(self):
        assert sharding_does_not_reduce_the_work()["total_unchanged"]

    def test_so_it_is_a_latency_structure_not_a_throughput_one(self):
        result = sharding_does_not_reduce_the_work()
        assert result["per_machine"] * 32 == result["thirty_two_machines"]

    def test_a_top_k_from_each_shard_is_enough(self):
        assert a_top_k_from_each_shard_is_enough_in_expectation()["exact"]

    def test_because_the_neighbours_are_spread(self):
        assert a_top_k_from_each_shard_is_enough_in_expectation()["expected_per_shard"] < 2.0

    def test_an_empty_shard_count_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            scatter_gather_is_exact(shard_counts=())

    def test_searching_no_shards_is_refused(self):
        with pytest.raises(ConfigError, match="no shards to search"):
            scatter_gather(torch.randn(2, 16), [], k=5)

    def test_an_empty_merge_is_refused(self):
        assert an_empty_merge_is_refused()

    def test_a_merged_capacity_below_k_is_refused(self):
        assert a_merged_capacity_below_k_is_refused()

    def test_a_zero_fetch_is_refused(self):
        assert a_zero_fetch_is_refused()

    def test_merging_more_than_the_candidates_is_refused(self):
        with pytest.raises(ConfigError, match="merged candidates"):
            merge([(torch.zeros(2, 3, dtype=torch.long), torch.zeros(2, 3))], k=10)


class TestFetch:
    def test_a_small_shard_count_concentrates_the_neighbours(self):
        rows = [row["recall"] for row in but_a_small_shard_count_concentrates_the_neighbours()]
        assert rows == sorted(rows)

    def test_and_more_shards_removes_the_loss(self):
        rows = {
            row["shards"]: row for row in but_a_small_shard_count_concentrates_the_neighbours()
        }
        assert rows[32]["recall"] == 1.0
        assert rows[4]["recall"] < 1.0

    def test_the_fetch_saturates_below_k(self):
        rows = {row["fetch"]: row for row in over_fetching_removes_the_variance()}
        assert rows[5]["recall"] == 1.0

    def test_so_fetching_k_from_each_shard_is_already_wasteful(self):
        rows = {row["fetch"]: row for row in over_fetching_removes_the_variance()}
        assert rows[10]["recall"] == rows[5]["recall"]

    def test_but_fetching_two_is_not_enough(self):
        rows = {row["fetch"]: row for row in over_fetching_removes_the_variance()}
        assert rows[2]["recall"] < 0.95

    def test_the_scan_cost_does_not_depend_on_the_fetch(self):
        rows = over_fetching_removes_the_variance()
        assert len({row["distances_per_query"] for row in rows}) == 1

    def test_an_empty_fetch_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            over_fetching_removes_the_variance(fetches=())

    def test_an_empty_shard_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            but_a_small_shard_count_concentrates_the_neighbours(counts=())


class TestRouting:
    def test_visiting_fewer_shards_costs_fewer_distances(self):
        rows = [
            row["distances_per_query"] for row in routing_reduces_the_work_and_can_be_wrong()
        ]
        assert rows == sorted(rows)

    def test_routing_works_on_structured_data(self):
        rows = {row["visit"]: row for row in routing_reduces_the_work_and_can_be_wrong()}
        assert rows[1]["recall"] == 1.0

    def test_and_much_less_well_without_structure(self):
        assert routing_is_worth_it_only_with_structure()["structure_helps"]

    def test_by_a_factor_of_three(self):
        assert routing_is_worth_it_only_with_structure()["ratio"] > 2.0

    def test_at_identical_cost(self):
        result = routing_is_worth_it_only_with_structure()
        assert (
            abs(
                result["clustered_distances_per_query"] - result["gaussian_distances_per_query"]
            )
            < 20
        )

    def test_the_wrong_answer_is_well_formed(self):
        assert a_query_visiting_the_wrong_shard_gets_a_clean_answer()["result_is_well_formed"]

    def test_with_correct_scores(self):
        assert a_query_visiting_the_wrong_shard_gets_a_clean_answer()["scores_are_correct"]

    def test_and_correct_ordering(self):
        assert a_query_visiting_the_wrong_shard_gets_a_clean_answer()["ordered"]

    def test_while_being_mostly_wrong(self):
        assert a_query_visiting_the_wrong_shard_gets_a_clean_answer()["recall"] < 0.5

    def test_visiting_more_shards_than_exist_is_refused(self):
        assert visiting_more_shards_than_exist_is_refused()

    def test_an_empty_visit_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            routing_reduces_the_work_and_can_be_wrong(visits=())


class TestLatencyAndMerge:
    def test_the_slowest_shard_sets_the_latency(self):
        assert the_slowest_shard_sets_the_latency()["latency_ratio"] > 1.0

    def test_leaving_part_of_the_cluster_idle(self):
        assert the_slowest_shard_sets_the_latency()["idle_fraction"] > 0.0

    def test_the_merge_grows_with_the_shard_count(self):
        rows = the_merge_cost_is_linear_in_the_shard_count(counts=(2, 8))
        assert rows[2]["merged_candidates"] > rows[0]["merged_candidates"]

    def test_and_stays_small_even_at_a_hundred_shards(self):
        assert sharding_scales_until_the_merge_does_not()["still_small"]

    def test_so_the_coordinator_is_not_the_bottleneck(self):
        assert sharding_scales_until_the_merge_does_not()["share_of_a_million"] < 0.01

    def test_an_empty_merge_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            the_merge_cost_is_linear_in_the_shard_count(counts=())

    def test_a_shard_can_run_a_full_index(self):
        assert a_shard_index_is_the_same_index()["recall"] > 0.5

    def test_and_its_right_partition_count_is_not_the_corpus_one(self):
        result = a_shard_index_is_the_same_index()
        assert result["square_root_of_a_shard"] < result["square_root_of_the_corpus"]

    def test_four_placements_are_compared(self):
        assert len(compare_placements()) == 4

    def test_random_placement_is_always_balanced(self):
        rows = [row for row in compare_placements() if row["placement"] == "random"]
        assert all(row["balance"] == 1.0 for row in rows)

    def test_and_always_exact(self):
        rows = [row for row in compare_placements() if row["placement"] == "random"]
        assert all(row["recall"] == 1.0 for row in rows)

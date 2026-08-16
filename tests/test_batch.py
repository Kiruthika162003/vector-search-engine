from __future__ import annotations

import pytest

from vse.errors import ConfigError
from vse.index.flat import FlatIndex
from vse.serve.batch import (
    BatchStats,
    a_batch_of_one_gains_nothing,
    a_busy_system_can_batch_more,
    a_ragged_final_batch_is_normal,
    a_rank_three_stream_is_refused,
    a_zero_arrival_rate_is_refused,
    a_zero_batch_is_refused,
    an_empty_batch_record_is_refused,
    batched_search,
    batching_changes_no_answers,
    compare_batch_sizes,
    the_corpus_read_is_shared,
    the_distance_count_cannot_see_batching,
    the_last_doubling_buys_almost_nothing,
    the_optimum_batch_depends_on_the_traffic,
    the_recall_column_never_moves,
    the_saving_is_in_memory_not_in_arithmetic,
    the_shared_cost_falls_as_one_over_the_batch,
    throughput_saturates,
    waiting_costs_latency,
)
from vse.vectors.dataset import gaussian, held_out


class TestFreeOptimisation:
    def test_the_batch_size_changes_no_answers(self):
        assert all(row["recall"] == 1.0 for row in batching_changes_no_answers())

    def test_at_every_size_tried(self):
        assert len(batching_changes_no_answers()) == 4

    def test_the_recall_column_never_moves(self):
        assert the_recall_column_never_moves()["identical"]

    def test_and_neither_does_the_distance_count(self):
        assert the_recall_column_never_moves()["distances_identical"]

    def test_a_batch_larger_than_the_stream_is_one_batch(self):
        rows = {row["batch"]: row for row in batching_changes_no_answers()}
        assert rows[512]["batches"] == 1

    def test_an_empty_batch_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            batching_changes_no_answers(batches=())


class TestSharedCost:
    def test_the_corpus_read_is_shared_across_the_batch(self):
        rows = [row["corpus_bytes_per_query"] for row in the_corpus_read_is_shared()]
        assert rows == sorted(rows, reverse=True)

    def test_falling_as_one_over_the_batch_size(self):
        assert the_shared_cost_falls_as_one_over_the_batch()["matches"]

    def test_by_exactly_sixty_four_at_a_batch_of_sixty_four(self):
        result = the_shared_cost_falls_as_one_over_the_batch()
        assert abs(result["ratio"] - result["predicted_ratio"]) < 4

    def test_a_batch_of_one_gains_nothing(self):
        assert a_batch_of_one_gains_nothing()["gain_at_one"] == 1.0

    def test_where_a_batch_of_eight_gains_eight(self):
        assert abs(a_batch_of_one_gains_nothing()["gain_at_eight"] - 8.0) < 0.1

    def test_the_saving_is_not_arithmetic(self):
        assert the_saving_is_in_memory_not_in_arithmetic()["distances_are_identical"]

    def test_the_distance_count_cannot_see_batching(self):
        assert the_distance_count_cannot_see_batching()["identical"]

    def test_while_the_bytes_read_fall_by_a_hundred_and_twenty_eight(self):
        result = the_distance_count_cannot_see_batching()
        ratio = (
            result["corpus_bytes_at_one"] / result["corpus_bytes_at_a_hundred_and_twenty_eight"]
        )
        assert ratio > 100

    def test_an_empty_shared_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            the_corpus_read_is_shared(batches=())


class TestQueueing:
    def test_waiting_costs_latency(self):
        assert waiting_costs_latency()["worst_wait_ms"] > 0

    def test_the_mean_wait_is_half_the_worst(self):
        result = waiting_costs_latency()
        assert abs(result["mean_wait_ms"] * 2 - result["worst_wait_ms"]) < 1e-6

    def test_a_batch_of_one_never_waits(self):
        assert waiting_costs_latency(batch=1)["worst_wait_ms"] == 0.0

    def test_a_faster_arrival_rate_waits_less(self):
        assert (
            waiting_costs_latency(arrival_rate=5000.0)["worst_wait_ms"]
            < waiting_costs_latency(arrival_rate=500.0)["worst_wait_ms"]
        )

    def test_a_zero_arrival_rate_is_refused(self):
        assert a_zero_arrival_rate_is_refused()

    def test_a_zero_batch_wait_is_refused(self):
        with pytest.raises(ConfigError, match="not a batch"):
            waiting_costs_latency(batch=0)

    def test_the_waiting_is_independent_of_the_index(self):
        assert waiting_costs_latency()["waiting_is_free_of_the_index"]


class TestOptimum:
    def test_a_busy_system_can_batch_more(self):
        assert a_busy_system_can_batch_more()["grows_with_traffic"]

    def test_by_two_orders_of_magnitude(self):
        result = a_busy_system_can_batch_more()
        assert result["at_ten_thousand"] > result["at_ten"] * 50

    def test_and_the_throughput_follows(self):
        result = a_busy_system_can_batch_more()
        assert result["throughput_at_ten_thousand"] > result["throughput_at_ten"] * 10

    def test_the_largest_batch_rises_with_the_arrival_rate(self):
        rows = [row["largest_batch"] for row in the_optimum_batch_depends_on_the_traffic()]
        assert rows == sorted(rows)

    def test_every_configuration_stays_inside_the_budget(self):
        rows = the_optimum_batch_depends_on_the_traffic(budget_ms=20.0)
        assert all(row["wait_ms"] + row["service_ms"] <= 20.0 + 1e-6 for row in rows)

    def test_an_empty_rate_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            the_optimum_batch_depends_on_the_traffic(rates=())

    def test_a_zero_budget_is_refused(self):
        with pytest.raises(ConfigError, match="latency budget"):
            the_optimum_batch_depends_on_the_traffic(budget_ms=0.0)


class TestSaturation:
    def test_throughput_rises_with_the_batch(self):
        rows = [row["throughput_per_second"] for row in throughput_saturates()]
        assert rows == sorted(rows)

    def test_and_the_per_query_time_falls(self):
        rows = [row["per_query_ms"] for row in throughput_saturates()]
        assert rows == sorted(rows, reverse=True)

    def test_the_first_doublings_are_worth_far_more(self):
        result = the_last_doubling_buys_almost_nothing()
        assert result["one_to_sixteen"] > result["two_fifty_six_to_a_thousand"] * 5

    def test_and_it_saturates(self):
        assert the_last_doubling_buys_almost_nothing()["saturates"]

    def test_an_empty_saturation_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            throughput_saturates(sizes=())


class TestMechanics:
    def test_a_ragged_final_batch_is_normal(self):
        assert a_ragged_final_batch_is_normal()["batches"] == 4

    def test_and_lowers_the_mean_batch(self):
        assert a_ragged_final_batch_is_normal()["mean_is_below_configured"]

    def test_so_the_mean_is_the_number_to_report(self):
        result = a_ragged_final_batch_is_normal()
        assert result["mean_batch"] < result["configured_batch"]

    def test_a_zero_batch_is_refused(self):
        assert a_zero_batch_is_refused()

    def test_a_rank_three_stream_is_refused(self):
        assert a_rank_three_stream_is_refused()

    def test_an_empty_batch_record_is_refused(self):
        assert an_empty_batch_record_is_refused()

    def test_an_empty_stat_divides_by_nothing_safely(self):
        assert BatchStats().mean_batch == 0.0

    def test_and_reports_no_shared_cost(self):
        assert BatchStats().corpus_bytes_per_query == 0.0

    def test_it_serialises(self):
        stats = BatchStats()
        stats.record(16, 4096, 100.0)
        assert stats.as_dict()["mean_batch"] == 16.0

    def test_the_whole_stream_is_answered(self):
        corpus = gaussian(count=512, dimension=16)
        searched, probes = held_out_probes(corpus)
        index = FlatIndex(16)
        index.build(searched)
        found, _, _ = batched_search(index, probes, k=5, batch=7)
        assert found.queries == int(probes.shape[0])

    def test_three_batch_sizes_are_compared(self):
        assert len(compare_batch_sizes()) == 3


def held_out_probes(corpus):
    """Split a corpus into a searchable part and a query stream."""
    searched, probes = held_out(corpus, count=100)
    return searched.vectors, probes

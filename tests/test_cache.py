from __future__ import annotations

import pytest
import torch

from vse.errors import ConfigError, DataError
from vse.index.flat import FlatIndex
from vse.serve.cache import (
    CacheStats,
    ExactCache,
    NearCache,
    a_batch_key_is_refused,
    a_hit_returns_a_stale_answer,
    a_near_cache_hits_where_the_exact_one_cannot,
    a_zero_nudge_stream_is_refused,
    a_zero_radius_is_refused,
    an_exact_cache_works_on_a_replayed_log,
    an_exact_repeat_stream_is_refused_below_its_unique_count,
    and_never_on_a_model_produced_stream,
    compare_cache_designs,
    eviction_keeps_the_cache_bounded,
    high_dimensions_make_reuse_safer,
    perturbed_stream,
    radius_sweep,
    replayed_stream,
    run_exact_cache,
    run_near_cache,
    the_cache_sizes_itself,
    the_capacity_knob_is_nearly_inert,
    the_lookup_is_itself_a_search,
    the_loss_falls_to_the_noise_floor,
    the_radius_is_a_cliff_not_a_slope,
)
from vse.vectors.dataset import gaussian, held_out
from vse.vectors.exact import Neighbours


def an_answer(k: int = 3) -> Neighbours:
    """A stand in result for tests that only care about storage."""
    return Neighbours(identifiers=torch.arange(k).reshape(1, k), scores=torch.zeros(1, k))


class TestExactKeying:
    def test_it_works_on_a_replayed_log(self):
        assert an_exact_cache_works_on_a_replayed_log()["hit_rate"] > 0.8

    def test_and_every_hit_agrees_with_the_index(self):
        assert an_exact_cache_works_on_a_replayed_log()["hits_are_exact"]

    def test_exactly_and_not_approximately(self):
        assert an_exact_cache_works_on_a_replayed_log()["agreement_with_the_index"] == 1.0

    def test_and_never_on_a_model_produced_stream(self):
        assert and_never_on_a_model_produced_stream()["hit_rate"] == 0.0

    def test_every_query_misses(self):
        result = and_never_on_a_model_produced_stream()
        assert result["hits"] == 0
        assert result["misses"] == 256

    def test_a_key_round_trips(self):
        cache = ExactCache()
        query = torch.randn(1, 8)
        assert cache.key(query) == cache.key(query.clone())

    def test_two_different_vectors_key_differently(self):
        cache = ExactCache()
        assert cache.key(torch.zeros(1, 8)) != cache.key(torch.ones(1, 8))

    def test_a_batch_key_is_refused(self):
        assert a_batch_key_is_refused()

    def test_a_rank_one_key_is_refused(self):
        with pytest.raises(DataError, match="one query"):
            ExactCache().key(torch.randn(8))

    def test_a_miss_returns_nothing(self):
        assert ExactCache().get(torch.randn(1, 8)) is None

    def test_a_stored_answer_comes_back(self):
        corpus = gaussian(count=256, dimension=8)
        index = FlatIndex(8)
        index.build(corpus.vectors)
        query = torch.randn(1, 8)
        found, _ = index.search(query, k=5)
        cache = ExactCache()
        cache.put(query, found)
        assert cache.get(query) is not None

    def test_it_evicts_at_capacity(self):
        cache = ExactCache(capacity=4)
        for row in range(16):
            cache.put(torch.full((1, 4), float(row)), an_answer())
        assert len(cache.entries) == 4


class TestNearKeying:
    def test_a_near_cache_hits_where_the_exact_one_cannot(self):
        assert a_near_cache_hits_where_the_exact_one_cannot()["recovers"]

    def test_and_recovers_most_of_the_hit_rate(self):
        assert a_near_cache_hits_where_the_exact_one_cannot()["near_hit_rate"] > 0.8

    def test_from_nothing_at_all(self):
        assert a_near_cache_hits_where_the_exact_one_cannot()["exact_hit_rate"] == 0.0

    def test_an_empty_cache_misses(self):
        found, cost = NearCache(radius=1.0).get(torch.randn(1, 8))
        assert found is None
        assert cost == 0.0

    def test_a_zero_radius_is_refused(self):
        assert a_zero_radius_is_refused()

    def test_a_negative_radius_is_refused(self):
        with pytest.raises(ConfigError, match="reuses nothing"):
            NearCache(radius=-1.0)

    def test_a_query_within_the_radius_hits(self):
        cache = NearCache(radius=1.0)
        cache.put(torch.zeros(1, 4), an_answer())
        assert cache.get(torch.full((1, 4), 0.1))[0] is not None

    def test_and_one_outside_it_misses(self):
        corpus = gaussian(count=64, dimension=4)
        index = FlatIndex(4)
        index.build(corpus.vectors)
        answer, _ = index.search(torch.zeros(1, 4), k=3)
        cache = NearCache(radius=0.5)
        cache.put(torch.zeros(1, 4), answer)
        assert cache.get(torch.full((1, 4), 10.0))[0] is None

    def test_the_lookup_cost_is_the_cache_size(self):
        corpus = gaussian(count=64, dimension=4)
        index = FlatIndex(4)
        index.build(corpus.vectors)
        cache = NearCache(radius=0.5)
        for row in range(5):
            answer, _ = index.search(torch.full((1, 4), float(row) * 10.0), k=3)
            cache.put(torch.full((1, 4), float(row) * 10.0), answer)
        assert cache.get(torch.full((1, 4), 1000.0))[1] == 5.0

    def test_it_grows_with_stored_answers(self):
        cache = NearCache(radius=1.0)
        for row in range(4):
            cache.put(torch.full((1, 4), float(row)), an_answer())
        assert cache.size == 4


class TestTheRecursion:
    def test_the_lookup_is_itself_a_search(self):
        assert the_lookup_is_itself_a_search()["lookup_cost"] > 0

    def test_and_it_still_pays(self):
        assert the_lookup_is_itself_a_search()["worth_it"]

    def test_by_a_wide_margin(self):
        result = the_lookup_is_itself_a_search()
        assert result["saved"] > result["lookup_cost"] * 10

    def test_because_the_cache_stops_growing(self):
        assert the_cache_sizes_itself()["larger_is_the_same"]

    def test_a_cache_smaller_than_the_traffic_thrashes(self):
        assert the_cache_sizes_itself()["too_small_is_worse"]

    def test_the_capacity_knob_is_inert_above_the_threshold(self):
        rows = {row["capacity"]: row for row in the_capacity_knob_is_nearly_inert()}
        assert rows[32]["lookup_cost"] == rows[1024]["lookup_cost"]

    def test_and_the_hit_rate_too(self):
        rows = {row["capacity"]: row for row in the_capacity_knob_is_nearly_inert()}
        assert rows[128]["hit_rate"] == rows[1024]["hit_rate"]

    def test_below_it_the_hit_rate_collapses(self):
        rows = {row["capacity"]: row for row in the_capacity_knob_is_nearly_inert()}
        assert rows[8]["hit_rate"] < rows[32]["hit_rate"] * 0.5

    def test_an_empty_capacity_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            the_capacity_knob_is_nearly_inert(capacities=())


class TestStaleness:
    def test_a_hit_returns_somebody_elses_answer(self):
        assert a_hit_returns_a_stale_answer()["hit_rate"] > 0.5

    def test_and_it_costs_nothing_measurable(self):
        assert a_hit_returns_a_stale_answer()["inside_the_index_error"]

    def test_the_cached_recall_matches_the_uncached_one(self):
        result = a_hit_returns_a_stale_answer()
        assert abs(result["index_recall"] - result["cached_recall"]) < 0.05

    def test_a_tight_radius_costs_nothing_and_gains_nothing(self):
        rows = {row["radius_share"]: row for row in radius_sweep()}
        assert rows[0.05]["hit_rate"] < 0.1

    def test_the_plateau_is_flat(self):
        assert the_radius_is_a_cliff_not_a_slope()["the_plateau_is_flat"]

    def test_and_the_cliff_is_steep(self):
        assert the_radius_is_a_cliff_not_a_slope()["the_cliff_is_steep"]

    def test_the_hit_rate_barely_moves_across_the_cliff(self):
        result = the_radius_is_a_cliff_not_a_slope()
        assert result["beyond_the_cliff_hit_rate"] - result["plateau_hit_rate"] < 0.15

    def test_while_the_recall_halves(self):
        result = the_radius_is_a_cliff_not_a_slope()
        assert result["beyond_the_cliff_recall"] < result["plateau_recall"] * 0.6

    def test_an_empty_radius_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            radius_sweep(shares=())


class TestWidth:
    def test_the_loss_falls_as_the_corpus_widens(self):
        assert the_loss_falls_to_the_noise_floor()["falls"]

    def test_and_reaches_the_floor_by_a_hundred_dimensions(self):
        assert the_loss_falls_to_the_noise_floor()["at_the_floor_by_a_hundred"]

    def test_the_hit_rate_does_not_depend_on_the_width(self):
        assert the_loss_falls_to_the_noise_floor()["hit_rate_is_flat"]

    def test_the_narrow_case_is_the_one_that_loses(self):
        rows = {row["dimension"]: row for row in high_dimensions_make_reuse_safer()}
        assert rows[8]["loss"] > 0.02

    def test_four_widths_are_measured(self):
        assert len(high_dimensions_make_reuse_safer()) == 4

    def test_an_empty_width_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            high_dimensions_make_reuse_safer(dimensions=())


class TestStreams:
    def test_a_replayed_stream_repeats(self):
        corpus = gaussian(count=512, dimension=8)
        stream = replayed_stream(corpus, count=128, unique=8)
        assert int(torch.unique(stream, dim=0).shape[0]) <= 8

    def test_and_is_the_right_length(self):
        corpus = gaussian(count=512, dimension=8)
        assert int(replayed_stream(corpus, count=128, unique=8).shape[0]) == 128

    def test_a_perturbed_stream_does_not_repeat(self):
        corpus = gaussian(count=512, dimension=8)
        stream = perturbed_stream(corpus, count=128, unique=8)
        assert int(torch.unique(stream, dim=0).shape[0]) == 128

    def test_but_stays_close_to_its_base(self):
        corpus = gaussian(count=512, dimension=8)
        base = replayed_stream(corpus, count=128, unique=8)
        moved = perturbed_stream(corpus, count=128, unique=8)
        assert float((moved - base).norm(dim=1).max()) < float(base.norm(dim=1).mean())

    def test_a_stream_shorter_than_its_unique_count_is_refused(self):
        assert an_exact_repeat_stream_is_refused_below_its_unique_count()

    def test_a_zero_unique_count_is_refused(self):
        with pytest.raises(ConfigError, match="not a stream"):
            replayed_stream(gaussian(count=128, dimension=8), unique=0)

    def test_a_zero_nudge_is_refused(self):
        assert a_zero_nudge_stream_is_refused()

    def test_a_negative_nudge_is_refused(self):
        with pytest.raises(ConfigError, match="exact repeat"):
            perturbed_stream(gaussian(count=128, dimension=8), nudge=-0.1)


class TestRuns:
    def test_an_exact_run_answers_the_whole_stream(self):
        corpus = gaussian(count=512, dimension=8)
        searched, _ = held_out(corpus, count=16)
        index = FlatIndex(8)
        index.build(searched.vectors)
        stream = replayed_stream(corpus, count=64, unique=16)
        found, stats = run_exact_cache(index, stream, k=5)
        assert int(found.identifiers.shape[0]) == 64
        assert stats.queries == 64

    def test_a_near_run_answers_the_whole_stream(self):
        corpus = gaussian(count=512, dimension=8)
        searched, _ = held_out(corpus, count=16)
        index = FlatIndex(8)
        index.build(searched.vectors)
        stream = perturbed_stream(corpus, count=64, unique=16)
        found, stats = run_near_cache(index, stream, radius=1.0, k=5)
        assert int(found.identifiers.shape[0]) == 64
        assert stats.queries == 64

    def test_the_shape_is_k_wide(self):
        corpus = gaussian(count=512, dimension=8)
        searched, _ = held_out(corpus, count=16)
        index = FlatIndex(8)
        index.build(searched.vectors)
        stream = replayed_stream(corpus, count=32, unique=16)
        found, _ = run_exact_cache(index, stream, k=7)
        assert int(found.identifiers.shape[1]) == 7

    def test_four_rows_compare_the_designs(self):
        assert len(compare_cache_designs()) == 4

    def test_the_exact_cache_is_useless_on_perturbed_traffic(self):
        rows = {(row["traffic"], row["cache"]): row for row in compare_cache_designs()}
        assert rows[("perturbed", "exact")]["hit_rate"] == 0.0

    def test_where_the_near_cache_works(self):
        rows = {(row["traffic"], row["cache"]): row for row in compare_cache_designs()}
        assert rows[("perturbed", "near")]["hit_rate"] > 0.8

    def test_and_both_work_on_a_replay(self):
        rows = {(row["traffic"], row["cache"]): row for row in compare_cache_designs()}
        assert rows[("replayed", "exact")]["hit_rate"] > 0.8
        assert rows[("replayed", "near")]["hit_rate"] > 0.8

    def test_eviction_keeps_the_cache_bounded(self):
        assert eviction_keeps_the_cache_bounded()["bounded"]

    def test_at_exactly_the_configured_capacity(self):
        assert eviction_keeps_the_cache_bounded()["size"] == 16


class TestStats:
    def test_an_empty_stat_has_no_hit_rate(self):
        assert CacheStats().hit_rate == 0.0

    def test_and_no_queries(self):
        assert CacheStats().queries == 0

    def test_the_hit_rate_is_hits_over_queries(self):
        assert CacheStats(hits=3, misses=1).hit_rate == 0.75

    def test_the_net_saving_subtracts_the_lookup(self):
        stats = CacheStats(saved_distances=100.0, lookup_distances=30.0)
        assert stats.net_saving == 70.0

    def test_and_can_be_negative(self):
        stats = CacheStats(saved_distances=10.0, lookup_distances=30.0)
        assert stats.net_saving < 0

    def test_it_serialises(self):
        stats = CacheStats(hits=1, misses=1, saved_distances=8.0)
        assert stats.as_dict()["hit_rate"] == 0.5

from __future__ import annotations

import pytest
import torch

from vse.errors import ConfigError, DataError, IndexStateError
from vse.index.base import (
    Quality,
    SearchStats,
    a_discount_lowers_the_charge,
    a_negative_charge_is_refused,
    a_quality_with_no_cost_reports_no_speedup,
    an_empty_stat_reports_nothing,
    an_unbuilt_index_refuses_to_search,
    compare,
    default_corpus,
    evaluate,
    evaluate_on,
    stats_add_up,
    the_speedup_is_the_corpus_over_the_count,
    the_three_numbers_are_independent,
)
from vse.index.flat import FlatIndex
from vse.vectors.dataset import gaussian, held_out
from vse.vectors.exact import search as exact_search


class TestStats:
    def test_charging_records_distances(self):
        stats = SearchStats(queries=2)
        stats.charge(100)
        assert stats.distances_per_query == 50.0

    def test_a_discount_lowers_the_charge(self):
        assert a_discount_lowers_the_charge()["ratio"] == 4.0

    def test_a_negative_charge_is_refused(self):
        assert a_negative_charge_is_refused()

    def test_a_negative_weight_is_refused(self):
        with pytest.raises(ConfigError, match="not a weight"):
            SearchStats().charge(10, weight=-1.0)

    def test_a_discount_above_one_is_refused(self):
        with pytest.raises(ConfigError, match="not a discount"):
            a_discount_lowers_the_charge(weight=2.0)

    def test_visiting_records_candidates(self):
        stats = SearchStats()
        stats.visit(7)
        assert stats.candidates == 7

    def test_a_negative_visit_is_refused(self):
        with pytest.raises(ConfigError, match="cannot visit"):
            SearchStats().visit(-1)

    def test_hopping_records_steps(self):
        stats = SearchStats()
        stats.hop(3)
        assert stats.hops == 3

    def test_a_negative_hop_is_refused(self):
        with pytest.raises(ConfigError, match="cannot hop"):
            SearchStats().hop(-1)

    def test_merging_adds_up(self):
        assert stats_add_up()["matches_the_mean"]

    def test_including_the_query_count(self):
        assert stats_add_up()["queries"] == 10

    def test_an_empty_stat_divides_by_nothing_safely(self):
        assert an_empty_stat_reports_nothing()["per_query"] == 0.0

    def test_it_serialises(self):
        stats = SearchStats(queries=4)
        stats.charge(400)
        assert stats.as_dict()["distances_per_query"] == 100.0


class TestQuality:
    def test_the_speedup_is_the_corpus_over_the_count(self):
        assert the_speedup_is_the_corpus_over_the_count()["speedup"] == 10.0

    def test_and_the_scanned_share_is_its_reciprocal(self):
        result = the_speedup_is_the_corpus_over_the_count()
        assert abs(result["scanned"] * result["speedup"] - 1.0) < 1e-6

    def test_a_quality_with_no_cost_reports_zero_not_infinity(self):
        assert a_quality_with_no_cost_reports_no_speedup()["speedup"] == 0.0

    def test_an_exhaustive_index_has_no_speedup(self):
        assert the_three_numbers_are_independent()["exhaustive"]["speedup"] == 1.0

    def test_and_a_lazy_one_has_no_recall(self):
        assert the_three_numbers_are_independent()["lazy"]["recall"] < 0.1

    def test_so_two_of_three_can_always_be_made_to_look_good(self):
        result = the_three_numbers_are_independent()
        assert result["lazy"]["speedup"] > result["exhaustive"]["speedup"]

    def test_a_comparison_sorts_by_recall(self):
        rows = compare(
            [
                Quality(index="low", recall=0.5, gap=0.1),
                Quality(index="high", recall=0.9, gap=0.2),
            ]
        )
        assert rows[0]["index"] == "high"

    def test_an_empty_comparison_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to compare"):
            compare([])

    def test_a_quality_with_no_corpus_scans_nothing(self):
        assert Quality(index="x", recall=1.0, gap=0.0).scanned == 0.0


class TestInterface:
    def test_an_unbuilt_index_refuses_to_search(self):
        assert an_unbuilt_index_refuses_to_search(FlatIndex(8))

    def test_a_zero_dimension_index_is_refused(self):
        with pytest.raises(ConfigError, match="not a width"):
            FlatIndex(0)

    def test_an_unknown_metric_is_refused(self):
        with pytest.raises(ConfigError, match="unknown metric"):
            FlatIndex(8, metric="manhattan")

    def test_the_name_comes_from_the_class(self):
        assert FlatIndex(8).name == "flat"

    def test_an_unbuilt_index_reports_no_size(self):
        assert FlatIndex(8).as_dict()["size"] == 0

    def test_a_built_one_reports_its_size(self):
        index = FlatIndex(8)
        index.build(gaussian(count=64, dimension=8).vectors)
        assert index.as_dict()["size"] == 64

    def test_building_with_the_wrong_width_is_refused(self):
        with pytest.raises(DataError, match="wide"):
            FlatIndex(8).build(torch.randn(16, 4))

    def test_building_with_nothing_is_refused(self):
        with pytest.raises(DataError, match="no vectors"):
            FlatIndex(8).build(torch.zeros(0, 8))

    def test_building_with_integers_is_refused(self):
        with pytest.raises(DataError, match="floating point"):
            FlatIndex(8).build(torch.zeros(16, 8, dtype=torch.int64))

    def test_a_rank_three_batch_is_refused(self):
        index = FlatIndex(8)
        index.build(gaussian(count=64, dimension=8).vectors)
        with pytest.raises(DataError, match="matrix of rows"):
            index.search(torch.randn(2, 3, 8), k=1)

    def test_asking_for_more_than_the_index_holds_is_refused(self):
        index = FlatIndex(8)
        index.build(gaussian(count=16, dimension=8).vectors)
        with pytest.raises(ConfigError, match="from 16 vectors"):
            index.search(torch.randn(2, 8), k=32)

    def test_searching_an_unbuilt_index_says_so(self):
        with pytest.raises(IndexStateError, match="has not been built"):
            FlatIndex(8).search(torch.randn(2, 8), k=1)


class TestEvaluation:
    def test_the_baseline_scores_perfectly_against_itself(self):
        corpus = gaussian(count=512, dimension=16)
        searched, probes = held_out(corpus, count=32)
        index = FlatIndex(16)
        index.build(searched.vectors)
        quality = evaluate(index, searched.vectors, probes, k=10)
        assert quality.recall == 1.0
        assert quality.gap == 0.0

    def test_evaluating_on_a_corpus_holds_queries_out(self):
        corpus = gaussian(count=512, dimension=16)
        quality = evaluate_on(FlatIndex(16), corpus, k=10, queries=32)
        assert quality.corpus_size == 480

    def test_the_default_corpus_is_the_hard_one(self):
        assert default_corpus().intrinsic == default_corpus().dimension

    def test_a_supplied_truth_is_used_rather_than_recomputed(self):
        corpus = gaussian(count=256, dimension=8)
        searched, probes = held_out(corpus, count=16)
        index = FlatIndex(8)
        index.build(searched.vectors)
        truth = exact_search(probes, searched.vectors, k=5)
        assert evaluate(index, searched.vectors, probes, k=5, truth=truth).recall == 1.0

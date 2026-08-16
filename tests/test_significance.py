from __future__ import annotations

import pytest
import torch

from vse.errors import ConfigError, DataError
from vse.eval.significance import (
    Estimate,
    a_hundred_queries_gives_two_points,
    a_mismatched_pair_is_refused,
    a_one_point_difference_needs_thousands_of_queries,
    a_paired_comparison_is_much_more_sensitive,
    an_estimate_of_one_sample_is_refused,
    an_interval_widens_with_the_error,
    comparing_a_configuration_with_itself_is_refused,
    estimate,
    how_many_queries_to_detect_a_gap,
    pairing_helps_at_a_small_gap_and_hurts_at_a_large_one,
    pairing_shrinks_the_error,
    per_query_recall,
    recall_at_ten_is_not_ten_coin_flips,
    the_error_falls_as_one_over_root_n,
    the_per_query_recall_is_not_bimodal,
    the_seed_moves_the_answer_by_more_than_the_error,
    the_tail_is_much_worse_than_the_mean,
    the_worst_queries_are_the_ones_to_report,
    two_estimates_far_apart_do_not_overlap,
    two_indexes_that_look_different_may_not_be,
)
from vse.vectors.exact import Neighbours


class TestEstimates:
    def test_an_estimate_carries_its_interval(self):
        result = Estimate(mean=0.5, error=0.01, samples=100)
        low, high = result.interval
        assert low < 0.5 < high

    def test_the_interval_is_two_errors_each_side(self):
        assert an_interval_widens_with_the_error()["two_standard_errors_each_side"]

    def test_a_larger_error_gives_a_wider_interval(self):
        assert an_interval_widens_with_the_error()["wider"]

    def test_two_estimates_far_apart_do_not_overlap(self):
        assert two_estimates_far_apart_do_not_overlap()["separated"]

    def test_and_two_close_ones_do(self):
        assert an_interval_widens_with_the_error()["they_overlap"]

    def test_an_estimate_serialises(self):
        row = Estimate(mean=0.5, error=0.01, samples=100).as_dict()
        assert row["samples"] == 100 and row["mean"] == 0.5

    def test_the_mean_of_a_constant_sample_has_no_error(self):
        assert estimate(torch.full((10,), 0.5)).error == 0.0

    def test_an_estimate_of_one_sample_is_refused(self):
        assert an_estimate_of_one_sample_is_refused()

    def test_an_empty_sample_is_refused(self):
        with pytest.raises(ConfigError, match="at least two samples"):
            estimate(torch.zeros(0))

    def test_the_error_is_the_spread_over_root_n(self):
        values = torch.tensor([0.0, 1.0, 0.0, 1.0])
        result = estimate(values)
        assert abs(result.error - float(values.std(unbiased=True)) / 2.0) < 1e-6


class TestSampleSize:
    def test_the_error_falls_with_the_sample(self):
        rows = [row["error"] for row in the_error_falls_as_one_over_root_n()]
        assert rows == sorted(rows, reverse=True)

    def test_by_roughly_the_square_root(self):
        rows = {row["queries"]: row for row in the_error_falls_as_one_over_root_n()}
        ratio = rows[10]["error"] / rows[1000]["error"]
        assert 5.0 < ratio < 18.0

    def test_a_hundred_queries_gives_a_few_points(self):
        assert a_hundred_queries_gives_two_points()["about_two_points"]

    def test_the_interval_is_several_points_wide(self):
        assert a_hundred_queries_gives_two_points()["width"] > 0.03

    def test_an_empty_size_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            the_error_falls_as_one_over_root_n(sizes=())

    def test_the_means_agree_across_sample_sizes(self):
        rows = [row["mean"] for row in the_error_falls_as_one_over_root_n()]
        assert max(rows) - min(rows) < 0.1


class TestTheDistribution:
    def test_the_binomial_understates_the_error(self):
        assert recall_at_ten_is_not_ten_coin_flips()["binomial_understates"]

    def test_but_only_by_a_fifth(self):
        assert recall_at_ten_is_not_ten_coin_flips()["ratio"] < 1.5

    def test_and_not_by_the_factor_of_three_expected(self):
        assert recall_at_ten_is_not_ten_coin_flips()["ratio"] < 2.0

    def test_the_recall_is_not_bimodal(self):
        assert the_per_query_recall_is_not_bimodal()["most_mass_is_in_the_middle"]

    def test_few_queries_get_nearly_everything(self):
        assert the_per_query_recall_is_not_bimodal()["share_above_nine_tenths"] < 0.1

    def test_and_few_get_nearly_nothing(self):
        assert the_per_query_recall_is_not_bimodal()["share_below_a_tenth"] < 0.2

    def test_the_tail_is_worse_than_the_mean(self):
        assert the_tail_is_much_worse_than_the_mean()["the_tail_is_worse"]

    def test_by_a_substantial_gap(self):
        assert the_tail_is_much_worse_than_the_mean()["gap"] > 0.1

    def test_the_quantiles_rise(self):
        rows = [row["recall"] for row in the_worst_queries_are_the_ones_to_report()]
        assert rows == sorted(rows)

    def test_an_empty_quantile_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            the_worst_queries_are_the_ones_to_report(quantiles=())


class TestPairing:
    def test_the_unpaired_intervals_overlap(self):
        assert a_paired_comparison_is_much_more_sensitive()["unpaired_intervals_overlap"]

    def test_where_the_paired_difference_does_not(self):
        assert a_paired_comparison_is_much_more_sensitive()["paired_difference_in_errors"] > 2.0

    def test_and_by_a_wide_margin(self):
        assert a_paired_comparison_is_much_more_sensitive()["paired_difference_in_errors"] > 5.0

    def test_the_paired_error_is_smaller(self):
        assert a_paired_comparison_is_much_more_sensitive()["paired_is_more_sensitive"]

    def test_pairing_helps_when_the_gap_is_small(self):
        assert pairing_helps_at_a_small_gap_and_hurts_at_a_large_one()["helps_when_close"]

    def test_and_hurts_when_it_is_large(self):
        assert pairing_helps_at_a_small_gap_and_hurts_at_a_large_one()["hurts_when_far"]

    def test_the_shrink_falls_with_the_gap(self):
        rows = [
            row["shrink"] for row in pairing_shrinks_the_error() if row["shrink"] is not None
        ]
        assert rows == sorted(rows, reverse=True)

    def test_the_baseline_has_no_paired_error(self):
        rows = pairing_shrinks_the_error()
        assert rows[0]["paired_error"] is None

    def test_a_sweep_of_one_setting_is_refused(self):
        with pytest.raises(ConfigError, match="at least two settings"):
            pairing_shrinks_the_error(probes=(4,))

    def test_comparing_a_configuration_with_itself_is_refused(self):
        assert comparing_a_configuration_with_itself_is_refused()


class TestPlanning:
    def test_a_one_point_gap_needs_thousands(self):
        assert a_one_point_difference_needs_thousands_of_queries()["one_point_needs_thousands"]

    def test_and_a_ten_point_one_needs_dozens(self):
        assert a_one_point_difference_needs_thousands_of_queries()["ten_points_needs_dozens"]

    def test_the_requirement_falls_with_the_gap(self):
        rows = [row["unpaired_queries"] for row in how_many_queries_to_detect_a_gap()]
        assert rows == sorted(rows, reverse=True)

    def test_by_the_square_of_the_ratio(self):
        rows = {row["gap"]: row for row in how_many_queries_to_detect_a_gap()}
        assert abs(rows[0.01]["unpaired_queries"] / rows[0.02]["unpaired_queries"] - 4.0) < 0.2

    def test_an_empty_gap_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            how_many_queries_to_detect_a_gap(gaps=())


class TestReadingOtherModules:
    def test_an_unmatched_comparison_is_flagged(self):
        assert two_indexes_that_look_different_may_not_be()["but_the_costs_are_not_matched"]

    def test_the_intervals_are_reported(self):
        result = two_indexes_that_look_different_may_not_be()
        assert "low" in result["ivf"] and "high" in result["forest"]

    def test_the_paired_difference_is_reported(self):
        assert "paired_difference" in two_indexes_that_look_different_may_not_be()

    def test_the_seed_matters_as_much_as_the_sample(self):
        assert the_seed_moves_the_answer_by_more_than_the_error()["comparable"]

    def test_five_seeds_are_measured(self):
        assert the_seed_moves_the_answer_by_more_than_the_error()["seeds"] == 5

    def test_the_means_differ_between_seeds(self):
        means = the_seed_moves_the_answer_by_more_than_the_error()["means"]
        assert len(set(means)) > 1

    def test_a_single_seed_cannot_show_a_spread(self):
        with pytest.raises(ConfigError, match="at least two seeds"):
            the_seed_moves_the_answer_by_more_than_the_error(seeds=(0,))


class TestPerQueryRecall:
    def test_a_perfect_result_scores_one_everywhere(self):
        truth = Neighbours(torch.arange(20).reshape(2, 10), torch.zeros(2, 10))
        assert bool(torch.all(per_query_recall(truth, truth) == 1.0))

    def test_a_disjoint_result_scores_zero(self):
        truth = Neighbours(torch.arange(20).reshape(2, 10), torch.zeros(2, 10))
        found = Neighbours(torch.arange(100, 120).reshape(2, 10), torch.zeros(2, 10))
        assert bool(torch.all(per_query_recall(truth, found) == 0.0))

    def test_a_half_overlap_scores_a_half(self):
        truth = Neighbours(torch.arange(10).reshape(1, 10), torch.zeros(1, 10))
        found = Neighbours(
            torch.cat([torch.arange(5), torch.arange(100, 105)]).reshape(1, 10),
            torch.zeros(1, 10),
        )
        assert float(per_query_recall(truth, found)[0]) == 0.5

    def test_it_returns_one_value_per_query(self):
        truth = Neighbours(torch.arange(50).reshape(5, 10), torch.zeros(5, 10))
        assert int(per_query_recall(truth, truth).numel()) == 5

    def test_a_mismatched_pair_is_refused(self):
        assert a_mismatched_pair_is_refused()

    def test_the_error_names_both_shapes(self):
        truth = Neighbours(torch.zeros(4, 10, dtype=torch.long), torch.zeros(4, 10))
        found = Neighbours(torch.zeros(2, 10, dtype=torch.long), torch.zeros(2, 10))
        with pytest.raises(DataError, match="truth and"):
            per_query_recall(truth, found)

from __future__ import annotations

import pytest
import torch

from vse.errors import ConfigError
from vse.index.base import SearchStats
from vse.serve.limits import (
    Budget,
    Served,
    a_bounded_search_still_returns_k,
    a_budget_costs_recall,
    a_budget_of_nothing_is_refused,
    a_budget_tracks_what_it_allowed,
    a_clustered_corpus_survives_a_deadline_better,
    a_deadline_levels_the_batch_down,
    a_graph_handles_a_budget_better,
    a_negative_budget_is_refused,
    a_negative_charge_is_refused,
    a_result_of_the_wrong_shape_is_refused,
    a_search_with_no_budget_is_refused,
    a_served_result_serialises,
    a_shared_budget_is_unfair,
    a_shared_search_with_no_budget_is_refused,
    an_empty_served_result_divides_safely,
    no_budget_is_ever_exceeded,
    partial_work_inside_a_partition_is_kept,
    the_budget_is_respected,
    the_recall_saturates_where_the_budget_stops_binding,
    the_shared_budget_splits_the_batch_in_two,
    truncating_costs_less_recall_than_work,
)
from vse.vectors.exact import Neighbours


class TestBudgets:
    def test_a_budget_of_nothing_is_refused(self):
        assert a_budget_of_nothing_is_refused()

    def test_a_negative_budget_is_refused(self):
        assert a_negative_budget_is_refused()

    def test_a_negative_charge_is_refused(self):
        assert a_negative_charge_is_refused()

    def test_a_budget_allows_what_it_has(self):
        assert a_budget_tracks_what_it_allowed()["second"] == 40.0

    def test_and_nothing_after_that(self):
        assert a_budget_tracks_what_it_allowed()["third"] == 0.0

    def test_it_never_overspends(self):
        assert a_budget_tracks_what_it_allowed()["never_overspent"]

    def test_a_fresh_budget_is_not_exhausted(self):
        assert not Budget(limit=100.0).exhausted

    def test_and_reports_its_whole_limit_as_remaining(self):
        assert Budget(limit=100.0).remaining == 100.0

    def test_a_spent_budget_is_exhausted(self):
        budget = Budget(limit=10.0)
        budget.charge(10.0)
        assert budget.exhausted

    def test_and_has_nothing_remaining(self):
        budget = Budget(limit=10.0)
        budget.charge(50.0)
        assert budget.remaining == 0.0

    def test_charging_nothing_is_allowed(self):
        budget = Budget(limit=10.0)
        assert budget.charge(0.0) == 0.0

    def test_it_serialises(self):
        budget = Budget(limit=100.0)
        budget.charge(30.0)
        assert budget.as_dict()["remaining"] == 70.0


class TestTheSweep:
    def test_the_recall_rises_with_the_budget(self):
        rows = [row["recall"] for row in a_budget_costs_recall()]
        assert rows == sorted(rows)

    def test_and_the_truncation_falls(self):
        assert the_recall_saturates_where_the_budget_stops_binding()["truncation_falls"]

    def test_a_tight_budget_truncates_everything(self):
        rows = {row["budget"]: row for row in a_budget_costs_recall()}
        assert rows[100]["truncated_share"] == 1.0

    def test_a_loose_one_truncates_nothing(self):
        rows = {row["budget"]: row for row in a_budget_costs_recall()}
        assert rows[1600]["truncated_share"] == 0.0

    def test_the_spend_matches_the_budget(self):
        rows = a_budget_costs_recall()
        assert all(row["spent"] <= row["budget"] + 1e-6 for row in rows)

    def test_an_empty_budget_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            a_budget_costs_recall(budgets=())


class TestTruncation:
    def test_recall_survives_better_than_work(self):
        assert truncating_costs_less_recall_than_work()["recall_survives_better_than_work"]

    def test_half_the_work_keeps_most_of_the_recall(self):
        result = truncating_costs_less_recall_than_work()
        assert result["recall_share"] > result["work_share"]

    def test_partial_work_inside_a_partition_is_kept(self):
        assert partial_work_inside_a_partition_is_kept()["partial_is_better"]

    def test_and_spends_the_whole_budget(self):
        result = partial_work_inside_a_partition_is_kept()
        assert result["partial_spent"] > result["whole_only_spent"]

    def test_a_bounded_search_returns_k(self):
        assert a_bounded_search_still_returns_k()["shape"] == (100, 10)

    def test_with_distinct_identifiers(self):
        assert a_bounded_search_still_returns_k()["distinct"]

    def test_and_sorted_scores(self):
        assert a_bounded_search_still_returns_k()["sorted"]

    def test_even_at_a_budget_that_searches_almost_nothing(self):
        assert a_bounded_search_still_returns_k()["truncated_share"] == 1.0


class TestFairness:
    def test_the_two_designs_have_similar_means(self):
        assert a_shared_budget_is_unfair()["means_are_close"]

    def test_and_spend_the_same(self):
        result = a_shared_budget_is_unfair()
        assert abs(result["fair_spent"] - result["shared_spent"]) < 1.0

    def test_the_shared_budget_leaves_queries_with_nothing(self):
        assert the_shared_budget_splits_the_batch_in_two()["shared_zero_share"] > 0.2

    def test_where_the_fair_one_leaves_none(self):
        assert the_shared_budget_splits_the_batch_in_two()["fair_zero_share"] == 0.0

    def test_the_shared_budget_is_more_spread(self):
        assert the_shared_budget_splits_the_batch_in_two()["shared_is_more_spread"]

    def test_a_deadline_levels_the_batch_down(self):
        assert a_deadline_levels_the_batch_down()["the_best_lose_more"]

    def test_the_worse_served_lose_less(self):
        result = a_deadline_levels_the_batch_down()
        assert result["loss_among_the_worst_half"] < result["loss_among_the_best_half"]

    def test_both_halves_lose_something(self):
        result = a_deadline_levels_the_batch_down()
        assert result["loss_among_the_worst_half"] > 0
        assert result["loss_among_the_best_half"] > 0


class TestStructures:
    def test_a_clustered_corpus_loses_less(self):
        assert a_clustered_corpus_survives_a_deadline_better()["clustered_loses_less"]

    def test_by_an_order_of_magnitude(self):
        result = a_clustered_corpus_survives_a_deadline_better()
        assert result["gaussian_loss"] > result["clustered_loss"] * 5

    def test_a_graph_handles_a_budget_better(self):
        result = a_graph_handles_a_budget_better()
        assert result["graph_recall"] > result["partitioned_recall"]

    def test_for_less_work(self):
        result = a_graph_handles_a_budget_better()
        assert result["graph_spent"] <= result["budget"]

    def test_a_budget_too_small_for_any_beam_is_refused(self):
        with pytest.raises(ConfigError, match="does not allow even the smallest beam"):
            a_graph_handles_a_budget_better(budget=1.0)


class TestGuards:
    def test_no_budget_is_ever_exceeded(self):
        assert no_budget_is_ever_exceeded()["all_within"]

    def test_with_no_overrun_at_all(self):
        assert no_budget_is_ever_exceeded()["worst_overrun"] <= 0.0

    def test_three_budgets_are_checked(self):
        assert len(the_budget_is_respected()) == 3

    def test_an_empty_respect_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            the_budget_is_respected(budgets=())

    def test_a_search_with_no_budget_is_refused(self):
        assert a_search_with_no_budget_is_refused()

    def test_a_shared_search_with_no_budget_is_refused(self):
        assert a_shared_search_with_no_budget_is_refused()

    def test_a_result_of_the_wrong_shape_is_refused(self):
        assert a_result_of_the_wrong_shape_is_refused()


class TestReporting:
    def test_an_empty_result_divides_safely(self):
        assert an_empty_served_result_divides_safely()["safe"]

    def test_and_reports_no_queries(self):
        assert an_empty_served_result_divides_safely()["queries"] == 0

    def test_a_served_result_serialises(self):
        assert a_served_result_serialises()["has_everything"]

    def test_with_the_truncated_share(self):
        assert a_served_result_serialises()["truncated_share"] == 0.25

    def test_and_the_distance_count(self):
        assert a_served_result_serialises()["distances_per_query"] == 100.0

    def test_the_query_count_is_the_two_halves(self):
        served = Served(
            found=Neighbours(torch.zeros(7, 5, dtype=torch.long), torch.zeros(7, 5)),
            stats=SearchStats(queries=7),
            completed=5,
            truncated=2,
        )
        assert served.queries == 7

    def test_a_fully_truncated_batch_reports_one(self):
        served = Served(
            found=Neighbours(torch.zeros(4, 5, dtype=torch.long), torch.zeros(4, 5)),
            stats=SearchStats(queries=4),
            completed=0,
            truncated=4,
        )
        assert served.truncated_share == 1.0

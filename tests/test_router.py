from __future__ import annotations

import pytest
import torch

from vse.errors import ConfigError, DataError
from vse.index.ivf import IVFIndex
from vse.serve.router import (
    Routed,
    Tier,
    _random_signal,
    _setup,
    _spread_signal,
    a_boundary_signal_needs_two_partitions,
    a_negative_share_is_refused,
    a_rank_one_query_batch_is_refused,
    a_routed_result_serialises,
    a_share_of_zero_is_always_cheap_plus_overhead,
    a_share_outside_the_unit_interval_is_refused,
    a_tier_serialises,
    always,
    an_empty_routed_result_divides_safely,
    an_up_front_share_outside_the_interval_is_refused,
    best_score,
    boundary_closeness,
    centrality,
    choose_up_front,
    choosing_up_front_is_cheaper_and_worse,
    compare_every_routing_scheme,
    escalate,
    escalating_beats_a_random_fifth,
    result_spread,
    routing_across_different_structures,
    routing_does_not_beat_tuning_the_cheap_index,
    structure_does_not_make_difficulty_more_predictable,
    the_best_signal_is_the_one_that_costs_a_search,
    the_escalated_queries_are_the_hard_ones,
    the_escalation_share_trades_cost_for_recall,
    the_routed_result_is_well_formed,
    the_signals_are_measured_against_difficulty,
    the_two_fixed_policies_bracket_everything,
)
from vse.vectors.dataset import gaussian
from vse.vectors.exact import Neighbours


class TestSignals:
    def test_four_signals_are_measured(self):
        assert len(the_signals_are_measured_against_difficulty()) == 4

    def test_two_of_them_need_a_search(self):
        rows = the_signals_are_measured_against_difficulty()
        assert sum(1 for row in rows if row["needs_the_cheap_search"]) == 2

    def test_the_paid_signal_is_stronger(self):
        assert the_best_signal_is_the_one_that_costs_a_search()["paid_is_stronger"]

    def test_the_spread_is_the_best_paid_one(self):
        assert the_best_signal_is_the_one_that_costs_a_search()["best_paid_signal"] == (
            "result spread"
        )

    def test_the_spread_correlates_negatively_with_difficulty(self):
        rows = {row["signal"]: row for row in the_signals_are_measured_against_difficulty()}
        assert rows["result spread"]["correlation"] < 0

    def test_centrality_is_nearly_useless(self):
        rows = {row["signal"]: row for row in the_signals_are_measured_against_difficulty()}
        assert abs(rows["centrality"]["correlation"]) < 0.1

    def test_centrality_returns_one_value_per_query(self):
        queries, centres = torch.randn(7, 8), torch.randn(4, 8)
        assert int(centrality(queries, centres).numel()) == 7

    def test_and_is_never_negative(self):
        queries, centres = torch.randn(7, 8), torch.randn(4, 8)
        assert bool(torch.all(centrality(queries, centres) >= 0))

    def test_boundary_closeness_is_never_negative(self):
        queries, centres = torch.randn(7, 8), torch.randn(4, 8)
        assert bool(torch.all(boundary_closeness(queries, centres) >= 0))

    def test_a_boundary_needs_two_partitions(self):
        assert a_boundary_signal_needs_two_partitions()

    def test_a_rank_one_query_batch_is_refused(self):
        assert a_rank_one_query_batch_is_refused()

    def test_the_spread_is_the_last_score_minus_the_first(self):
        found = Neighbours(
            torch.zeros(2, 4, dtype=torch.long), torch.tensor([[1.0, 2, 3, 4], [0.0, 1, 2, 9]])
        )
        assert result_spread(found).tolist() == [3.0, 9.0]

    def test_the_best_score_is_the_first(self):
        found = Neighbours(
            torch.zeros(2, 4, dtype=torch.long), torch.tensor([[1.0, 2, 3, 4], [0.5, 1, 2, 9]])
        )
        assert best_score(found).tolist() == [1.0, 0.5]


class TestBrackets:
    def test_the_expensive_tier_is_more_accurate(self):
        result = the_two_fixed_policies_bracket_everything()
        assert result["expensive_recall"] > result["cheap_recall"]

    def test_and_costs_far_more(self):
        assert the_two_fixed_policies_bracket_everything()["cost_ratio"] > 5

    def test_the_gap_is_wide(self):
        assert the_two_fixed_policies_bracket_everything()["recall_gap"] > 0.5

    def test_a_share_of_zero_matches_always_cheap(self):
        assert a_share_of_zero_is_always_cheap_plus_overhead()["zero_matches_cheap"]

    def test_and_costs_the_same(self):
        assert a_share_of_zero_is_always_cheap_plus_overhead()["zero_costs_the_same"]

    def test_a_share_of_one_matches_always_expensive(self):
        assert a_share_of_zero_is_always_cheap_plus_overhead()["one_matches_expensive"]

    def test_but_costs_more(self):
        assert a_share_of_zero_is_always_cheap_plus_overhead()["one_costs_more"]

    def test_by_exactly_the_cheap_search(self):
        result = a_share_of_zero_is_always_cheap_plus_overhead()
        brackets = the_two_fixed_policies_bracket_everything()
        assert abs(result["overhead"] - brackets["cheap_distances"]) < 1.0


class TestTheSignalWorks:
    def test_it_beats_a_random_fifth(self):
        assert escalating_beats_a_random_fifth()["signal_is_worth_something"]

    def test_at_the_same_cost(self):
        assert escalating_beats_a_random_fifth()["same_cost"]

    def test_the_escalated_queries_are_the_hard_ones(self):
        assert the_escalated_queries_are_the_hard_ones()["the_escalated_are_worse"]

    def test_by_a_visible_margin(self):
        assert the_escalated_queries_are_the_hard_ones()["gap"] > 0.05

    def test_the_share_trades_cost_for_recall(self):
        rows = the_escalation_share_trades_cost_for_recall()
        assert [row["recall"] for row in rows] == sorted(row["recall"] for row in rows)

    def test_and_the_cost_rises_too(self):
        rows = the_escalation_share_trades_cost_for_recall()
        assert [row["distances"] for row in rows] == sorted(row["distances"] for row in rows)

    def test_the_escalated_count_matches_the_share(self):
        rows = {row["share"]: row for row in the_escalation_share_trades_cost_for_recall()}
        assert rows[0.2]["escalated"] == 40

    def test_an_empty_share_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            the_escalation_share_trades_cost_for_recall(shares=())


class TestTheSignalIsNotEnough:
    def test_routing_does_not_beat_tuning(self):
        assert not routing_does_not_beat_tuning_the_cheap_index()["above_the_line"]

    def test_but_it_is_close(self):
        assert routing_does_not_beat_tuning_the_cheap_index()["inside_the_noise"]

    def test_structure_does_not_help(self):
        assert structure_does_not_make_difficulty_more_predictable()["both_inside_the_noise"]

    def test_the_gains_have_opposite_signs(self):
        result = structure_does_not_make_difficulty_more_predictable()
        assert result["gaussian_gain"] > 0 > result["clustered_gain"]

    def test_the_signal_does_not_transfer_between_structures(self):
        assert not routing_across_different_structures()["signal_transfers"]

    def test_and_the_gap_is_small(self):
        assert abs(routing_across_different_structures()["gap"]) < 0.05

    def test_the_mixed_tiers_still_bracket_it(self):
        result = routing_across_different_structures()
        assert result["cheap_recall"] < result["informed_recall"] < result["expensive_recall"]


class TestTheOtherDesign:
    def test_choosing_up_front_is_cheaper(self):
        assert choosing_up_front_is_cheaper_and_worse()["up_front_is_cheaper"]

    def test_and_less_accurate(self):
        assert choosing_up_front_is_cheaper_and_worse()["escalating_is_more_accurate"]

    def test_six_schemes_are_compared(self):
        assert len(compare_every_routing_scheme()) == 6

    def test_every_escalating_scheme_sends_the_same_volume(self):
        rows = {row["scheme"]: row for row in compare_every_routing_scheme()}
        assert (
            rows["escalate on spread"]["escalated"]
            == rows["escalate at random"]["escalated"]
            == rows["choose up front"]["escalated"]
        )

    def test_the_informed_scheme_leads_the_random_one(self):
        rows = {row["scheme"]: row for row in compare_every_routing_scheme()}
        assert rows["escalate on spread"]["recall"] > rows["escalate at random"]["recall"]

    def test_and_both_sit_between_the_brackets(self):
        rows = {row["scheme"]: row for row in compare_every_routing_scheme()}
        low, high = rows["always cheap"]["recall"], rows["always expensive"]["recall"]
        assert low < rows["escalate on spread"]["recall"] < high


class TestMechanics:
    def test_the_routed_result_is_well_formed(self):
        result = the_routed_result_is_well_formed()
        assert result["distinct"] and result["sorted"]

    def test_and_accounts_for_every_query(self):
        assert the_routed_result_is_well_formed()["accounted_for"]

    def test_a_share_above_one_is_refused(self):
        assert a_share_outside_the_unit_interval_is_refused()

    def test_a_negative_share_is_refused(self):
        assert a_negative_share_is_refused()

    def test_the_up_front_design_refuses_it_too(self):
        assert an_up_front_share_outside_the_interval_is_refused()

    def test_an_empty_result_divides_safely(self):
        assert an_empty_routed_result_divides_safely()["safe"]

    def test_a_routed_result_serialises(self):
        assert a_routed_result_serialises()["has_everything"]

    def test_with_the_escalation_rate(self):
        assert a_routed_result_serialises()["escalation_rate"] == 0.2

    def test_a_tier_serialises(self):
        assert a_tier_serialises()["name"] == "cheap"

    def test_always_sends_everything_to_one_tier(self):
        cheap, _, probes, _, _ = _setup(count=512, queries=16)
        routed = always(cheap, probes, 5)
        assert routed.sent == {"cheap": 16} and routed.escalated == 0

    def test_escalating_sends_the_right_number(self):
        cheap, expensive, probes, _, _ = _setup(count=512, queries=20)
        routed = escalate(cheap, expensive, probes, 5, _spread_signal, share=0.25)
        assert routed.escalated == 5

    def test_and_reports_the_rate(self):
        cheap, expensive, probes, _, _ = _setup(count=512, queries=20)
        routed = escalate(cheap, expensive, probes, 5, _spread_signal, share=0.25)
        assert routed.escalation_rate == 0.25

    def test_choosing_up_front_sends_the_right_number(self):
        cheap, expensive, probes, _, centres = _setup(count=512, queries=20)
        routed = choose_up_front(
            cheap,
            expensive,
            probes,
            5,
            lambda queries, found: centrality(queries, centres),  # noqa: ARG005
            share=0.25,
        )
        assert routed.escalated == 5

    def test_a_random_signal_has_one_value_per_query(self):
        queries = torch.randn(12, 8)
        assert int(_random_signal(queries, None).numel()) == 12

    def test_a_tier_holds_its_index(self):
        corpus = gaussian(count=256, dimension=8)
        index = IVFIndex(8, partitions=8, probe=2)
        index.build(corpus.vectors)
        assert Tier("cheap", index, 1.0).index.size == 256

    def test_an_empty_routed_result_has_no_escalations(self):
        routed = Routed(
            found=Neighbours(torch.zeros(0, 5, dtype=torch.long), torch.zeros(0, 5))
        )
        assert routed.escalated == 0

    def test_a_centrality_signal_refuses_a_rank_one_centre_set(self):
        with pytest.raises(DataError, match="batch of queries"):
            centrality(torch.randn(4, 8), torch.randn(8))

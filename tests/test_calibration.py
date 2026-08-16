from __future__ import annotations

import pytest
import torch

from vse.errors import ConfigError, DataError
from vse.eval.calibration import (
    SIGNALS,
    Calibration,
    Flagged,
    _ranks,
    _setup,
    a_correlation_of_two_points_is_refused,
    a_mismatched_flag_is_refused,
    a_random_flag_is_the_baseline,
    a_share_outside_the_range_is_refused,
    a_single_neighbour_has_no_shape,
    correlation,
    escalating_the_flagged_queries,
    even_a_perfect_flag_loses,
    every_signal,
    flag_the_worst,
    it_does_not_beat_the_signal_from_before_the_search,
    measure,
    no_signal_is_strong,
    no_signal_travels_between_corpora,
    per_query_recall,
    signals,
    spending_the_same_everywhere_wins,
    summarise,
    the_precision_beats_the_base_rate,
    the_signal_fades_as_the_index_improves,
    there_is_nothing_left_to_catch_at_the_top,
    what_a_flag_catches,
)
from vse.vectors.exact import Neighbours


class TestTheSignals:
    def test_every_signal_is_measured(self):
        assert {row["signal"] for row in every_signal()} == set(SIGNALS)

    def test_they_come_back_sorted_by_strength(self):
        strengths = [abs(row["correlation"]) for row in every_signal()]
        assert strengths == sorted(strengths, reverse=True)

    def test_the_best_signal_is_the_nearest_distance(self):
        assert no_signal_is_strong()["best"] == "nearest"

    def test_and_it_points_the_right_way(self):
        assert no_signal_is_strong()["it_is_negative"]

    def test_but_it_is_weak(self):
        assert no_signal_is_strong()["it_is_weak"]

    def test_and_its_lead_is_inside_the_noise(self):
        assert no_signal_is_strong()["the_margin_is_noise"]

    def test_every_correlation_is_small(self):
        assert all(abs(row["correlation"]) < 0.2 for row in every_signal())

    def test_four_hundred_queries_are_used(self):
        assert all(row["queries"] == 400 for row in every_signal())


class TestAgainstTheRouter:
    def test_the_post_search_signal_is_worse(self):
        assert it_does_not_beat_the_signal_from_before_the_search()["it_is_worse"]

    def test_by_more_than_half(self):
        assert it_does_not_beat_the_signal_from_before_the_search()["by_more_than_half"]

    def test_the_ratio_of_the_two_is_reported(self):
        assert it_does_not_beat_the_signal_from_before_the_search()["ratio"] < 1.0


class TestTravel:
    def test_no_signal_keeps_its_sign(self):
        assert no_signal_travels_between_corpora()["nothing_keeps_its_sign"]

    def test_and_all_of_them_stay_near_zero(self):
        assert no_signal_travels_between_corpora()["all_are_near_zero"]

    def test_three_corpora_are_measured(self):
        rows = no_signal_travels_between_corpora()["rows"]
        assert all(len(values) == 3 for values in rows.values())

    def test_an_empty_corpus_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            no_signal_travels_between_corpora(kinds=())

    def test_an_unknown_corpus_is_refused(self):
        with pytest.raises(ConfigError, match="is not a corpus"):
            _setup(kind="spiral")


class TestTheFlag:
    def test_four_budgets_are_measured(self):
        assert len(what_a_flag_catches()) == 4

    def test_the_precision_falls_as_the_flag_widens(self):
        precisions = [row["precision"] for row in what_a_flag_catches()]
        assert precisions == sorted(precisions, reverse=True)

    def test_the_sensitivity_rises_as_the_flag_widens(self):
        sensitivities = [row["sensitivity"] for row in what_a_flag_catches()]
        assert sensitivities == sorted(sensitivities)

    def test_the_precision_beats_the_base_rate(self):
        assert the_precision_beats_the_base_rate()["it_beats_the_base_rate"]

    def test_and_by_a_visible_margin(self):
        assert the_precision_beats_the_base_rate()["by"] > 0.1

    def test_a_random_flag_lands_on_the_base_rate(self):
        assert a_random_flag_is_the_baseline()["it_lands_on_the_base_rate"]

    def test_and_the_real_rule_beats_it(self):
        assert a_random_flag_is_the_baseline()["the_real_rule_beats_it"]

    def test_a_single_random_trial_is_refused(self):
        with pytest.raises(ConfigError, match="not enough trials"):
            a_random_flag_is_the_baseline(trials=1)

    def test_an_empty_share_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            what_a_flag_catches(shares=())


class TestEscalation:
    def test_escalating_the_flagged_queries_helps(self):
        assert escalating_the_flagged_queries()["it_helped"]

    def test_and_costs_something(self):
        assert escalating_the_flagged_queries()["overhead"] > 0.0

    def test_a_multiplier_of_one_is_refused(self):
        with pytest.raises(ConfigError, match="not an escalation"):
            escalating_the_flagged_queries(multiplier=1)

    def test_but_uniform_spending_wins(self):
        assert spending_the_same_everywhere_wins()["the_flag_loses"]

    def test_and_by_more_than_the_noise(self):
        assert spending_the_same_everywhere_wins()["and_by_more_than_the_noise"]

    def test_the_uniform_probe_is_above_the_baseline(self):
        assert spending_the_same_everywhere_wins()["uniform_probe"] > 4

    def test_the_uniform_sweep_rises(self):
        recalls = [row["recall"] for row in spending_the_same_everywhere_wins()["rows"]]
        assert recalls == sorted(recalls)


class TestTheOracle:
    def test_the_oracle_beats_the_rule(self):
        assert even_a_perfect_flag_loses()["the_oracle_is_ahead"]

    def test_but_only_just(self):
        assert even_a_perfect_flag_loses()["but_only_just"]

    def test_the_rule_captures_most_of_what_is_there(self):
        assert even_a_perfect_flag_loses()["most_of_it_is_captured"]

    def test_and_the_oracle_still_loses_to_uniform_spending(self):
        oracle = even_a_perfect_flag_loses()["with_the_oracle"]
        assert oracle < spending_the_same_everywhere_wins()["uniform_recall"]


class TestFading:
    def test_five_probe_settings_are_measured(self):
        assert len(the_signal_fades_as_the_index_improves()) == 5

    def test_the_correlation_fades(self):
        rows = the_signal_fades_as_the_index_improves()
        strengths = [abs(row["correlation"]) for row in rows]
        assert strengths == sorted(strengths, reverse=True)

    def test_it_is_strongest_at_the_cheap_end(self):
        assert there_is_nothing_left_to_catch_at_the_top()["it_is_strongest_at_the_cheap_end"]

    def test_the_base_rate_collapses(self):
        assert there_is_nothing_left_to_catch_at_the_top()["the_base_rate_collapses"]

    def test_and_so_does_the_correlation(self):
        assert there_is_nothing_left_to_catch_at_the_top()["the_correlation_collapses_too"]

    def test_at_the_cheap_end_it_beats_the_router(self):
        first = the_signal_fades_as_the_index_improves()[0]
        assert abs(first["correlation"]) > 0.254

    def test_an_empty_probe_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            the_signal_fades_as_the_index_improves(probe_values=())


class TestSummary:
    def test_the_summary_names_the_best_signal(self):
        assert summarise()["best_signal"] == "nearest"

    def test_and_reports_that_the_flag_loses(self):
        assert summarise()["the_flag_loses"]

    def test_and_that_the_precision_beat_the_base_rate(self):
        result = summarise()
        assert result["precision_at_a_tenth"] > result["base_rate"]


class TestMechanics:
    def test_a_perfect_correlation_is_one(self):
        assert correlation([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == pytest.approx(1.0)

    def test_a_reversed_one_is_minus_one(self):
        assert correlation([1.0, 2.0, 3.0], [6.0, 4.0, 2.0]) == pytest.approx(-1.0)

    def test_a_constant_side_correlates_with_nothing(self):
        assert correlation([1.0, 1.0, 1.0], [2.0, 4.0, 6.0]) == 0.0

    def test_mismatched_lengths_are_refused(self):
        with pytest.raises(DataError, match="against"):
            correlation([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_two_points_are_refused(self):
        assert a_correlation_of_two_points_is_refused()

    def test_ranks_start_at_zero(self):
        assert _ranks([5.0, 1.0, 3.0]) == [2.0, 0.0, 1.0]

    def test_ties_share_a_rank(self):
        assert _ranks([1.0, 1.0, 3.0]) == [0.5, 0.5, 2.0]

    def test_every_signal_comes_back(self):
        found = Neighbours(torch.zeros(4, 5, dtype=torch.long), torch.rand(4, 5) + 1.0)
        assert set(signals(found)) == set(SIGNALS)

    def test_each_signal_has_one_value_per_query(self):
        found = Neighbours(torch.zeros(4, 5, dtype=torch.long), torch.rand(4, 5) + 1.0)
        assert all(len(values) == 4 for values in signals(found).values())

    def test_the_gap_is_the_furthest_minus_the_nearest(self):
        scores = torch.tensor([[1.0, 2.0, 4.0]])
        found = Neighbours(torch.zeros(1, 3, dtype=torch.long), scores)
        assert signals(found)["gap"] == [3.0]

    def test_the_ratio_is_the_nearest_over_the_furthest(self):
        scores = torch.tensor([[1.0, 2.0, 4.0]])
        found = Neighbours(torch.zeros(1, 3, dtype=torch.long), scores)
        assert signals(found)["ratio"] == [pytest.approx(0.25)]

    def test_a_single_neighbour_has_no_shape(self):
        assert a_single_neighbour_has_no_shape()

    def test_a_one_dimensional_result_never_reaches_the_signals(self):
        with pytest.raises(DataError, match="matrix of rows"):
            Neighbours(torch.zeros(4, dtype=torch.long), torch.zeros(4))

    def test_per_query_recall_scores_the_truth_perfectly(self):
        _, _, truth = _setup(count=512, queries=8)
        assert per_query_recall(truth, truth) == [1.0] * 8

    def test_a_mismatched_batch_is_refused(self):
        _, _, truth = _setup(count=512, queries=8)
        short = Neighbours(truth.identifiers[:4], truth.scores[:4])
        with pytest.raises(DataError, match="truths against"):
            per_query_recall(truth, short)

    def test_a_calibration_of_two_queries_is_refused(self):
        with pytest.raises(ConfigError, match="at least three queries"):
            Calibration(signal="ratio", values=[1.0, 2.0], recalls=[0.1, 0.2])

    def test_a_calibration_with_mismatched_lengths_is_refused(self):
        with pytest.raises(DataError, match="values against"):
            Calibration(signal="ratio", values=[1.0, 2.0, 3.0], recalls=[0.1, 0.2])

    def test_a_calibration_reports_its_strength(self):
        calibration = Calibration(
            signal="ratio", values=[1.0, 2.0, 3.0], recalls=[3.0, 2.0, 1.0]
        )
        assert calibration.strength == pytest.approx(1.0)

    def test_a_calibration_serialises(self):
        calibration = Calibration(
            signal="ratio", values=[1.0, 2.0, 3.0], recalls=[3.0, 2.0, 1.0]
        )
        assert calibration.as_dict()["signal"] == "ratio"

    def test_a_flagged_result_reports_precision_and_sensitivity(self):
        flagged = Flagged(share=0.5, caught=3, missed=1, false_alarms=1, kept=5)
        assert flagged.precision == 0.75 and flagged.sensitivity == 0.75

    def test_a_flag_that_caught_nothing_has_no_precision(self):
        flagged = Flagged(share=0.5, caught=0, missed=0, false_alarms=0, kept=5)
        assert flagged.precision == 0.0 and flagged.sensitivity == 0.0

    def test_a_flagged_result_serialises(self):
        flagged = Flagged(share=0.5, caught=3, missed=1, false_alarms=1, kept=5)
        assert flagged.as_dict()["caught"] == 3

    def test_a_share_of_everything_is_refused(self):
        assert a_share_outside_the_range_is_refused()

    def test_a_share_of_nothing_is_refused(self):
        with pytest.raises(ConfigError, match="not a share"):
            flag_the_worst([1.0, 2.0, 3.0], [0.1, 0.2, 0.3], share=0.0)

    def test_a_mismatched_flag_is_refused(self):
        assert a_mismatched_flag_is_refused()

    def test_a_flag_always_takes_at_least_one_query(self):
        flagged = flag_the_worst([1.0, 2.0, 3.0], [0.1, 0.2, 0.3], share=0.01)
        assert flagged.caught + flagged.false_alarms == 1

    def test_a_flag_on_the_other_direction_is_allowed(self):
        flagged = flag_the_worst(
            [1.0, 2.0, 3.0], [0.9, 0.2, 0.1], share=0.34, largest_is_worse=False
        )
        assert flagged.caught == 0

    def test_a_measurement_lines_up_its_pieces(self):
        result = measure(prepared=_setup(count=512, queries=8))
        assert len(result["recalls"]) == 8 and len(result["signals"]["ratio"]) == 8

    def test_a_measurement_reports_its_cost(self):
        assert measure(prepared=_setup(count=512, queries=8))["distances"] > 0.0

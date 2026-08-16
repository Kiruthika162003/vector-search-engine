from __future__ import annotations

import pytest

from vse.errors import ConfigError
from vse.eval.autotune import (
    BEAM_VALUES,
    PROBE_VALUES,
    Setting,
    Tuning,
    a_clustered_corpus_needs_a_different_setting,
    a_higher_target_costs_more,
    a_margin_fixes_it,
    a_setting_tuned_on_one_sample_holds_on_another,
    a_target_picks_a_setting,
    a_wider_probe_opens_a_superset,
    a_zero_target_is_refused,
    an_empty_sweep_is_refused,
    an_empty_tuning_has_no_best,
    an_impossible_target_is_refused,
    an_unreachable_target_says_so,
    cheapest_that_clears,
    set_beam,
    set_probe,
    sweep_setting,
    the_cheapest_clearing_setting_is_not_always_the_smallest,
    the_cost_curve_is_monotone,
    the_curve_is_monotone_for_every_query_not_just_on_average,
    the_last_points_of_recall_are_the_expensive_ones,
    the_recall_curve_is_monotone,
    the_setting_does_not_transfer_between_indexes,
    the_two_structures_need_different_numbers,
    tune,
    tuning_the_hierarchy,
    tuning_to_the_target_exactly_misses_half_the_time,
)
from vse.index.graph import GraphIndex
from vse.index.ivf import IVFIndex
from vse.vectors.dataset import gaussian, held_out
from vse.vectors.exact import search


def a_small_setup(count: int = 1024, dimension: int = 16, queries: int = 40):
    """A built index with its queries and their true answers."""
    corpus = gaussian(count=count, dimension=dimension)
    searched, probes = held_out(corpus, count=queries)
    truth = search(probes, searched.vectors, k=5)
    index = IVFIndex(dimension, partitions=32, probe=1)
    index.build(searched.vectors)
    return index, searched.vectors, probes, truth


class TestPicking:
    def test_a_target_picks_a_setting(self):
        assert a_target_picks_a_setting()["met"]

    def test_and_reports_what_it_costs(self):
        assert a_target_picks_a_setting()["distances"] > 0

    def test_the_chosen_setting_clears_the_target(self):
        result = a_target_picks_a_setting()
        assert result["recall"] >= result["target"]

    def test_a_higher_target_picks_a_higher_setting(self):
        rows = [
            row["probe"] for row in a_higher_target_costs_more() if row["probe"] is not None
        ]
        assert rows == sorted(rows)

    def test_and_costs_more(self):
        rows = [
            row["distances"]
            for row in a_higher_target_costs_more()
            if row["distances"] is not None
        ]
        assert rows == sorted(rows)

    def test_the_cost_grows_faster_than_the_target(self):
        assert the_last_points_of_recall_are_the_expensive_ones()["cost_grows_faster"]

    def test_by_more_than_a_factor_of_three(self):
        assert the_last_points_of_recall_are_the_expensive_ones()["cost_ratio"] > 3.0

    def test_a_target_of_ninety_nine_is_out_of_reach_on_this_sweep(self):
        rows = {row["target"]: row for row in a_higher_target_costs_more()}
        assert not rows[0.99]["met"]

    def test_an_empty_target_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            a_higher_target_costs_more(targets=())


class TestMonotonicity:
    def test_the_recall_curve_is_monotone(self):
        assert the_recall_curve_is_monotone()["monotone"]

    def test_with_no_drops_at_all(self):
        assert the_recall_curve_is_monotone()["drops"] == 0

    def test_and_it_rises(self):
        assert the_recall_curve_is_monotone()["rises_overall"]

    def test_the_cost_curve_is_monotone_too(self):
        assert the_cost_curve_is_monotone()["monotone"]

    def test_and_rises_by_an_order_of_magnitude(self):
        result = the_cost_curve_is_monotone()
        assert result["last"] > result["first"] * 10

    def test_it_holds_per_query_not_just_on_average(self):
        assert the_curve_is_monotone_for_every_query_not_just_on_average()[
            "monotone_everywhere"
        ]

    def test_across_thousands_of_pairs(self):
        result = the_curve_is_monotone_for_every_query_not_just_on_average()
        assert result["graph_pairs"] > 5000
        assert result["ivf_pairs"] > 5000

    def test_on_the_graph(self):
        assert the_curve_is_monotone_for_every_query_not_just_on_average()["graph_drops"] == 0

    def test_and_on_the_inverted_file(self):
        assert the_curve_is_monotone_for_every_query_not_just_on_average()["ivf_drops"] == 0

    def test_too_few_queries_is_refused(self):
        with pytest.raises(ConfigError, match="cannot show a distribution"):
            the_curve_is_monotone_for_every_query_not_just_on_average(queries=1)

    def test_the_mechanism_is_inclusion(self):
        assert a_wider_probe_opens_a_superset()["nested"]

    def test_a_single_probe_opens_one_partition(self):
        assert len(a_wider_probe_opens_a_superset()["opened_at_one"]) == 1

    def test_and_sixteen_opens_sixteen(self):
        assert a_wider_probe_opens_a_superset()["opened_at_sixteen"] == 16

    def test_one_probe_count_cannot_show_nesting(self):
        with pytest.raises(ConfigError, match="at least two"):
            a_wider_probe_opens_a_superset(probes_tried=(4,))


class TestGeneralisation:
    def test_a_setting_holds_on_fresh_samples(self):
        result = a_setting_tuned_on_one_sample_holds_on_another()
        assert result["fresh_mean"] > 0.85

    def test_but_lands_on_both_sides_of_the_target(self):
        assert tuning_to_the_target_exactly_misses_half_the_time()["some_miss"]

    def test_the_worst_fresh_sample_is_below_the_target(self):
        result = tuning_to_the_target_exactly_misses_half_the_time()
        assert result["fresh_worst"] < result["target"]

    def test_but_not_far_below(self):
        result = tuning_to_the_target_exactly_misses_half_the_time()
        assert result["fresh_worst"] > result["target"] - 0.05

    def test_a_margin_moves_the_setting_up(self):
        result = a_margin_fixes_it()
        assert result["probe_with"] > result["probe_without"]

    def test_and_clears_every_fresh_sample(self):
        assert a_margin_fixes_it()["below_with"] == 0

    def test_where_the_unpadded_tuning_missed_most(self):
        assert a_margin_fixes_it()["below_without"] > 2

    def test_it_helps(self):
        assert a_margin_fixes_it()["helps"]

    def test_a_single_sample_cannot_show_variation(self):
        with pytest.raises(ConfigError, match="cannot show variation"):
            a_setting_tuned_on_one_sample_holds_on_another(samples=1)

    def test_a_margin_of_a_whole_point_is_refused(self):
        with pytest.raises(ConfigError, match="not a margin"):
            a_margin_fixes_it(margin=0.9)


class TestTransfer:
    def test_two_structures_are_compared(self):
        assert len(the_setting_does_not_transfer_between_indexes()) == 2

    def test_both_meet_the_target(self):
        assert all(row["met"] for row in the_setting_does_not_transfer_between_indexes())

    def test_at_different_knob_values(self):
        assert the_two_structures_need_different_numbers()["different"]

    def test_the_graph_costs_far_less_at_the_same_recall(self):
        result = the_two_structures_need_different_numbers()
        assert result["graph_distances"] < result["ivf_distances"] / 2

    def test_even_though_its_knob_value_is_larger(self):
        result = the_two_structures_need_different_numbers()
        assert result["graph_beam"] > result["ivf_probe"]

    def test_the_hierarchy_tunes_the_same_way(self):
        assert tuning_the_hierarchy()["met"]

    def test_with_a_beam_rather_than_a_probe(self):
        assert tuning_the_hierarchy()["beam"] in BEAM_VALUES

    def test_a_clustered_corpus_needs_a_different_setting(self):
        assert a_clustered_corpus_needs_a_different_setting()["differ"]

    def test_and_a_much_lower_one(self):
        result = a_clustered_corpus_needs_a_different_setting()
        assert result["clustered_probe"] < result["gaussian_probe"]

    def test_costing_six_times_less(self):
        result = a_clustered_corpus_needs_a_different_setting()
        assert result["gaussian_distances"] > result["clustered_distances"] * 5


class TestFailure:
    def test_an_unreachable_target_says_so(self):
        assert not an_unreachable_target_says_so()["met"]

    def test_and_returns_nothing_rather_than_the_best_it_found(self):
        assert an_unreachable_target_says_so()["chosen"] is None

    def test_but_reports_the_best_it_found(self):
        assert an_unreachable_target_says_so()["best_recall"] > 0

    def test_an_empty_sweep_is_refused(self):
        assert an_empty_sweep_is_refused()

    def test_an_impossible_target_is_refused(self):
        assert an_impossible_target_is_refused()

    def test_a_zero_target_is_refused(self):
        assert a_zero_target_is_refused()

    def test_a_negative_target_is_refused(self):
        with pytest.raises(ConfigError, match="not a target"):
            cheapest_that_clears([Setting(1, 0.5, 10.0)], target=-0.5)

    def test_an_empty_tuning_has_no_best(self):
        assert an_empty_tuning_has_no_best()

    def test_a_sweep_that_clears_nothing_returns_nothing(self):
        assert cheapest_that_clears([Setting(1, 0.5, 10.0)], target=0.9) is None


class TestPickingByCost:
    def test_the_cheapest_is_not_the_smallest_value(self):
        assert the_cheapest_clearing_setting_is_not_always_the_smallest()["picked_by_cost"]

    def test_it_picks_the_lower_cost(self):
        result = the_cheapest_clearing_setting_is_not_always_the_smallest()
        assert result["chosen_distances"] == 200.0

    def test_and_not_the_smaller_knob(self):
        result = the_cheapest_clearing_setting_is_not_always_the_smallest()
        assert result["chosen_value"] != result["smallest_clearing_value"]


class TestMechanics:
    def test_a_sweep_returns_one_setting_per_value(self):
        index, corpus, probes, truth = a_small_setup()
        settings = sweep_setting(index, corpus, probes, truth, (1, 2, 4), set_probe, k=5)
        assert len(settings) == 3

    def test_each_carries_its_value(self):
        index, corpus, probes, truth = a_small_setup()
        settings = sweep_setting(index, corpus, probes, truth, (1, 2, 4), set_probe, k=5)
        assert [setting.value for setting in settings] == [1, 2, 4]

    def test_an_empty_value_list_is_refused(self):
        index, corpus, probes, truth = a_small_setup()
        with pytest.raises(ConfigError, match="nothing to sweep"):
            sweep_setting(index, corpus, probes, truth, (), set_probe, k=5)

    def test_tune_returns_the_whole_sweep(self):
        index, corpus, probes, truth = a_small_setup()
        result = tune(index, corpus, probes, truth, (1, 2, 4, 8), set_probe, target=0.5, k=5)
        assert len(result.sweep) == 4

    def test_and_records_whether_it_met_the_target(self):
        index, corpus, probes, truth = a_small_setup()
        result = tune(index, corpus, probes, truth, (1, 2, 4, 8), set_probe, target=0.5, k=5)
        assert result.met == (result.chosen is not None)

    def test_the_sweep_leaves_the_knob_at_the_last_value_tried(self):
        index, corpus, probes, truth = a_small_setup()
        sweep_setting(index, corpus, probes, truth, (1, 2, 4), set_probe, k=5)
        assert index.probe == 4

    def test_set_probe_applies_to_the_index(self):
        index, _, _, _ = a_small_setup()
        set_probe(index, 7)
        assert index.probe == 7

    def test_set_beam_applies_to_the_index(self):
        index = GraphIndex(8, degree=8, ef=10)
        set_beam(index, 21)
        assert index.ef == 21

    def test_the_best_available_is_the_highest_recall(self):
        sweep = [Setting(1, 0.4, 10.0), Setting(2, 0.8, 20.0), Setting(4, 0.6, 40.0)]
        assert Tuning(target=0.9, sweep=sweep, chosen=None).best_available.value == 2

    def test_a_setting_serialises(self):
        assert Setting(4, 0.912345, 100.0).as_dict()["recall"] == 0.9123

    def test_a_tuning_serialises(self):
        sweep = [Setting(1, 0.95, 10.0)]
        result = Tuning(target=0.9, sweep=sweep, chosen=sweep[0])
        assert result.as_dict()["chosen"]["value"] == 1

    def test_and_records_how_many_settings_it_tried(self):
        sweep = [Setting(1, 0.95, 10.0), Setting(2, 0.99, 20.0)]
        assert Tuning(target=0.9, sweep=sweep, chosen=sweep[0]).as_dict()["settings_tried"] == 2

    def test_the_probe_values_start_at_one(self):
        assert PROBE_VALUES[0] == 1

    def test_the_beam_values_start_at_ten(self):
        assert BEAM_VALUES[0] == 10

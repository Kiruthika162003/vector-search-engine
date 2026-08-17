from __future__ import annotations

import pytest
import torch

from vse.errors import ConfigError, DataError
from vse.vectors.drift import (
    ADVERSE,
    KINDS,
    Damage,
    Drift,
    _setup,
    a_negative_magnitude_is_refused,
    a_rotation_is_harmless_by_construction,
    a_shift_is_nearly_harmless,
    an_unknown_drift_is_refused,
    compare_the_drifts,
    drifted,
    only_the_scaling_hurts,
    probing_more_repairs_a_shrunk_population,
    rebuilding_on_the_drifted_queries_does_not_help,
    rotate,
    scale,
    scoring_against_the_old_truth_invents_a_collapse,
    shift,
    shrinking_costs_more_than_stretching,
    summarise,
    the_query_sits_no_worse_in_the_partitions,
    the_scatter_tracks_the_damage,
    the_true_neighbours_are_what_scatter,
    the_truth_moves_with_the_queries,
)


class TestTheHarmlessDrifts:
    def test_a_shift_costs_almost_nothing(self):
        rows = a_shift_is_nearly_harmless()
        assert all(abs(row["loss"]) < 0.05 for row in rows)

    def test_the_largest_shift_does_not_hurt(self):
        assert a_shift_is_nearly_harmless()[-1]["loss"] <= 0.0

    def test_five_shifts_are_measured(self):
        assert len(a_shift_is_nearly_harmless()) == 5

    def test_the_cost_does_not_move_with_the_shift(self):
        costs = [row["distances"] for row in a_shift_is_nearly_harmless()]
        assert max(costs) - min(costs) < 10.0

    def test_a_rotation_is_within_the_noise(self):
        assert all(abs(row["loss"]) < 0.02 for row in a_rotation_is_harmless_by_construction())

    def test_a_rotation_of_nothing_costs_nothing(self):
        assert a_rotation_is_harmless_by_construction()[0]["loss"] == 0.0

    def test_an_empty_shift_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            a_shift_is_nearly_harmless(magnitudes=())

    def test_an_empty_rotation_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            a_rotation_is_harmless_by_construction(magnitudes=())


class TestTheScaling:
    def test_shrinking_hurts(self):
        assert shrinking_costs_more_than_stretching()["shrinking_hurts"]

    def test_stretching_helps(self):
        assert shrinking_costs_more_than_stretching()["stretching_helps"]

    def test_the_asymmetry_is_real(self):
        assert shrinking_costs_more_than_stretching()["the_asymmetry_is_real"]

    def test_and_lopsided(self):
        result = shrinking_costs_more_than_stretching()
        assert result["shrinking_costs_more_than_stretching_gains"]

    def test_the_shrunk_recall_is_well_below_the_baseline(self):
        result = shrinking_costs_more_than_stretching()
        assert result["undrifted"] - result["shrunk"] > 0.15

    def test_the_loss_falls_as_the_queries_stretch(self):
        rows = only_the_scaling_hurts(magnitudes=(0.25, 0.5, 1.0, 2.0))
        losses = [row["loss"] for row in rows]
        assert losses == sorted(losses, reverse=True)

    def test_a_shrunk_search_costs_more_distances(self):
        rows = {row["magnitude"]: row for row in only_the_scaling_hurts()}
        assert rows[0.25]["distances"] > rows[4.0]["distances"]

    def test_an_empty_scale_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            only_the_scaling_hurts(magnitudes=())


class TestTheMechanism:
    def test_the_boundary_ratio_points_the_wrong_way(self):
        rows = the_query_sits_no_worse_in_the_partitions(magnitudes=(1.0, 2.0, 4.0))
        ratios = [row["mean_ratio"] for row in rows]
        assert ratios == sorted(ratios)

    def test_the_scatter_points_the_right_way(self):
        assert the_scatter_tracks_the_damage()["the_scatter_falls_while_the_recall_rises"]

    def test_the_worst_case_is_the_most_scattered(self):
        assert the_scatter_tracks_the_damage()["the_worst_case_is_the_most_scattered"]

    def test_the_answer_never_fits_in_the_probe_budget(self):
        rows = the_true_neighbours_are_what_scatter()
        assert all(row["partitions_holding_the_answer"] > 4.0 for row in rows)

    def test_and_never_exceeds_the_result_width(self):
        rows = the_true_neighbours_are_what_scatter()
        assert all(row["partitions_holding_the_answer"] <= 10.0 for row in rows)

    def test_an_empty_boundary_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            the_query_sits_no_worse_in_the_partitions(magnitudes=())

    def test_an_empty_scatter_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            the_true_neighbours_are_what_scatter(magnitudes=())


class TestTheRepairs:
    def test_more_probes_repair_it(self):
        rows = probing_more_repairs_a_shrunk_population()
        assert rows[-1]["recall"] > rows[0]["recall"] + 0.3

    def test_the_recall_rises_with_the_probe(self):
        recalls = [row["recall"] for row in probing_more_repairs_a_shrunk_population()]
        assert recalls == sorted(recalls)

    def test_and_so_does_the_cost(self):
        costs = [row["distances"] for row in probing_more_repairs_a_shrunk_population()]
        assert costs == sorted(costs)

    def test_an_empty_probe_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            probing_more_repairs_a_shrunk_population(probe_values=())

    def test_a_rebuild_does_not_repair_it(self):
        assert rebuilding_on_the_drifted_queries_does_not_help()["a_reseed_changes_little"]

    def test_and_leaves_the_recall_where_it_was(self):
        result = rebuilding_on_the_drifted_queries_does_not_help()
        assert result["after_a_reseed"] < 0.3


class TestScoring:
    def test_the_true_answer_moves_under_drift(self):
        rows = the_truth_moves_with_the_queries()
        drifted_rows = [row for row in rows if row["magnitude"] != 1.0]
        assert all(row["overlap_with_the_original_answer"] < 0.5 for row in drifted_rows)

    def test_an_undrifted_query_keeps_its_answer(self):
        rows = {row["magnitude"]: row for row in the_truth_moves_with_the_queries()}
        assert rows[1.0]["overlap_with_the_original_answer"] == 1.0

    def test_an_empty_truth_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            the_truth_moves_with_the_queries(magnitudes=())

    def test_the_old_truth_understates_the_index(self):
        result = scoring_against_the_old_truth_invents_a_collapse()
        assert result["the_old_truth_understates_it"]

    def test_by_a_third(self):
        assert scoring_against_the_old_truth_invents_a_collapse()["by_a_third"]


class TestComparison:
    def test_every_drift_appears(self):
        assert {row["kind"] for row in compare_the_drifts()} == set(KINDS)

    def test_the_scaling_is_the_worst(self):
        assert compare_the_drifts()[0]["kind"] == "scale"

    def test_and_the_only_one_that_loses_anything(self):
        rows = compare_the_drifts()
        assert all(row["loss"] <= 0.0 for row in rows[1:])

    def test_the_adverse_settings_are_used(self):
        rows = {row["kind"]: row for row in compare_the_drifts()}
        assert all(rows[kind]["magnitude"] == ADVERSE[kind] for kind in KINDS)

    def test_a_custom_setting_is_honoured(self):
        rows = compare_the_drifts(magnitudes={"scale": 0.5})
        assert len(rows) == 1 and rows[0]["magnitude"] == 0.5

    def test_an_empty_comparison_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            compare_the_drifts(magnitudes={})

    def test_the_summary_names_the_worst_drift(self):
        assert summarise()["worst_drift"] == "scale"

    def test_and_reports_the_asymmetry(self):
        assert summarise()["the_asymmetry_is_real"]


class TestMechanics:
    def test_a_shift_moves_every_query_the_same_way(self):
        queries = torch.zeros(4, 8)
        moved = shift(queries, 2.0).queries
        assert torch.allclose(moved, moved[0].expand_as(moved))

    def test_a_shift_of_nothing_changes_nothing(self):
        queries = torch.randn(4, 8)
        assert torch.equal(shift(queries, 0.0).queries, queries)

    def test_a_shift_moves_by_the_magnitude(self):
        queries = torch.zeros(1, 8)
        assert float(shift(queries, 3.0).queries.norm()) == pytest.approx(3.0, abs=1e-5)

    def test_a_scale_multiplies(self):
        queries = torch.ones(2, 4)
        assert torch.equal(scale(queries, 3.0).queries, queries * 3.0)

    def test_a_scale_of_one_changes_nothing(self):
        queries = torch.randn(4, 8)
        assert torch.equal(scale(queries, 1.0).queries, queries)

    def test_a_rotation_preserves_lengths(self):
        queries = torch.randn(6, 16)
        turned = rotate(queries, 0.5).queries
        assert torch.allclose(queries.norm(dim=1), turned.norm(dim=1), atol=1e-5)

    def test_a_rotation_of_nothing_changes_nothing(self):
        queries = torch.randn(4, 8)
        assert torch.allclose(rotate(queries, 0.0).queries, queries, atol=1e-5)

    def test_a_rotation_needs_two_dimensions(self):
        with pytest.raises(ConfigError, match="at least two dimensions"):
            rotate(torch.randn(4, 1), 0.5)

    def test_a_negative_shift_is_refused(self):
        assert a_negative_magnitude_is_refused()

    def test_a_negative_scale_is_refused(self):
        with pytest.raises(ConfigError, match="not a magnitude"):
            scale(torch.randn(4, 8), -1.0)

    def test_a_negative_rotation_is_refused(self):
        with pytest.raises(ConfigError, match="not a magnitude"):
            rotate(torch.randn(4, 8), -0.5)

    def test_an_unknown_drift_is_refused(self):
        assert an_unknown_drift_is_refused()

    def test_every_named_drift_dispatches(self):
        queries = torch.randn(4, 8)
        for kind in KINDS:
            assert drifted(queries, kind, 1.0).kind == kind

    def test_a_drift_reports_its_count(self):
        assert shift(torch.randn(7, 8), 1.0).count == 7

    def test_a_drift_serialises(self):
        assert scale(torch.randn(4, 8), 2.0).as_dict()["magnitude"] == 2.0

    def test_a_one_dimensional_query_block_is_refused(self):
        with pytest.raises(DataError, match="two dimensional"):
            Drift(kind="shift", magnitude=1.0, queries=torch.zeros(8))

    def test_a_negative_magnitude_on_the_record_is_refused(self):
        with pytest.raises(ConfigError, match="not a magnitude"):
            Drift(kind="shift", magnitude=-1.0, queries=torch.zeros(4, 8))

    def test_damage_reports_its_loss(self):
        damage = Damage(kind="scale", magnitude=0.25, recall=0.2, distances=100.0, baseline=0.4)
        assert damage.loss == pytest.approx(0.2)

    def test_damage_serialises(self):
        damage = Damage(kind="scale", magnitude=0.25, recall=0.2, distances=100.0, baseline=0.4)
        assert damage.as_dict()["kind"] == "scale"

    def test_the_setup_returns_matching_widths(self):
        corpus, probes = _setup(count=256, queries=8)
        assert corpus.shape[1] == probes.shape[1]

    def test_the_setup_holds_nothing_out(self):
        corpus, probes = _setup(count=256, queries=8)
        assert int(corpus.shape[0]) == 256 and int(probes.shape[0]) == 8

from __future__ import annotations

import io
import json

import pytest

from vse.cli.main import (
    CORPORA,
    INDEXES,
    Outcome,
    _knob_for,
    a_bad_corpus_is_reported_by_name,
    a_bad_setting_is_reported_as_one_line,
    a_missing_file_is_reported,
    a_shortlist_target_outside_the_range_is_refused,
    a_structure_with_no_format_refuses_to_build,
    a_structure_with_no_knob_says_so,
    a_sweep_produces_a_curve,
    a_zero_count_corpus_is_refused,
    a_zero_dimension_is_refused,
    an_outcome_reports_success,
    an_unknown_index_is_refused,
    an_unreachable_shortlist_target_is_reported_not_raised,
    build_corpus,
    build_index,
    building_and_inspecting_round_trip,
    dispatch,
    errors_go_to_the_error_stream,
    every_command_is_reachable,
    every_index_can_be_measured,
    main,
    measuring_one_index_works,
    parser,
    the_drift_command_names_the_one_that_matters,
    the_error_names_what_is_available,
    the_json_output_parses,
    the_report_command_prints_its_precision,
    the_shortlist_command_prices_against_a_scan,
    the_stability_command_prints_both_noise_sources,
    the_stability_table_carries_a_deviation_per_row,
    the_sweep_picks_the_right_knob,
    the_text_output_says_both_numbers,
    the_verify_command_fails_when_it_finds_something,
)
from vse.errors import ConfigError


@pytest.fixture(autouse=True)
def in_a_scratch_directory(tmp_path, monkeypatch):
    """The build command writes files, so give each test its own directory."""
    monkeypatch.chdir(tmp_path)


class TestMeasure:
    def test_measuring_one_index_works(self):
        assert measuring_one_index_works()["code"] == 0

    def test_it_reports_both_numbers(self):
        assert measuring_one_index_works()["has_both"]

    def test_and_a_speedup(self):
        assert measuring_one_index_works()["has_a_speedup"]

    def test_the_text_output_mentions_recall(self):
        assert the_text_output_says_both_numbers()["mentions_recall"]

    def test_and_distances(self):
        assert the_text_output_says_both_numbers()["mentions_distances"]

    def test_and_a_speedup_too(self):
        assert the_text_output_says_both_numbers()["mentions_a_speedup"]

    def test_every_index_is_reachable(self):
        assert every_index_can_be_measured()["all_reachable"]

    def test_nine_indexes_are_offered(self):
        assert len(INDEXES) == 9

    def test_three_corpora_are_offered(self):
        assert len(CORPORA) == 3

    def test_a_flat_index_measures(self):
        result = dispatch(
            ["measure", "--index", "flat", "--count", "512", "--dimension", "8", "--json"]
        )
        assert json.loads(result.output)["recall"] == 1.0

    def test_a_clustered_corpus_measures(self):
        result = dispatch(
            ["measure", "--corpus", "clustered", "--count", "1024", "--dimension", "16"]
        )
        assert result.code == 0 and "clustered" in result.output


class TestSweep:
    def test_a_sweep_produces_a_curve(self):
        assert a_sweep_produces_a_curve()["code"] == 0

    def test_with_several_points(self):
        assert a_sweep_produces_a_curve()["points"] > 3

    def test_the_cost_rises(self):
        assert a_sweep_produces_a_curve()["cost_rises"]

    def test_and_so_does_the_recall(self):
        assert a_sweep_produces_a_curve()["recall_rises"]

    def test_each_structure_gets_its_own_knob(self):
        assert the_sweep_picks_the_right_knob()["they_differ"]

    def test_and_they_all_have_one(self):
        assert the_sweep_picks_the_right_knob()["all_have_one"]

    def test_the_inverted_file_sweeps_probe(self):
        assert the_sweep_picks_the_right_knob()["ivf"] == "probe"

    def test_the_graph_sweeps_the_beam(self):
        assert the_sweep_picks_the_right_knob()["graph"] == "ef"

    def test_a_structure_with_no_knob_says_so(self):
        assert a_structure_with_no_knob_says_so()["refused"]

    def test_and_explains_why(self):
        assert a_structure_with_no_knob_says_so()["says_why"]

    def test_the_beam_sweep_starts_at_ten(self):
        args = parser().parse_args(["sweep", "--index", "graph"])
        _, values = _knob_for("graph", args)
        assert values[0] == 10

    def test_and_the_probe_sweep_at_one(self):
        args = parser().parse_args(["sweep", "--index", "ivf"])
        _, values = _knob_for("ivf", args)
        assert values[0] == 1


class TestErrors:
    def test_a_bad_setting_is_one_line(self):
        assert a_bad_setting_is_reported_as_one_line()["one_line"]

    def test_and_names_the_error_type(self):
        assert a_bad_setting_is_reported_as_one_line()["names_the_error"]

    def test_and_writes_nothing_to_stdout(self):
        assert a_bad_setting_is_reported_as_one_line()["no_output"]

    def test_errors_go_to_the_error_stream(self):
        result = errors_go_to_the_error_stream()
        assert result["output_is_empty"] and result["error_is_not"]

    def test_a_missing_file_is_reported(self):
        assert a_missing_file_is_reported()["names_the_file"]

    def test_without_a_traceback(self):
        assert a_missing_file_is_reported()["no_traceback"]

    def test_a_bad_corpus_is_refused_by_argparse(self):
        assert a_bad_corpus_is_reported_by_name()["argparse_refuses_it"]

    def test_and_by_the_library(self):
        assert a_bad_corpus_is_reported_by_name()["and_the_library_would_too"]

    def test_the_error_lists_the_corpora(self):
        assert the_error_names_what_is_available()["corpus_lists_options"]

    def test_and_the_indexes(self):
        assert the_error_names_what_is_available()["index_lists_options"]

    def test_a_zero_count_corpus_is_refused(self):
        assert a_zero_count_corpus_is_refused()

    def test_a_zero_dimension_is_refused(self):
        assert a_zero_dimension_is_refused()

    def test_an_unknown_index_is_refused(self):
        assert an_unknown_index_is_refused()

    def test_an_unknown_corpus_is_refused(self):
        with pytest.raises(ConfigError, match="is not a corpus"):
            build_corpus("spirals", 100, 8)

    def test_the_index_builder_refuses_by_name(self):
        args = parser().parse_args(["measure"])
        with pytest.raises(ConfigError, match="is not an index"):
            build_index("quantum", 8, args)


class TestPersistence:
    def test_building_and_inspecting_round_trip(self):
        result = building_and_inspecting_round_trip()
        assert result["build_code"] == 0 and result["inspect_code"] == 0

    def test_the_kind_survives(self):
        assert building_and_inspecting_round_trip()["kind_matches"]

    def test_and_the_count(self):
        assert building_and_inspecting_round_trip()["count_matches"]

    def test_and_the_digest(self):
        assert building_and_inspecting_round_trip()["digest_matches"]

    def test_and_the_settings(self):
        assert building_and_inspecting_round_trip()["settings_survived"]

    def test_a_structure_with_no_format_refuses(self):
        assert a_structure_with_no_format_refuses_to_build()["refused"]

    def test_and_names_the_index(self):
        assert a_structure_with_no_format_refuses_to_build()["names_the_index"]

    def test_inspecting_prints_a_summary(self):
        dispatch(
            [
                "build",
                "--index",
                "flat",
                "--count",
                "512",
                "--dimension",
                "8",
                "--out",
                "flat.vse",
            ]
        )
        result = dispatch(["inspect", "--path", "flat.vse"])
        assert result.code == 0 and "FlatIndex" in result.output


class TestOutput:
    def test_the_json_output_parses(self):
        assert the_json_output_parses()["all_parse"]

    def test_for_measure(self):
        assert the_json_output_parses()["measure"]

    def test_for_sweep(self):
        assert the_json_output_parses()["sweep"]

    def test_for_verify(self):
        assert the_json_output_parses()["verify"]

    def test_the_report_prints_its_precision(self):
        assert the_report_command_prints_its_precision()["mentions_the_error"]

    def test_and_its_query_count(self):
        assert the_report_command_prints_its_precision()["mentions_queries"]

    def test_and_warns_about_ties(self):
        assert the_report_command_prints_its_precision()["mentions_ties"]

    def test_the_report_exits_zero(self):
        assert the_report_command_prints_its_precision()["code"] == 0


class TestVerify:
    def test_it_exits_non_zero_on_violations(self):
        assert the_verify_command_fails_when_it_finds_something()[
            "exits_non_zero_on_violations"
        ]

    def test_and_lists_them(self):
        assert the_verify_command_fails_when_it_finds_something()["lists_them"]

    def test_the_first_line_is_a_summary(self):
        result = the_verify_command_fails_when_it_finds_something()
        assert "checks" in result["output"]


class TestPlumbing:
    def test_every_command_is_reachable(self):
        assert every_command_is_reachable()["they_agree"]

    def test_six_commands_exist(self):
        assert len(every_command_is_reachable()["declared"]) == 6

    def test_none_are_unreachable(self):
        assert every_command_is_reachable()["unreachable"] == []

    def test_an_outcome_reports_success(self):
        assert an_outcome_reports_success()["ok"]

    def test_and_failure(self):
        assert not an_outcome_reports_success()["not_ok"]

    def test_emitting_writes_the_output(self):
        out, err = io.StringIO(), io.StringIO()
        code = Outcome(0, "the result").emit(out, err)
        assert code == 0 and "the result" in out.getvalue()

    def test_and_the_error_separately(self):
        out, err = io.StringIO(), io.StringIO()
        Outcome(1, "", "the problem").emit(out, err)
        assert out.getvalue() == "" and "the problem" in err.getvalue()

    def test_an_empty_outcome_writes_nothing(self):
        out, err = io.StringIO(), io.StringIO()
        Outcome(0, "").emit(out, err)
        assert out.getvalue() == "" and err.getvalue() == ""

    def test_main_returns_the_exit_code(self, capsys):
        code = main(["measure", "--count", "512", "--dimension", "8"])
        capsys.readouterr()
        assert code == 0

    def test_and_non_zero_on_a_failure(self, capsys):
        code = main(["inspect", "--path", "nowhere.vse"])
        capsys.readouterr()
        assert code == 1

    def test_a_command_is_required(self):
        with pytest.raises(SystemExit):
            parser().parse_args([])

    def test_every_subcommand_takes_json(self):
        for name in ("measure", "report", "sweep", "verify"):
            args = parser().parse_args([name] + (["--out", "x"] if name == "build" else []))
            assert hasattr(args, "json")


class TestDrift:
    def test_the_drift_command_runs(self):
        assert the_drift_command_names_the_one_that_matters()["code"] == 0

    def test_it_names_the_scaling(self):
        assert the_drift_command_names_the_one_that_matters()["names_the_scaling"]

    def test_and_the_repair(self):
        assert the_drift_command_names_the_one_that_matters()["says_the_repair"]

    def test_it_produces_json(self):
        result = dispatch(["drift", "--json"])
        assert len(json.loads(result.output)) == 3

    def test_the_json_rows_carry_a_loss(self):
        rows = json.loads(dispatch(["drift", "--json"]).output)
        assert all("loss" in row for row in rows)

    def test_the_worst_drift_leads(self):
        rows = json.loads(dispatch(["drift", "--json"]).output)
        assert rows[0]["loss"] == max(row["loss"] for row in rows)


class TestStability:
    def test_both_noise_sources_are_reported(self):
        result = the_stability_command_prints_both_noise_sources()
        assert result["has_the_seed_noise"] and result["has_the_query_noise"]

    def test_they_come_out_level(self):
        assert the_stability_command_prints_both_noise_sources()["they_are_level"]

    def test_five_structures_are_measured(self):
        assert the_stability_command_prints_both_noise_sources()["structures"] == 5

    def test_every_row_carries_a_deviation(self):
        assert the_stability_table_carries_a_deviation_per_row()["every_row_has_one"]

    def test_the_deterministic_ones_report_zero(self):
        assert the_stability_table_carries_a_deviation_per_row()["some_are_zero"]

    def test_and_the_seeded_ones_do_not(self):
        assert the_stability_table_carries_a_deviation_per_row()["some_are_not"]

    def test_the_text_output_warns_about_small_gaps(self):
        result = dispatch(["stability"])
        assert "have not been measured" in result.output


class TestShortlist:
    def test_it_prices_against_a_scan(self):
        assert the_shortlist_command_prices_against_a_scan()["mentions_a_scan"]

    def test_and_reports_the_ratio(self):
        assert the_shortlist_command_prices_against_a_scan()["mentions_the_ratio"]

    def test_it_has_a_row_per_depth(self):
        assert the_shortlist_command_prices_against_a_scan()["has_a_row_per_depth"]

    def test_an_unreachable_target_is_reported(self):
        assert an_unreachable_shortlist_target_is_reported_not_raised()["says_so"]

    def test_and_still_exits_zero(self):
        assert an_unreachable_shortlist_target_is_reported_not_raised()["still_succeeds"]

    def test_a_target_above_one_is_refused(self):
        assert a_shortlist_target_outside_the_range_is_refused()["refused"]

    def test_and_the_refusal_names_the_target(self):
        assert a_shortlist_target_outside_the_range_is_refused()["names_the_target"]

    def test_a_target_of_nothing_is_refused(self):
        assert dispatch(["shortlist", "--target", "0"]).code == 1

    def test_it_produces_json(self):
        row = json.loads(dispatch(["shortlist", "--json"]).output)
        assert row["cheapest_depth"] == 1600

    def test_the_json_carries_the_full_scan_cost(self):
        row = json.loads(dispatch(["shortlist", "--json"]).output)
        assert row["full_scan_cost"] > row["cheapest_cost"]

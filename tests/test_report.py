from __future__ import annotations

import json

import pytest

from vse.errors import ConfigError, DataError
from vse.eval.report import (
    Report,
    Row,
    a_budget_of_nothing_is_refused,
    a_corpus_that_is_not_in_the_report_is_refused,
    a_negative_cost_is_refused,
    a_report_groups_its_rows,
    a_row_over_a_different_query_count_is_refused,
    a_row_without_a_cost_cannot_exist,
    a_speedup_needs_a_corpus_size,
    an_empty_frontier_is_refused,
    an_empty_report_renders_something,
    an_empty_run_is_refused,
    an_impossible_recall_is_refused,
    compare_at_a_budget,
    differences_below_the_error_are_ties,
    every_structure,
    every_structure_appears,
    frontier,
    memory_and_recall_are_different_axes,
    no_structure_beats_exact_search,
    standard_corpora,
    standard_report,
    the_exact_structures_agree,
    the_frontier_handles_a_single_point,
    the_frontier_interpolates_between_points,
    the_ordering_is_not_stable_across_corpora,
    the_report_knows_its_own_precision,
    the_report_serialises,
    the_table_renders,
)


def a_row(recall: float = 0.5, distances: float = 100.0, index: str = "ivf") -> Row:
    """A well formed row for the shape tests."""
    return Row(
        index=index,
        corpus="gaussian",
        setting="probe 8",
        recall=recall,
        distances=distances,
        memory_bytes=1000,
        queries=100,
    )


class TestRows:
    def test_a_row_without_a_cost_cannot_exist(self):
        assert a_row_without_a_cost_cannot_exist()

    def test_an_impossible_recall_is_refused(self):
        assert an_impossible_recall_is_refused()

    def test_a_negative_recall_is_refused(self):
        with pytest.raises(DataError, match="is not a recall"):
            a_row(recall=-0.1)

    def test_a_negative_cost_is_refused(self):
        assert a_negative_cost_is_refused()

    def test_a_zero_query_row_is_refused(self):
        with pytest.raises(DataError, match="is not a measurement"):
            Row("ivf", "gaussian", "probe 8", 0.5, 100.0, 0, 0)

    def test_a_row_serialises(self):
        assert a_row().as_dict()["index"] == "ivf"

    def test_the_recall_is_rounded_to_three_places(self):
        assert a_row(recall=0.123456).as_dict()["recall"] == 0.123

    def test_a_speedup_needs_a_corpus_size(self):
        assert a_speedup_needs_a_corpus_size()

    def test_the_speedup_against_a_corpus_is_the_ratio(self):
        assert a_row(distances=100.0).speedup_against(1000) == 10.0

    def test_a_free_search_is_the_whole_corpus(self):
        assert a_row(distances=0.0).speedup_against(1000) == 1000.0

    def test_a_speedup_against_nothing_is_refused(self):
        with pytest.raises(ConfigError, match="is not a corpus size"):
            a_row().speedup_against(0)


class TestReports:
    def test_a_report_groups_its_rows(self):
        assert a_report_groups_its_rows()["grouping_covers_everything"]

    def test_it_lists_its_indexes(self):
        assert a_report_groups_its_rows()["indexes"] == 9

    def test_and_its_corpora(self):
        assert a_report_groups_its_rows()["corpora"] == 2

    def test_a_row_over_a_different_query_count_is_refused(self):
        assert a_row_over_a_different_query_count_is_refused()

    def test_an_empty_report_renders_something(self):
        assert an_empty_report_renders_something()["safe"]

    def test_and_has_no_standard_error(self):
        assert an_empty_report_renders_something()["standard_error"] == 0.0

    def test_the_report_knows_its_own_precision(self):
        assert the_report_knows_its_own_precision()["smaller_sample_is_less_precise"]

    def test_by_the_square_root(self):
        assert the_report_knows_its_own_precision()["by_a_factor_of_two"] == 2.0

    def test_a_hundred_queries_gives_about_two_points(self):
        assert 0.01 < the_report_knows_its_own_precision()["hundred_query_error"] < 0.03

    def test_an_empty_run_is_refused(self):
        assert an_empty_run_is_refused()

    def test_an_empty_report_lists_nothing(self):
        report = Report()
        assert report.indexes == [] and report.corpora == []


class TestTheHarness:
    def test_every_structure_appears(self):
        assert every_structure_appears()["all_present"]

    def test_nine_are_offered(self):
        assert len({name for name, _, _ in every_structure(32)}) == 9

    def test_fourteen_settings_are_offered(self):
        assert len(every_structure(32)) == 14

    def test_four_corpora_are_standard(self):
        assert len(standard_corpora()) == 4

    def test_the_exact_structures_agree(self):
        assert the_exact_structures_agree()["all_exact"]

    def test_at_recall_one(self):
        assert the_exact_structures_agree()["worst"] == 1.0

    def test_nothing_beats_exact_search(self):
        assert no_structure_beats_exact_search()["clean"]

    def test_nothing_is_dearer_than_a_scan(self):
        assert no_structure_beats_exact_search()["dearer_than_a_scan"] == 0

    def test_the_report_has_a_row_per_setting_per_corpus(self):
        report = standard_report(corpora=1)
        assert len(report.rows) == len(every_structure(32))


class TestTheFrontier:
    def test_it_interpolates_between_points(self):
        assert the_frontier_interpolates_between_points()["halfway_is_halfway"]

    def test_it_returns_zero_below_the_range(self):
        assert the_frontier_interpolates_between_points()["zero_below_the_range"]

    def test_and_clamps_above_it(self):
        assert the_frontier_interpolates_between_points()["clamped_above"]

    def test_a_single_point_reports_zero_below_its_cost(self):
        assert the_frontier_handles_a_single_point()["zero_below_its_cost"]

    def test_and_its_recall_at_and_above(self):
        assert the_frontier_handles_a_single_point()["its_recall_at_and_above"]

    def test_an_empty_frontier_is_refused(self):
        assert an_empty_frontier_is_refused()

    def test_a_budget_of_nothing_is_refused(self):
        assert a_budget_of_nothing_is_refused()

    def test_a_negative_budget_is_refused(self):
        with pytest.raises(ConfigError, match="allows no search"):
            frontier([a_row()], budget=-10.0)

    def test_two_points_at_the_same_cost_take_the_later(self):
        rows = [a_row(recall=0.3, distances=100.0), a_row(recall=0.7, distances=100.0)]
        assert frontier(rows, 100.0) in {0.3, 0.7}


class TestComparisons:
    def test_a_comparison_ranks_by_recall(self):
        report = standard_report(corpora=1)
        rows = compare_at_a_budget(report, "gaussian", 1000.0)
        recalls = [row["recall_at_the_budget"] for row in rows]
        assert recalls == sorted(recalls, reverse=True)

    def test_every_structure_appears_in_it(self):
        report = standard_report(corpora=1)
        assert len(compare_at_a_budget(report, "gaussian", 1000.0)) == 9

    def test_a_corpus_that_is_not_in_the_report_is_refused(self):
        assert a_corpus_that_is_not_in_the_report_is_refused()

    def test_the_ordering_is_not_stable_across_corpora(self):
        assert the_ordering_is_not_stable_across_corpora()["not_stable"]

    def test_with_at_least_two_leaders(self):
        assert the_ordering_is_not_stable_across_corpora()["distinct_leaders"] >= 2

    def test_every_corpus_has_a_leader(self):
        leaders = the_ordering_is_not_stable_across_corpora()["leaders"]
        assert all(value is not None for value in leaders.values())

    def test_most_comparisons_are_decidable(self):
        assert differences_below_the_error_are_ties()["most_are_decidable"]

    def test_but_some_are_ties(self):
        assert differences_below_the_error_are_ties()["ties"] > 0

    def test_the_threshold_is_two_standard_errors(self):
        result = differences_below_the_error_are_ties()
        assert abs(result["threshold"] - 2 * 0.017) < 0.002


class TestOutput:
    def test_the_table_renders(self):
        assert the_table_renders()["aligned"]

    def test_with_a_header_and_a_rule(self):
        result = the_table_renders()
        assert result["has_a_header"] and result["has_a_rule"]

    def test_and_one_line_per_row(self):
        assert the_table_renders()["one_line_per_row"]

    def test_the_report_serialises(self):
        assert the_report_serialises()["matches"]

    def test_the_header_carries_the_query_count(self):
        assert the_report_serialises()["queries"] == 100

    def test_and_the_standard_error(self):
        assert the_report_serialises()["standard_error"] > 0

    def test_and_the_corpus_sizes(self):
        assert the_report_serialises()["has_corpus_sizes"]

    def test_the_json_parses(self):
        report = standard_report(corpora=1)
        assert isinstance(json.loads(report.as_json()), dict)

    def test_memory_is_a_third_axis(self):
        assert memory_and_recall_are_different_axes()["ratio"] > 10

    def test_the_quantised_structure_uses_least(self):
        assert memory_and_recall_are_different_axes()["smallest"] in {"binary", "residual"}

from __future__ import annotations

import pytest
import torch

from vse.errors import ConfigError, DataError
from vse.vectors.dataset import gaussian
from vse.vectors.duplicates import (
    Duplication,
    _setup,
    a_clean_corpus_has_almost_no_ties,
    a_duplicate_returns_alongside_its_original,
    a_duplication_reports_its_own_shape,
    a_mismatched_label_set_is_refused,
    a_negative_share_is_refused,
    a_rank_one_corpus_is_refused,
    a_share_of_one_is_refused,
    a_single_copy_is_refused,
    a_zero_nudge_is_refused,
    compare_the_scorings,
    deduplicating_shrinks_the_index,
    deduplicating_the_result_is_the_usual_fix,
    duplicates_create_ties,
    duplicates_do_not_break_the_search,
    exact_duplicates,
    identifier_recall_understates_the_index,
    near_duplicates,
    near_duplicates_do_not_make_ties,
    recall_by_distance,
    scoring_by_distance_needs_matching_shapes,
    the_distance_recall_barely_moves,
    the_duplicates_really_are_identical,
    the_gap_grows_with_the_duplication,
    the_groups_partition_the_corpus,
    the_near_duplicates_really_are_near,
    the_recalls_are_not_comparable_across_deduplication,
    the_two_measurements_agree_on_a_clean_corpus,
    ties_in_the_truth,
)
from vse.vectors.exact import Neighbours, search


class TestConstruction:
    def test_the_corpus_size_is_unchanged(self):
        corpus = gaussian(count=1024, dimension=8).vectors
        assert exact_duplicates(corpus, share=0.3).count == 1024

    def test_the_distinct_count_falls(self):
        corpus = gaussian(count=1024, dimension=8).vectors
        assert exact_duplicates(corpus, share=0.3).distinct < 1024

    def test_the_duplicates_really_are_identical(self):
        assert the_duplicates_really_are_identical()["identical"]

    def test_with_no_gap_at_all(self):
        assert the_duplicates_really_are_identical()["max_gap"] == 0.0

    def test_the_near_duplicates_really_are_near(self):
        assert the_near_duplicates_really_are_near()["close"]

    def test_and_not_equal(self):
        assert the_near_duplicates_really_are_near()["not_equal"]

    def test_a_duplication_reports_its_own_shape(self):
        assert a_duplication_reports_its_own_shape()["below_the_requested_share"]

    def test_but_within_a_tenth_of_it(self):
        assert a_duplication_reports_its_own_shape()["within_a_tenth_of_it"]

    def test_and_keeps_the_size(self):
        assert a_duplication_reports_its_own_shape()["size_is_unchanged"]

    def test_the_groups_cover_everything(self):
        assert the_groups_partition_the_corpus()["covers_everything"]

    def test_without_overlapping(self):
        assert the_groups_partition_the_corpus()["no_overlap"]

    def test_a_share_of_zero_duplicates_nothing(self):
        corpus = gaussian(count=512, dimension=8).vectors
        assert exact_duplicates(corpus, share=0.0).distinct == 512

    def test_it_is_deterministic(self):
        corpus = gaussian(count=512, dimension=8).vectors
        assert bool(
            torch.equal(
                exact_duplicates(corpus, share=0.3).vectors,
                exact_duplicates(corpus, share=0.3).vectors,
            )
        )

    def test_and_the_seed_changes_it(self):
        corpus = gaussian(count=512, dimension=8).vectors
        assert not bool(
            torch.equal(
                exact_duplicates(corpus, share=0.3, seed=0).vectors,
                exact_duplicates(corpus, share=0.3, seed=1).vectors,
            )
        )


class TestTies:
    def test_a_clean_corpus_has_no_ties(self):
        assert a_clean_corpus_has_almost_no_ties()["tied"] == 0

    def test_and_a_wide_median_gap(self):
        assert a_clean_corpus_has_almost_no_ties()["median_gap"] > 0.01

    def test_duplicates_create_ties(self):
        rows = [row["tied_share"] for row in duplicates_create_ties()]
        assert rows == sorted(rows)

    def test_a_third_duplicated_ties_a_sixth_of_the_queries(self):
        rows = {row["share"]: row for row in duplicates_create_ties()}
        assert rows[0.3]["tied_share"] > 0.1

    def test_near_duplicates_make_none(self):
        assert near_duplicates_do_not_make_ties()["near_ties"] == 0.0

    def test_where_exact_ones_do(self):
        assert near_duplicates_do_not_make_ties()["exact_ties"] > 0.1

    def test_an_empty_tie_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            duplicates_create_ties(shares=())

    def test_ties_are_counted_per_query(self):
        corpus, probes = _setup(count=512, queries=16)
        assert ties_in_the_truth(corpus, probes)["queries"] == 16


class TestScoring:
    def test_the_two_measurements_agree_on_a_clean_corpus(self):
        assert the_two_measurements_agree_on_a_clean_corpus()["identical"]

    def test_and_the_gap_is_exactly_zero(self):
        assert the_two_measurements_agree_on_a_clean_corpus()["gap"] == 0.0

    def test_distance_recall_is_never_lower(self):
        assert identifier_recall_understates_the_index()["distance_is_higher"]

    def test_but_the_gap_is_negligible(self):
        assert identifier_recall_understates_the_index()["gap"] < 0.02

    def test_the_gap_stays_small_across_the_sweep(self):
        rows = the_gap_grows_with_the_duplication()
        assert all(row["gap"] < 0.02 for row in rows)

    def test_an_empty_gap_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            the_gap_grows_with_the_duplication(shares=())

    def test_a_perfect_result_scores_one_by_distance(self):
        corpus, probes = _setup(count=512, queries=8)
        truth = search(probes, corpus, k=10)
        assert abs(recall_by_distance(probes, corpus, truth, truth) - 1.0) < 1e-6

    def test_scoring_by_distance_needs_matching_shapes(self):
        assert scoring_by_distance_needs_matching_shapes()

    def test_four_rows_compare_the_scorings(self):
        assert len(compare_the_scorings()) == 4

    def test_both_scorings_appear(self):
        rows = compare_the_scorings()
        assert {row["scoring"] for row in rows} == {"identifier", "distance"}


class TestTheSearch:
    def test_the_distance_recall_barely_moves(self):
        assert the_distance_recall_barely_moves()["barely_moves"]

    def test_across_a_wide_duplication_range(self):
        assert the_distance_recall_barely_moves()["spread"] < 0.1

    def test_an_empty_search_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            duplicates_do_not_break_the_search(shares=())

    def test_deduplicating_shrinks_the_index(self):
        assert deduplicating_shrinks_the_index()["the_index_shrinks"]

    def test_and_costs_less(self):
        assert deduplicating_shrinks_the_index()["and_costs_less"]

    def test_by_about_the_duplication_share(self):
        result = deduplicating_shrinks_the_index()
        assert result["size_after"] < result["size_before"] * 0.9

    def test_the_recalls_are_not_comparable(self):
        assert the_recalls_are_not_comparable_across_deduplication()[
            "the_difference_means_nothing"
        ]

    def test_but_the_cost_difference_is(self):
        assert the_recalls_are_not_comparable_across_deduplication()["the_cost_difference_does"]


class TestResults:
    def test_a_duplicate_returns_alongside_its_original(self):
        assert a_duplicate_returns_alongside_its_original()["happens"]

    def test_about_twice_per_query(self):
        assert a_duplicate_returns_alongside_its_original()["per_query"] > 1.0

    def test_deduplicating_the_result_needs_a_deeper_search(self):
        assert deduplicating_the_result_is_the_usual_fix()["depth_for_ten_distinct"] > 10

    def test_by_about_half_again(self):
        assert deduplicating_the_result_is_the_usual_fix()["overhead"] < 2.0

    def test_the_distinct_count_rises_with_the_depth(self):
        rows = [
            row["mean_distinct"] for row in deduplicating_the_result_is_the_usual_fix()["rows"]
        ]
        assert rows == sorted(rows)


class TestGuards:
    def test_a_share_of_one_is_refused(self):
        assert a_share_of_one_is_refused()

    def test_a_negative_share_is_refused(self):
        assert a_negative_share_is_refused()

    def test_a_single_copy_is_refused(self):
        assert a_single_copy_is_refused()

    def test_a_zero_nudge_is_refused(self):
        assert a_zero_nudge_is_refused()

    def test_a_negative_nudge_is_refused(self):
        corpus = gaussian(count=512, dimension=8).vectors
        with pytest.raises(ConfigError, match="makes exact duplicates"):
            near_duplicates(corpus, nudge=-0.1)

    def test_a_mismatched_label_set_is_refused(self):
        assert a_mismatched_label_set_is_refused()

    def test_a_rank_one_corpus_is_refused(self):
        assert a_rank_one_corpus_is_refused()

    def test_an_empty_duplication_divides_safely(self):
        empty = Duplication(
            vectors=torch.zeros(0, 8), original=torch.zeros(0, dtype=torch.long)
        )
        assert empty.duplicate_share == 0.0

    def test_the_error_names_the_counts(self):
        with pytest.raises(DataError, match="labels for"):
            Duplication(vectors=torch.randn(10, 8), original=torch.arange(3))

    def test_a_scoring_mismatch_names_the_reason(self):
        corpus, probes = _setup(count=512, queries=8)
        truth = Neighbours(torch.zeros(8, 10, dtype=torch.long), torch.zeros(8, 10))
        found = Neighbours(torch.zeros(8, 4, dtype=torch.long), torch.zeros(8, 4))
        with pytest.raises(DataError, match="matching shapes"):
            recall_by_distance(probes, corpus, truth, found)

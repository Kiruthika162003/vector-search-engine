from __future__ import annotations

import pytest
import torch

from vse.errors import ConfigError, DataError
from vse.eval.recall import (
    Scores,
    a_perfect_answer_scores_one_everywhere,
    a_zero_offset_near_miss_is_refused,
    but_the_gain_sees_position_when_the_result_is_mixed,
    compare_on_a_real_index,
    comparing_different_query_counts_is_refused,
    comparing_different_shapes_is_refused,
    discounted_gain,
    distance_ratio,
    k_changes_what_recall_means,
    near_misses,
    neither_set_measure_sees_a_pure_reordering,
    recall_at_k,
    recall_at_one_is_the_hard_one,
    recall_cannot_see_how_wrong_a_miss_is,
    reciprocal_rank,
    reciprocal_rank_only_sees_the_first_hit,
    score_all,
    shuffled,
    the_gain_can_exceed_the_recall,
    the_measures_agree_at_the_top_and_separate_at_the_bottom,
    the_measures_can_rank_two_results_oppositely,
)
from vse.vectors.dataset import gaussian, held_out
from vse.vectors.exact import Neighbours, search


class TestCalibration:
    def test_a_perfect_answer_scores_one_on_every_measure(self):
        assert a_perfect_answer_scores_one_everywhere()["all_one"]

    def test_the_distance_ratio_of_a_perfect_answer_is_exactly_one(self):
        assert a_perfect_answer_scores_one_everywhere()["distance_ratio"] == 1.0

    def test_comparing_different_shapes_is_refused(self):
        assert comparing_different_shapes_is_refused()

    def test_comparing_different_query_counts_is_refused(self):
        assert comparing_different_query_counts_is_refused()

    def test_a_zero_offset_near_miss_is_refused(self):
        assert a_zero_offset_near_miss_is_refused()

    def test_a_near_miss_shares_nothing_with_the_truth(self):
        corpus = gaussian(count=512, dimension=16)
        truth = search(corpus.vectors[:8], corpus.vectors, k=10)
        close = near_misses(corpus.vectors[:8], corpus.vectors, k=10, offset=10)
        assert recall_at_k(truth, close) == 0.0

    def test_the_scores_serialise(self):
        assert Scores(1.0, 1.0, 1.0, 1.0, 10).as_dict()["k"] == 10

    def test_a_mismatched_gain_is_refused(self):
        corpus = gaussian(count=512, dimension=16)
        truth = search(corpus.vectors[:8], corpus.vectors, k=10)
        other = search(corpus.vectors[:8], corpus.vectors, k=5)
        with pytest.raises(DataError, match="top 10 against top 5"):
            discounted_gain(truth, other)


class TestOrdering:
    def test_recall_cannot_see_a_reordering(self):
        assert neither_set_measure_sees_a_pure_reordering()["recall_unchanged"]

    def test_and_neither_can_the_gain(self):
        # Every item in a correct result is relevant, so permuting them permutes the weights
        # among relevant items and the total does not move.
        assert neither_set_measure_sees_a_pure_reordering()["gain_unchanged"]

    def test_but_the_gain_separates_a_mixed_result(self):
        assert but_the_gain_sees_position_when_the_result_is_mixed()["gain_separates"]

    def test_at_identical_recall(self):
        assert but_the_gain_sees_position_when_the_result_is_mixed()["recall_identical"]

    def test_by_thirty_points(self):
        result = but_the_gain_sees_position_when_the_result_is_mixed()
        assert result["front_gain"] - result["back_gain"] > 0.25

    def test_shuffling_preserves_the_identifiers(self):
        corpus = gaussian(count=512, dimension=16)
        truth = search(corpus.vectors[:8], corpus.vectors, k=10)
        mixed = shuffled(truth)
        for row in range(8):
            assert set(truth.row(row)) == set(mixed.row(row))


class TestSeverity:
    def test_recall_cannot_tell_a_near_miss_from_noise(self):
        assert recall_cannot_see_how_wrong_a_miss_is()["recall_cannot_tell"]

    def test_the_distance_ratio_can(self):
        assert recall_cannot_see_how_wrong_a_miss_is()["distance_can"]

    def test_by_a_factor_of_six_in_the_excess(self):
        assert recall_cannot_see_how_wrong_a_miss_is()["excess_ratio"] > 4.0

    def test_though_only_a_third_in_the_ratio_itself(self):
        # Concentration of distances leaves the measure less room than it looks like it has.
        result = recall_cannot_see_how_wrong_a_miss_is()
        assert result["random_ratio"] / result["near_miss_ratio"] < 1.5

    def test_a_near_miss_is_a_few_percent_further(self):
        assert recall_cannot_see_how_wrong_a_miss_is()["near_miss_ratio"] < 1.15


class TestDisagreement:
    def test_the_gain_can_exceed_the_recall(self):
        assert the_gain_can_exceed_the_recall()["gain_exceeds_recall_somewhere"]

    def test_but_they_agree_at_the_extremes(self):
        assert the_gain_can_exceed_the_recall()["they_agree_at_the_extremes"]

    def test_two_measures_can_rank_two_results_oppositely(self):
        assert the_measures_can_rank_two_results_oppositely()["they_disagree"]

    def test_recall_prefers_the_one_with_more_hits(self):
        assert the_measures_can_rank_two_results_oppositely()["recall_prefers_the_first"]

    def test_and_the_distance_prefers_the_one_with_closer_misses(self):
        assert the_measures_can_rank_two_results_oppositely()["distance_prefers_the_second"]

    def test_the_reciprocal_rank_ignores_everything_after_the_first_hit(self):
        assert reciprocal_rank_only_sees_the_first_hit()["identical_reciprocal"]

    def test_while_the_recall_of_the_same_result_is_a_tenth(self):
        assert reciprocal_rank_only_sees_the_first_hit()["first_right_recall"] == 0.1

    def test_a_result_with_no_hits_has_no_reciprocal_rank(self):
        corpus = gaussian(count=512, dimension=16)
        truth = search(corpus.vectors[:8], corpus.vectors, k=10)
        close = near_misses(corpus.vectors[:8], corpus.vectors, k=10, offset=20)
        assert reciprocal_rank(truth, close) == 0.0


class TestK:
    def test_recall_at_one_is_the_hardest(self):
        result = recall_at_one_is_the_hard_one()
        assert result["at_one"] < result["at_fifty"]

    def test_the_same_shift_scores_zero_at_one_and_most_of_the_way_at_fifty(self):
        result = recall_at_one_is_the_hard_one()
        assert result["at_one"] == 0.0
        assert result["at_fifty"] > 0.5

    def test_while_the_distance_ratio_barely_moves(self):
        assert recall_at_one_is_the_hard_one()["recall_moved_more"]

    def test_the_ratio_is_close_to_one_at_every_k(self):
        rows = k_changes_what_recall_means()
        assert all(1.0 <= row["ratio"] < 1.2 for row in rows)

    def test_an_empty_k_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            k_changes_what_recall_means(values=())


class TestOnARealIndex:
    def test_three_settings_are_measured(self):
        assert len(compare_on_a_real_index()) == 3

    def test_every_measure_improves_with_the_probe_count(self):
        rows = compare_on_a_real_index()
        assert [row["recall"] for row in rows] == sorted(row["recall"] for row in rows)
        assert [row["gain"] for row in rows] == sorted(row["gain"] for row in rows)

    def test_and_the_distance_ratio_falls_towards_one(self):
        rows = [row["distance_ratio"] for row in compare_on_a_real_index()]
        assert rows == sorted(rows, reverse=True)

    def test_the_recall_moves_much_further_than_the_distance_ratio(self):
        assert the_measures_agree_at_the_top_and_separate_at_the_bottom()["recall_moves_more"]

    def test_by_a_factor_of_seven(self):
        result = the_measures_agree_at_the_top_and_separate_at_the_bottom()
        assert result["recall_spread"] > result["ratio_spread"] * 4

    def test_so_the_flattering_measure_is_the_one_a_user_would_notice(self):
        result = the_measures_agree_at_the_top_and_separate_at_the_bottom()
        assert result["at_one_probe"]["distance_ratio"] < 1.2
        assert result["at_one_probe"]["recall"] < 0.3

    def test_the_reciprocal_rank_saturates_first(self):
        rows = {row["probe"]: row for row in compare_on_a_real_index()}
        assert rows[4]["reciprocal_rank"] == 1.0
        assert rows[4]["recall"] < 1.0


class TestMechanics:
    def test_the_gain_of_an_empty_hit_set_is_zero(self):
        corpus = gaussian(count=512, dimension=16)
        truth = search(corpus.vectors[:8], corpus.vectors, k=10)
        far = near_misses(corpus.vectors[:8], corpus.vectors, k=10, offset=50)
        assert discounted_gain(truth, far) == 0.0

    def test_the_distance_ratio_is_never_below_one(self):
        corpus = gaussian(count=512, dimension=16)
        searched, probes = held_out(corpus, count=16)
        truth = search(probes, searched.vectors, k=10)
        close = near_misses(probes, searched.vectors, k=10, offset=5)
        assert distance_ratio(probes, searched.vectors, truth, close) >= 1.0

    def test_scoring_everything_at_once_matches_the_pieces(self):
        corpus = gaussian(count=512, dimension=16)
        searched, probes = held_out(corpus, count=16)
        truth = search(probes, searched.vectors, k=10)
        close = near_misses(probes, searched.vectors, k=10, offset=5)
        scores = score_all(probes, searched.vectors, truth, close)
        assert scores.recall == recall_at_k(truth, close)
        assert scores.gain == discounted_gain(truth, close)

    def test_a_result_of_random_identifiers_scores_near_zero(self):
        corpus = gaussian(count=512, dimension=16)
        truth = search(corpus.vectors[:8], corpus.vectors, k=10)
        noise = Neighbours(
            identifiers=torch.randint(0, 512, (8, 10)), scores=torch.zeros(8, 10)
        )
        assert recall_at_k(truth, noise) < 0.2

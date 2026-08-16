from __future__ import annotations

import pytest
import torch

from vse.errors import ConfigError
from vse.eval.stability import (
    FIXED,
    SEEDED,
    Spread,
    _setup,
    a_clustered_corpus_is_less_seed_sensitive,
    a_single_trial_null_is_refused,
    a_spread_of_nothing_is_refused,
    across_seeds,
    an_empty_seed_list_is_refused,
    an_unknown_index_is_refused,
    averaging_over_seeds_narrows_the_interval,
    build,
    close_pairs_flip_and_distant_ones_do_not,
    compare_the_structures,
    does_the_leader_change_with_the_seed,
    how_many_seeds_a_comparison_needs,
    per_query_recall,
    query_standard_error,
    summarise,
    the_corpus_seed_moves_it_no_further_than_the_index_seed,
    the_cost_moves_with_the_seed_too,
    the_deterministic_structures_do_not_move,
    the_flip_rate_follows_the_gap,
    the_noise_is_worst_in_the_middle,
    the_number_of_random_draws_does_not_predict_the_spread,
    the_seed_and_the_query_noise_are_the_same_size,
    the_seeded_structures_move,
    the_spread_across_the_probe_range,
    two_identical_methods_flip_half_the_time,
)
from vse.vectors.exact import Neighbours


class TestTheControl:
    def test_the_deterministic_structures_do_not_move(self):
        assert the_deterministic_structures_do_not_move()["nothing_moved"]

    def test_nor_do_their_costs(self):
        assert the_deterministic_structures_do_not_move()["and_the_costs_did_not_either"]

    def test_the_kd_tree_is_exact(self):
        assert the_deterministic_structures_do_not_move()["recalls"]["tree"] == 1.0

    def test_both_fixed_structures_are_measured(self):
        assert set(the_deterministic_structures_do_not_move()["ranges"]) == set(FIXED)


class TestTheSpreads:
    def test_all_three_seeded_structures_are_measured(self):
        assert {row["index"] for row in the_seeded_structures_move()} == set(SEEDED)

    def test_every_one_of_them_moves(self):
        assert all(row["range"] > 0.0 for row in the_seeded_structures_move())

    def test_but_none_by_much(self):
        assert all(row["deviation"] < 0.02 for row in the_seeded_structures_move())

    def test_eight_seeds_are_used(self):
        assert all(row["seeds"] == 8 for row in the_seeded_structures_move())

    def test_an_empty_seed_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            the_seeded_structures_move(seeds=())

    def test_the_number_of_draws_does_not_predict_the_spread(self):
        assert the_number_of_random_draws_does_not_predict_the_spread()[
            "they_are_all_within_a_third"
        ]

    def test_the_hash_is_not_the_worst(self):
        result = the_number_of_random_draws_does_not_predict_the_spread()
        assert result["the_hash_is_not_the_worst"]


class TestAgainstTheQueryNoise:
    def test_the_seed_and_the_queries_are_level(self):
        assert the_seed_and_the_query_noise_are_the_same_size()["they_are_level"]

    def test_the_combined_noise_is_about_two_points(self):
        assert the_seed_and_the_query_noise_are_the_same_size()["combined"] < 0.02

    def test_neither_source_is_negligible(self):
        result = the_seed_and_the_query_noise_are_the_same_size()
        assert 0.5 < result["ratio"] < 2.0

    def test_a_standard_error_needs_two_queries(self):
        truth = Neighbours(torch.zeros(1, 5, dtype=torch.long), torch.zeros(1, 5))
        with pytest.raises(ConfigError, match="at least two queries"):
            query_standard_error(truth, truth)


class TestTheLeader:
    def test_the_leader_is_the_inverted_file(self):
        assert does_the_leader_change_with_the_seed()["by_mean"][0] == "ivf"

    def test_and_every_seed_agrees(self):
        assert does_the_leader_change_with_the_seed()["every_seed_agreed"]

    def test_because_the_gaps_are_wide(self):
        assert does_the_leader_change_with_the_seed()["the_narrowest_gap"] > 0.05

    def test_an_empty_seed_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            does_the_leader_change_with_the_seed(seeds=())

    def test_no_pair_flips(self):
        assert the_flip_rate_follows_the_gap()["nothing_flips_at_all"]

    def test_three_pairs_are_compared(self):
        assert len(close_pairs_flip_and_distant_ones_do_not()) == 3

    def test_the_pairs_come_back_sorted_by_gap(self):
        gaps = [row["gap"] for row in close_pairs_flip_and_distant_ones_do_not()]
        assert gaps == sorted(gaps)

    def test_the_widest_pair_never_flips(self):
        assert the_flip_rate_follows_the_gap()["the_widest_never_flips"]

    def test_an_empty_pair_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            close_pairs_flip_and_distant_ones_do_not(seeds=())


class TestTheNull:
    def test_two_identical_methods_split_about_evenly(self):
        assert two_identical_methods_flip_half_the_time()["it_is_near_a_half"]

    def test_and_their_means_agree(self):
        assert two_identical_methods_flip_half_the_time()["the_means_are_level"]

    def test_the_trial_count_is_reported(self):
        assert two_identical_methods_flip_half_the_time(trials=6)["trials"] == 6

    def test_a_single_trial_is_refused(self):
        assert a_single_trial_null_is_refused()


class TestAveraging:
    def test_the_deviation_falls_with_the_count(self):
        rows = [row["deviation"] for row in averaging_over_seeds_narrows_the_interval()]
        assert rows == sorted(rows, reverse=True)

    def test_and_roughly_as_the_root_law_says(self):
        rows = averaging_over_seeds_narrows_the_interval()
        assert all(row["deviation"] <= row["predicted"] * 1.2 for row in rows)

    def test_a_single_seed_matches_the_law_exactly(self):
        first = averaging_over_seeds_narrows_the_interval(counts=(1,))[0]
        assert first["deviation"] == first["predicted"]

    def test_more_seeds_than_measured_is_refused(self):
        with pytest.raises(ConfigError, match="were not measured"):
            averaging_over_seeds_narrows_the_interval(counts=(32,))

    def test_an_empty_count_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            averaging_over_seeds_narrows_the_interval(counts=())

    def test_a_two_point_gap_needs_a_couple_of_seeds(self):
        assert how_many_seeds_a_comparison_needs()["most"] <= 4

    def test_and_that_is_affordable(self):
        assert how_many_seeds_a_comparison_needs()["all_are_affordable"]

    def test_a_narrower_gap_needs_more_seeds(self):
        wide = how_many_seeds_a_comparison_needs(gap=0.05)["most"]
        narrow = how_many_seeds_a_comparison_needs(gap=0.005)["most"]
        assert narrow > wide

    def test_a_gap_of_nothing_is_refused(self):
        with pytest.raises(ConfigError, match="is not a gap"):
            how_many_seeds_a_comparison_needs(gap=0.0)


class TestAcrossTheRange:
    def test_six_probe_settings_are_measured(self):
        assert len(the_spread_across_the_probe_range()) == 6

    def test_the_recall_rises_with_the_probe(self):
        recalls = [row["recall"] for row in the_spread_across_the_probe_range()]
        assert recalls == sorted(recalls)

    def test_an_exhaustive_probe_has_no_spread(self):
        assert the_spread_across_the_probe_range()[-1]["deviation"] == 0.0

    def test_the_noise_peaks_inside_the_range(self):
        assert the_noise_is_worst_in_the_middle()["the_peak_is_interior"]

    def test_and_both_ends_are_quieter(self):
        assert the_noise_is_worst_in_the_middle()["the_ends_are_quieter"]

    def test_the_peak_is_at_low_recall(self):
        assert the_noise_is_worst_in_the_middle()["peak_recall"] < 0.5

    def test_an_empty_probe_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            the_spread_across_the_probe_range(probe_values=())


class TestOtherSources:
    def test_the_corpus_seed_does_not_dominate(self):
        result = the_corpus_seed_moves_it_no_further_than_the_index_seed()
        assert result["the_corpus_does_not_dominate"]

    def test_the_two_are_level(self):
        assert the_corpus_seed_moves_it_no_further_than_the_index_seed()["they_are_level"]

    def test_an_empty_corpus_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            the_corpus_seed_moves_it_no_further_than_the_index_seed(seeds=())

    def test_a_clustered_corpus_moves_less(self):
        assert a_clustered_corpus_is_less_seed_sensitive()["the_clustered_corpus_moves_less"]

    def test_because_it_saturates(self):
        assert a_clustered_corpus_is_less_seed_sensitive()["because_it_saturates"]

    def test_and_scores_higher(self):
        assert a_clustered_corpus_is_less_seed_sensitive()["and_scores_higher"]

    def test_an_unknown_corpus_is_refused(self):
        with pytest.raises(ConfigError, match="is not a corpus"):
            _setup(kind="spiral")


class TestCost:
    def test_the_forest_cost_is_steady(self):
        assert the_cost_moves_with_the_seed_too()["the_forest_is_steady"]

    def test_the_others_are_not(self):
        assert the_cost_moves_with_the_seed_too()["the_others_are_not"]

    def test_the_hash_moves_most(self):
        assert the_cost_moves_with_the_seed_too()["worst"] == "lsh"

    def test_but_still_under_a_tenth(self):
        assert the_cost_moves_with_the_seed_too()["worst_relative"] < 0.1


class TestComparison:
    def test_every_structure_appears(self):
        rows = compare_the_structures()
        assert {row["index"] for row in rows} == set(SEEDED) | set(FIXED)

    def test_the_rows_come_back_sorted_by_recall(self):
        means = [row["mean"] for row in compare_the_structures()]
        assert means == sorted(means, reverse=True)

    def test_the_fixed_structures_have_no_deviation(self):
        rows = {row["index"]: row for row in compare_the_structures()}
        assert all(rows[name]["deviation"] == 0.0 for name in FIXED)

    def test_an_empty_comparison_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            compare_the_structures(seeds=())

    def test_the_summary_names_the_leader(self):
        assert summarise()["leader_by_mean"] == "ivf"

    def test_and_reports_both_noise_sources(self):
        result = summarise()
        assert result["seed_noise"] > 0.0 and result["query_noise"] > 0.0

    def test_and_how_many_seeds_agreed(self):
        assert summarise()["seeds_agreeing"] == 8


class TestMechanics:
    def test_every_named_index_builds(self):
        for name in (*SEEDED, *FIXED):
            assert build(name, dimension=8, seed=0) is not None

    def test_an_unknown_index_is_refused(self):
        assert an_unknown_index_is_refused()

    def test_an_empty_seed_list_is_refused(self):
        assert an_empty_seed_list_is_refused()

    def test_a_spread_with_no_recalls_is_refused(self):
        assert a_spread_of_nothing_is_refused()

    def test_a_spread_of_one_seed_has_no_deviation(self):
        spread = Spread(name="ivf", recalls=[0.5], distances=[10.0])
        assert spread.deviation == 0.0 and spread.range == 0.0

    def test_a_spread_reports_its_mean(self):
        spread = Spread(name="ivf", recalls=[0.4, 0.6], distances=[10.0, 12.0])
        assert spread.mean == 0.5 and spread.mean_distances == 11.0

    def test_a_spread_reports_its_cost_range(self):
        spread = Spread(name="ivf", recalls=[0.4, 0.6], distances=[10.0, 12.0])
        assert spread.cost_range == 2.0

    def test_a_spread_serialises(self):
        spread = Spread(name="ivf", recalls=[0.4, 0.6], distances=[10.0, 12.0])
        assert spread.as_dict()["index"] == "ivf" and spread.as_dict()["seeds"] == 2

    def test_per_query_recall_returns_one_score_per_query(self):
        corpus, probes, truth = _setup(count=512, queries=8)
        index = build("ivf", int(corpus.shape[1]), seed=0)
        index.build(corpus)
        found, _ = index.search(probes, k=10)
        assert len(per_query_recall(truth, found)) == 8

    def test_every_score_is_inside_the_unit_interval(self):
        corpus, probes, truth = _setup(count=512, queries=8)
        index = build("ivf", int(corpus.shape[1]), seed=0)
        index.build(corpus)
        found, _ = index.search(probes, k=10)
        assert all(0.0 <= score <= 1.0 for score in per_query_recall(truth, found))

    def test_the_truth_scores_perfectly_against_itself(self):
        _, _, truth = _setup(count=512, queries=8)
        assert per_query_recall(truth, truth) == [1.0] * 8

    def test_a_spread_over_one_seed_is_allowed(self):
        corpus, probes, truth = _setup(count=512, queries=8)
        spread = across_seeds("ivf", seeds=(0,), corpus=corpus, probes=probes, truth=truth)
        assert len(spread.recalls) == 1

    def test_a_spread_builds_its_own_corpus_when_given_none(self):
        assert across_seeds("lsh", seeds=(0, 1)).mean > 0.0

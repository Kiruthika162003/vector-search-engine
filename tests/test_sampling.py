from __future__ import annotations

import pytest
import torch

from vse.build.sampling import (
    Fit,
    a_biased_sample_is_worse_than_a_small_one,
    a_biased_sample_of_the_wrong_size_is_refused,
    a_clustered_corpus_samples_more_easily,
    a_fit_reports_what_it_was_fitted_on,
    a_full_fit_and_a_full_sample_agree,
    a_low_rank_corpus_samples_easily,
    a_sample_larger_than_the_corpus_is_refused,
    a_sample_smaller_than_the_partitions_is_refused,
    a_zero_sharpness_bias_is_refused,
    an_empty_fit_divides_safely,
    an_empty_sample_is_refused,
    at_a_fixed_probe_a_worse_fit_looks_better,
    at_matched_cost_a_fuller_fit_wins,
    dense_sample,
    fit_on_sample,
    index_from,
    more_partitions_than_vectors_is_refused,
    sampling_without_replacement_matters_when_the_sample_is_large,
    the_assignment_becomes_the_floor,
    the_balance_keeps_improving_past_the_recommendation,
    the_build_cost_falls_with_the_sample,
    the_centroids_move_more_than_the_balance_does,
    the_probe_that_fits_the_budget_falls_with_the_fit,
    the_recall_falls_because_the_scan_grows,
    the_rule_of_thumb_is_roughly_right,
    the_sample_buys_balance,
    the_sample_does_not_grow_with_the_corpus,
    uniform_sample,
    what_matters_is_samples_per_centroid,
)
from vse.errors import ConfigError
from vse.vectors.dataset import gaussian


class TestSamplers:
    def test_a_uniform_sample_is_the_right_size(self):
        corpus = gaussian(count=1024, dimension=8).vectors
        assert int(uniform_sample(corpus, 100).shape[0]) == 100

    def test_and_has_no_duplicates(self):
        corpus = gaussian(count=1024, dimension=8).vectors
        sample = uniform_sample(corpus, 100)
        assert int(torch.unique(sample, dim=0).shape[0]) == 100

    def test_it_is_deterministic(self):
        corpus = gaussian(count=1024, dimension=8).vectors
        assert bool(torch.equal(uniform_sample(corpus, 50), uniform_sample(corpus, 50)))

    def test_and_the_seed_changes_it(self):
        corpus = gaussian(count=1024, dimension=8).vectors
        assert not bool(
            torch.equal(uniform_sample(corpus, 50, seed=0), uniform_sample(corpus, 50, seed=1))
        )

    def test_a_sample_larger_than_the_corpus_is_refused(self):
        assert a_sample_larger_than_the_corpus_is_refused()

    def test_an_empty_sample_is_refused(self):
        assert an_empty_sample_is_refused()

    def test_a_dense_sample_is_the_right_size(self):
        corpus = gaussian(count=1024, dimension=8).vectors
        assert int(dense_sample(corpus, 100).shape[0]) == 100

    def test_and_sits_nearer_the_middle(self):
        corpus = gaussian(count=2048, dimension=8).vectors
        centre = corpus.mean(dim=0, keepdim=True)
        biased = float(((dense_sample(corpus, 200) - centre) ** 2).sum(dim=1).mean())
        plain = float(((uniform_sample(corpus, 200) - centre) ** 2).sum(dim=1).mean())
        assert biased < plain

    def test_a_biased_sample_of_the_wrong_size_is_refused(self):
        assert a_biased_sample_of_the_wrong_size_is_refused()

    def test_a_zero_sharpness_bias_is_refused(self):
        assert a_zero_sharpness_bias_is_refused()

    def test_replacement_only_bites_at_scale(self):
        result = sampling_without_replacement_matters_when_the_sample_is_large()
        assert result["replacement_only_bites_at_scale"]

    def test_drawing_from_its_own_size_loses_a_third(self):
        result = sampling_without_replacement_matters_when_the_sample_is_large()
        assert (
            abs(
                result["distinct_drawn_from_its_own_size"] - result["predicted_at_its_own_size"]
            )
            < 15
        )

    def test_and_without_replacement_loses_nothing(self):
        assert sampling_without_replacement_matters_when_the_sample_is_large()[
            "without_is_exact"
        ]


class TestTheInversion:
    def test_at_a_fixed_probe_less_sample_looks_better(self):
        assert the_recall_falls_because_the_scan_grows()["recall_looks_better_with_less_sample"]

    def test_because_it_scans_far_more(self):
        assert the_recall_falls_because_the_scan_grows()["but_it_scans_far_more"]

    def test_and_the_efficiency_is_worse(self):
        assert the_recall_falls_because_the_scan_grows()["recall_per_distance_is_worse"]

    def test_the_distance_count_falls_with_the_sample(self):
        rows = [row["distances"] for row in at_a_fixed_probe_a_worse_fit_looks_better()]
        assert rows == sorted(rows, reverse=True)

    def test_and_so_does_the_partition_spread(self):
        rows = [row["partition_spread"] for row in at_a_fixed_probe_a_worse_fit_looks_better()]
        assert rows == sorted(rows, reverse=True)

    def test_an_empty_fixed_probe_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            at_a_fixed_probe_a_worse_fit_looks_better(shares=())


class TestMatchedCost:
    def test_at_matched_cost_a_fuller_fit_wins(self):
        assert the_sample_buys_balance()["and_so_does_recall"]

    def test_and_the_balance_improves(self):
        assert the_sample_buys_balance()["balance_improves"]

    def test_by_an_order_of_magnitude(self):
        result = the_sample_buys_balance()
        assert result["spread_at_one_percent"] > result["spread_on_everything"] * 5

    def test_the_largest_partition_shrinks(self):
        result = the_sample_buys_balance()
        assert result["largest_at_one_percent"] > result["largest_on_everything"] * 5

    def test_the_probe_that_fits_the_budget_falls(self):
        assert the_probe_that_fits_the_budget_falls_with_the_fit()["falls"]

    def test_and_every_row_is_inside_the_budget(self):
        assert the_probe_that_fits_the_budget_falls_with_the_fit()[
            "every_configuration_is_inside_the_budget"
        ]

    def test_an_empty_matched_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            at_matched_cost_a_fuller_fit_wins(shares=())

    def test_a_budget_of_nothing_is_refused(self):
        with pytest.raises(ConfigError, match="allows no search"):
            at_matched_cost_a_fuller_fit_wins(budget=0.0)


class TestTheRuleOfThumb:
    def test_the_balance_improves_throughout(self):
        assert the_balance_keeps_improving_past_the_recommendation()[
            "balance_improves_throughout"
        ]

    def test_and_so_does_the_recall(self):
        assert the_balance_keeps_improving_past_the_recommendation()["recall_improves_too"]

    def test_more_rounds_are_needed_with_more_sample(self):
        rows = [row["rounds"] for row in the_rule_of_thumb_is_roughly_right()]
        assert rows == sorted(rows)

    def test_the_sample_size_scales_with_the_setting(self):
        rows = {row["per_centroid"]: row for row in the_rule_of_thumb_is_roughly_right()}
        assert rows[16]["sample_size"] == rows[4]["sample_size"] * 4

    def test_an_empty_rule_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            the_rule_of_thumb_is_roughly_right(per_centroid=())


class TestScaling:
    def test_the_sample_share_falls_as_the_corpus_grows(self):
        rows = [row["sample_share"] for row in what_matters_is_samples_per_centroid()]
        assert rows == sorted(rows, reverse=True)

    def test_and_the_balance_improves(self):
        assert the_sample_does_not_grow_with_the_corpus()["balance_improves"]

    def test_by_a_wide_margin(self):
        assert the_sample_does_not_grow_with_the_corpus()["share_falls_sixteenfold"]

    def test_an_empty_corpus_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            what_matters_is_samples_per_centroid(counts=())


class TestWhatSamplingBreaks:
    def test_the_centroids_move_a_long_way(self):
        assert the_centroids_move_more_than_the_balance_does()["move_relative_to_spread"] > 1.0

    def test_and_the_balance_is_what_differs(self):
        assert the_centroids_move_more_than_the_balance_does()["the_balance_is_what_differs"]

    def test_a_biased_sample_costs_recall(self):
        assert a_biased_sample_is_worse_than_a_small_one()["biased_is_worse"]

    def test_which_the_spread_does_not_show(self):
        assert a_biased_sample_is_worse_than_a_small_one()["the_spread_does_not_show_it"]

    def test_but_the_largest_partition_does(self):
        assert a_biased_sample_is_worse_than_a_small_one()["the_largest_partition_does"]

    def test_a_clustered_corpus_loses_nothing(self):
        assert a_clustered_corpus_samples_more_easily()["clustered_loses_nothing"]

    def test_and_reaches_perfect_recall(self):
        assert a_clustered_corpus_samples_more_easily()["clustered_full"] == 1.0

    def test_a_low_rank_corpus_samples_for_free(self):
        assert a_low_rank_corpus_samples_easily()["nearly_free"]

    def test_with_no_loss_at_all(self):
        assert a_low_rank_corpus_samples_easily()["loss"] == 0.0


class TestBuildCost:
    def test_the_fitting_cost_falls_with_the_sample(self):
        rows = [row["fitting_distances"] for row in the_build_cost_falls_with_the_sample()]
        assert rows == sorted(rows)

    def test_the_assignment_cost_does_not(self):
        rows = {row["share"]: row for row in the_build_cost_falls_with_the_sample()}
        assert rows[0.01]["assigning_distances"] == rows[1.0]["assigning_distances"]

    def test_the_assignment_dominates_a_small_sample(self):
        assert the_assignment_becomes_the_floor()["assignment_dominates_a_small_sample"]

    def test_and_sampling_saves_most_of_the_build(self):
        assert the_assignment_becomes_the_floor()["saving"] > 0.9

    def test_an_empty_cost_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            the_build_cost_falls_with_the_sample(shares=())


class TestFits:
    def test_a_fit_reports_everything(self):
        assert a_fit_reports_what_it_was_fitted_on()["has_everything"]

    def test_the_samples_per_centroid_is_right(self):
        assert a_fit_reports_what_it_was_fitted_on()["samples_per_centroid"] == 32.0

    def test_and_the_share(self):
        assert a_fit_reports_what_it_was_fitted_on()["sample_share"] == 0.25

    def test_an_empty_fit_divides_safely(self):
        assert an_empty_fit_divides_safely()["safe"]

    def test_a_fit_knows_its_partition_count(self):
        assert Fit(torch.zeros(7, 4), 100, 1000, 3, 1.0).partitions == 7

    def test_a_full_sample_is_the_same_as_no_sample(self):
        assert a_full_fit_and_a_full_sample_agree()["identical_centres"]

    def test_with_the_same_inertia(self):
        assert a_full_fit_and_a_full_sample_agree()["same_inertia"]

    def test_a_sample_smaller_than_the_partitions_is_refused(self):
        assert a_sample_smaller_than_the_partitions_is_refused()

    def test_more_partitions_than_vectors_is_refused(self):
        assert more_partitions_than_vectors_is_refused()

    def test_a_zero_partition_fit_is_refused(self):
        with pytest.raises(ConfigError, match="not a clustering"):
            fit_on_sample(torch.randn(100, 8), partitions=0)

    def test_an_index_built_from_a_fit_searches(self):
        corpus = gaussian(count=2048, dimension=16).vectors
        index = index_from(fit_on_sample(corpus, 32, sample_size=512), corpus, probe=4)
        found, _ = index.search(corpus[:8], k=5)
        assert tuple(found.identifiers.shape) == (8, 5)

    def test_and_finds_the_query_itself(self):
        corpus = gaussian(count=2048, dimension=16).vectors
        index = index_from(fit_on_sample(corpus, 32, sample_size=512), corpus, probe=8)
        found, _ = index.search(corpus[:1], k=1)
        assert int(found.identifiers[0, 0]) == 0

    def test_it_holds_the_whole_corpus(self):
        corpus = gaussian(count=2048, dimension=16).vectors
        index = index_from(fit_on_sample(corpus, 32, sample_size=512), corpus)
        assert index.size == 2048

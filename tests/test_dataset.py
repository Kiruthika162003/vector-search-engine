from __future__ import annotations

import pytest
import torch

from vse.errors import ConfigError, DataError
from vse.vectors.dataset import (
    Corpus,
    a_corpus_of_one_vector_is_refused,
    a_random_nudge_destabilises_low_dimensions_more,
    ambient_dimension_is_not_what_matters,
    an_intrinsic_wider_than_the_ambient_is_refused,
    clustered,
    clusters_are_easier_than_noise,
    compare_corpora,
    contrast_by_dimension,
    contrast_collapses_with_dimension,
    estimate_intrinsic_dimension,
    gaussian,
    held_out,
    more_queries_than_vectors_is_refused,
    nearest_neighbour_stability,
    neighbour_margin,
    noise_raises_the_estimate,
    on_a_subspace,
    perturbed_queries,
    perturbed_queries_measure_the_nudge,
    recall_charges_full_price_for_a_small_error,
    relative_contrast,
    the_contrast_ordering_matches_the_difficulty,
    the_estimator_recovers_a_known_dimension,
    the_estimator_sees_through_the_rotation,
    the_expected_contrast_at_a_dimension,
    the_margin_collapses_with_dimension,
    the_measurement_follows_the_closed_form,
    the_perturbed_contrast_is_flat_in_dimension,
    typical_distance,
)


class TestCorpora:
    def test_a_gaussian_corpus_has_the_shape_asked_for(self):
        corpus = gaussian(count=512, dimension=24)
        assert (corpus.count, corpus.dimension) == (512, 24)

    def test_its_intrinsic_dimension_is_its_ambient_one(self):
        assert gaussian(count=64, dimension=8).intrinsic == 8

    def test_a_subspace_corpus_is_wide_and_thin(self):
        corpus = on_a_subspace(count=256, dimension=128, intrinsic=4)
        assert corpus.dimension == 128
        assert corpus.intrinsic == 4

    def test_and_really_lies_on_a_subspace(self):
        corpus = on_a_subspace(count=256, dimension=32, intrinsic=4)
        assert int(torch.linalg.matrix_rank(corpus.vectors)) == 4

    def test_noise_takes_it_off_the_subspace(self):
        corpus = on_a_subspace(count=256, dimension=32, intrinsic=4, noise=0.1)
        assert int(torch.linalg.matrix_rank(corpus.vectors)) > 4

    def test_a_clustered_corpus_has_tight_groups(self):
        # Every point is within a few spreads of its centre, so distances within a group are
        # much smaller than distances between them.
        corpus = clustered(count=1024, dimension=16, clusters=8, spread=0.1)
        assert typical_distance(corpus) > 1.0

    def test_a_corpus_of_one_vector_is_refused(self):
        assert a_corpus_of_one_vector_is_refused()

    def test_an_intrinsic_wider_than_the_ambient_is_refused(self):
        assert an_intrinsic_wider_than_the_ambient_is_refused()

    def test_a_rank_three_corpus_is_refused(self):
        with pytest.raises(DataError, match="matrix of rows"):
            Corpus(vectors=torch.randn(4, 4, 4))

    def test_more_clusters_than_vectors_is_refused(self):
        with pytest.raises(ConfigError, match="leaves some empty"):
            clustered(count=8, dimension=4, clusters=32)

    def test_a_zero_spread_is_refused(self):
        with pytest.raises(ConfigError, match="zero width"):
            clustered(count=64, dimension=4, clusters=2, spread=0.0)

    def test_an_intrinsic_larger_than_the_ambient_is_refused_at_construction(self):
        with pytest.raises(ConfigError, match="does not fit"):
            on_a_subspace(count=64, dimension=8, intrinsic=32)

    def test_it_serialises(self):
        assert gaussian(count=100, dimension=10).as_dict()["bytes"] == 4000


class TestQueries:
    def test_held_out_queries_leave_the_corpus_smaller(self):
        corpus = gaussian(count=512, dimension=8)
        remaining, queries = held_out(corpus, count=64)
        assert remaining.count == 448
        assert queries.shape[0] == 64

    def test_and_are_not_in_it(self):
        corpus = gaussian(count=256, dimension=8)
        remaining, queries = held_out(corpus, count=32)
        assert not bool(
            (remaining.vectors.unsqueeze(0) == queries.unsqueeze(1)).all(dim=2).any()
        )

    def test_holding_out_the_whole_corpus_is_refused(self):
        with pytest.raises(ConfigError, match="held out of"):
            held_out(gaussian(count=64, dimension=4), count=64)

    def test_more_queries_than_vectors_is_refused(self):
        assert more_queries_than_vectors_is_refused()

    def test_a_zero_nudge_is_refused(self):
        with pytest.raises(ConfigError, match="does not move"):
            perturbed_queries(gaussian(count=64, dimension=4), count=8, nudge=0.0)


class TestContrast:
    def test_the_contrast_falls_with_dimension(self):
        assert contrast_collapses_with_dimension()["fell"]

    def test_by_a_factor_of_six(self):
        assert contrast_collapses_with_dimension()["ratio"] > 6.0

    def test_and_is_nearly_flat_above_a_hundred(self):
        assert contrast_collapses_with_dimension()["flat_above_a_hundred"] < 0.15

    def test_at_five_hundred_it_is_barely_above_one(self):
        rows = {row["dimension"]: row for row in contrast_by_dimension()}
        assert rows[512]["contrast"] < 1.2

    def test_the_curve_decreases_all_the_way(self):
        rows = [row["contrast"] for row in contrast_by_dimension()]
        assert rows == sorted(rows, reverse=True)

    def test_the_typical_distance_grows_with_the_square_root(self):
        rows = {row["dimension"]: row for row in contrast_by_dimension()}
        assert abs(rows[512]["typical_distance"] / rows[128]["typical_distance"] - 2.0) < 0.05

    def test_an_empty_dimension_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            contrast_by_dimension(dimensions=())

    def test_the_measured_curve_matches_the_closed_form(self):
        assert the_measurement_follows_the_closed_form()["largest_relative_gap"] < 0.1

    def test_and_the_formula_is_only_checked_where_it_means_anything(self):
        # Its subtraction goes negative around sixteen dimensions.
        assert the_expected_contrast_at_a_dimension(512) < 1.3

    def test_a_zero_dimension_is_refused(self):
        with pytest.raises(ConfigError, match="not a corpus"):
            the_expected_contrast_at_a_dimension(0)


class TestWhatDrivesIt:
    def test_ambient_width_does_not_drive_the_contrast(self):
        result = ambient_dimension_is_not_what_matters()
        assert abs(result["eight_within_five_hundred"] - result["eight_dimensional"]) < 1e-6

    def test_though_it_drives_the_storage(self):
        assert ambient_dimension_is_not_what_matters()["storage_ratio"] == 64

    def test_and_the_wide_unstructured_corpus_is_much_harder(self):
        result = ambient_dimension_is_not_what_matters()
        assert result["five_hundred_dimensional"] < result["eight_within_five_hundred"]

    def test_clusters_raise_the_contrast(self):
        assert clusters_are_easier_than_noise()["ratio"] > 3.0

    def test_at_the_same_width(self):
        assert clusters_are_easier_than_noise()["same_dimension"]

    def test_the_hardest_fixture_is_the_unstructured_one(self):
        assert the_contrast_ordering_matches_the_difficulty()["hardest"] == "gaussian 32d"

    def test_and_the_easiest_is_the_clustered_one(self):
        assert the_contrast_ordering_matches_the_difficulty()["easiest"] == "clustered 32d"

    def test_all_three_fixtures_are_the_same_width(self):
        assert the_contrast_ordering_matches_the_difficulty()["all_the_same_width"]

    def test_three_corpora_are_compared(self):
        assert len(compare_corpora()) == 3


class TestPerturbedQueries:
    def test_the_perturbed_contrast_is_one_over_the_nudge(self):
        for row in perturbed_queries_measure_the_nudge(nudges=(0.1, 0.25)):
            assert abs(row["contrast"] - row["one_over_the_nudge"]) < 0.3

    def test_and_barely_moves_with_dimension(self):
        assert the_perturbed_contrast_is_flat_in_dimension()["perturbed_change"] < 0.05

    def test_where_the_honest_measurement_moves_a_lot(self):
        assert the_perturbed_contrast_is_flat_in_dimension()["held_out_change"] > 0.5

    def test_so_the_benchmark_would_report_the_wrong_conclusion(self):
        result = the_perturbed_contrast_is_flat_in_dimension()
        assert result["perturbed_change"] < result["held_out_change"] / 10

    def test_an_empty_nudge_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            perturbed_queries_measure_the_nudge(nudges=())

    def test_a_larger_nudge_gives_a_smaller_contrast(self):
        rows = perturbed_queries_measure_the_nudge(nudges=(0.1, 0.5), dimensions=(64,))
        assert rows[0]["contrast"] > rows[1]["contrast"]


class TestIntrinsicDimension:
    def test_the_estimator_is_close_at_low_dimensions(self):
        rows = {row["true"]: row for row in the_estimator_recovers_a_known_dimension()}
        assert abs(rows[8]["ratio"] - 1.0) < 0.05

    def test_and_underestimates_at_higher_ones(self):
        # Two thousand points do not fill thirty two dimensions densely enough for the local
        # uniformity the derivation assumes.
        rows = {row["true"]: row for row in the_estimator_recovers_a_known_dimension()}
        assert rows[32]["ratio"] < 0.8

    def test_the_error_only_goes_one_way(self):
        rows = the_estimator_recovers_a_known_dimension()
        assert all(row["ratio"] <= 1.05 for row in rows)

    def test_it_sees_through_a_rotation(self):
        assert the_estimator_sees_through_the_rotation()["closer_to_the_intrinsic"]

    def test_giving_the_same_answer_as_on_the_narrow_corpus(self):
        result = the_estimator_sees_through_the_rotation()
        assert result["estimated"] == result["estimated_on_the_narrow_corpus"]

    def test_noise_raises_the_estimate(self):
        rows = [row["estimated"] for row in noise_raises_the_estimate()]
        assert rows == sorted(rows)

    def test_and_lowers_the_contrast(self):
        rows = [row["contrast"] for row in noise_raises_the_estimate()]
        assert rows == sorted(rows, reverse=True)

    def test_with_no_threshold_below_which_it_is_ignored(self):
        rows = {row["noise"]: row for row in noise_raises_the_estimate()}
        assert rows[0.01]["estimated"] > rows[0.0]["estimated"]

    def test_a_tiny_sample_is_refused(self):
        with pytest.raises(ConfigError, match="will not fit"):
            estimate_intrinsic_dimension(gaussian(count=64, dimension=4), sample=4)

    def test_an_empty_noise_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            noise_raises_the_estimate(levels=())


class TestStabilityAndMargin:
    def test_a_random_nudge_destabilises_low_dimensions_more(self):
        measured = a_random_nudge_destabilises_low_dimensions_more()
        rows = {row["dimension"]: row for row in measured}
        assert rows[256]["stable"] > rows[4]["stable"]

    def test_which_is_the_opposite_of_what_the_contrast_suggests(self):
        measured = a_random_nudge_destabilises_low_dimensions_more()
        rows = {row["dimension"]: row for row in measured}
        assert rows[256]["contrast"] < rows[4]["contrast"]

    def test_a_zero_nudge_is_refused(self):
        with pytest.raises(ConfigError, match="does not move"):
            nearest_neighbour_stability(gaussian(count=128, dimension=8), nudge=0.0)

    def test_the_margin_collapses_with_dimension(self):
        assert recall_charges_full_price_for_a_small_error()["collapsed"]

    def test_by_a_factor_of_thirty(self):
        assert recall_charges_full_price_for_a_small_error()["ratio"] > 25.0

    def test_at_five_hundred_the_runner_up_is_within_one_percent(self):
        assert recall_charges_full_price_for_a_small_error()["margin_at_five_hundred"] < 0.01

    def test_while_recall_charges_the_whole_query(self):
        assert recall_charges_full_price_for_a_small_error()["recall_cost_of_a_miss"] == 1.0

    def test_the_margin_is_not_the_contrast(self):
        corpus = gaussian(count=1024, dimension=32)
        assert neighbour_margin(corpus) < relative_contrast(corpus)

    def test_an_empty_margin_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            the_margin_collapses_with_dimension(dimensions=())

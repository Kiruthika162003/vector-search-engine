from __future__ import annotations

import pytest
import torch

from vse.errors import ConfigError, DataError
from vse.vectors.dataset import gaussian, on_a_subspace
from vse.vectors.preprocess import (
    Transform,
    a_centre_of_the_wrong_width_is_refused,
    a_corpus_on_a_subspace_reduces_for_free,
    a_high_floor_turns_whitening_into_a_rotation,
    a_projection_wider_than_the_source_is_refused,
    a_random_projection_needs_no_corpus,
    a_reduction_beyond_the_dimension_is_refused,
    a_single_vector_has_no_covariance,
    a_transform_of_the_wrong_width_is_refused,
    a_zero_target_is_refused,
    a_zero_variance_floor_is_refused,
    an_orthonormal_projection_is_a_rotation_at_full_width,
    centring_changes_nothing,
    compare_every_transform,
    fit_centring,
    fit_random_projection,
    fit_reduction,
    fit_whitening,
    fitting_pays_where_there_is_structure,
    normalising_is_a_transform_too,
    orthonormalising_is_free_accuracy,
    principal_directions,
    recall_after,
    reduction_costs_recall_before_any_index_runs,
    reduction_makes_the_index_cheaper_and_the_ceiling_lower,
    reduction_raises_the_contrast,
    the_cheapest_transform_is_the_one_that_does_nothing,
    the_contrast_explains_the_share,
    the_damage_is_estimation_error,
    the_index_gets_closer_to_a_lower_ceiling,
    the_spread_and_the_damage_converge_together,
    the_variance_floor_is_what_makes_whitening_survivable,
    transforms_compose,
    variance_kept_is_not_recall_kept,
    which_reduction_wins_depends_on_the_corpus,
    whitening_changes_the_question,
    whitening_costs_more_on_an_isotropic_corpus,
)


class TestCentring:
    def test_centring_changes_nothing(self):
        assert centring_changes_nothing()["exact"]

    def test_the_recall_is_exactly_one(self):
        assert centring_changes_nothing()["recall"] == 1.0

    def test_a_centring_transform_is_square(self):
        corpus = gaussian(count=256, dimension=16)
        transform = fit_centring(corpus.vectors)
        assert transform.source == transform.target == 16

    def test_and_moves_the_mean_to_the_origin(self):
        corpus = gaussian(count=256, dimension=16)
        moved = fit_centring(corpus.vectors).apply(corpus.vectors)
        assert float(moved.mean(dim=0).abs().max()) < 1e-4

    def test_it_serialises(self):
        corpus = gaussian(count=256, dimension=16)
        assert fit_centring(corpus.vectors).as_dict()["compression"] == 1.0


class TestWhitening:
    def test_whitening_is_not_distance_preserving(self):
        assert whitening_changes_the_question()["recall"] < 1.0

    def test_where_centring_is(self):
        assert whitening_changes_the_question()["centring_recall"] == 1.0

    def test_it_costs_more_on_an_isotropic_corpus(self):
        assert whitening_costs_more_on_an_isotropic_corpus()["isotropic_is_worse"]

    def test_the_damage_is_estimation_error(self):
        assert the_spread_and_the_damage_converge_together()["spread_falls"]

    def test_and_the_recall_recovers_with_the_sample(self):
        assert the_spread_and_the_damage_converge_together()["recall_rises"]

    def test_a_small_sample_has_a_wide_spectrum(self):
        assert the_spread_and_the_damage_converge_together()["spread_at_two_hundred"] > 5.0

    def test_and_a_large_one_does_not(self):
        result = the_spread_and_the_damage_converge_together()
        assert result["spread_at_sixteen_thousand"] < 1.5

    def test_the_recall_at_a_large_sample_is_high(self):
        assert (
            the_spread_and_the_damage_converge_together()["recall_at_sixteen_thousand"] > 0.85
        )

    def test_the_spread_falls_monotonically(self):
        rows = [row["eigenvalue_spread"] for row in the_damage_is_estimation_error()]
        assert rows == sorted(rows, reverse=True)

    def test_and_the_recall_rises_monotonically(self):
        rows = [row["recall"] for row in the_damage_is_estimation_error()]
        assert rows == sorted(rows)

    def test_an_empty_sample_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            the_damage_is_estimation_error(counts=())

    def test_a_full_floor_is_a_rotation(self):
        assert a_high_floor_turns_whitening_into_a_rotation()["a_full_floor_is_a_rotation"]

    def test_a_higher_floor_never_hurts(self):
        rows = [
            row["recall"] for row in the_variance_floor_is_what_makes_whitening_survivable()
        ]
        assert rows[-1] >= rows[0]

    def test_a_zero_floor_is_refused(self):
        assert a_zero_variance_floor_is_refused()

    def test_an_empty_floor_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            the_variance_floor_is_what_makes_whitening_survivable(floors=())

    def test_whitening_makes_the_covariance_the_identity(self):
        corpus = gaussian(count=4096, dimension=8)
        moved = fit_whitening(corpus.vectors).apply(corpus.vectors)
        covariance = (moved.T @ moved) / (4096 - 1)
        assert float((covariance - torch.eye(8)).abs().max()) < 0.1


class TestReduction:
    def test_reduction_costs_recall(self):
        rows = [row["recall"] for row in reduction_costs_recall_before_any_index_runs()]
        assert rows == sorted(rows)

    def test_full_width_is_lossless(self):
        rows = {row["target"]: row for row in reduction_costs_recall_before_any_index_runs()}
        assert rows[64]["recall"] == 1.0

    def test_the_variance_kept_rises_with_the_target(self):
        rows = [row["kept_variance"] for row in reduction_costs_recall_before_any_index_runs()]
        assert rows == sorted(rows)

    def test_variance_kept_overstates_recall_kept(self):
        assert variance_kept_is_not_recall_kept()["variance_overstates_recall"]

    def test_by_a_wide_margin(self):
        assert variance_kept_is_not_recall_kept()["gap"] > 0.2

    def test_a_subspace_corpus_reduces_for_free(self):
        assert a_corpus_on_a_subspace_reduces_for_free()["free_at_the_rank"]

    def test_but_not_below_its_rank(self):
        assert a_corpus_on_a_subspace_reduces_for_free()["not_free_below_it"]

    def test_the_intrinsic_dimension_finds_the_rank(self):
        result = a_corpus_on_a_subspace_reduces_for_free()
        assert abs(result["estimated_intrinsic"] - result["true_rank"]) < 1.5

    def test_reducing_beyond_the_dimension_is_refused(self):
        assert a_reduction_beyond_the_dimension_is_refused()

    def test_a_zero_target_is_refused(self):
        assert a_zero_target_is_refused()

    def test_an_empty_reduction_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            reduction_costs_recall_before_any_index_runs(targets=())

    def test_a_reduction_reports_its_kept_variance(self):
        corpus = on_a_subspace(count=1024, dimension=32, intrinsic=4)
        assert fit_reduction(corpus.vectors, 4).kept_variance > 0.95


class TestRandomProjection:
    def test_it_needs_no_corpus(self):
        transform = fit_random_projection(64, 16)
        assert transform.source == 64
        assert transform.target == 16

    def test_an_orthonormal_projection_is_lossless_at_full_width(self):
        assert orthonormalising_is_free_accuracy()["full_width_is_lossless"]

    def test_and_the_plain_one_is_not(self):
        assert orthonormalising_is_free_accuracy()["and_the_plain_one_is_not"]

    def test_orthonormalising_never_loses_by_more_than_noise(self):
        rows = an_orthonormal_projection_is_a_rotation_at_full_width()
        assert all(row["orthonormal_recall"] >= row["plain_recall"] - 0.005 for row in rows)

    def test_and_matters_most_at_full_width(self):
        rows = {
            row["target"]: row
            for row in an_orthonormal_projection_is_a_rotation_at_full_width()
        }
        wide = rows[64]["orthonormal_recall"] - rows[64]["plain_recall"]
        narrow = rows[8]["orthonormal_recall"] - rows[8]["plain_recall"]
        assert wide > narrow

    def test_a_thin_gaussian_matrix_is_already_nearly_orthogonal(self):
        rows = {
            row["target"]: row
            for row in an_orthonormal_projection_is_a_rotation_at_full_width()
        }
        assert abs(rows[8]["orthonormal_recall"] - rows[8]["plain_recall"]) < 0.01

    def test_and_by_a_lot_at_half_width(self):
        result = orthonormalising_is_free_accuracy()
        assert result["orthonormal_at_half_width"] > result["plain_at_half_width"]

    def test_a_projection_wider_than_the_source_is_refused(self):
        assert a_projection_wider_than_the_source_is_refused()

    def test_a_zero_width_projection_is_refused(self):
        with pytest.raises(ConfigError, match="cannot project"):
            fit_random_projection(64, 0)

    def test_an_empty_orthonormal_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            an_orthonormal_projection_is_a_rotation_at_full_width(targets=())

    def test_the_recall_rises_with_the_width(self):
        rows = [row["recall"] for row in a_random_projection_needs_no_corpus()]
        assert rows == sorted(rows)

    def test_an_empty_projection_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            a_random_projection_needs_no_corpus(targets=())

    def test_fitting_pays_where_there_is_structure(self):
        assert fitting_pays_where_there_is_structure()["fitting_pays_more_on_structure"]

    def test_and_barely_where_there_is_none(self):
        assert fitting_pays_where_there_is_structure()["gaussian_gap"] < 0.1

    def test_two_corpora_are_compared(self):
        assert len(which_reduction_wins_depends_on_the_corpus()) == 2


class TestTheCeiling:
    def test_reduction_lowers_the_ceiling(self):
        assert the_index_gets_closer_to_a_lower_ceiling()["ceiling_falls"]

    def test_and_raises_the_share_reached(self):
        assert the_index_gets_closer_to_a_lower_ceiling()["share_rises"]

    def test_the_index_nearly_saturates_a_low_ceiling(self):
        assert the_index_gets_closer_to_a_lower_ceiling()["share_at_eight"] > 0.85

    def test_and_falls_well_short_of_a_high_one(self):
        assert the_index_gets_closer_to_a_lower_ceiling()["share_at_sixty_four"] < 0.6

    def test_the_achieved_recall_never_beats_the_ceiling_by_much(self):
        rows = reduction_makes_the_index_cheaper_and_the_ceiling_lower()
        assert all(row["achieved"] <= row["ceiling"] + 0.02 for row in rows)

    def test_the_contrast_rises_as_the_dimension_falls(self):
        rows = [row["contrast"] for row in reduction_raises_the_contrast()]
        assert rows == sorted(rows, reverse=True)

    def test_and_explains_the_share(self):
        assert the_contrast_explains_the_share()["contrast_rises_as_dimension_falls"]
        assert the_contrast_explains_the_share()["and_so_does_the_share"]

    def test_an_empty_ceiling_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            reduction_makes_the_index_cheaper_and_the_ceiling_lower(targets=())

    def test_an_empty_contrast_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            reduction_raises_the_contrast(targets=())


class TestMechanics:
    def test_transforms_compose(self):
        assert transforms_compose()["same_up_to_sign"]

    def test_normalising_is_not_one_of_these(self):
        assert not normalising_is_a_transform_too()["linear"]

    def test_and_changes_the_answers(self):
        assert normalising_is_a_transform_too()["recall_against_raw_truth"] < 1.0

    def test_four_transforms_are_compared(self):
        assert len(compare_every_transform()) == 4

    def test_centring_is_the_only_exact_one(self):
        assert the_cheapest_transform_is_the_one_that_does_nothing()["centring_is_exact"]

    def test_and_everything_else_costs(self):
        assert the_cheapest_transform_is_the_one_that_does_nothing()["everything_else_costs"]

    def test_a_single_vector_has_no_covariance(self):
        assert a_single_vector_has_no_covariance()

    def test_a_rank_one_corpus_is_refused(self):
        with pytest.raises(DataError, match="a corpus is a matrix"):
            principal_directions(torch.randn(16))

    def test_the_directions_are_orthonormal(self):
        corpus = gaussian(count=512, dimension=16)
        directions, _ = principal_directions(corpus.vectors)
        product = directions.T @ directions
        assert float((product - torch.eye(16)).abs().max()) < 1e-4

    def test_the_variances_come_out_sorted(self):
        corpus = on_a_subspace(count=1024, dimension=32, intrinsic=8)
        _, variance = principal_directions(corpus.vectors)
        assert bool(torch.all(variance[:-1] >= variance[1:] - 1e-6))

    def test_a_transform_of_the_wrong_width_is_refused(self):
        assert a_transform_of_the_wrong_width_is_refused()

    def test_a_centre_of_the_wrong_width_is_refused(self):
        assert a_centre_of_the_wrong_width_is_refused()

    def test_a_rank_one_transform_matrix_is_refused(self):
        with pytest.raises(DataError, match="a transform is a matrix"):
            Transform(matrix=torch.zeros(8), centre=torch.zeros(1, 8), name="broken")

    def test_a_rank_one_centre_is_refused(self):
        with pytest.raises(DataError, match="a centre is one row"):
            Transform(matrix=torch.eye(8), centre=torch.zeros(8), name="broken")

    def test_a_rank_one_batch_is_refused(self):
        corpus = gaussian(count=256, dimension=8)
        with pytest.raises(DataError, match="a batch is a matrix"):
            fit_centring(corpus.vectors).apply(torch.randn(8))

    def test_recall_after_reports_the_transform_name(self):
        corpus = gaussian(count=512, dimension=16)
        assert recall_after(corpus, fit_centring(corpus.vectors))["transform"] == "centred"

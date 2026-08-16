from __future__ import annotations

import pytest
import torch

from vse.errors import ConfigError, DataError
from vse.quantize.opq import (
    Rotated,
    a_corpus_with_no_variance_is_refused,
    a_non_square_rotation_is_refused,
    a_rotation_is_orthogonal,
    a_rotation_levels_the_variance,
    a_rotation_of_the_wrong_width_is_refused,
    a_rotation_preserves_distances,
    an_unknown_rotation_is_refused,
    and_buys_nothing_on_isotropic_data,
    compare_rotations,
    identity_rotation,
    pca_rotation,
    random_rotation,
    rotate,
    rotation_helps_on_skewed_data,
    search_rotated,
    skewed,
    subspace_variance,
    the_component_ordering_matters,
    the_decoded_vectors_live_in_the_rotated_space,
    the_rotation_costs_one_matrix_product,
    the_skewed_corpus_is_unbalanced,
    train_rotated,
)
from vse.vectors.dataset import gaussian
from vse.vectors.exact import identifier_overlap, search


class TestRotations:
    def test_both_constructions_are_orthogonal(self):
        assert a_rotation_is_orthogonal()["both_orthogonal"]

    def test_they_preserve_every_distance(self):
        assert a_rotation_preserves_distances()["preserved"]

    def test_to_about_a_thousandth_in_relative_terms(self):
        # Which is float32 doing a sixty four wide matrix product, not the rotation.
        assert a_rotation_preserves_distances()["relative"] < 1e-2

    def test_the_identity_is_the_identity(self):
        assert torch.equal(identity_rotation(8), torch.eye(8))

    def test_a_random_rotation_is_reproducible(self):
        assert torch.equal(random_rotation(16, seed=3), random_rotation(16, seed=3))

    def test_and_different_seeds_differ(self):
        assert not torch.equal(random_rotation(16, seed=1), random_rotation(16, seed=2))

    def test_a_zero_width_rotation_is_refused(self):
        with pytest.raises(ConfigError, match="not a width"):
            random_rotation(0)

    def test_a_rotation_of_the_wrong_width_is_refused(self):
        assert a_rotation_of_the_wrong_width_is_refused()

    def test_a_non_square_rotation_is_refused(self):
        assert a_non_square_rotation_is_refused()

    def test_a_rank_three_rotation_is_refused(self):
        codes = train_rotated(
            gaussian(count=512, dimension=32).vectors, subspaces=4, centroids=64
        )
        with pytest.raises(DataError, match="a rotation is a matrix"):
            Rotated(codes=codes.codes, rotation=torch.randn(2, 2, 2))

    def test_an_unknown_rotation_is_refused(self):
        assert an_unknown_rotation_is_refused()

    def test_a_pca_rotation_needs_a_dividing_width(self):
        with pytest.raises(ConfigError, match="does not divide"):
            pca_rotation(torch.randn(256, 30), subspaces=8)


class TestSkew:
    def test_the_skewed_fixture_is_genuinely_skewed(self):
        assert the_skewed_corpus_is_unbalanced()["skew_is_real"]

    def test_and_the_gaussian_one_is_already_level(self):
        assert the_skewed_corpus_is_unbalanced()["gaussian_is_level"]

    def test_the_first_subspace_carries_hundreds_of_times_the_last(self):
        result = the_skewed_corpus_is_unbalanced()
        assert result["skewed_first"] > result["skewed_last"] * 100

    def test_a_decay_of_one_is_refused(self):
        with pytest.raises(ConfigError, match="not a decay"):
            skewed(decay=1.0)

    def test_a_corpus_with_no_variance_is_refused(self):
        assert a_corpus_with_no_variance_is_refused()

    def test_a_width_that_does_not_split_is_refused(self):
        with pytest.raises(ConfigError, match="does not split"):
            subspace_variance(torch.randn(64, 30), subspaces=8)

    def test_the_variance_shares_add_up(self):
        corpus = skewed(count=512, dimension=32)
        centred = corpus.vectors - corpus.vectors.mean(dim=0, keepdim=True)
        total = float(centred.pow(2).mean(dim=0).sum())
        assert abs(float(subspace_variance(corpus.vectors, 4).sum()) - total) < 1e-3


class TestBalanceIsNotTheObjective:
    def test_both_rotations_level_the_variance(self):
        assert a_rotation_levels_the_variance()["both_help"]

    def test_the_random_one_levels_it_better(self):
        assert a_rotation_levels_the_variance()["random_levels_it_better"]

    def test_but_the_principal_component_one_has_better_recall(self):
        result = rotation_helps_on_skewed_data()
        assert result["pca_recall"] > result["random_recall"]

    def test_so_balance_is_not_what_a_rotation_is_for(self):
        levels = a_rotation_levels_the_variance()
        recalls = rotation_helps_on_skewed_data()
        assert levels["random"] > levels["pca"]
        assert recalls["random_recall"] < recalls["pca_recall"]

    def test_by_sixteen_points_of_recall(self):
        result = rotation_helps_on_skewed_data()
        assert result["pca_recall"] - result["random_recall"] > 0.1

    def test_both_beat_no_rotation(self):
        result = rotation_helps_on_skewed_data()
        assert result["random_helps"] and result["pca_helps"]

    def test_and_the_best_is_the_principal_component_one(self):
        assert rotation_helps_on_skewed_data()["best"] == "pca"


class TestOrdering:
    def test_dealing_the_components_is_the_whole_advantage(self):
        result = the_component_ordering_matters()
        assert result["dealt_balance"] > result["ordered_balance"] * 50

    def test_leaving_them_in_variance_order_is_no_better_than_nothing(self):
        assert the_component_ordering_matters()["ordering_alone_is_worse"]

    def test_because_it_puts_everything_in_the_first_subspace(self):
        result = the_component_ordering_matters()
        assert result["ordered_balance"] < 0.01

    def test_a_pca_rotation_is_orthogonal_after_the_dealing(self):
        corpus = skewed(count=512, dimension=32)
        matrix = pca_rotation(corpus.vectors, subspaces=4)
        product = matrix @ matrix.transpose(0, 1)
        assert torch.allclose(product, torch.eye(32), atol=1e-4)


class TestIsotropic:
    def test_a_rotation_buys_almost_nothing_on_gaussian_data(self):
        assert and_buys_nothing_on_isotropic_data()["within_noise"]

    def test_the_spread_is_three_points(self):
        assert and_buys_nothing_on_isotropic_data()["spread"] < 0.05

    def test_six_rows_in_the_comparison(self):
        assert len(compare_rotations()) == 6

    def test_the_skewed_corpus_shows_the_effect(self):
        rows = {(row["corpus"], row["rotation"]): row for row in compare_rotations()}
        assert rows[("skewed", "pca")]["recall"] > rows[("skewed", "none")]["recall"] * 1.5

    def test_and_the_gaussian_one_does_not(self):
        rows = {(row["corpus"], row["rotation"]): row for row in compare_rotations()}
        assert rows[("gaussian", "pca")]["recall"] < rows[("gaussian", "none")]["recall"] * 1.2

    def test_the_gaussian_corpus_is_balanced_before_any_rotation(self):
        rows = {(row["corpus"], row["rotation"]): row for row in compare_rotations()}
        assert rows[("gaussian", "none")]["balance"] > 0.9


class TestMechanics:
    def test_the_rotation_costs_one_matrix_product(self):
        result = the_rotation_costs_one_matrix_product()
        assert result["rotation_operations"] == 64 * 64

    def test_which_is_a_fraction_of_a_scan(self):
        assert the_rotation_costs_one_matrix_product()["share"] < 0.05

    def test_a_zero_width_cost_is_refused(self):
        with pytest.raises(ConfigError, match="not a width"):
            the_rotation_costs_one_matrix_product(dimension=0)

    def test_the_codes_describe_the_rotated_space(self):
        assert the_decoded_vectors_live_in_the_rotated_space()["rotated_is_much_smaller"]

    def test_comparing_against_the_originals_measures_the_rotation(self):
        result = the_decoded_vectors_live_in_the_rotated_space()
        assert result["against_original"] > result["against_rotated"] * 10

    def test_searching_a_rotated_index_returns_the_right_shape(self):
        corpus = skewed(count=512, dimension=32)
        model = train_rotated(corpus.vectors, subspaces=4, centroids=64)
        assert search_rotated(corpus.vectors[:8], model, k=5).k == 5

    def test_a_rotated_index_finds_something(self):
        corpus = skewed(count=1024, dimension=32)
        model = train_rotated(corpus.vectors, subspaces=8, centroids=64)
        truth = search(corpus.vectors[:32], corpus.vectors, k=10)
        found = search_rotated(corpus.vectors[:32], model, k=10)
        assert identifier_overlap(truth, found) > 0.3

    def test_the_rotation_is_counted_in_the_memory(self):
        corpus = skewed(count=512, dimension=32)
        model = train_rotated(corpus.vectors, subspaces=4, centroids=64)
        assert model.bytes_used() > model.codes.bytes_used()

    def test_it_serialises(self):
        corpus = skewed(count=512, dimension=32)
        model = train_rotated(corpus.vectors, subspaces=4, centroids=64)
        assert model.as_dict()["rotation_bytes"] == 32 * 32 * 4

    def test_rotating_twice_by_the_transpose_returns_the_original(self):
        corpus = skewed(count=256, dimension=32)
        matrix = random_rotation(32)
        there = rotate(corpus.vectors, matrix)
        back = there @ matrix
        assert torch.allclose(back, corpus.vectors, atol=1e-3)

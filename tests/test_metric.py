from __future__ import annotations

import pytest
import torch

from vse.errors import ConfigError, DataError
from vse.vectors.metric import (
    BY_NAME,
    COSINE,
    INNER_PRODUCT,
    L2,
    METRICS,
    Metric,
    a_query_of_the_wrong_width_is_refused,
    a_similarity_fails_the_first_axiom,
    a_zero_vector_survives_normalising,
    an_empty_corpus_is_refused,
    asking_for_more_neighbours_than_exist_is_refused,
    compare_metrics,
    cosine,
    cosine_is_inner_product_after_normalising,
    distances,
    inner_product,
    metric_named,
    metric_table,
    normalise,
    off_the_sphere_they_are_not,
    on_the_sphere_they_are_the_same_ordering,
    only_euclidean_is_a_metric,
    orderings_agree,
    rank_by,
    scaled_vectors,
    squared_l2,
    the_expansion_matches_the_difference,
    the_square_is_not_the_metric,
    triangle_inequality_holds_for,
    unit_vectors,
    which_metrics_allow_pruning,
)


class TestDistances:
    def test_the_distance_from_a_vector_to_itself_is_zero(self):
        corpus = unit_vectors(count=32)
        assert float(squared_l2(corpus[:4], corpus).diagonal().abs().max()) < 1e-5

    def test_the_expansion_matches_subtracting_the_rows(self):
        assert the_expansion_matches_the_difference()["close_enough"]

    def test_and_never_comes_out_negative(self):
        # It can without the clamp, and a negative squared distance becomes a nan under a root.
        assert the_expansion_matches_the_difference()["never_negative"]

    def test_the_inner_product_of_a_unit_vector_with_itself_is_one(self):
        corpus = unit_vectors(count=16)
        assert abs(float(inner_product(corpus[:1], corpus[:1])) - 1.0) < 1e-5

    def test_cosine_is_bounded(self):
        scores = cosine(unit_vectors(count=8, seed=1), scaled_vectors(count=32))
        assert float(scores.abs().max()) <= 1.0 + 1e-5

    def test_a_query_of_the_wrong_width_is_refused(self):
        assert a_query_of_the_wrong_width_is_refused()

    def test_an_empty_corpus_is_refused(self):
        assert an_empty_corpus_is_refused()

    def test_a_vector_of_zero_width_is_refused(self):
        with pytest.raises(DataError, match="zero dimensions"):
            squared_l2(torch.randn(2, 0), torch.randn(2, 0))

    def test_a_rank_three_tensor_is_refused(self):
        with pytest.raises(DataError, match="matrix of rows"):
            squared_l2(torch.randn(2, 3, 4), torch.randn(8, 4))

    def test_an_integer_corpus_is_refused(self):
        with pytest.raises(DataError, match="floating point"):
            squared_l2(torch.randn(2, 4), torch.zeros(8, 4, dtype=torch.int64))


class TestRanking:
    def test_the_closest_vector_under_euclidean_is_the_smallest_score(self):
        corpus = unit_vectors(count=64)
        scores = squared_l2(corpus[:1], corpus)
        assert int(rank_by(scores, "l2", 1)[0, 0]) == int(scores.argmin())

    def test_the_closest_under_inner_product_is_the_largest(self):
        corpus = scaled_vectors(count=64)
        scores = inner_product(corpus[:1], corpus)
        assert int(rank_by(scores, "ip", 1)[0, 0]) == int(scores.argmax())

    def test_a_vector_is_its_own_nearest_neighbour_under_euclidean(self):
        corpus = scaled_vectors(count=64)
        found = rank_by(squared_l2(corpus, corpus), "l2", 1)
        assert found.flatten().tolist() == list(range(corpus.shape[0]))

    def test_but_not_under_inner_product(self):
        assert not a_similarity_fails_the_first_axiom()["self_is_always_the_closest"]

    def test_because_its_own_score_is_its_squared_length(self):
        result = a_similarity_fails_the_first_axiom()
        assert not result["self_similarity_is_zero"]
        assert result["largest_self_score"] > result["smallest_self_score"]

    def test_asking_for_no_neighbours_is_refused(self):
        with pytest.raises(ConfigError, match="not a query"):
            rank_by(torch.randn(2, 8), "l2", 0)

    def test_asking_for_more_than_exist_is_refused(self):
        assert asking_for_more_neighbours_than_exist_is_refused()


class TestEquivalence:
    def test_on_the_sphere_the_two_orderings_are_identical(self):
        assert on_the_sphere_they_are_the_same_ordering()["agreement"] == 1.0

    def test_because_one_is_an_affine_map_of_the_other(self):
        assert on_the_sphere_they_are_the_same_ordering()["exact"]

    def test_off_the_sphere_they_agree_about_nothing(self):
        assert off_the_sphere_they_are_not()["agreement"] == 0.0

    def test_not_even_about_the_single_nearest_vector(self):
        assert off_the_sphere_they_are_not()["top_one_agreement"] == 0.0

    def test_cosine_is_inner_product_on_normalised_vectors(self):
        result = cosine_is_inner_product_after_normalising()
        assert result["against_normalised_inner_product"] == 1.0

    def test_and_euclidean_too_once_normalised(self):
        result = cosine_is_inner_product_after_normalising()
        assert result["and_against_euclidean_once_normalised"] == 1.0

    def test_but_not_on_the_raw_scaled_vectors(self):
        assert cosine_is_inner_product_after_normalising()["on_scaled_vectors"] == 0.0

    def test_a_metric_always_agrees_with_itself(self):
        queries = unit_vectors(count=8, seed=5)
        corpus = scaled_vectors(count=64)
        assert orderings_agree(queries, corpus, "l2", "l2", 10) == 1.0


class TestAxioms:
    def test_euclidean_obeys_the_triangle_inequality(self):
        assert triangle_inequality_holds_for("l2", 300)["violations"] == 0

    def test_but_only_once_the_root_is_taken(self):
        assert triangle_inequality_holds_for("l2", 300, root=False)["violations"] > 0

    def test_the_squared_distance_breaks_it_about_one_time_in_seventy(self):
        assert 0.005 < the_square_is_not_the_metric()["squared_share"] < 0.03

    def test_while_ranking_identically(self):
        # Which is why nothing takes the root, and why a pruning bound has to.
        assert the_square_is_not_the_metric()["same_ordering"] == 1.0

    def test_and_the_excess_is_not_marginal(self):
        assert the_square_is_not_the_metric()["worst_excess"] > 1.0

    def test_inner_product_breaks_it_more_than_half_the_time(self):
        assert triangle_inequality_holds_for("ip", 300)["share"] > 0.5

    def test_and_so_does_cosine(self):
        assert triangle_inequality_holds_for("cosine", 300)["share"] > 0.5

    def test_only_euclidean_survives(self):
        assert only_euclidean_is_a_metric(300)["clean"] == ["l2"]

    def test_which_is_what_the_flag_records(self):
        assert only_euclidean_is_a_metric(300)["matches_the_flag"]

    def test_and_the_flag_is_about_the_ordering_not_the_function(self):
        assert only_euclidean_is_a_metric(300)["and_only_with_the_root"]

    def test_one_metric_permits_pruning(self):
        assert which_metrics_allow_pruning()["prunable"] == ["l2"]

    def test_and_two_do_not(self):
        assert len(which_metrics_allow_pruning()["not_prunable"]) == 2

    def test_an_empty_metric_list_is_refused(self):
        with pytest.raises(ConfigError, match="no metrics to check"):
            which_metrics_allow_pruning(names=())


class TestNormalising:
    def test_every_row_comes_out_unit_length(self):
        normalised = normalise(scaled_vectors(count=64))
        assert float((normalised.pow(2).sum(dim=1) - 1.0).abs().max()) < 1e-5

    def test_a_zero_row_does_not_become_a_nan(self):
        assert not a_zero_vector_survives_normalising()["any_nan"]

    def test_it_stays_at_zero(self):
        assert a_zero_vector_survives_normalising()["zero_row_norm"] == 0.0

    def test_without_disturbing_its_neighbours_in_the_batch(self):
        assert abs(a_zero_vector_survives_normalising()["unit_row_norm"] - 1.0) < 1e-6

    def test_a_zero_epsilon_is_refused(self):
        with pytest.raises(ConfigError, match="does not protect"):
            normalise(torch.randn(4, 8), epsilon=0.0)

    def test_normalising_twice_changes_nothing(self):
        once = normalise(scaled_vectors(count=32))
        assert torch.allclose(once, normalise(once), atol=1e-6)


class TestTable:
    def test_three_metrics_are_defined(self):
        assert len(METRICS) == 3

    def test_each_one_is_reachable_by_name(self):
        assert all(metric_named(name).name == name for name in METRICS)

    def test_an_unknown_metric_is_refused(self):
        with pytest.raises(ConfigError, match="unknown metric"):
            metric_named("manhattan")

    def test_and_at_construction(self):
        with pytest.raises(ConfigError, match="unknown metric"):
            Metric(name="manhattan", smaller_is_closer=True, is_a_metric=True)

    def test_only_euclidean_ranks_smallest_first(self):
        assert [name for name in METRICS if BY_NAME[name].smaller_is_closer] == ["l2"]

    def test_the_table_covers_every_metric(self):
        assert len(metric_table()) == len(METRICS)

    def test_the_comparison_covers_both_corpora(self):
        assert len({row["corpus"] for row in compare_metrics()}) == 2

    def test_and_every_metric_on_each(self):
        assert len(compare_metrics()) == 2 * len(METRICS)

    def test_dispatching_by_name_matches_dispatching_by_object(self):
        queries, corpus = unit_vectors(count=4, seed=2), unit_vectors(count=32)
        assert torch.allclose(
            distances(queries, corpus, "l2"), distances(queries, corpus, L2), atol=1e-6
        )

    def test_the_two_similarity_metrics_share_a_sign_convention(self):
        assert not INNER_PRODUCT.smaller_is_closer
        assert not COSINE.smaller_is_closer

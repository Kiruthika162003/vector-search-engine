from __future__ import annotations

import pytest
import torch

from vse.errors import ConfigError, DataError
from vse.quantize.product import (
    ProductCodes,
    a_query_of_the_wrong_width_is_refused,
    a_width_that_does_not_divide_is_refused,
    and_loses_on_accuracy,
    asymmetric_scores,
    compare_quantisers,
    decode,
    distance_table,
    it_beats_scalar_quantisation_on_memory,
    it_needs_a_much_longer_shortlist_than_scalar,
    more_centroids_than_a_byte_is_refused,
    reconstruction_error,
    rerank,
    reranking_rescues_it,
    search_codes,
    structure_helps_here_too,
    subspace_sweep,
    the_compression_is_enormous,
    the_return_per_byte_increases,
    the_table_makes_scoring_cheap,
    the_table_matches_a_direct_computation,
    too_few_vectors_to_train_is_refused,
    train,
)
from vse.vectors.dataset import gaussian


class TestCodes:
    def test_a_code_is_a_byte(self):
        codes = train(gaussian(count=512, dimension=32).vectors, subspaces=4, centroids=64)
        assert codes.codes.dtype == torch.uint8

    def test_there_is_one_code_per_subspace(self):
        codes = train(gaussian(count=512, dimension=32).vectors, subspaces=4, centroids=64)
        assert codes.codes.shape == (512, 4)

    def test_the_codebooks_cover_the_whole_width(self):
        codes = train(gaussian(count=512, dimension=32).vectors, subspaces=4, centroids=64)
        assert codes.dimension == 32

    def test_decoding_gives_back_the_original_shape(self):
        vectors = gaussian(count=512, dimension=32).vectors
        codes = train(vectors, subspaces=4, centroids=64)
        assert decode(codes).shape == vectors.shape

    def test_a_width_that_does_not_divide_is_refused(self):
        assert a_width_that_does_not_divide_is_refused()

    def test_more_centroids_than_a_byte_is_refused(self):
        assert more_centroids_than_a_byte_is_refused()

    def test_too_few_vectors_to_train_is_refused(self):
        assert too_few_vectors_to_train_is_refused()

    def test_zero_subspaces_is_refused(self):
        with pytest.raises(ConfigError, match="not a split"):
            train(torch.randn(512, 32), subspaces=0)

    def test_a_rank_three_input_is_refused(self):
        with pytest.raises(DataError, match="matrix of rows"):
            train(torch.randn(4, 4, 4))

    def test_a_mismatched_codebook_count_is_refused(self):
        with pytest.raises(DataError, match="codes per vector against"):
            ProductCodes(
                codes=torch.zeros(4, 8, dtype=torch.uint8), codebooks=torch.randn(4, 16, 2)
            )

    def test_a_flat_codebook_is_refused(self):
        with pytest.raises(DataError, match="subspace by centroid"):
            ProductCodes(
                codes=torch.zeros(4, 2, dtype=torch.uint8), codebooks=torch.randn(2, 4)
            )

    def test_it_serialises(self):
        codes = train(gaussian(count=512, dimension=32).vectors, subspaces=4, centroids=64)
        assert codes.as_dict()["subspaces"] == 4


class TestTheTable:
    def test_the_table_matches_decoding_and_measuring(self):
        assert the_table_matches_a_direct_computation()["agree"]

    def test_to_the_rounding_unit(self):
        result = the_table_matches_a_direct_computation()
        assert result["largest_gap"] < result["typical"] / 1000

    def test_the_table_has_one_row_per_subspace(self):
        codes = train(gaussian(count=512, dimension=32).vectors, subspaces=4, centroids=64)
        assert distance_table(torch.randn(8, 32), codes).shape == (8, 4, 64)

    def test_a_query_of_the_wrong_width_is_refused(self):
        assert a_query_of_the_wrong_width_is_refused()

    def test_scoring_a_code_is_cheaper_than_a_full_distance(self):
        assert the_table_makes_scoring_cheap()["ratio"] > 1.0

    def test_by_a_factor_of_eight(self):
        assert the_table_makes_scoring_cheap()["ratio"] == 8.0

    def test_the_table_amortises_quickly(self):
        assert the_table_makes_scoring_cheap()["table_amortises_after"] < 1000

    def test_a_codebook_of_one_centroid_is_refused(self):
        with pytest.raises(ConfigError, match="not a codebook"):
            the_table_makes_scoring_cheap(centroids=1)

    def test_every_score_is_non_negative(self):
        codes = train(gaussian(count=512, dimension=32).vectors, subspaces=4, centroids=64)
        assert float(asymmetric_scores(torch.randn(8, 32), codes).min()) >= -1e-4


class TestCompression:
    def test_the_codes_are_sixty_four_to_one(self):
        assert the_compression_is_enormous()["code_ratio"] == 64.0

    def test_but_the_index_is_only_seven_to_one(self):
        assert the_compression_is_enormous()["total_ratio"] < 10.0

    def test_because_the_codebooks_dominate_at_this_size(self):
        assert the_compression_is_enormous()["codebooks_dominate"]

    def test_it_beats_scalar_quantisation_by_sixteen_times(self):
        assert it_beats_scalar_quantisation_on_memory()["factor"] == 16.0

    def test_at_the_cost_of_a_training_pass(self):
        assert it_beats_scalar_quantisation_on_memory()["product_needs_training"]

    def test_and_of_a_great_deal_of_accuracy(self):
        result = and_loses_on_accuracy()
        assert result["scalar_recall"] > result["product_recall"] * 4

    def test_the_errors_are_orders_apart(self):
        result = and_loses_on_accuracy()
        assert result["product_error"] > result["scalar_error"] * 1000

    def test_which_is_the_one_place_the_two_measures_agree(self):
        result = and_loses_on_accuracy()
        assert result["product_error"] > result["scalar_error"]
        assert result["product_recall"] < result["scalar_recall"]


class TestSubspaces:
    def test_recall_rises_with_the_subspace_count(self):
        rows = [row["recall"] for row in subspace_sweep()]
        assert rows == sorted(rows)

    def test_and_the_error_falls(self):
        rows = [row["error"] for row in subspace_sweep()]
        assert rows == sorted(rows, reverse=True)

    def test_the_return_per_byte_increases(self):
        assert the_return_per_byte_increases()["increasing"]

    def test_rather_than_diminishing(self):
        assert not the_return_per_byte_increases()["diminishing"]

    def test_the_last_doubling_beats_the_first_by_five_times(self):
        result = the_return_per_byte_increases()
        assert result["last_doubling"] > result["first_doubling"] * 3

    def test_two_subspaces_are_nearly_useless(self):
        assert the_return_per_byte_increases()["at_two"] < 0.1

    def test_an_empty_subspace_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            subspace_sweep(counts=())

    def test_the_bytes_per_vector_is_the_subspace_count(self):
        rows = subspace_sweep(counts=(4, 8))
        assert all(row["bytes_per_vector"] == row["subspaces"] for row in rows)


class TestReranking:
    def test_reranking_rescues_the_recall(self):
        rows = {row["shortlist"]: row for row in reranking_rescues_it()}
        assert rows[400]["recall"] > 0.9

    def test_from_a_much_longer_shortlist_than_scalar_needed(self):
        result = it_needs_a_much_longer_shortlist_than_scalar()
        assert result["at_four_hundred"] > result["at_a_hundred"] > result["at_ten"]

    def test_twenty_times_longer(self):
        assert it_needs_a_much_longer_shortlist_than_scalar()["scalar_needed"] == 20

    def test_but_still_a_fifth_of_the_corpus(self):
        result = it_needs_a_much_longer_shortlist_than_scalar()
        assert result["still_a_fraction_of_the_corpus"]

    def test_the_gap_falls_with_the_shortlist(self):
        rows = [row["gap"] for row in reranking_rescues_it()]
        assert rows == sorted(rows, reverse=True)

    def test_an_empty_shortlist_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            reranking_rescues_it(shortlists=())

    def test_a_shortlist_shorter_than_k_is_refused(self):
        corpus = gaussian(count=512, dimension=32)
        codes = train(corpus.vectors, subspaces=4, centroids=64)
        with pytest.raises(ConfigError, match="cannot produce"):
            rerank(corpus.vectors[:4], corpus.vectors, codes, k=10, shortlist=5)

    def test_asking_for_more_codes_than_exist_is_refused(self):
        corpus = gaussian(count=512, dimension=32)
        codes = train(corpus.vectors, subspaces=4, centroids=64)
        with pytest.raises(ConfigError, match="from 512 codes"):
            search_codes(corpus.vectors[:4], codes, k=1024)


class TestStructure:
    def test_clustered_data_is_easier(self):
        assert structure_helps_here_too()["clustered_is_easier"]

    def test_the_error_moves_much_more_than_the_recall(self):
        # Twenty eight times better error, twice the recall. The same disagreement as
        # everywhere else in this package.
        result = structure_helps_here_too()
        error_ratio = result["gaussian_error"] / result["clustered_error"]
        recall_ratio = result["clustered_recall"] / result["gaussian_recall"]
        assert error_ratio > recall_ratio * 5

    def test_both_quantisers_appear_in_the_comparison(self):
        assert len(compare_quantisers()) == 2

    def test_reranking_closes_most_of_the_gap_between_them(self):
        rows = {row["method"]: row for row in compare_quantisers()}
        assert rows["product"]["reranked"] > rows["product"]["recall"] * 4

    def test_the_product_codes_are_sixteen_times_smaller(self):
        rows = {row["method"]: row for row in compare_quantisers()}
        assert rows["scalar"]["bytes_per_vector"] == rows["product"]["bytes_per_vector"] * 16

    def test_a_reconstruction_of_the_wrong_shape_is_refused(self):
        vectors = gaussian(count=512, dimension=32).vectors
        codes = train(vectors, subspaces=4, centroids=64)
        with pytest.raises(DataError, match="rebuilt against"):
            reconstruction_error(vectors[:64], codes)

from __future__ import annotations

import pytest
import torch

from vse.errors import BuildError, ConfigError, DataError, IndexStateError
from vse.quantize.residual import (
    ResidualCodes,
    ResidualIndex,
    a_bigger_codebook_or_more_stages,
    a_codebook_of_one_is_refused,
    a_corpus_smaller_than_the_codebook_is_refused,
    a_negative_rerank_is_refused,
    a_rank_one_corpus_is_refused,
    a_rerank_below_k_is_refused,
    a_rerank_matters_more_than_the_stages,
    a_single_stage_is_plain_quantisation,
    against_product_quantisation_at_matched_storage,
    codes_that_do_not_match_their_books_are_refused,
    decode,
    decoding_a_subset_matches_decoding_everything,
    each_stage_removes_the_same_fraction,
    fit,
    insertion_reuses_the_fitted_codebooks,
    more_stages_buy_recall,
    one_big_codebook_beats_several_small_ones,
    removal_takes_a_row_out,
    residual_norms,
    the_effective_codebook_multiplies,
    the_gap_depends_on_the_corpus,
    the_marginal_stage_is_worth_about_the_same_each_time,
    the_reconstruction_is_the_sum_of_the_codes,
    the_residual_shrinks_fast,
    the_scoring_cost_is_where_it_loses,
    the_shortlist_buys_more_than_a_stage,
    the_split_matters_more_on_correlated_coordinates,
    zero_stages_are_refused,
)
from vse.vectors.dataset import gaussian


class TestFitting:
    def test_a_fit_has_one_code_per_stage(self):
        corpus = gaussian(count=512, dimension=8).vectors
        codes = fit(corpus, stages=3, entries=16)
        assert tuple(codes.codes.shape) == (512, 3)

    def test_and_one_book_per_stage(self):
        corpus = gaussian(count=512, dimension=8).vectors
        codes = fit(corpus, stages=3, entries=16)
        assert tuple(codes.books.shape) == (3, 16, 8)

    def test_it_reports_its_shape(self):
        corpus = gaussian(count=512, dimension=8).vectors
        row = fit(corpus, stages=2, entries=16).as_dict()
        assert row["stages"] == 2 and row["entries"] == 16

    def test_the_effective_codebook_is_the_product(self):
        corpus = gaussian(count=512, dimension=8).vectors
        assert fit(corpus, stages=3, entries=16).effective_entries == 4096

    def test_a_byte_per_stage_at_two_hundred_and_fifty_six_entries(self):
        corpus = gaussian(count=512, dimension=8).vectors
        assert fit(corpus, stages=2, entries=256).bytes_per_vector == 2

    def test_and_still_a_byte_at_sixty_four(self):
        corpus = gaussian(count=512, dimension=8).vectors
        assert fit(corpus, stages=2, entries=64).bytes_per_vector == 2

    def test_zero_stages_are_refused(self):
        assert zero_stages_are_refused()

    def test_a_codebook_of_one_is_refused(self):
        assert a_codebook_of_one_is_refused()

    def test_a_corpus_smaller_than_the_codebook_is_refused(self):
        assert a_corpus_smaller_than_the_codebook_is_refused()

    def test_a_rank_one_corpus_is_refused(self):
        assert a_rank_one_corpus_is_refused()

    def test_codes_that_do_not_match_their_books_are_refused(self):
        assert codes_that_do_not_match_their_books_are_refused()

    def test_a_rank_one_code_matrix_is_refused(self):
        with pytest.raises(DataError, match="codes are a matrix"):
            ResidualCodes(codes=torch.zeros(10, dtype=torch.long), books=torch.zeros(2, 4, 8))

    def test_a_rank_two_book_stack_is_refused(self):
        with pytest.raises(DataError, match="stages by entries by width"):
            ResidualCodes(codes=torch.zeros(10, 2, dtype=torch.long), books=torch.zeros(4, 8))

    def test_a_negative_stage_count_is_refused(self):
        with pytest.raises(ConfigError, match="quantises nothing"):
            fit(torch.randn(512, 8), stages=-1)

    def test_a_tiny_corpus_is_refused(self):
        with pytest.raises(BuildError, match="cannot fit a codebook"):
            fit(torch.randn(10, 8), entries=64)


class TestDecoding:
    def test_the_reconstruction_is_the_sum_of_the_codes(self):
        assert the_reconstruction_is_the_sum_of_the_codes()["identical"]

    def test_exactly(self):
        assert the_reconstruction_is_the_sum_of_the_codes()["max_gap"] == 0.0

    def test_decoding_a_subset_matches(self):
        assert decoding_a_subset_matches_decoding_everything()["identical"]

    def test_a_single_stage_is_plain_quantisation(self):
        assert a_single_stage_is_plain_quantisation()["identical"]

    def test_decoding_returns_the_right_shape(self):
        corpus = gaussian(count=256, dimension=8).vectors
        codes = fit(corpus, stages=2, entries=16)
        assert tuple(decode(codes).shape) == (256, 8)

    def test_and_a_subset_returns_its_own_shape(self):
        corpus = gaussian(count=256, dimension=8).vectors
        codes = fit(corpus, stages=2, entries=16)
        assert tuple(decode(codes, torch.tensor([1, 2, 3])).shape) == (3, 8)

    def test_the_reconstruction_is_closer_than_the_mean(self):
        corpus = gaussian(count=1024, dimension=16).vectors
        codes = fit(corpus, stages=2, entries=64)
        reconstructed = decode(codes)
        mean = corpus.mean(dim=0, keepdim=True).expand_as(corpus)
        assert float((corpus - reconstructed).norm()) < float((corpus - mean).norm())


class TestTheResidual:
    def test_it_shrinks_at_every_stage(self):
        rows = [row["residual_share"] for row in the_residual_shrinks_fast()]
        assert rows == sorted(rows, reverse=True)

    def test_each_stage_removes_the_same_fraction(self):
        assert each_stage_removes_the_same_fraction()["relative_is_constant"]

    def test_while_the_absolute_amount_falls(self):
        assert each_stage_removes_the_same_fraction()["absolute_falls"]

    def test_the_norms_have_one_entry_per_stage(self):
        corpus = gaussian(count=512, dimension=8).vectors
        codes = fit(corpus, stages=3, entries=16)
        assert int(residual_norms(corpus, codes).numel()) == 3

    def test_and_are_all_below_one(self):
        corpus = gaussian(count=512, dimension=8).vectors
        codes = fit(corpus, stages=3, entries=16)
        assert bool(torch.all(residual_norms(corpus, codes) < 1.0))

    def test_a_zero_stage_measurement_is_refused(self):
        with pytest.raises(ConfigError, match="measures nothing"):
            the_residual_shrinks_fast(stages=0)


class TestStages:
    def test_more_stages_buy_recall(self):
        rows = [row["recall"] for row in more_stages_buy_recall()]
        assert rows == sorted(rows)

    def test_and_cost_a_byte_each(self):
        rows = {row["stages"]: row for row in more_stages_buy_recall()}
        assert rows[4]["bytes_per_vector"] == 4

    def test_the_marginal_gain_is_flat(self):
        assert the_marginal_stage_is_worth_about_the_same_each_time()["flat"]

    def test_with_no_knee(self):
        assert the_marginal_stage_is_worth_about_the_same_each_time()["no_knee_over_this_range"]

    def test_an_empty_stage_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            more_stages_buy_recall(stages=())

    def test_the_effective_codebook_multiplies(self):
        rows = {row["stages"]: row for row in the_effective_codebook_multiplies()}
        assert rows[2]["effective_entries"] == 65536

    def test_and_a_single_codebook_that_size_would_be_enormous(self):
        rows = {row["stages"]: row for row in the_effective_codebook_multiplies()}
        assert rows[2]["single_codebook_table_bytes"] > 200000

    def test_an_empty_multiplication_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            the_effective_codebook_multiplies(stages=())


class TestAgainstProduct:
    def test_the_product_code_wins_on_a_gaussian_corpus(self):
        assert not against_product_quantisation_at_matched_storage()["residual_wins"]

    def test_at_the_same_storage(self):
        assert against_product_quantisation_at_matched_storage()["bytes_each"] == 2

    def test_the_residual_code_wins_on_a_subspace_corpus(self):
        assert the_gap_depends_on_the_corpus()["subspace_gap"] > 0.1

    def test_and_loses_on_a_gaussian_one(self):
        assert the_gap_depends_on_the_corpus()["gaussian_gap"] < 0

    def test_the_subspace_gap_is_the_widest(self):
        assert the_gap_depends_on_the_corpus()["widest"] == "subspace"

    def test_three_corpora_are_compared(self):
        assert len(the_split_matters_more_on_correlated_coordinates()) == 3

    def test_the_scoring_cost_is_where_it_loses(self):
        assert the_scoring_cost_is_where_it_loses()["residual_is_dearer"]

    def test_by_a_factor_of_three(self):
        assert the_scoring_cost_is_where_it_loses()["ratio"] > 2.0


class TestTheShortlist:
    def test_the_shortlist_buys_more_than_a_stage(self):
        assert the_shortlist_buys_more_than_a_stage()["shortlist_wins"]

    def test_and_they_compose(self):
        rows = {
            (row["stages"], row["shortlist"]): row
            for row in a_rerank_matters_more_than_the_stages()
        }
        assert rows[(4, 200)]["recall"] > rows[(4, 0)]["recall"]
        assert rows[(4, 200)]["recall"] > rows[(1, 200)]["recall"]

    def test_the_recall_rises_with_the_shortlist_at_every_stage_count(self):
        rows = {
            (row["stages"], row["shortlist"]): row
            for row in a_rerank_matters_more_than_the_stages()
        }
        for stages in (1, 2, 4):
            assert rows[(stages, 0)]["recall"] < rows[(stages, 200)]["recall"]

    def test_a_rerank_never_lowers_the_recall(self):
        rows = {
            (row["stages"], row["shortlist"]): row
            for row in a_rerank_matters_more_than_the_stages()
        }
        for stages in (1, 2, 4):
            assert rows[(stages, 50)]["recall"] >= rows[(stages, 0)]["recall"]

    def test_an_empty_shortlist_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            a_rerank_matters_more_than_the_stages(shortlists=())

    def test_an_empty_stage_list_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            a_rerank_matters_more_than_the_stages(stages=())


class TestTheDecomposition:
    def test_one_big_codebook_is_more_accurate(self):
        assert one_big_codebook_beats_several_small_ones()["single_is_more_accurate"]

    def test_and_costs_far_more_to_store(self):
        assert one_big_codebook_beats_several_small_ones()["and_costs_more_to_store"]

    def test_by_two_orders_of_magnitude(self):
        result = one_big_codebook_beats_several_small_ones()
        assert result["single_codebook_bytes"] > result["four_stage_codebook_bytes"] * 100

    def test_every_setting_expresses_the_same_number_of_points(self):
        rows = a_bigger_codebook_or_more_stages()
        assert all(row["effective_entries"] == 4096 for row in rows)

    def test_an_empty_setting_list_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            a_bigger_codebook_or_more_stages(settings=())


class TestTheIndex:
    def test_it_returns_k_neighbours(self):
        corpus = gaussian(count=1024, dimension=16).vectors
        index = ResidualIndex(16, stages=2, entries=32)
        index.build(corpus)
        found, _ = index.search(corpus[:8], k=7)
        assert tuple(found.identifiers.shape) == (8, 7)

    def test_a_rerank_still_returns_k(self):
        corpus = gaussian(count=1024, dimension=16).vectors
        index = ResidualIndex(16, stages=2, entries=32, rerank=100)
        index.build(corpus)
        found, _ = index.search(corpus[:8], k=7)
        assert tuple(found.identifiers.shape) == (8, 7)

    def test_a_rerank_finds_the_query_itself(self):
        corpus = gaussian(count=1024, dimension=16).vectors
        index = ResidualIndex(16, stages=2, entries=32, rerank=200)
        index.build(corpus)
        found, _ = index.search(corpus[:1], k=1)
        assert int(found.identifiers[0, 0]) == 0

    def test_a_reranked_result_is_sorted(self):
        corpus = gaussian(count=1024, dimension=16).vectors
        index = ResidualIndex(16, stages=2, entries=32, rerank=100)
        index.build(corpus)
        found, _ = index.search(corpus[:4], k=10)
        assert bool(torch.all(found.scores[:, 1:] >= found.scores[:, :-1] - 1e-5))

    def test_a_negative_rerank_is_refused(self):
        assert a_negative_rerank_is_refused()

    def test_a_rerank_below_k_is_refused(self):
        assert a_rerank_below_k_is_refused()

    def test_searching_before_building_is_refused(self):
        with pytest.raises(IndexStateError):
            ResidualIndex(16).search(torch.randn(1, 16), k=5)

    def test_insertion_reuses_the_fitted_codebooks(self):
        assert insertion_reuses_the_fitted_codebooks()["books_unchanged"]

    def test_and_grows_the_index(self):
        assert insertion_reuses_the_fitted_codebooks()["size"] == 1500

    def test_removal_takes_a_row_out(self):
        assert not removal_takes_a_row_out()["still_present"]

    def test_and_lowers_the_size(self):
        assert removal_takes_a_row_out()["size"] == 1023

    def test_and_still_returns_k(self):
        assert removal_takes_a_row_out()["still_returns_k"]

    def test_removing_a_row_that_is_not_there_is_refused(self):
        corpus = gaussian(count=512, dimension=16).vectors
        index = ResidualIndex(16, stages=2, entries=32)
        index.build(corpus)
        with pytest.raises(ConfigError, match="is not one of"):
            index.remove([9999])

    def test_the_memory_counts_the_codes_and_the_books(self):
        corpus = gaussian(count=1024, dimension=16).vectors
        index = ResidualIndex(16, stages=2, entries=32)
        index.build(corpus)
        assert index.memory_bytes() > 1024 * 2

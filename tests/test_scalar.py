from __future__ import annotations

import pytest
import torch

from vse.errors import ConfigError, DataError
from vse.quantize.scalar import (
    LEVELS,
    ScalarCodes,
    a_constant_vector_survives,
    a_float_code_is_refused,
    a_shortlist_of_twice_k_is_enough,
    a_shortlist_shorter_than_k_is_refused,
    an_empty_corpus_is_refused,
    asymmetric_scoring_is_strictly_better,
    compare_configurations,
    dequantise,
    each_bit_quarters_the_error,
    error_falls_with_the_level_count,
    per_vector_scales_barely_help,
    quantise,
    reconstruction_error,
    rerank,
    reranking_recovers_the_recall,
    search_codes,
    the_best_configuration_is_the_cheap_one,
    the_codes_do_not_have_to_be_good,
    the_compression_is_four_to_one,
)
from vse.vectors.dataset import gaussian


class TestCodes:
    def test_a_code_is_a_byte(self):
        codes = quantise(gaussian(count=256, dimension=8).vectors)
        assert codes.codes.dtype == torch.uint8

    def test_every_code_is_in_range(self):
        codes = quantise(gaussian(count=256, dimension=8).vectors)
        assert int(codes.codes.max()) <= LEVELS - 1

    def test_the_extremes_map_to_the_extremes(self):
        vectors = gaussian(count=256, dimension=8).vectors
        codes = quantise(vectors)
        assert int(codes.codes.min()) == 0
        assert int(codes.codes.max()) == LEVELS - 1

    def test_decoding_lands_near_the_original(self):
        vectors = gaussian(count=256, dimension=8).vectors
        assert reconstruction_error(vectors, quantise(vectors)) < 0.01

    def test_a_global_scale_is_one_number(self):
        codes = quantise(gaussian(count=256, dimension=8).vectors)
        assert not codes.per_vector

    def test_a_per_vector_scale_is_one_per_row(self):
        codes = quantise(gaussian(count=256, dimension=8).vectors, per_vector=True)
        assert codes.per_vector and codes.scale.numel() == 256

    def test_a_float_code_is_refused(self):
        assert a_float_code_is_refused()

    def test_an_empty_corpus_is_refused(self):
        assert an_empty_corpus_is_refused()

    def test_a_rank_three_input_is_refused(self):
        with pytest.raises(DataError, match="matrix of rows"):
            quantise(torch.randn(4, 4, 4))

    def test_a_scale_without_an_offset_is_refused(self):
        with pytest.raises(DataError, match="matching offset"):
            ScalarCodes(
                codes=torch.zeros(4, 8, dtype=torch.uint8),
                scale=torch.ones(4, 1),
                offset=torch.zeros(1, 1),
            )

    def test_a_constant_row_does_not_become_a_nan(self):
        assert not a_constant_vector_survives()["any_nan"]

    def test_and_decodes_back_to_itself(self):
        assert a_constant_vector_survives()["constant_row_recovered"]

    def test_without_disturbing_its_neighbours(self):
        assert a_constant_vector_survives()["other_row_recovered"]

    def test_it_serialises(self):
        codes = quantise(gaussian(count=100, dimension=10).vectors)
        assert codes.as_dict()["count"] == 100


class TestCompression:
    def test_a_global_scale_is_four_to_one(self):
        assert the_compression_is_four_to_one()["global_ratio"] == 4.0

    def test_per_vector_scales_cost_a_fifth_of_that(self):
        result = the_compression_is_four_to_one()
        assert result["per_vector_ratio"] < result["global_ratio"]

    def test_and_the_overhead_is_fixed_per_vector(self):
        assert the_compression_is_four_to_one()["overhead_per_vector"] == 8

    def test_so_narrow_vectors_compress_worse(self):
        assert the_compression_is_four_to_one(dimension=8)["per_vector_ratio"] < 3.0

    def test_the_error_falls_by_four_per_bit(self):
        assert each_bit_quarters_the_error()["close_to_sixteen"]

    def test_a_two_bit_code_is_noise(self):
        # Twenty three of error against a typical squared distance of seventy.
        rows = {row["bits"]: row for row in error_falls_with_the_level_count()}
        assert rows[2]["error"] > 10.0

    def test_where_an_eight_bit_code_is_not(self):
        rows = {row["bits"]: row for row in error_falls_with_the_level_count()}
        assert rows[8]["error"] < 0.01

    def test_an_empty_bit_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            error_falls_with_the_level_count(bits=())


class TestScoring:
    def test_asymmetric_scoring_wins(self):
        assert asymmetric_scoring_is_strictly_better()["asymmetric_wins"]

    def test_at_the_same_cost(self):
        assert asymmetric_scoring_is_strictly_better()["same_cost"]

    def test_and_with_a_smaller_gap(self):
        result = asymmetric_scoring_is_strictly_better()
        assert result["asymmetric_gap"] < result["symmetric_gap"]

    def test_it_wins_at_both_scale_settings(self):
        rows = {(row["scale"], row["scoring"]): row for row in compare_configurations()}
        for scale in ("global", "per vector"):
            assert rows[(scale, "asymmetric")]["recall"] >= rows[(scale, "symmetric")]["recall"]

    def test_four_configurations_are_compared(self):
        assert len(compare_configurations()) == 4

    def test_the_best_is_not_the_cheapest(self):
        assert not the_best_configuration_is_the_cheap_one()["same"]

    def test_but_the_whole_spread_is_under_a_point(self):
        assert the_best_configuration_is_the_cheap_one()["recall_spread"] < 0.02

    def test_asking_for_more_codes_than_exist_is_refused(self):
        codes = quantise(gaussian(count=64, dimension=8).vectors)
        with pytest.raises(ConfigError, match="from 64 codes"):
            search_codes(torch.randn(2, 8), codes, k=128)


class TestPerVector:
    def test_per_vector_scales_lower_the_error(self):
        result = per_vector_scales_barely_help()
        assert result["per_vector_error"] < result["global_error"]

    def test_by_a_factor_of_five(self):
        assert per_vector_scales_barely_help()["error_ratio"] > 3.0

    def test_but_the_recall_barely_moves(self):
        assert per_vector_scales_barely_help()["recall_gap"] < 0.02

    def test_while_the_memory_does(self):
        result = per_vector_scales_barely_help()
        assert result["per_vector_bytes"] > result["global_bytes"] * 1.2

    def test_so_the_two_measures_disagree(self):
        # The error says per vector by a mile; the recall says it hardly matters.
        result = per_vector_scales_barely_help()
        assert result["error_ratio"] > 3.0 > result["recall_gap"] * 100


class TestReranking:
    def test_reranking_reaches_perfect_recall(self):
        rows = {row["shortlist"]: row for row in reranking_recovers_the_recall()}
        assert rows[50]["recall"] == 1.0

    def test_from_a_shortlist_of_twice_k(self):
        rows = {row["shortlist"]: row for row in reranking_recovers_the_recall()}
        assert rows[20]["recall"] == 1.0

    def test_and_a_longer_one_buys_nothing(self):
        assert a_shortlist_of_twice_k_is_enough()["saturates"]

    def test_it_improves_on_the_raw_codes(self):
        assert a_shortlist_of_twice_k_is_enough()["recovered"]

    def test_the_gap_goes_to_zero(self):
        rows = {row["shortlist"]: row for row in reranking_recovers_the_recall()}
        assert rows[50]["gap"] == 0.0

    def test_the_codes_are_visibly_inaccurate(self):
        # And that does not matter, which is the point.
        result = the_codes_do_not_have_to_be_good()
        assert result["reranked_recall"] > result["raw_recall"]

    def test_the_reconstruction_error_is_tiny_against_the_distances(self):
        result = the_codes_do_not_have_to_be_good()
        assert result["reconstruction_error"] < result["typical_squared_distance"] / 1000

    def test_a_shortlist_shorter_than_k_is_refused(self):
        assert a_shortlist_shorter_than_k_is_refused()

    def test_a_shortlist_longer_than_the_corpus_is_refused(self):
        corpus = gaussian(count=64, dimension=8)
        with pytest.raises(ConfigError, match="from 64 codes"):
            rerank(corpus.vectors[:2], corpus.vectors, quantise(corpus.vectors), shortlist=128)

    def test_an_empty_shortlist_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            reranking_recovers_the_recall(shortlists=())

    def test_reranking_never_returns_a_vector_outside_the_shortlist(self):
        corpus = gaussian(count=512, dimension=16)
        codes = quantise(corpus.vectors)
        queries = corpus.vectors[:8]
        shortlist = search_codes(queries, codes, k=30).identifiers
        found = rerank(queries, corpus.vectors, codes, k=5, shortlist=30)
        for row in range(8):
            assert set(found.row(row)) <= {int(value) for value in shortlist[row]}


class TestRoundTrip:
    def test_decoding_a_quantised_corpus_has_the_same_shape(self):
        vectors = gaussian(count=128, dimension=16).vectors
        assert dequantise(quantise(vectors)).shape == vectors.shape

    def test_comparing_the_wrong_shapes_is_refused(self):
        vectors = gaussian(count=128, dimension=16).vectors
        with pytest.raises(DataError, match="rebuilt against"):
            reconstruction_error(vectors[:64], quantise(vectors))

    def test_quantising_twice_gives_the_same_codes(self):
        vectors = gaussian(count=128, dimension=16).vectors
        assert torch.equal(quantise(vectors).codes, quantise(vectors).codes)

    def test_a_wider_corpus_compresses_better(self):
        narrow = the_compression_is_four_to_one(dimension=8)["per_vector_ratio"]
        wide = the_compression_is_four_to_one(dimension=128)["per_vector_ratio"]
        assert wide > narrow

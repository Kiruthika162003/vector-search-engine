from __future__ import annotations

import math

import pytest
import torch

from vse.errors import ConfigError, DataError
from vse.quantize.binary import (
    WORD_BITS,
    BinaryCodes,
    BinaryIndex,
    _popcount,
    a_dimension_that_is_not_a_multiple_of_a_word_wastes_the_tail,
    a_mismatched_rotation_is_refused,
    a_negative_rerank_is_refused,
    a_rerank_below_k_is_refused,
    a_rerank_is_part_of_the_method,
    a_rotation_helps_a_clustered_corpus_and_not_a_gaussian_one,
    a_zero_dimension_is_refused,
    an_unknown_rotation_is_refused,
    angle_from_hamming,
    binary_alone_barely_ranks,
    binary_beats_product_quantisation_on_speed_and_loses_on_accuracy,
    centring_matters_more_than_rotating,
    codes_of_the_wrong_width_are_refused,
    encode_queries,
    hamming,
    hamming_is_a_metric,
    insertion_reuses_the_fitted_centre,
    normalising_is_worth_more_than_any_amount_of_bits,
    pack,
    packing_round_trips,
    quantise,
    removal_takes_a_row_out_of_the_result,
    the_bit_count_estimates_the_angle,
    the_codes_measure_an_angle_so_the_corpus_should_be_normalised,
    the_error_falls_as_one_over_root_d,
    the_gap_and_the_error_are_the_same_size,
    the_population_count_is_right,
    the_rate_matches_the_prediction,
    the_recall_does_not_move_with_the_dimension,
    the_rerank_recovers_most_of_the_loss,
    the_rotation_is_worth_more_on_structure,
    the_signal_falls_at_the_same_rate_as_the_noise,
    thirty_two_to_one,
    unpack,
    words_needed,
)
from vse.vectors.dataset import gaussian


class TestPacking:
    def test_a_word_holds_sixty_four_bits(self):
        assert words_needed(64) == 1

    def test_and_sixty_five_needs_two(self):
        assert words_needed(65) == 2

    def test_five_hundred_and_twelve_needs_eight(self):
        assert words_needed(512) == 8

    def test_a_zero_dimension_is_refused(self):
        assert a_zero_dimension_is_refused()

    def test_a_negative_dimension_is_refused(self):
        with pytest.raises(ConfigError, match="not a dimension"):
            words_needed(-8)

    def test_packing_round_trips(self):
        assert packing_round_trips()["identical"]

    def test_at_a_width_that_is_not_word_aligned(self):
        assert packing_round_trips(dimension=300)["identical"]

    def test_and_at_one_that_is(self):
        assert packing_round_trips(dimension=256)["identical"]

    def test_a_single_bit_packs(self):
        bits = torch.tensor([[True]])
        assert bool(torch.equal(unpack(pack(bits), 1), bits))

    def test_a_rank_one_input_is_refused(self):
        with pytest.raises(DataError, match="bits are a matrix"):
            pack(torch.tensor([True, False]))

    def test_a_rank_one_word_matrix_is_refused(self):
        with pytest.raises(DataError, match="words are a matrix"):
            unpack(torch.zeros(4, dtype=torch.int64), 8)

    def test_the_population_count_is_right(self):
        assert the_population_count_is_right()["identical"]

    def test_with_no_disagreement_at_all(self):
        assert the_population_count_is_right()["max_disagreement"] == 0

    def test_an_empty_word_counts_zero(self):
        assert int(_popcount(torch.zeros(1, 1, dtype=torch.int64))[0, 0]) == 0

    def test_a_full_word_counts_sixty_four(self):
        assert int(_popcount(torch.full((1, 1), -1, dtype=torch.int64))[0, 0]) == WORD_BITS


class TestTheAngleEstimate:
    def test_the_bit_count_estimates_the_angle(self):
        assert the_bit_count_estimates_the_angle()["close"]

    def test_to_within_a_few_degrees(self):
        assert the_bit_count_estimates_the_angle()["mean_error_degrees"] < 5.0

    def test_the_error_falls_with_the_dimension(self):
        rows = [row["mean_error_degrees"] for row in the_error_falls_as_one_over_root_d()]
        assert rows == sorted(rows, reverse=True)

    def test_at_the_predicted_rate(self):
        assert the_rate_matches_the_prediction()["matches"]

    def test_and_so_does_the_signal(self):
        assert the_signal_falls_at_the_same_rate_as_the_noise()["both_fall"]

    def test_so_the_correlation_does_not_move(self):
        assert the_signal_falls_at_the_same_rate_as_the_noise()["the_ratio_holds"]

    def test_across_sixty_four_times_the_bits(self):
        result = the_signal_falls_at_the_same_rate_as_the_noise()
        assert result["error_at_thirty_two"] > result["error_at_two_thousand"] * 5

    def test_a_measurement_too_small_to_mean_anything_is_refused(self):
        with pytest.raises(ConfigError, match="not a measurement"):
            the_bit_count_estimates_the_angle(trials=1)

    def test_a_dimension_too_small_is_refused(self):
        with pytest.raises(ConfigError, match="not a measurement"):
            the_bit_count_estimates_the_angle(dimension=4)

    def test_an_empty_dimension_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            the_error_falls_as_one_over_root_d(dimensions=())

    def test_a_full_bit_count_is_half_a_turn(self):
        assert abs(float(angle_from_hamming(torch.tensor([64.0]), 64)) - math.pi) < 1e-6

    def test_and_no_differing_bits_is_no_angle(self):
        assert float(angle_from_hamming(torch.tensor([0.0]), 64)) == 0.0

    def test_a_zero_dimension_angle_is_refused(self):
        with pytest.raises(ConfigError, match="not a dimension"):
            angle_from_hamming(torch.tensor([1.0]), 0)


class TestTheRatio:
    def test_the_gap_and_the_error_are_the_same_size(self):
        rows = the_gap_and_the_error_are_the_same_size()
        for row in rows:
            ratio = row["angular_gap_degrees"] / row["estimator_error_degrees"]
            assert 0.85 < ratio < 1.15

    def test_the_gap_falls_with_the_dimension(self):
        rows = [row["angular_gap_degrees"] for row in the_gap_and_the_error_are_the_same_size()]
        assert rows == sorted(rows, reverse=True)

    def test_and_the_recall_does_not(self):
        assert the_recall_does_not_move_with_the_dimension()["recall_is_flat"]

    def test_even_though_the_gap_fell_by_four(self):
        assert the_recall_does_not_move_with_the_dimension()["gap_fell_by_four"]

    def test_four_dimensions_are_measured(self):
        assert len(the_gap_and_the_error_are_the_same_size()) == 4

    def test_an_empty_ratio_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            the_gap_and_the_error_are_the_same_size(dimensions=())


class TestCompression:
    def test_thirty_two_to_one(self):
        assert thirty_two_to_one()["compression"] == 32.0

    def test_sixty_four_bytes_for_five_hundred_and_twelve_dimensions(self):
        assert thirty_two_to_one()["bytes_per_vector"] == 64

    def test_a_word_aligned_width_wastes_nothing(self):
        rows = {
            row["dimension"]: row
            for row in a_dimension_that_is_not_a_multiple_of_a_word_wastes_the_tail()
        }
        assert rows[128]["wasted_bits"] == 0

    def test_a_hundred_wastes_a_fifth(self):
        rows = {
            row["dimension"]: row
            for row in a_dimension_that_is_not_a_multiple_of_a_word_wastes_the_tail()
        }
        assert rows[100]["wasted_share"] > 0.2

    def test_three_hundred_wastes_little(self):
        rows = {
            row["dimension"]: row
            for row in a_dimension_that_is_not_a_multiple_of_a_word_wastes_the_tail()
        }
        assert rows[300]["wasted_share"] < 0.1

    def test_an_empty_padding_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            a_dimension_that_is_not_a_multiple_of_a_word_wastes_the_tail(dimensions=())

    def test_codes_report_their_compression(self):
        corpus = gaussian(count=256, dimension=128)
        assert quantise(corpus.vectors).compression == 32.0

    def test_and_serialise(self):
        corpus = gaussian(count=256, dimension=128)
        assert quantise(corpus.vectors).as_dict()["words"] == 2

    def test_codes_of_the_wrong_width_are_refused(self):
        assert codes_of_the_wrong_width_are_refused()

    def test_a_rank_one_code_matrix_is_refused(self):
        with pytest.raises(DataError, match="codes are a matrix"):
            BinaryCodes(words=torch.zeros(4, dtype=torch.int64), dimension=64)

    def test_a_zero_dimension_code_is_refused(self):
        with pytest.raises(ConfigError, match="not a dimension"):
            BinaryCodes(words=torch.zeros(4, 1, dtype=torch.int64), dimension=0)


class TestSearchQuality:
    def test_binary_alone_barely_ranks(self):
        assert binary_alone_barely_ranks()["recall"] < 0.2

    def test_and_the_shortlist_is_what_makes_it_work(self):
        assert the_rerank_recovers_most_of_the_loss()["recovers"]

    def test_a_shortlist_of_four_hundred_is_far_better_than_none(self):
        result = the_rerank_recovers_most_of_the_loss()
        assert result["recall_at_four_hundred"] > result["recall_without_rerank"] * 5

    def test_the_recall_rises_with_the_shortlist(self):
        rows = [row["recall"] for row in a_rerank_is_part_of_the_method()]
        assert rows == sorted(rows)

    def test_and_so_does_the_cost(self):
        rows = [row["distances_per_query"] for row in a_rerank_is_part_of_the_method()]
        assert rows == sorted(rows)

    def test_an_empty_shortlist_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            a_rerank_is_part_of_the_method(shortlists=())

    def test_normalising_helps(self):
        assert normalising_is_worth_more_than_any_amount_of_bits()["helps"]

    def test_by_more_than_a_tenth_at_a_shortlist_of_a_hundred(self):
        assert normalising_is_worth_more_than_any_amount_of_bits()["gain_at_a_hundred"] > 0.1

    def test_and_at_four_hundred_too(self):
        result = normalising_is_worth_more_than_any_amount_of_bits()
        assert result["normalised_at_four_hundred"] > result["raw_at_four_hundred"]

    def test_an_empty_normalisation_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            the_codes_measure_an_angle_so_the_corpus_should_be_normalised(shortlists=())


class TestPreprocessing:
    def test_centring_helps(self):
        assert centring_matters_more_than_rotating()["centring_helps"]

    def test_by_an_order_of_magnitude(self):
        result = centring_matters_more_than_rotating()
        assert result["centred"] > result["raw"] * 10

    def test_and_beats_rotating(self):
        assert centring_matters_more_than_rotating()["centring_beats_rotating"]

    def test_an_uncentred_corpus_is_nearly_useless(self):
        assert centring_matters_more_than_rotating()["raw"] < 0.02

    def test_the_rotation_is_worth_more_on_structure(self):
        assert the_rotation_is_worth_more_on_structure()["worth_more_on_structure"]

    def test_it_does_nothing_on_a_gaussian_corpus(self):
        assert abs(the_rotation_is_worth_more_on_structure()["gaussian_gain"]) < 0.01

    def test_and_something_on_a_clustered_one(self):
        assert the_rotation_is_worth_more_on_structure()["clustered_gain"] > 0.01

    def test_three_rotations_are_compared(self):
        rows = a_rotation_helps_a_clustered_corpus_and_not_a_gaussian_one()
        assert all({"none", "random", "pca"} <= set(row) for row in rows)

    def test_a_mismatched_rotation_is_refused(self):
        assert a_mismatched_rotation_is_refused()

    def test_an_unknown_rotation_is_refused(self):
        assert an_unknown_rotation_is_refused()

    def test_a_rank_one_corpus_is_refused(self):
        with pytest.raises(DataError, match="a corpus is a matrix"):
            quantise(torch.randn(16))

    def test_queries_get_the_same_treatment_as_the_corpus(self):
        corpus = gaussian(count=256, dimension=64)
        codes = quantise(corpus.vectors)
        assert int(encode_queries(corpus.vectors[:4], codes).shape[1]) == 1


class TestAgainstProduct:
    def test_the_storage_is_matched(self):
        assert binary_beats_product_quantisation_on_speed_and_loses_on_accuracy()[
            "storage_is_matched"
        ]

    def test_product_codes_are_more_accurate(self):
        assert binary_beats_product_quantisation_on_speed_and_loses_on_accuracy()[
            "product_is_more_accurate"
        ]

    def test_by_a_factor_of_two(self):
        result = binary_beats_product_quantisation_on_speed_and_loses_on_accuracy()
        assert result["product_recall"] > result["binary_recall"] * 2


class TestTheMetric:
    def test_hamming_obeys_the_triangle_inequality(self):
        assert hamming_is_a_metric(trials=200)["holds"]

    def test_with_no_violations(self):
        assert hamming_is_a_metric(trials=200)["violations"] == 0

    def test_a_vector_is_zero_from_itself(self):
        rows = pack(torch.randn(4, 64) > 0)
        assert int(hamming(rows[:1], rows[:1])[0, 0]) == 0

    def test_a_vector_is_the_full_width_from_its_negation(self):
        bits = torch.randn(1, 64) > 0
        assert int(hamming(pack(bits), pack(~bits))[0, 0]) == 64

    def test_it_is_symmetric(self):
        rows = pack(torch.randn(4, 64) > 0)
        assert int(hamming(rows[:1], rows[1:2])[0, 0]) == int(
            hamming(rows[1:2], rows[:1])[0, 0]
        )

    def test_mismatched_widths_are_refused(self):
        with pytest.raises(DataError, match="cannot be compared"):
            hamming(torch.zeros(2, 1, dtype=torch.int64), torch.zeros(2, 4, dtype=torch.int64))

    def test_a_rank_one_comparison_is_refused(self):
        with pytest.raises(DataError, match="two matrices"):
            hamming(torch.zeros(4, dtype=torch.int64), torch.zeros(2, 4, dtype=torch.int64))


class TestTheIndex:
    def test_it_returns_k_neighbours(self):
        corpus = gaussian(count=512, dimension=64)
        index = BinaryIndex(64)
        index.build(corpus.vectors)
        found, _ = index.search(corpus.vectors[:8], k=7)
        assert tuple(found.identifiers.shape) == (8, 7)

    def test_a_rerank_still_returns_k(self):
        corpus = gaussian(count=512, dimension=64)
        index = BinaryIndex(64, rerank=50)
        index.build(corpus.vectors)
        found, _ = index.search(corpus.vectors[:8], k=7)
        assert tuple(found.identifiers.shape) == (8, 7)

    def test_a_rerank_finds_the_query_itself(self):
        corpus = gaussian(count=512, dimension=64)
        index = BinaryIndex(64, rerank=100)
        index.build(corpus.vectors)
        found, _ = index.search(corpus.vectors[:1], k=1)
        assert int(found.identifiers[0, 0]) == 0

    def test_a_negative_rerank_is_refused(self):
        assert a_negative_rerank_is_refused()

    def test_a_rerank_below_k_is_refused(self):
        assert a_rerank_below_k_is_refused()

    def test_searching_before_building_is_refused(self):
        from vse.errors import IndexStateError

        with pytest.raises(IndexStateError):
            BinaryIndex(64).search(torch.randn(1, 64), k=5)

    def test_removal_takes_a_row_out(self):
        assert not removal_takes_a_row_out_of_the_result()["still_present"]

    def test_and_lowers_the_size(self):
        assert removal_takes_a_row_out_of_the_result()["size_fell"]

    def test_and_still_returns_k(self):
        assert removal_takes_a_row_out_of_the_result()["still_returns_k"]

    def test_removing_a_row_that_is_not_there_is_refused(self):
        corpus = gaussian(count=256, dimension=32)
        index = BinaryIndex(32)
        index.build(corpus.vectors)
        with pytest.raises(ConfigError, match="is not one of"):
            index.remove([9999])

    def test_removing_twice_counts_once(self):
        corpus = gaussian(count=256, dimension=32)
        index = BinaryIndex(32)
        index.build(corpus.vectors)
        assert index.remove([4]) == 1
        assert index.remove([4]) == 0

    def test_insertion_reuses_the_fitted_centre(self):
        assert insertion_reuses_the_fitted_centre()["centre_unchanged"]

    def test_and_grows_the_index(self):
        assert insertion_reuses_the_fitted_centre()["size"] == 992

    def test_the_memory_is_the_packed_codes(self):
        corpus = gaussian(count=1024, dimension=128)
        index = BinaryIndex(128)
        index.build(corpus.vectors)
        assert index.memory_bytes() == 1024 * 16

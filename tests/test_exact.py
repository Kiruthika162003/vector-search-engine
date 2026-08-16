from __future__ import annotations

import pytest
import torch

from vse.errors import ConfigError, DataError
from vse.vectors.exact import (
    Neighbours,
    a_mismatched_result_is_refused,
    a_vector_is_its_own_nearest_neighbour,
    asking_for_more_than_the_corpus_is_refused,
    batching_changes_nothing,
    batching_costs_a_second_selection,
    batching_saves_the_matrix,
    comparing_different_shapes_is_refused,
    duplicated_corpus,
    identifier_overlap,
    k_sweep,
    random_corpus,
    random_queries,
    score_gap,
    scores_for,
    search,
    search_batched,
    ties_make_identifiers_ambiguous,
    without_ties_the_two_measures_agree,
)


class TestSearch:
    def test_a_vector_is_its_own_nearest_neighbour(self):
        assert a_vector_is_its_own_nearest_neighbour()["all_self"]

    def test_at_a_distance_of_zero(self):
        assert a_vector_is_its_own_nearest_neighbour()["largest_score"] < 1e-4

    def test_but_not_under_inner_product(self):
        # Corroborates the axiom result in the metric module from a different direction.
        assert not a_vector_is_its_own_nearest_neighbour("ip")["all_self"]

    def test_the_results_come_back_closest_first(self):
        found = search(random_queries(count=8), random_corpus(), k=10)
        assert bool((found.scores[:, 1:] >= found.scores[:, :-1] - 1e-6).all())

    def test_the_result_has_the_shape_that_was_asked_for(self):
        found = search(random_queries(count=8), random_corpus(), k=7)
        assert (found.queries, found.k) == (8, 7)

    def test_the_scores_match_a_rescoring_of_the_identifiers(self):
        corpus, queries = random_corpus(), random_queries(count=8)
        found = search(queries, corpus, k=10)
        assert torch.allclose(
            found.scores, scores_for(queries, corpus, found.identifiers), atol=1e-4
        )

    def test_a_row_can_be_read_out(self):
        found = search(random_queries(count=4), random_corpus(), k=5)
        assert len(found.row(0)) == 5

    def test_asking_for_a_row_that_was_not_queried_is_refused(self):
        found = search(random_queries(count=4), random_corpus(), k=5)
        with pytest.raises(ConfigError, match="not one of the 4"):
            found.row(9)

    def test_asking_for_more_neighbours_than_the_corpus_is_refused(self):
        assert asking_for_more_than_the_corpus_is_refused()

    def test_asking_for_none_is_refused(self):
        with pytest.raises(ConfigError, match="not a query"):
            search(random_queries(count=2), random_corpus(), k=0)

    def test_an_empty_corpus_is_refused(self):
        with pytest.raises(DataError, match="is empty"):
            search(random_queries(count=2), torch.zeros(0, 32))

    def test_a_mismatched_result_is_refused(self):
        assert a_mismatched_result_is_refused()

    def test_it_serialises(self):
        assert search(random_queries(count=4), random_corpus(), k=5).as_dict()["k"] == 5


class TestBatching:
    def test_the_batched_search_returns_the_same_identifiers(self):
        assert batching_changes_nothing()["identical_identifiers"]

    def test_and_the_same_scores(self):
        assert batching_changes_nothing()["identical_scores"]

    def test_at_every_batch_size(self):
        for batch in (1, 17, 512, 100000):
            assert batching_changes_nothing(batch=batch)["overlap"] == 1.0

    def test_a_batch_larger_than_the_corpus_is_one_batch(self):
        assert batching_changes_nothing(batch=100000)["batches"] == 1

    def test_it_saves_the_score_matrix(self):
        assert batching_saves_the_matrix()["ratio"] > 1.0

    def test_by_the_number_of_batches(self):
        result = batching_saves_the_matrix()
        assert abs(result["ratio"] - result["batches"]) < 1e-6

    def test_and_costs_a_second_selection(self):
        result = batching_costs_a_second_selection()
        assert result["candidates_merged"] == result["batches"] * 10

    def test_which_is_a_fraction_of_the_corpus(self):
        assert batching_costs_a_second_selection()["share_of_the_corpus"] < 0.05

    def test_a_zero_batch_is_refused(self):
        with pytest.raises(ConfigError, match="not a batch"):
            search_batched(random_queries(count=2), random_corpus(), batch=0)

    def test_a_batch_smaller_than_k_still_works(self):
        # Each block contributes fewer than k candidates and the merge still fills the result.
        corpus, queries = (
            random_corpus(count=256, dimension=16),
            random_queries(count=4, dimension=16),
        )
        assert search_batched(queries, corpus, k=10, batch=3).k == 10

    def test_and_agrees_with_the_unbatched_answer(self):
        corpus, queries = (
            random_corpus(count=256, dimension=16),
            random_queries(count=4, dimension=16),
        )
        plain = search(queries, corpus, k=10)
        chunked = search_batched(queries, corpus, k=10, batch=3)
        assert torch.equal(plain.identifiers, chunked.identifiers)


class TestTies:
    def test_duplicates_make_the_identifier_measure_disagree(self):
        assert ties_make_identifiers_ambiguous()["overlap"] < 0.9

    def test_while_both_answers_are_exactly_correct(self):
        assert ties_make_identifiers_ambiguous()["both_are_correct"]

    def test_the_score_gap_is_zero(self):
        assert ties_make_identifiers_ambiguous()["score_gap"] == 0.0

    def test_a_k_that_lands_on_a_group_boundary_is_unambiguous(self):
        assert ties_make_identifiers_ambiguous()["overlap_on_whole_groups"] == 1.0

    def test_without_duplicates_the_two_measures_agree(self):
        assert without_ties_the_two_measures_agree()["overlap"] == 1.0

    def test_so_the_ambiguity_is_the_corpus_and_not_the_measure(self):
        result = without_ties_the_two_measures_agree()
        assert result["score_gap"] == 0.0

    def test_a_single_copy_corpus_is_refused(self):
        with pytest.raises(ConfigError, match="does not duplicate"):
            duplicated_corpus(copies=1)

    def test_a_count_that_does_not_divide_is_refused(self):
        with pytest.raises(ConfigError, match="does not divide"):
            duplicated_corpus(count=10, copies=4)

    def test_the_duplicated_corpus_really_has_duplicates(self):
        corpus = duplicated_corpus(count=64, copies=4)
        assert torch.equal(corpus[0], corpus[16])


class TestComparison:
    def test_a_result_overlaps_itself_completely(self):
        found = search(random_queries(count=8), random_corpus(), k=10)
        assert identifier_overlap(found, found) == 1.0

    def test_and_has_no_gap_against_itself(self):
        corpus, queries = random_corpus(), random_queries(count=8)
        found = search(queries, corpus, k=10)
        assert score_gap(queries, corpus, found, found) == 0.0

    def test_a_worse_result_has_a_positive_gap(self):
        corpus, queries = random_corpus(), random_queries(count=8)
        best = search(queries, corpus, k=5)
        worse = Neighbours(
            identifiers=search(queries, corpus, k=50).identifiers[:, -5:],
            scores=search(queries, corpus, k=50).scores[:, -5:],
        )
        assert score_gap(queries, corpus, best, worse) > 0.0

    def test_comparing_different_shapes_is_refused(self):
        assert comparing_different_shapes_is_refused()

    def test_comparing_different_query_counts_is_refused(self):
        corpus = random_corpus()
        with pytest.raises(DataError, match="queries against"):
            identifier_overlap(
                search(random_queries(count=4), corpus, k=5),
                search(random_queries(count=8), corpus, k=5),
            )

    def test_rescoring_the_wrong_number_of_rows_is_refused(self):
        corpus, queries = random_corpus(), random_queries(count=8)
        with pytest.raises(DataError, match="identifier rows"):
            scores_for(queries, corpus, torch.zeros(3, 5, dtype=torch.long))


class TestCost:
    def test_the_distance_count_does_not_depend_on_k(self):
        rows = k_sweep()
        assert len({row["distances_computed"] for row in rows}) == 1

    def test_so_asking_for_more_is_nearly_free(self):
        rows = {row["k"]: row for row in k_sweep()}
        assert rows[100]["share_returned"] < 0.03

    def test_the_result_size_grows_with_k(self):
        rows = k_sweep()
        assert [row["results"] for row in rows] == sorted(row["results"] for row in rows)

    def test_an_empty_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            k_sweep(values=())

    def test_a_corpus_of_nothing_is_refused(self):
        with pytest.raises(ConfigError, match="not a corpus"):
            random_corpus(count=0)

from __future__ import annotations

import pytest
import torch

from vse.errors import ConfigError, DataError
from vse.filter.predicate import (
    Predicate,
    a_correlated_condition_behaves_differently,
    a_float_predicate_is_refused,
    a_predicate_that_matches_nothing_is_refused,
    a_selectivity_above_one_is_refused,
    asking_for_more_than_the_matching_set_is_refused,
    clustered_predicate,
    compare_strategies,
    exact_filtered,
    filter_then_search,
    filter_then_search_is_exact_and_scales_the_other_way,
    filtering_inside_a_graph_strands_the_walk,
    filtering_inside_a_partitioned_index_works,
    random_predicate,
    search_then_filter,
    search_then_filter_returns_nothing,
    selectivity_sweep,
    the_cheap_strategy_collapses,
    the_cheap_strategy_is_never_cheaper_here,
    the_exact_answer_only_contains_matching_vectors,
    the_graph_fails_where_the_partitions_do_not,
)
from vse.vectors.dataset import clustered, gaussian


class TestPredicates:
    def test_a_predicate_reports_its_selectivity(self):
        predicate = random_predicate(1000, selectivity=0.1)
        assert abs(predicate.selectivity - 0.1) < 0.01

    def test_and_how_many_pass(self):
        assert random_predicate(1000, selectivity=0.25).matching == 250

    def test_the_rows_are_the_passing_indices(self):
        predicate = random_predicate(100, selectivity=0.2)
        assert bool(predicate.mask[predicate.rows()].all())

    def test_a_clustered_predicate_passes_a_region(self):
        corpus = clustered(count=512, dimension=16, clusters=8)
        predicate = clustered_predicate(corpus, selectivity=0.1)
        assert predicate.matching == 51

    def test_a_predicate_that_matches_nothing_is_refused(self):
        assert a_predicate_that_matches_nothing_is_refused()

    def test_a_float_predicate_is_refused(self):
        assert a_float_predicate_is_refused()

    def test_a_selectivity_above_one_is_refused(self):
        assert a_selectivity_above_one_is_refused()

    def test_a_zero_selectivity_is_refused(self):
        with pytest.raises(ConfigError, match="not a share"):
            random_predicate(100, selectivity=0.0)

    def test_a_rank_two_predicate_is_refused(self):
        with pytest.raises(DataError, match="vector of flags"):
            Predicate(mask=torch.ones(4, 4, dtype=torch.bool))

    def test_an_empty_corpus_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to filter"):
            random_predicate(0, selectivity=0.5)

    def test_it_serialises(self):
        assert random_predicate(1000, selectivity=0.5).as_dict()["matching"] == 500


class TestGroundTruth:
    def test_the_exact_answer_only_contains_matching_vectors(self):
        assert the_exact_answer_only_contains_matching_vectors()["all_match"]

    def test_asking_for_more_than_the_matching_set_is_refused(self):
        assert asking_for_more_than_the_matching_set_is_refused()

    def test_it_returns_the_full_result(self):
        result = the_exact_answer_only_contains_matching_vectors()
        assert result["returned"] == 32 * 10

    def test_filter_then_search_is_the_exact_answer(self):
        corpus = gaussian(count=512, dimension=16)
        predicate = random_predicate(512, selectivity=0.2)
        truth = exact_filtered(corpus.vectors[:8], corpus.vectors, predicate, k=5)
        found, _ = filter_then_search(corpus.vectors[:8], corpus.vectors, predicate, k=5)
        assert torch.equal(truth.identifiers, found.identifiers)


class TestTheCliff:
    def test_the_cheap_strategy_is_exact_at_half_selectivity(self):
        assert the_cheap_strategy_collapses()["at_half"] == 1.0

    def test_and_collapses_below_two_percent(self):
        assert the_cheap_strategy_collapses()["at_half_a_percent"] < 0.1

    def test_the_collapse_is_a_cliff(self):
        assert the_cheap_strategy_collapses()["collapsed"]

    def test_while_the_cost_does_not_move(self):
        assert the_cheap_strategy_collapses()["cost_unchanged"]

    def test_the_result_comes_back_short_rather_than_wrong(self):
        rows = {row["selectivity"]: row for row in search_then_filter_returns_nothing()}
        assert rows[0.005]["slots_filled"] < 0.5

    def test_where_at_half_selectivity_it_is_full(self):
        rows = {row["selectivity"]: row for row in search_then_filter_returns_nothing()}
        assert rows[0.5]["slots_filled"] == 1.0

    def test_an_over_fetch_below_one_is_refused(self):
        corpus = gaussian(count=256, dimension=16)
        predicate = random_predicate(256, selectivity=0.5)
        with pytest.raises(ConfigError, match="fetches less"):
            search_then_filter(corpus.vectors[:4], corpus.vectors, predicate, over_fetch=0)

    def test_an_empty_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            search_then_filter_returns_nothing(selectivities=())


class TestRestricting:
    def test_restricting_is_exact_at_every_selectivity(self):
        rows = filter_then_search_is_exact_and_scales_the_other_way()
        assert all(row["recall"] == 1.0 for row in rows)

    def test_and_gets_cheaper_as_the_condition_tightens(self):
        rows = [
            row["distances_per_query"]
            for row in filter_then_search_is_exact_and_scales_the_other_way()
        ]
        assert rows == sorted(rows, reverse=True)

    def test_its_cost_is_the_matching_set(self):
        rows = filter_then_search_is_exact_and_scales_the_other_way()
        for row in rows:
            assert abs(row["share_of_the_corpus"] - row["selectivity"]) < 0.005

    def test_there_is_no_crossover_over_a_flat_scan(self):
        assert the_cheap_strategy_is_never_cheaper_here()["restricting_is_cheaper"]

    def test_and_restricting_is_also_more_accurate(self):
        assert the_cheap_strategy_is_never_cheaper_here()["restricting_is_also_more_accurate"]

    def test_so_the_cheap_strategy_is_never_the_right_choice_here(self):
        rows = selectivity_sweep()
        assert all(row["restricted_cost"] <= row["cheap_cost"] for row in rows)

    def test_and_never_more_accurate(self):
        rows = selectivity_sweep()
        assert all(row["restricted_recall"] >= row["cheap_recall"] for row in rows)

    def test_an_empty_selectivity_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            selectivity_sweep(shares=())


class TestFilteringInside:
    def test_a_partitioned_index_filters_cleanly(self):
        rows = {row["selectivity"]: row for row in filtering_inside_a_partitioned_index_works()}
        assert rows[0.5]["recall"] == 1.0

    def test_and_holds_up_at_two_percent(self):
        rows = {row["selectivity"]: row for row in filtering_inside_a_partitioned_index_works()}
        assert rows[0.02]["recall"] > 0.5

    def test_getting_cheaper_as_the_condition_tightens(self):
        rows = [
            row["distances_per_query"] for row in filtering_inside_a_partitioned_index_works()
        ]
        assert rows == sorted(rows, reverse=True)

    def test_a_graph_index_does_not(self):
        assert the_graph_fails_where_the_partitions_do_not()["graph_collapses"]

    def test_where_the_partitioned_one_holds(self):
        assert the_graph_fails_where_the_partitions_do_not()["partitioned_holds"]

    def test_the_graph_cost_does_not_move_as_it_fails(self):
        rows = [
            row["distances_per_query"] for row in filtering_inside_a_graph_strands_the_walk()
        ]
        assert len(set(rows)) == 1

    def test_which_is_why_it_fails_silently(self):
        rows = {row["selectivity"]: row for row in filtering_inside_a_graph_strands_the_walk()}
        assert rows[0.02]["recall"] < 0.2
        assert rows[0.02]["distances_per_query"] == rows[0.5]["distances_per_query"]

    def test_an_empty_partition_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            filtering_inside_a_partitioned_index_works(selectivities=())

    def test_an_empty_graph_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            filtering_inside_a_graph_strands_the_walk(selectivities=())


class TestCorrelation:
    def test_a_correlated_condition_is_harder_not_easier(self):
        assert not a_correlated_condition_behaves_differently()["correlated_is_easier"]

    def test_by_a_factor_of_three(self):
        result = a_correlated_condition_behaves_differently()
        assert result["random"] > result["clustered"] * 2

    def test_so_the_random_measurement_is_the_optimistic_one(self):
        result = a_correlated_condition_behaves_differently()
        assert result["clustered"] < result["random"]

    def test_three_strategies_are_compared(self):
        assert len(compare_strategies()) == 3

    def test_filtering_during_the_traversal_is_cheapest(self):
        rows = compare_strategies()
        cheapest = min(rows, key=lambda row: row["distances_per_query"])
        assert cheapest["strategy"] == "filter during"

    def test_and_nearly_as_accurate_as_restricting(self):
        rows = {row["strategy"]: row for row in compare_strategies()}
        assert rows["filter during"]["recall"] > rows["filter then search"]["recall"] - 0.05

    def test_while_the_cheap_one_is_neither(self):
        rows = {row["strategy"]: row for row in compare_strategies()}
        assert rows["search then filter"]["recall"] < rows["filter then search"]["recall"]
        assert (
            rows["search then filter"]["distances_per_query"]
            > rows["filter then search"]["distances_per_query"]
        )

from __future__ import annotations

import pytest
import torch

from vse.build.kmeans import lloyd
from vse.errors import ConfigError, IndexStateError
from vse.index.spill import (
    Spilled,
    SpillIndex,
    a_clustered_corpus_needs_less_spilling,
    a_corpus_smaller_than_the_partitions_is_refused,
    a_rank_one_assignment_is_refused,
    a_spilled_assignment_reports_its_shape,
    adaptive_spilling_needs_two_partitions,
    adaptive_spilling_spends_less_for_the_same,
    against_probing_more,
    an_adaptive_share_outside_the_interval_is_refused,
    compare_the_schemes,
    more_copies_than_partitions_are_refused,
    one_copy_is_a_plain_inverted_file,
    probing_more_is_usually_the_better_spend,
    probing_more_partitions_than_exist_is_refused,
    removal_and_insertion_work,
    spill_adaptively,
    spill_uniformly,
    spilling_buys_recall,
    spilling_fixes_the_boundary_attack,
    spilling_grows_the_lists,
    the_adaptive_curve_is_better_than_the_uniform_one,
    the_full_share_matches_uniform_spilling,
    the_growth_is_the_copy_count,
    the_lists_are_deduplicated,
    the_result_is_well_formed,
    zero_copies_are_refused,
)
from vse.vectors.dataset import gaussian
from vse.vectors.metric import squared_l2


class TestAssignment:
    def test_uniform_spilling_gives_one_column_per_copy(self):
        corpus = gaussian(count=1024, dimension=8).vectors
        run = lloyd(corpus, k=16, seed=0)
        assert spill_uniformly(corpus, run.centres, 3).copies == 3

    def test_and_the_columns_are_distinct(self):
        corpus = gaussian(count=1024, dimension=8).vectors
        run = lloyd(corpus, k=16, seed=0)
        assignment = spill_uniformly(corpus, run.centres, 3).assignment
        for row in range(int(assignment.shape[0])):
            assert int(torch.unique(assignment[row]).numel()) == 3

    def test_the_first_column_is_the_nearest_centroid(self):
        corpus = gaussian(count=512, dimension=8).vectors
        run = lloyd(corpus, k=16, seed=0)
        spilled = spill_uniformly(corpus, run.centres, 2)
        nearest = squared_l2(corpus, run.centres).argmin(dim=1)
        assert bool(torch.equal(spilled.assignment[:, 0], nearest))

    def test_adaptive_spilling_duplicates_a_share(self):
        corpus = gaussian(count=1000, dimension=8).vectors
        run = lloyd(corpus, k=16, seed=0)
        spilled = spill_adaptively(corpus, run.centres, share=0.3)
        duplicated = int((spilled.assignment[:, 0] != spilled.assignment[:, 1]).sum())
        assert duplicated == 300

    def test_a_share_of_zero_duplicates_nothing(self):
        corpus = gaussian(count=512, dimension=8).vectors
        run = lloyd(corpus, k=16, seed=0)
        spilled = spill_adaptively(corpus, run.centres, share=0.0)
        assert int((spilled.assignment[:, 0] != spilled.assignment[:, 1]).sum()) == 0

    def test_a_share_of_one_duplicates_everything(self):
        corpus = gaussian(count=512, dimension=8).vectors
        run = lloyd(corpus, k=16, seed=0)
        spilled = spill_adaptively(corpus, run.centres, share=1.0)
        assert int((spilled.assignment[:, 0] != spilled.assignment[:, 1]).sum()) == 512

    def test_an_assignment_reports_its_shape(self):
        assert a_spilled_assignment_reports_its_shape()["growth_is_the_copies"]

    def test_zero_copies_are_refused(self):
        assert zero_copies_are_refused()

    def test_more_copies_than_partitions_are_refused(self):
        assert more_copies_than_partitions_are_refused()

    def test_an_adaptive_share_outside_the_interval_is_refused(self):
        assert an_adaptive_share_outside_the_interval_is_refused()

    def test_adaptive_spilling_needs_two_partitions(self):
        assert adaptive_spilling_needs_two_partitions()

    def test_a_rank_one_assignment_is_refused(self):
        assert a_rank_one_assignment_is_refused()


class TestGrowth:
    def test_the_growth_is_exactly_the_copy_count(self):
        assert the_growth_is_the_copy_count()["exact"]

    def test_from_one_to_four(self):
        result = the_growth_is_the_copy_count()
        assert result["growth_at_one"] == 1.0 and result["growth_at_four"] == 4.0

    def test_the_memory_grows_much_less_than_the_lists(self):
        assert the_growth_is_the_copy_count()["memory_ratio"] < 1.5

    def test_the_lists_are_deduplicated(self):
        assert the_lists_are_deduplicated()["no_growth"]

    def test_and_the_results_stay_distinct(self):
        assert the_lists_are_deduplicated()["results_are_distinct"]

    def test_an_empty_growth_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            spilling_grows_the_lists(copies=())


class TestRecall:
    def test_more_copies_buy_recall(self):
        rows = [row["recall"] for row in spilling_buys_recall()]
        assert rows == sorted(rows)

    def test_and_cost_more(self):
        rows = [row["distances"] for row in spilling_buys_recall()]
        assert rows == sorted(rows)

    def test_two_copies_buy_a_lot(self):
        rows = {row["copies"]: row for row in spilling_buys_recall()}
        assert rows[2]["recall"] - rows[1]["recall"] > 0.1

    def test_an_empty_recall_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            spilling_buys_recall(copies=())

    def test_one_copy_is_a_plain_inverted_file(self):
        assert one_copy_is_a_plain_inverted_file()["recall_matches"]

    def test_at_the_same_cost(self):
        assert one_copy_is_a_plain_inverted_file()["cost_matches"]


class TestAgainstProbing:
    def test_three_budgets_are_compared(self):
        assert len(against_probing_more()) == 3

    def test_every_budget_has_a_winner(self):
        rows = against_probing_more()
        assert all(row["spilled_recall"] > 0 and row["plain_recall"] > 0 for row in rows)

    def test_the_settings_are_reported(self):
        rows = against_probing_more()
        assert all("copies" in row["spilled_setting"] for row in rows)

    def test_the_comparison_is_summarised(self):
        result = probing_more_is_usually_the_better_spend()
        assert result["spilling_wins"] + result["probing_wins"] == result["budgets"]

    def test_an_empty_budget_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            against_probing_more(budgets=())


class TestTheBoundaryAttack:
    def test_spilling_raises_the_midpoint_recall(self):
        assert spilling_fixes_the_boundary_attack()["midpoint_recall_rises"]

    def test_by_a_lot(self):
        result = spilling_fixes_the_boundary_attack()
        assert result["spilled_midpoint"] - result["plain_midpoint"] > 0.1

    def test_but_the_relative_gap_widens(self):
        assert spilling_fixes_the_boundary_attack()["but_the_gap_widens"]

    def test_because_ordinary_queries_gain_more(self):
        result = spilling_fixes_the_boundary_attack()
        ordinary_gain = result["spilled_ordinary"] - result["plain_ordinary"]
        midpoint_gain = result["spilled_midpoint"] - result["plain_midpoint"]
        assert ordinary_gain > midpoint_gain


class TestAdaptive:
    def test_the_full_share_matches_uniform_spilling(self):
        assert the_full_share_matches_uniform_spilling()["recall_matches"]

    def test_on_growth_too(self):
        assert the_full_share_matches_uniform_spilling()["growth_matches"]

    def test_the_growth_rises_with_the_share(self):
        rows = [row["growth"] for row in adaptive_spilling_spends_less_for_the_same()]
        assert rows == sorted(rows)

    def test_and_so_does_the_recall(self):
        rows = [row["recall"] for row in adaptive_spilling_spends_less_for_the_same()]
        assert rows == sorted(rows)

    def test_a_third_costs_a_third_of_the_growth(self):
        result = the_adaptive_curve_is_better_than_the_uniform_one()
        assert 0.2 < result["adaptive_share_of_the_cost"] < 0.45

    def test_and_buys_some_of_the_gain(self):
        result = the_adaptive_curve_is_better_than_the_uniform_one()
        assert result["adaptive_share_of_the_gain"] > 0

    def test_an_empty_share_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            adaptive_spilling_spends_less_for_the_same(shares=())


class TestTheIndex:
    def test_the_result_is_well_formed(self):
        result = the_result_is_well_formed()
        assert result["distinct"] and result["sorted"]

    def test_it_returns_k(self):
        assert the_result_is_well_formed()["shape"] == (64, 10)

    def test_insertion_and_removal_work(self):
        result = removal_and_insertion_work()
        assert result["insert_worked"] and result["remove_worked"]

    def test_and_leave_it_searchable(self):
        assert removal_and_insertion_work()["still_searchable"]

    def test_a_corpus_smaller_than_the_partitions_is_refused(self):
        assert a_corpus_smaller_than_the_partitions_is_refused()

    def test_probing_more_partitions_than_exist_is_refused(self):
        assert probing_more_partitions_than_exist_is_refused()

    def test_searching_before_building_is_refused(self):
        with pytest.raises(IndexStateError):
            SpillIndex(8).search(torch.randn(1, 8), k=5)

    def test_a_zero_probe_is_refused(self):
        with pytest.raises(ConfigError, match="is not a search"):
            SpillIndex(8, probe=0)

    def test_a_zero_partition_index_is_refused(self):
        with pytest.raises(ConfigError, match="not a partitioning"):
            SpillIndex(8, partitions=0)

    def test_removing_a_row_that_is_not_there_is_refused(self):
        corpus = gaussian(count=512, dimension=8).vectors
        index = SpillIndex(8, partitions=8, probe=2, copies=2)
        index.build(corpus)
        with pytest.raises(ConfigError, match="is not one of"):
            index.remove([9999])

    def test_it_finds_the_query_itself(self):
        corpus = gaussian(count=1024, dimension=16).vectors
        index = SpillIndex(16, partitions=16, probe=4, copies=2)
        index.build(corpus)
        found, _ = index.search(corpus[:1], k=1)
        assert int(found.identifiers[0, 0]) == 0

    def test_the_size_counts_vectors_not_entries(self):
        corpus = gaussian(count=1024, dimension=16).vectors
        index = SpillIndex(16, partitions=16, probe=4, copies=3)
        index.build(corpus)
        assert index.size == 1024

    def test_the_lists_hold_three_times_as_much(self):
        corpus = gaussian(count=1024, dimension=16).vectors
        index = SpillIndex(16, partitions=16, probe=4, copies=3)
        index.build(corpus)
        assert sum(index.list_lengths()) == 3072


class TestCorpusShape:
    def test_a_clustered_corpus_needs_less_spilling(self):
        assert a_clustered_corpus_needs_less_spilling()["structure_needs_less"]

    def test_because_it_starts_higher(self):
        result = a_clustered_corpus_needs_less_spilling()
        assert result["clustered_one"] > result["gaussian_one"]

    def test_four_schemes_are_compared(self):
        assert len(compare_the_schemes()) == 4

    def test_the_plain_scheme_has_no_growth(self):
        rows = {row["scheme"]: row for row in compare_the_schemes()}
        assert rows["plain"]["growth"] == 1.0

    def test_and_the_adaptive_one_sits_between(self):
        rows = {row["scheme"]: row for row in compare_the_schemes()}
        assert 1.0 < rows["adaptive a third"]["growth"] < rows["uniform two"]["growth"]

    def test_an_empty_assignment_column_is_refused(self):
        with pytest.raises(ConfigError, match="at least one partition"):
            Spilled(assignment=torch.zeros(10, 0, dtype=torch.long), centres=torch.randn(4, 8))

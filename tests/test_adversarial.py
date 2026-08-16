from __future__ import annotations

import pytest
import torch

from vse.errors import ConfigError, DataError
from vse.eval.adversarial import (
    Attack,
    _setup,
    a_clustered_corpus_is_more_exposed,
    a_higher_setting_closes_the_gap,
    a_midpoint_needs_two_partitions,
    a_rank_one_attack_is_refused,
    a_zero_count_attack_is_refused,
    an_attack_reports_what_it_needs,
    an_attack_with_no_queries_is_refused,
    between_partitions,
    compare_the_attacks_on_one_index,
    each_attack_hits_its_own_structure_hardest,
    every_structure_under_every_attack,
    far_from_the_entry_point,
    from_sparse_regions,
    midpoints_between_partitions_are_harder,
    sampling_more_than_the_corpus_is_refused,
    score,
    selecting_more_than_the_pool_is_refused,
    sparse_regions_are_easier,
    the_attacks_are_deterministic,
    the_attacks_do_not_change_the_cost,
    the_average_case_is_the_baseline,
    the_constructible_attacks_find_some_of_it,
    the_exposure_is_not_the_same_for_every_structure,
    the_gap_closes_more_slowly_than_the_mean_rises,
    the_midpoints_really_are_equidistant,
    the_selected_worst_is_the_upper_bound,
    the_sparse_queries_really_are_sparse,
    the_worst_answered,
)
from vse.index.graph import GraphIndex
from vse.index.ivf import IVFIndex
from vse.vectors.dataset import gaussian


class TestConstructions:
    def test_a_sparse_attack_is_the_right_size(self):
        corpus = gaussian(count=2048, dimension=16).vectors
        assert from_sparse_regions(corpus, count=40).count == 40

    def test_and_names_what_it_needs(self):
        corpus = gaussian(count=2048, dimension=16).vectors
        assert from_sparse_regions(corpus, count=10).needs == "the corpus"

    def test_the_sparse_queries_really_are_sparse(self):
        assert the_sparse_queries_really_are_sparse()["sparser"]

    def test_by_a_measurable_margin(self):
        assert the_sparse_queries_really_are_sparse()["ratio"] > 1.1

    def test_a_midpoint_attack_is_the_right_size(self):
        corpus = gaussian(count=2048, dimension=16).vectors
        index = IVFIndex(16, partitions=32, probe=4)
        index.build(corpus)
        assert between_partitions(index, count=25).count == 25

    def test_the_midpoints_really_are_equidistant(self):
        assert the_midpoints_really_are_equidistant()["near_one"]

    def test_to_within_six_percent_at_worst(self):
        assert the_midpoints_really_are_equidistant()["worst_ratio"] < 1.1

    def test_an_entry_point_attack_is_the_right_size(self):
        corpus = gaussian(count=1024, dimension=16).vectors
        index = GraphIndex(16, degree=16, ef=32)
        index.build(corpus)
        assert far_from_the_entry_point(index, corpus, count=30).count == 30

    def test_the_worst_answered_selects_from_the_pool(self):
        corpus, probes, _ = _setup(count=1024, queries=64)
        index = IVFIndex(32, partitions=16, probe=2)
        index.build(corpus)
        assert the_worst_answered(index, corpus, probes, count=20).count == 20

    def test_and_names_the_truth_as_its_requirement(self):
        corpus, probes, _ = _setup(count=1024, queries=64)
        index = IVFIndex(32, partitions=16, probe=2)
        index.build(corpus)
        attack = the_worst_answered(index, corpus, probes, count=20)
        assert attack.needs == "the ground truth"

    def test_the_attacks_are_deterministic(self):
        result = the_attacks_are_deterministic()
        assert result["sparse_identical"] and result["midpoint_identical"]

    def test_and_the_seed_changes_them(self):
        assert the_attacks_are_deterministic()["seeds_differ"]

    def test_four_distinct_requirements(self):
        assert an_attack_reports_what_it_needs()["four_distinct_requirements"]

    def test_all_the_same_size(self):
        assert an_attack_reports_what_it_needs()["all_the_same_size"]


class TestWhatWorks:
    def test_the_baseline_is_reported_first(self):
        assert 0.0 < the_average_case_is_the_baseline()["recall"] < 1.0

    def test_sparse_regions_are_easier_not_harder(self):
        assert sparse_regions_are_easier()["easier_not_harder"]

    def test_by_about_eight_points(self):
        assert sparse_regions_are_easier()["gap"] < -0.05

    def test_at_a_similar_cost(self):
        assert sparse_regions_are_easier()["distances_are_similar"]

    def test_midpoints_are_harder(self):
        assert midpoints_between_partitions_are_harder()["harder"]

    def test_by_about_nine_points(self):
        assert midpoints_between_partitions_are_harder()["gap"] > 0.05

    def test_the_selected_worst_is_much_worse(self):
        assert the_selected_worst_is_the_upper_bound()["gap"] > 0.2

    def test_the_midpoint_attack_finds_a_quarter_of_it(self):
        assert 0.15 < the_constructible_attacks_find_some_of_it()["midpoint_share"] < 0.5

    def test_the_sparse_attack_finds_none_of_it(self):
        assert the_constructible_attacks_find_some_of_it()["sparse_share"] < 0

    def test_no_constructible_attack_reaches_the_ceiling(self):
        assert the_constructible_attacks_find_some_of_it()["constructible_reaches_less"]


class TestAcrossStructures:
    def test_twenty_rows_are_measured(self):
        assert len(every_structure_under_every_attack()) == 20

    def test_five_structures_appear(self):
        rows = every_structure_under_every_attack()
        assert len({row["index"] for row in rows}) == 5

    def test_and_four_query_sets(self):
        rows = every_structure_under_every_attack()
        assert len({row["attack"] for row in rows}) == 4

    def test_the_exposure_differs_between_structures(self):
        assert the_exposure_is_not_the_same_for_every_structure()["they_differ"]

    def test_by_a_wide_spread(self):
        assert the_exposure_is_not_the_same_for_every_structure()["spread"] > 0.2

    def test_the_midpoint_attack_is_targeted(self):
        assert each_attack_hits_its_own_structure_hardest()["midpoint_is_targeted"]

    def test_and_so_is_the_entry_point_attack(self):
        assert each_attack_hits_its_own_structure_hardest()["entry_is_targeted"]

    def test_the_attacks_do_not_change_the_cost(self):
        assert the_attacks_do_not_change_the_cost()["cost_is_stable"]

    def test_by_more_than_a_tenth(self):
        assert the_attacks_do_not_change_the_cost()["widest_ratio"] < 1.2


class TestDefences:
    def test_the_gap_closes_at_full_probe(self):
        assert the_gap_closes_more_slowly_than_the_mean_rises()["closes_completely"]

    def test_and_the_ordinary_recall_reaches_one(self):
        result = the_gap_closes_more_slowly_than_the_mean_rises()
        assert result["ordinary_at_sixty_four"] == 1.0

    def test_the_ordinary_recall_rises_with_the_probe(self):
        rows = [row["ordinary_recall"] for row in a_higher_setting_closes_the_gap()]
        assert rows == sorted(rows)

    def test_and_so_does_the_adversarial_one(self):
        rows = [row["adversarial_recall"] for row in a_higher_setting_closes_the_gap()]
        assert rows == sorted(rows)

    def test_an_empty_probe_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            a_higher_setting_closes_the_gap(probes=())

    def test_a_clustered_corpus_is_answered_exactly(self):
        assert a_clustered_corpus_is_more_exposed()["clustered_ordinary"] == 1.0

    def test_and_has_no_sparse_gap(self):
        assert a_clustered_corpus_is_more_exposed()["clustered_gap"] == 0.0


class TestGuards:
    def test_an_attack_with_no_queries_is_refused(self):
        assert an_attack_with_no_queries_is_refused()

    def test_a_rank_one_attack_is_refused(self):
        assert a_rank_one_attack_is_refused()

    def test_a_zero_count_attack_is_refused(self):
        assert a_zero_count_attack_is_refused()

    def test_sampling_more_than_the_corpus_is_refused(self):
        assert sampling_more_than_the_corpus_is_refused()

    def test_a_midpoint_needs_two_partitions(self):
        assert a_midpoint_needs_two_partitions()

    def test_selecting_more_than_the_pool_is_refused(self):
        assert selecting_more_than_the_pool_is_refused()

    def test_a_zero_count_midpoint_attack_is_refused(self):
        corpus = gaussian(count=512, dimension=8).vectors
        index = IVFIndex(8, partitions=8, probe=2)
        index.build(corpus)
        with pytest.raises(ConfigError, match="is not an attack"):
            between_partitions(index, count=0)

    def test_a_zero_count_entry_point_attack_is_refused(self):
        corpus = gaussian(count=512, dimension=8).vectors
        index = GraphIndex(8, degree=8, ef=16)
        index.build(corpus)
        with pytest.raises(ConfigError, match="is not an attack"):
            far_from_the_entry_point(index, corpus, count=0)

    def test_a_rank_three_attack_is_refused(self):
        with pytest.raises(DataError, match="queries are a matrix"):
            Attack(torch.randn(2, 3, 4), "cube", "nothing")


class TestSummary:
    def test_five_attacks_are_compared(self):
        assert len(compare_the_attacks_on_one_index()) == 5

    def test_each_says_what_it_needs(self):
        rows = compare_the_attacks_on_one_index()
        assert all(row["needs"] for row in rows)

    def test_the_ordinary_row_comes_first(self):
        assert compare_the_attacks_on_one_index()[0]["attack"] == "ordinary"

    def test_and_the_selected_worst_comes_last(self):
        assert compare_the_attacks_on_one_index()[-1]["attack"] == "the worst answered"

    def test_the_worst_answered_scores_lowest(self):
        rows = compare_the_attacks_on_one_index()
        assert rows[-1]["recall"] == min(row["recall"] for row in rows)

    def test_scoring_an_attack_reports_its_name(self):
        corpus, probes, _ = _setup(count=1024, queries=32)
        index = IVFIndex(32, partitions=16, probe=4)
        index.build(corpus)
        row = score(index, corpus, Attack(probes, "ordinary", "nothing"))
        assert row["attack"] == "ordinary" and row["queries"] == 32

    def test_and_both_numbers(self):
        corpus, probes, _ = _setup(count=1024, queries=32)
        index = IVFIndex(32, partitions=16, probe=4)
        index.build(corpus)
        row = score(index, corpus, Attack(probes, "ordinary", "nothing"))
        assert "recall" in row and "distances" in row

    def test_an_attack_serialises(self):
        corpus = gaussian(count=512, dimension=8).vectors
        assert from_sparse_regions(corpus, count=10).as_dict()["queries"] == 10

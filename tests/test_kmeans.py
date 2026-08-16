from __future__ import annotations

import pytest
import torch

from vse.build.kmeans import (
    Clustering,
    a_clustering_with_no_centres_is_refused,
    a_partition_that_does_not_exist_is_refused,
    a_smart_start_buys_nothing_on_unstructured_data,
    and_something_on_data_with_regions,
    assign,
    balance_swings_where_the_objective_does_not,
    compare_starts,
    empty_partitions_never_happened_here,
    inertia_of,
    lloyd,
    minibatch,
    minibatch_is_cheaper_and_worse,
    more_centres_than_vectors_is_refused,
    more_partitions_keep_fewer_neighbours,
    neighbours_in_the_same_partition,
    plus_plus_init,
    random_init,
    the_imbalance_is_a_cost_tail,
    the_local_optimum_depends_on_the_seed,
    the_objective_never_increases,
    the_objective_only_ranks_the_top_correctly,
    the_repair_lowers_the_objective,
    the_structured_corpus_is_the_unbalanced_one,
    the_trade_is_visible_before_any_index_exists,
)
from vse.errors import BuildError, ConfigError, DataError
from vse.vectors.dataset import clustered, gaussian


class TestConvergence:
    def test_the_objective_never_increases(self):
        assert the_objective_never_increases()["monotone"]

    def test_and_it_actually_falls(self):
        assert the_objective_never_increases()["total_drop"] > 0

    def test_it_takes_more_than_one_round(self):
        # An infinite starting objective makes the relative tolerance infinite, so the first
        # round always looks converged. It does not, and this is what catches that.
        run = lloyd(gaussian(count=1024, dimension=16).vectors, k=16)
        assert run.iterations > 1

    def test_a_tighter_tolerance_runs_longer(self):
        vectors = gaussian(count=1024, dimension=16).vectors
        loose = lloyd(vectors, k=16, tolerance=1e-2)
        tight = lloyd(vectors, k=16, tolerance=1e-6)
        assert tight.iterations >= loose.iterations

    def test_the_round_limit_is_respected(self):
        run = lloyd(gaussian(count=1024, dimension=16).vectors, k=16, rounds=3, tolerance=1e-12)
        assert run.iterations <= 3

    def test_a_zero_round_run_is_refused(self):
        with pytest.raises(ConfigError, match="not a run"):
            lloyd(gaussian(count=64, dimension=4).vectors, rounds=0)

    def test_a_zero_tolerance_is_refused(self):
        with pytest.raises(ConfigError, match="never stops"):
            lloyd(gaussian(count=64, dimension=4).vectors, tolerance=0.0)

    def test_a_rank_three_input_is_refused(self):
        with pytest.raises(DataError, match="matrix of rows"):
            lloyd(torch.randn(4, 4, 4))


class TestInitialisation:
    def test_both_starts_pick_actual_data_points(self):
        vectors = gaussian(count=256, dimension=8).vectors
        for centres in (random_init(vectors, 8), plus_plus_init(vectors, 8)):
            assert all(bool((vectors == centre).all(dim=1).any()) for centre in centres)

    def test_the_spread_start_buys_nothing_without_regions(self):
        assert a_smart_start_buys_nothing_on_unstructured_data()["relative_gap"] < 0.001

    def test_it_does_not_even_win(self):
        assert not a_smart_start_buys_nothing_on_unstructured_data()["smart_wins"]

    def test_though_it_converges_in_fewer_rounds(self):
        assert a_smart_start_buys_nothing_on_unstructured_data()["fewer_rounds"]

    def test_and_it_wins_by_a_lot_with_regions(self):
        assert and_something_on_data_with_regions()["smart_wins"]

    def test_by_most_of_the_objective(self):
        assert and_something_on_data_with_regions()["relative_gap"] > 0.5

    def test_and_its_worst_run_beats_the_other_start_entirely(self):
        result = and_something_on_data_with_regions()
        assert result["smart_worst"] < result["random_worst"]

    def test_more_centres_than_vectors_is_refused(self):
        assert more_centres_than_vectors_is_refused()

    def test_a_zero_centre_start_is_refused(self):
        with pytest.raises(ConfigError, match="not a partition"):
            random_init(torch.randn(16, 4), k=0)

    def test_an_empty_seed_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            a_smart_start_buys_nothing_on_unstructured_data(seeds=())


class TestLocalOptima:
    def test_the_objective_barely_moves_between_seeds(self):
        assert the_local_optimum_depends_on_the_seed()["spread"] < 0.02

    def test_while_the_assignments_agree_about_nothing(self):
        assert the_local_optimum_depends_on_the_seed()["assignment_agreement"] < 0.1

    def test_a_single_run_is_not_a_spread(self):
        with pytest.raises(ConfigError, match="at least two runs"):
            the_local_optimum_depends_on_the_seed(seeds=(0,))

    def test_the_balance_swings_where_the_objective_does_not(self):
        assert balance_swings_where_the_objective_does_not()["balance_ratio"] > 20.0

    def test_and_the_objective_looks_perfectly_stable(self):
        assert balance_swings_where_the_objective_does_not()["objective_is_stable"]

    def test_a_single_balance_run_is_refused(self):
        with pytest.raises(ConfigError, match="at least two runs"):
            balance_swings_where_the_objective_does_not(seeds=(0,))


class TestBalance:
    def test_the_structured_corpus_is_the_unbalanced_one(self):
        assert the_structured_corpus_is_the_unbalanced_one()["structured_is_worse"]

    def test_the_unstructured_one_comes_out_nearly_even(self):
        assert the_structured_corpus_is_the_unbalanced_one()["gaussian_ratio"] < 2.5

    def test_where_the_clustered_one_is_several_times_off(self):
        assert the_structured_corpus_is_the_unbalanced_one()["clustered_ratio"] > 5.0

    def test_the_largest_partition_is_twice_the_mean(self):
        assert the_imbalance_is_a_cost_tail()["tail_ratio"] > 1.5

    def test_where_an_even_split_would_be_the_mean(self):
        result = the_imbalance_is_a_cost_tail()
        assert result["even_would_be"] == int(result["mean_partition"])

    def test_the_sizes_add_up_to_the_corpus(self):
        run = lloyd(gaussian(count=1024, dimension=8).vectors, k=16)
        assert int(run.sizes.sum()) == 1024

    def test_a_balance_of_an_empty_clustering_is_refused(self):
        empty = Clustering(
            centres=torch.randn(4, 2), assignment=torch.zeros(0, dtype=torch.long)
        )
        with pytest.raises(BuildError, match="every partition is empty"):
            empty.as_dict()


class TestEmptyPartitions:
    def test_the_repair_never_fires_on_an_ordinary_run(self):
        assert empty_partitions_never_happened_here()["never_fired"]

    def test_at_any_partition_count_tried(self):
        assert empty_partitions_never_happened_here()["largest_k"] == 512

    def test_across_both_starts(self):
        assert empty_partitions_never_happened_here()["runs"] == 10

    def test_but_it_works_when_it_is_needed(self):
        assert the_repair_lowers_the_objective()["fixed"] == 1

    def test_and_it_lowers_the_objective(self):
        assert the_repair_lowers_the_objective()["improved"]

    def test_an_empty_partition_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            empty_partitions_never_happened_here(counts=())

    def test_a_clustering_with_no_centres_is_refused(self):
        assert a_clustering_with_no_centres_is_refused()

    def test_a_partition_that_does_not_exist_is_refused(self):
        assert a_partition_that_does_not_exist_is_refused()


class TestMinibatch:
    def test_it_lands_worse_than_the_full_algorithm(self):
        assert minibatch_is_cheaper_and_worse()["quality_ratio"] > 1.0

    def test_by_a_few_percent(self):
        assert minibatch_is_cheaper_and_worse()["quality_ratio"] < 1.1

    def test_and_touches_a_third_of_the_vectors(self):
        assert minibatch_is_cheaper_and_worse()["visit_ratio"] < 0.5

    def test_it_produces_a_usable_clustering(self):
        run = minibatch(gaussian(count=1024, dimension=8).vectors, k=8, rounds=50, batch=64)
        assert run.k == 8
        assert int(run.sizes.sum()) == 1024

    def test_a_batch_larger_than_the_corpus_is_refused(self):
        with pytest.raises(ConfigError, match="a batch of"):
            minibatch(gaussian(count=64, dimension=4).vectors, batch=256)

    def test_a_zero_round_run_is_refused(self):
        with pytest.raises(ConfigError, match="not a run"):
            minibatch(gaussian(count=64, dimension=4).vectors, rounds=0, batch=8)


class TestWhatTheIndexNeeds:
    def test_the_top_of_the_ranking_agrees(self):
        assert the_objective_only_ranks_the_top_correctly()["the_top_agrees"]

    def test_but_a_quarter_of_the_pairs_are_inverted(self):
        assert the_objective_only_ranks_the_top_correctly()["inverted_share"] > 0.15

    def test_and_not_all_of_them(self):
        assert the_objective_only_ranks_the_top_correctly()["inverted_share"] < 0.5

    def test_too_few_runs_to_rank_is_refused(self):
        with pytest.raises(ConfigError, match="at least three runs"):
            the_objective_only_ranks_the_top_correctly(seeds=(0, 1))

    def test_more_partitions_keep_fewer_neighbours(self):
        rows = [row["kept"] for row in more_partitions_keep_fewer_neighbours()]
        assert rows == sorted(rows, reverse=True)

    def test_while_scanning_less(self):
        rows = [row["mean_partition"] for row in more_partitions_keep_fewer_neighbours()]
        assert rows == sorted(rows, reverse=True)

    def test_the_trade_is_visible_before_any_index_exists(self):
        assert the_trade_is_visible_before_any_index_exists()["fell"]

    def test_four_partitions_keep_more_than_half(self):
        assert the_trade_is_visible_before_any_index_exists()["kept_at_four"] > 0.5

    def test_an_empty_partition_count_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            more_partitions_keep_fewer_neighbours(counts=())

    def test_a_perfect_clustering_keeps_everything(self):
        corpus = clustered(count=512, dimension=8, clusters=4, spread=0.02)
        run = lloyd(corpus.vectors, k=4)
        assert neighbours_in_the_same_partition(corpus, run, corpus.vectors[:32]) > 0.95


class TestMechanics:
    def test_assignment_picks_the_nearest_centre(self):
        vectors = gaussian(count=128, dimension=4).vectors
        centres = random_init(vectors, 8)
        chosen = assign(vectors, centres)
        assert torch.equal(chosen, assign(vectors, centres, batch=7))

    def test_a_width_mismatch_is_refused(self):
        with pytest.raises(DataError, match="wide"):
            assign(torch.randn(8, 4), torch.randn(4, 8))

    def test_a_zero_batch_is_refused(self):
        with pytest.raises(ConfigError, match="not a batch"):
            assign(torch.randn(8, 4), torch.randn(4, 4), batch=0)

    def test_the_inertia_of_a_perfect_fit_is_zero(self):
        vectors = torch.randn(8, 4)
        assert inertia_of(vectors, vectors, torch.arange(8)) < 1e-9

    def test_a_clustering_serialises(self):
        assert lloyd(gaussian(count=256, dimension=8).vectors, k=4).as_dict()["k"] == 4

    def test_members_of_a_partition_all_belong_to_it(self):
        run = lloyd(gaussian(count=256, dimension=8).vectors, k=4)
        assert bool((run.assignment[run.members(2)] == 2).all())

    def test_both_starts_appear_in_the_comparison(self):
        assert len({row["start"] for row in compare_starts()}) == 2

    def test_an_empty_start_comparison_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            compare_starts(seeds=())

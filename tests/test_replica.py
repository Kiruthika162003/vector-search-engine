from __future__ import annotations

import pytest
import torch

from vse.errors import ConfigError, DataError
from vse.index.ivf import IVFIndex
from vse.serve.replica import (
    Fleet,
    Replica,
    _setup,
    a_fleet_of_nothing_is_refused,
    a_fleet_reports_how_it_was_built,
    a_lag_of_everything_is_refused,
    a_query_pinned_to_one_replica_is_consistent,
    a_replica_reports_its_own_state,
    a_shared_seed_is_the_real_fix,
    a_stale_replica_answers_from_the_past,
    a_stale_replica_is_not_a_broken_one,
    a_zero_replica_build_is_refused,
    agreement,
    build_fleet,
    compare_the_designs,
    comparing_answers_of_different_widths_is_refused,
    comparing_one_answer_is_refused,
    copied_replicas_agree_exactly,
    independently_built_replicas_disagree,
    merge,
    merging_beats_probing_more_at_the_same_cost,
    merging_nothing_is_refused,
    merging_one_replica_is_that_replica,
    merging_replicas_beats_any_one,
    more_replicas_disagree_more,
    order_matters_more_than_membership,
    set_agreement,
    the_disagreement_is_not_a_recall_loss,
    the_lag_costs_about_the_write_share,
    the_merge_is_well_formed,
    unanimity_falls_faster_than_pairwise_agreement,
)
from vse.vectors.dataset import gaussian
from vse.vectors.exact import Neighbours


class TestTheControl:
    def test_copied_replicas_agree_exactly(self):
        assert copied_replicas_agree_exactly()["exact"]

    def test_on_slots_and_on_sets(self):
        result = copied_replicas_agree_exactly()
        assert result["slot_agreement"] == 1.0 and result["set_agreement"] == 1.0

    def test_the_fleet_knows_it_was_built_identically(self):
        assert copied_replicas_agree_exactly()["identically_built"]

    def test_a_shared_seed_costs_no_recall(self):
        assert a_shared_seed_is_the_real_fix()["and_costs_no_recall"]

    def test_and_gives_consistency_for_free(self):
        assert a_shared_seed_is_the_real_fix()["consistency_is_free"]


class TestDisagreement:
    def test_independent_replicas_disagree(self):
        assert independently_built_replicas_disagree()["slot_agreement"] < 0.5

    def test_while_their_recalls_stay_together(self):
        assert the_disagreement_is_not_a_recall_loss()["recalls_stay_together"]

    def test_the_agreement_falls_from_one(self):
        assert the_disagreement_is_not_a_recall_loss()["agreement_falls"]

    def test_order_is_part_of_the_disagreement(self):
        assert order_matters_more_than_membership()["order_is_part_of_it"]

    def test_and_a_substantial_part(self):
        assert order_matters_more_than_membership()["gap"] > 0.2

    def test_pairwise_agreement_is_flat_across_fleet_sizes(self):
        assert unanimity_falls_faster_than_pairwise_agreement()["pairwise_is_flat"]

    def test_and_nothing_is_ever_unanimous(self):
        assert unanimity_falls_faster_than_pairwise_agreement()["nothing_is_ever_unanimous"]

    def test_four_fleet_sizes_are_measured(self):
        assert len(more_replicas_disagree_more()) == 4

    def test_an_empty_fleet_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            more_replicas_disagree_more(counts=())


class TestMerging:
    def test_merging_beats_any_one_replica(self):
        rows = {row["replicas"]: row for row in merging_replicas_beats_any_one()}
        assert rows[3]["merged_recall"] > rows[3]["single_recall"]

    def test_and_the_gain_grows_with_the_fleet(self):
        rows = [row["merged_recall"] for row in merging_replicas_beats_any_one()]
        assert rows == sorted(rows)

    def test_by_a_lot(self):
        rows = {row["replicas"]: row for row in merging_replicas_beats_any_one()}
        assert rows[5]["merged_recall"] > rows[1]["merged_recall"] + 0.3

    def test_merging_one_replica_changes_nothing(self):
        rows = {row["replicas"]: row for row in merging_replicas_beats_any_one()}
        assert rows[1]["merged_recall"] == rows[1]["single_recall"]

    def test_and_returns_it_unchanged(self):
        assert merging_one_replica_is_that_replica()["identical"]

    def test_with_the_same_scores(self):
        assert merging_one_replica_is_that_replica()["scores_identical"]

    def test_merging_edges_out_doubling_the_probe(self):
        assert merging_beats_probing_more_at_the_same_cost()["merging_wins"]

    def test_at_a_comparable_cost(self):
        result = merging_beats_probing_more_at_the_same_cost()
        assert abs(result["merged_distances"] - result["doubled_probe_distances"]) < 200

    def test_the_merge_is_well_formed(self):
        result = the_merge_is_well_formed()
        assert result["distinct"] and result["sorted"]

    def test_and_returns_k(self):
        assert the_merge_is_well_formed()["shape"] == (200, 10)

    def test_an_empty_merge_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            merging_replicas_beats_any_one(counts=())

    def test_merging_nothing_is_refused(self):
        assert merging_nothing_is_refused()


class TestStaleness:
    def test_a_stale_replica_is_right_about_what_it_has(self):
        assert a_stale_replica_answers_from_the_past()["it_is_right_about_what_it_has"]

    def test_and_holds_fewer_vectors(self):
        result = a_stale_replica_answers_from_the_past()
        assert result["stale_size"] < result["current_size"]

    def test_the_loss_is_about_the_lag(self):
        assert a_stale_replica_is_not_a_broken_one()["loss_is_about_the_lag"]

    def test_and_it_scores_better_against_its_own_corpus(self):
        assert a_stale_replica_is_not_a_broken_one()["better_against_its_own"]

    def test_the_loss_grows_across_the_wider_lags(self):
        rows = {row["lag"]: row for row in the_lag_costs_about_the_write_share()}
        assert rows[0.2]["loss"] > rows[0.1]["loss"] > rows[0.02]["loss"]

    def test_a_small_lag_costs_nothing_measurable(self):
        rows = {row["lag"]: row for row in the_lag_costs_about_the_write_share()}
        assert abs(rows[0.02]["loss"]) < 0.02

    def test_an_empty_lag_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            the_lag_costs_about_the_write_share(shares=())

    def test_a_lag_of_everything_is_refused(self):
        assert a_lag_of_everything_is_refused()

    def test_a_lag_of_nothing_is_refused(self):
        with pytest.raises(ConfigError, match="is not a lag"):
            a_stale_replica_answers_from_the_past(share=0.0)


class TestFixes:
    def test_pinning_a_query_is_exact(self):
        assert a_query_pinned_to_one_replica_is_consistent()["pinning_is_exact"]

    def test_where_the_fleet_is_not(self):
        assert a_query_pinned_to_one_replica_is_consistent()["and_the_fleet_is_not"]

    def test_three_designs_are_compared(self):
        assert len(compare_the_designs()) == 3

    def test_copying_and_merging_both_agree(self):
        rows = {row["design"]: row for row in compare_the_designs()}
        assert rows["copied"]["agreement"] == rows["merged"]["agreement"] == 1.0

    def test_merging_has_the_best_recall(self):
        rows = {row["design"]: row for row in compare_the_designs()}
        assert rows["merged"]["recall"] > rows["copied"]["recall"]

    def test_and_costs_the_most(self):
        rows = {row["design"]: row for row in compare_the_designs()}
        assert rows["merged"]["cost_multiple"] == 3


class TestMechanics:
    def test_a_fleet_reports_how_it_was_built(self):
        assert a_fleet_reports_how_it_was_built()["the_flag_distinguishes_them"]

    def test_a_replica_reports_its_size(self):
        assert a_replica_reports_its_own_state()["sizes_agree"]

    def test_and_its_seed(self):
        assert a_replica_reports_its_own_state()["seeds_differ"]

    def test_a_fleet_of_nothing_is_refused(self):
        assert a_fleet_of_nothing_is_refused()

    def test_a_zero_replica_build_is_refused(self):
        assert a_zero_replica_build_is_refused()

    def test_comparing_one_answer_is_refused(self):
        assert comparing_one_answer_is_refused()

    def test_comparing_different_widths_is_refused(self):
        assert comparing_answers_of_different_widths_is_refused()

    def test_set_agreement_also_needs_two(self):
        left = Neighbours(torch.zeros(4, 5, dtype=torch.long), torch.zeros(4, 5))
        with pytest.raises(ConfigError, match="at least two answers"):
            set_agreement([left])

    def test_two_identical_answers_agree_completely(self):
        answer = Neighbours(torch.arange(20).reshape(4, 5), torch.zeros(4, 5))
        assert agreement([answer, answer]) == 1.0

    def test_two_disjoint_answers_agree_on_nothing(self):
        left = Neighbours(torch.arange(20).reshape(4, 5), torch.zeros(4, 5))
        right = Neighbours(torch.arange(100, 120).reshape(4, 5), torch.zeros(4, 5))
        assert agreement([left, right]) == 0.0

    def test_a_fleet_of_one_is_allowed(self):
        corpus, _, _ = _setup(count=512, queries=16)
        assert build_fleet(corpus, count=1, partitions=16, probe=4).size == 1

    def test_a_replica_serialises(self):
        corpus = gaussian(count=256, dimension=8)
        index = IVFIndex(8, partitions=8, probe=2)
        index.build(corpus.vectors)
        assert Replica(index=index, seed=3, label="one").as_dict()["seed"] == 3

    def test_a_fleet_serialises(self):
        corpus, _, _ = _setup(count=512, queries=16)
        fleet = build_fleet(corpus, count=2, partitions=16, probe=4)
        assert fleet.as_dict()["replicas"] == 2

    def test_a_merge_deduplicates(self):
        answer = Neighbours(
            torch.arange(10).reshape(1, 10), torch.arange(10).float().reshape(1, 10)
        )
        merged = merge([answer, answer], 10)
        assert int(torch.unique(merged.identifiers[0]).numel()) == 10

    def test_and_takes_the_better_score(self):
        left = Neighbours(torch.zeros(1, 2, dtype=torch.long), torch.tensor([[5.0, 6.0]]))
        right = Neighbours(torch.zeros(1, 2, dtype=torch.long), torch.tensor([[1.0, 2.0]]))
        merged = merge([left, right], 1)
        assert float(merged.scores[0, 0]) == 1.0

    def test_an_empty_replica_list_is_refused(self):
        with pytest.raises(ConfigError, match="at least one replica"):
            Fleet(replicas=[])

    def test_mismatched_shapes_name_both(self):
        left = Neighbours(torch.zeros(4, 10, dtype=torch.long), torch.zeros(4, 10))
        right = Neighbours(torch.zeros(4, 5, dtype=torch.long), torch.zeros(4, 5))
        with pytest.raises(DataError, match="cannot be compared"):
            agreement([left, right])

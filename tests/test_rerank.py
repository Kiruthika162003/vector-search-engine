from __future__ import annotations

import pytest
import torch

from vse.errors import ConfigError, DataError
from vse.serve.rerank import (
    SCORERS,
    Shortlist,
    Staged,
    _setup,
    a_deeper_list_helps_less_than_a_wider_probe,
    a_mismatched_query_count_is_refused,
    a_rank_outside_the_dimension_is_refused,
    a_shortlist_longer_than_the_corpus_is_refused,
    a_shortlist_shorter_than_the_answer_is_refused,
    compare_the_first_stages,
    exact_shortlist,
    hamming_scores,
    how_deep_the_shortlist_must_be,
    partition_shortlist,
    projected_shortlist,
    projection,
    rerank,
    reranking_a_coverage_error_is_also_an_identity,
    reranking_a_scoring_error_recovers_most_of_the_loss,
    reranking_an_exact_first_stage_is_an_identity,
    reranking_cannot_fix_a_shortlist_that_missed,
    shortlist_recall,
    sign_codes,
    sign_shortlist,
    staged_search,
    summarise,
    the_cheapest_projection_rank_is_in_the_middle,
    the_corpus_changes_the_depth,
    the_depth_needed_grows_as_the_first_stage_gets_cruder,
    the_gain_from_reranking_is_the_scoring_error,
    the_shortlist_is_a_ceiling,
    where_the_rerank_overtakes_the_first_stage,
)
from vse.vectors.exact import Neighbours


class TestTheControls:
    def test_an_exact_first_stage_survives_reranking(self):
        assert reranking_an_exact_first_stage_is_an_identity()["identical"]

    def test_with_the_same_scores(self):
        assert reranking_an_exact_first_stage_is_an_identity()["scores_identical"]

    def test_and_it_was_perfect_to_begin_with(self):
        assert reranking_an_exact_first_stage_is_an_identity()["is_perfect"]

    def test_with_no_headroom(self):
        assert reranking_an_exact_first_stage_is_an_identity()["headroom"] == 0.0

    def test_a_coverage_error_survives_reranking_too(self):
        assert reranking_a_coverage_error_is_also_an_identity()["reranking_changed_nothing"]

    def test_and_stays_under_its_ceiling(self):
        assert reranking_a_coverage_error_is_also_an_identity()["and_the_ceiling_is_the_limit"]

    def test_the_partition_recall_is_what_the_ivf_reaches(self):
        result = reranking_a_coverage_error_is_also_an_identity()
        assert 0.3 < result["before_reranking"] < 0.45


class TestTheCeiling:
    def test_the_final_recall_never_exceeds_the_shortlist(self):
        assert all(row["under_the_ceiling"] for row in the_shortlist_is_a_ceiling())

    def test_and_always_meets_it(self):
        assert all(row["meets_the_ceiling"] for row in the_shortlist_is_a_ceiling())

    def test_the_ceiling_rises_with_the_depth(self):
        ceilings = [row["ceiling"] for row in the_shortlist_is_a_ceiling()]
        assert ceilings == sorted(ceilings)

    def test_six_depths_are_measured(self):
        assert len(the_shortlist_is_a_ceiling()) == 6

    def test_an_empty_depth_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            the_shortlist_is_a_ceiling(depths=())

    def test_a_shortlist_as_wide_as_the_answer_gains_nothing(self):
        assert reranking_cannot_fix_a_shortlist_that_missed()["nothing_changed"]

    def test_and_the_ceiling_is_already_the_answer(self):
        assert reranking_cannot_fix_a_shortlist_that_missed()["the_ceiling_is_the_answer"]


class TestScoringErrors:
    def test_reranking_a_sign_code_helps(self):
        assert reranking_a_scoring_error_recovers_most_of_the_loss()["reranking_helps"]

    def test_and_recovers_the_whole_of_the_ordering_loss(self):
        result = reranking_a_scoring_error_recovers_most_of_the_loss()
        assert result["and_reaches_the_ceiling_exactly"]

    def test_the_codes_alone_are_poor(self):
        assert reranking_a_scoring_error_recovers_most_of_the_loss()["first_stage_alone"] < 0.2

    def test_the_gain_separates_the_two_error_kinds(self):
        assert the_gain_from_reranking_is_the_scoring_error()["the_split_is_clean"]

    def test_exact_scorers_gain_nothing_at_all(self):
        assert the_gain_from_reranking_is_the_scoring_error()["exact_scorers_gain_nothing"]

    def test_approximate_scorers_gain_a_lot(self):
        assert the_gain_from_reranking_is_the_scoring_error()["approximate_scorers_gain_a_lot"]

    def test_the_sign_stage_gains_more_than_the_projection(self):
        gains = the_gain_from_reranking_is_the_scoring_error()["gains"]
        assert gains["sign"] > gains["projected"]


class TestDepth:
    def test_a_target_needs_a_depth(self):
        assert how_deep_the_shortlist_must_be()["cheapest_depth"] == 1600

    def test_and_it_beats_a_full_scan(self):
        result = how_deep_the_shortlist_must_be()
        assert result["cheapest_cost"] < result["full_scan_cost"]

    def test_but_not_by_much(self):
        result = how_deep_the_shortlist_must_be()
        assert result["full_scan_cost"] / result["cheapest_cost"] < 3.0

    def test_the_recall_rises_with_the_depth(self):
        recalls = [row["recall"] for row in how_deep_the_shortlist_must_be()["rows"]]
        assert recalls == sorted(recalls)

    def test_an_unreachable_target_is_reported_rather_than_raised(self):
        result = how_deep_the_shortlist_must_be(target=0.999, depths=(10, 20))
        assert not result["a_target_is_reachable"] and result["cheapest_depth"] is None

    def test_a_target_outside_the_unit_interval_is_refused(self):
        with pytest.raises(ConfigError, match="not a recall target"):
            how_deep_the_shortlist_must_be(target=1.5)

    def test_a_target_of_nothing_is_refused(self):
        with pytest.raises(ConfigError, match="not a recall target"):
            how_deep_the_shortlist_must_be(target=0.0)

    def test_an_empty_depth_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            how_deep_the_shortlist_must_be(depths=())


class TestTheRankTrade:
    def test_a_cruder_stage_needs_a_deeper_list(self):
        rows = the_depth_needed_grows_as_the_first_stage_gets_cruder()
        depths = [row["depth_needed"] for row in rows]
        assert depths == sorted(depths, reverse=True)

    def test_every_rank_reaches_the_target_somewhere(self):
        rows = the_depth_needed_grows_as_the_first_stage_gets_cruder()
        assert all(row["depth_needed"] is not None for row in rows)

    def test_the_cheapest_rank_is_neither_end(self):
        assert the_cheapest_projection_rank_is_in_the_middle()["the_optimum_is_interior"]

    def test_the_cheapest_rank_is_half_the_dimension(self):
        assert the_cheapest_projection_rank_is_in_the_middle()["cheapest_rank"] == 16

    def test_the_crudest_rank_costs_more(self):
        result = the_cheapest_projection_rank_is_in_the_middle()
        assert result["crudest_cost"] > result["cheapest_cost"]

    def test_and_so_does_the_finest(self):
        result = the_cheapest_projection_rank_is_in_the_middle()
        assert result["finest_cost"] > result["cheapest_cost"]

    def test_sign_codes_beat_every_projection(self):
        cheapest = the_cheapest_projection_rank_is_in_the_middle()["cheapest_cost"]
        assert how_deep_the_shortlist_must_be()["cheapest_cost"] < cheapest

    def test_an_empty_rank_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            the_depth_needed_grows_as_the_first_stage_gets_cruder(ranks=())

    def test_an_empty_depth_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            the_depth_needed_grows_as_the_first_stage_gets_cruder(depths=())

    def test_an_unreachable_target_is_refused_by_the_comparison(self):
        with pytest.raises(ConfigError, match="nothing reached"):
            the_cheapest_projection_rank_is_in_the_middle(target=0.9, depths=(10, 20))


class TestCost:
    def test_the_rerank_overtakes_the_scan(self):
        assert where_the_rerank_overtakes_the_first_stage()["crossing_depth"] == 200

    def test_the_scan_price_does_not_move_with_the_depth(self):
        rows = where_the_rerank_overtakes_the_first_stage()["rows"]
        assert len({row["first_stage"] for row in rows}) == 1

    def test_the_rerank_price_is_the_depth(self):
        rows = where_the_rerank_overtakes_the_first_stage()["rows"]
        assert all(row["rerank"] == float(row["depth"]) for row in rows)

    def test_an_empty_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            where_the_rerank_overtakes_the_first_stage(depths=())

    def test_the_deep_list_does_not_help_a_partition_stage(self):
        assert a_deeper_list_helps_less_than_a_wider_probe()["depth_did_not_help"]

    def test_but_probing_more_does(self):
        assert a_deeper_list_helps_less_than_a_wider_probe()["probing_more_did"]

    def test_the_probe_recall_rises(self):
        rows = a_deeper_list_helps_less_than_a_wider_probe()["rows"]
        recalls = [row["recall"] for row in rows]
        assert recalls == sorted(recalls)

    def test_an_empty_probe_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            a_deeper_list_helps_less_than_a_wider_probe(probes_tried=())


class TestAcrossCorpora:
    def test_three_corpora_are_measured(self):
        assert len(the_corpus_changes_the_depth()) == 3

    def test_the_subspace_corpus_suits_sign_codes_best(self):
        rows = {row["corpus"]: row for row in the_corpus_changes_the_depth()}
        assert rows["subspace"]["reranked"] > rows["clustered"]["reranked"]

    def test_and_the_clustered_one_beats_the_gaussian(self):
        rows = {row["corpus"]: row for row in the_corpus_changes_the_depth()}
        assert rows["clustered"]["reranked"] > rows["gaussian"]["reranked"]

    def test_reranking_helps_on_all_three(self):
        assert all(row["reranked"] > row["alone"] for row in the_corpus_changes_the_depth())

    def test_an_empty_corpus_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            the_corpus_changes_the_depth(kinds=())

    def test_an_unknown_corpus_is_refused(self):
        with pytest.raises(ConfigError, match="is not a corpus"):
            _setup(kind="spiral")


class TestComparison:
    def test_every_scorer_appears(self):
        assert {row["stage"] for row in compare_the_first_stages()} == set(SCORERS)

    def test_the_exact_stage_is_perfect(self):
        rows = {row["stage"]: row for row in compare_the_first_stages()}
        assert rows["exact"]["reranked"] == 1.0

    def test_and_the_most_expensive(self):
        rows = compare_the_first_stages()
        assert max(rows, key=lambda row: row["cost"])["stage"] == "exact"

    def test_the_sign_stage_is_the_cheapest(self):
        assert summarise()["cheapest"] == "sign"

    def test_the_partition_stage_beats_the_sign_stage_on_recall(self):
        rows = {row["stage"]: row for row in compare_the_first_stages()}
        assert rows["partition"]["reranked"] > rows["sign"]["reranked"]

    def test_every_row_meets_its_ceiling(self):
        rows = compare_the_first_stages()
        assert all(row["reranked"] == row["ceiling"] for row in rows)

    def test_the_summary_names_the_split(self):
        assert summarise()["the_split_is_clean"]

    def test_the_summary_carries_the_depth(self):
        assert summarise(depth=50)["depth"] == 50


class TestMechanics:
    def test_sign_codes_are_one_bit_per_dimension(self):
        vectors = torch.tensor([[1.0, -2.0, 0.5]])
        assert sign_codes(vectors).tolist() == [[True, False, True]]

    def test_a_one_dimensional_corpus_is_refused(self):
        with pytest.raises(DataError, match="two dimensional"):
            sign_codes(torch.zeros(8))

    def test_hamming_counts_differing_bits(self):
        left = torch.tensor([[True, True, False, False]])
        right = torch.tensor([[True, False, True, False]])
        assert float(hamming_scores(left, right)) == 2.0

    def test_identical_codes_are_at_zero(self):
        codes = torch.tensor([[True, False, True]])
        assert float(hamming_scores(codes, codes)) == 0.0

    def test_mismatched_widths_are_refused(self):
        left = torch.zeros(1, 4, dtype=torch.bool)
        right = torch.zeros(1, 8, dtype=torch.bool)
        with pytest.raises(DataError, match="bits against"):
            hamming_scores(left, right)

    def test_a_projection_is_orthonormal(self):
        basis = projection(dimension=16, rank=4)
        product = basis.T @ basis
        assert torch.allclose(product, torch.eye(4), atol=1e-5)

    def test_a_projection_has_the_asked_for_shape(self):
        assert tuple(projection(dimension=16, rank=5).shape) == (16, 5)

    def test_a_rank_of_nothing_is_refused(self):
        with pytest.raises(ConfigError, match="not inside"):
            projection(dimension=8, rank=0)

    def test_a_rank_past_the_dimension_is_refused(self):
        assert a_rank_outside_the_dimension_is_refused()

    def test_a_shortlist_reports_its_shape(self):
        corpus, probes, _ = _setup(count=512, queries=8)
        shortlist = sign_shortlist(probes, corpus, depth=20)
        assert shortlist.depth == 20 and shortlist.queries == 8

    def test_a_shortlist_serialises(self):
        corpus, probes, _ = _setup(count=512, queries=8)
        assert sign_shortlist(probes, corpus, depth=20).as_dict()["depth"] == 20

    def test_mismatched_shortlist_tensors_are_refused(self):
        with pytest.raises(DataError, match="identifiers against"):
            Shortlist(torch.zeros(4, 5, dtype=torch.long), torch.zeros(4, 6), 1.0)

    def test_a_one_dimensional_shortlist_is_refused(self):
        with pytest.raises(DataError, match="two dimensional"):
            Shortlist(torch.zeros(5, dtype=torch.long), torch.zeros(5), 1.0)

    def test_a_shortlist_narrower_than_k_cannot_answer(self):
        corpus, probes, _ = _setup(count=512, queries=8)
        shortlist = sign_shortlist(probes, corpus, depth=5)
        with pytest.raises(ConfigError, match="cannot answer for"):
            shortlist.as_neighbours(10)

    def test_a_depth_of_nothing_is_refused(self):
        corpus, probes, _ = _setup(count=512, queries=8)
        with pytest.raises(ConfigError, match="not a shortlist depth"):
            sign_shortlist(probes, corpus, depth=0)

    def test_a_depth_past_the_corpus_is_refused(self):
        assert a_shortlist_longer_than_the_corpus_is_refused()

    def test_reranking_below_k_is_refused(self):
        assert a_shortlist_shorter_than_the_answer_is_refused()

    def test_reranking_a_different_batch_is_refused(self):
        assert a_mismatched_query_count_is_refused()

    def test_a_result_width_of_nothing_is_refused(self):
        corpus, probes, _ = _setup(count=512, queries=8)
        shortlist = sign_shortlist(probes, corpus, depth=20)
        with pytest.raises(ConfigError, match="not a result width"):
            rerank(probes, corpus, shortlist, k=0)

    def test_the_rerank_returns_its_own_cost(self):
        corpus, probes, _ = _setup(count=512, queries=8)
        shortlist = sign_shortlist(probes, corpus, depth=20)
        _, cost = rerank(probes, corpus, shortlist, k=10)
        assert cost == 20.0

    def test_the_rerank_returns_k_columns(self):
        corpus, probes, _ = _setup(count=512, queries=8)
        shortlist = sign_shortlist(probes, corpus, depth=20)
        answer, _ = rerank(probes, corpus, shortlist, k=7)
        assert tuple(answer.identifiers.shape) == (8, 7)

    def test_the_rerank_returns_sorted_scores(self):
        corpus, probes, _ = _setup(count=512, queries=8)
        shortlist = sign_shortlist(probes, corpus, depth=20)
        answer, _ = rerank(probes, corpus, shortlist, k=10)
        assert bool((answer.scores[:, 1:] >= answer.scores[:, :-1]).all())

    def test_the_rerank_returns_distinct_identifiers(self):
        corpus, probes, _ = _setup(count=512, queries=8)
        shortlist = sign_shortlist(probes, corpus, depth=20)
        answer, _ = rerank(probes, corpus, shortlist, k=10)
        assert all(len(set(row.tolist())) == 10 for row in answer.identifiers)

    def test_the_rerank_returns_only_shortlisted_identifiers(self):
        corpus, probes, _ = _setup(count=512, queries=8)
        shortlist = sign_shortlist(probes, corpus, depth=20)
        answer, _ = rerank(probes, corpus, shortlist, k=10)
        for row in range(8):
            offered = set(shortlist.identifiers[row].tolist())
            assert set(answer.identifiers[row].tolist()) <= offered

    def test_shortlist_recall_of_the_exact_stage_is_one(self):
        corpus, probes, truth = _setup(count=512, queries=8)
        shortlist = exact_shortlist(probes, corpus, depth=20)
        assert shortlist_recall(truth, shortlist) == 1.0

    def test_shortlist_recall_against_the_wrong_batch_is_refused(self):
        corpus, probes, truth = _setup(count=512, queries=8)
        shortlist = sign_shortlist(probes[:4], corpus, depth=20)
        with pytest.raises(DataError, match="truths against"):
            shortlist_recall(truth, shortlist)

    def test_a_staged_result_adds_its_costs(self):
        staged = Staged(
            answer=Neighbours(torch.zeros(1, 1, dtype=torch.long), torch.zeros(1, 1)),
            shortlist_recall=0.8,
            final_recall=0.8,
            first_stage_cost=100.0,
            rerank_cost=50.0,
        )
        assert staged.total_cost == 150.0 and staged.headroom == 0.0

    def test_a_staged_result_serialises(self):
        corpus, probes, truth = _setup(count=512, queries=8)
        shortlist = sign_shortlist(probes, corpus, depth=20)
        staged = staged_search(probes, corpus, truth, shortlist, k=10)
        assert staged.as_dict()["headroom"] == 0.0

    def test_the_partition_stage_reports_a_real_distance_count(self):
        corpus, probes, _ = _setup(count=512, queries=8)
        shortlist = partition_shortlist(probes, corpus, depth=20, partitions=16, probe=4)
        assert 0.0 < shortlist.cost_per_query <= float(corpus.shape[0])

    def test_the_projected_stage_is_priced_by_its_rank(self):
        corpus, probes, _ = _setup(count=512, queries=8)
        shortlist = projected_shortlist(probes, corpus, depth=20, rank=8)
        assert shortlist.cost_per_query == float(corpus.shape[0]) * 8 / 32

    def test_the_sign_stage_is_priced_at_a_thirty_second(self):
        corpus, probes, _ = _setup(count=512, queries=8)
        shortlist = sign_shortlist(probes, corpus, depth=20)
        assert shortlist.cost_per_query == float(corpus.shape[0]) / 32.0

from __future__ import annotations

import pytest
import torch

from vse.errors import ConfigError, DataError
from vse.eval.fusion import (
    Ranking,
    a_mismatched_weight_list_is_refused,
    a_negative_constant_is_refused,
    a_range_normalisation_of_one_result_is_refused,
    a_rank_one_ranking_is_refused,
    a_ranking_whose_scores_do_not_match_is_refused,
    a_small_constant_makes_depth_free,
    an_empty_fusion_is_refused,
    fuse_by_rank,
    fuse_by_score,
    fusing_a_list_with_itself_changes_nothing,
    fusing_three_is_better_than_fusing_two,
    fusing_with_a_useless_retriever_costs_something,
    normalise_by_maximum,
    normalise_by_range,
    normalise_by_the_best,
    normalise_globally,
    rank_fusion_beats_score_fusion,
    rank_fusion_is_exactly_invariant,
    rankings_of_different_query_counts_are_refused,
    score_fusion_breaks_under_a_rescaling,
    the_best_weight_is_interior,
    the_ceiling_is_the_union,
    the_constant_decides_whether_rank_fusion_wins,
    the_constant_is_what_makes_rank_fusion_work,
    the_conventional_constant_is_wrong_for_short_lists,
    the_depth_and_the_constant_are_one_parameter,
    the_normalisation_barely_matters,
    the_normalisations_all_preserve_the_order,
    the_two_retrievers_disagree,
    weighting_a_better_retriever_helps,
)


def a_ranking(queries: int = 4, depth: int = 6, seed: int = 0) -> Ranking:
    """A small ranking with distinct identifiers per row."""
    generator = torch.Generator().manual_seed(seed)
    identifiers = torch.stack(
        [torch.randperm(50, generator=generator)[:depth] for _ in range(queries)]
    )
    scores = torch.sort(torch.rand(queries, depth, generator=generator), dim=1).values
    return Ranking(identifiers=identifiers, scores=scores, name=f"seed {seed}")


class TestInputs:
    def test_the_two_retrievers_disagree(self):
        assert the_two_retrievers_disagree()["agreement"] < 0.5

    def test_and_each_finds_things_the_other_missed(self):
        result = the_two_retrievers_disagree()
        assert result["only_left"] > 0
        assert result["only_right"] > 0

    def test_so_there_is_something_to_gain(self):
        assert the_two_retrievers_disagree()["there_is_something_to_gain"]

    def test_the_ceiling_is_above_the_better_retriever(self):
        assert the_ceiling_is_the_union()["headroom"] > 0

    def test_by_a_substantial_share(self):
        assert the_ceiling_is_the_union()["headroom_share"] > 0.2


class TestTheConstant:
    def test_the_conventional_constant_loses(self):
        assert the_constant_decides_whether_rank_fusion_wins()[
            "conventional_loses_to_doing_nothing"
        ]

    def test_and_the_tuned_one_wins(self):
        assert the_constant_decides_whether_rank_fusion_wins()["tuned_beats_score"]

    def test_the_gap_between_them_is_enormous(self):
        assert the_constant_decides_whether_rank_fusion_wins()["the_constant_is_worth"] > 0.2

    def test_a_small_constant_is_better_here(self):
        assert the_conventional_constant_is_wrong_for_short_lists()["small_is_better_here"]

    def test_sixty_is_far_from_the_best(self):
        assert the_conventional_constant_is_wrong_for_short_lists()[
            "sixty_is_far_from_the_best"
        ]

    def test_the_recall_falls_as_the_constant_rises(self):
        rows = [row["recall"] for row in the_constant_is_what_makes_rank_fusion_work()]
        assert rows == sorted(rows, reverse=True)

    def test_a_huge_constant_is_the_same_as_sixty(self):
        rows = {row["constant"]: row for row in the_constant_is_what_makes_rank_fusion_work()}
        assert rows[500.0]["recall"] == rows[60.0]["recall"]

    def test_an_empty_constant_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            the_constant_is_what_makes_rank_fusion_work(constants=())

    def test_a_negative_constant_is_refused(self):
        assert a_negative_constant_is_refused()


class TestFamilies:
    def test_eight_methods_are_compared(self):
        assert len(rank_fusion_beats_score_fusion()) == 8

    def test_every_score_fusion_beats_both_retrievers(self):
        rows = {row["method"]: row["recall"] for row in rank_fusion_beats_score_fusion()}
        best_single = max(rows["ivf alone"], rows["forest alone"])
        assert all(
            value > best_single for name, value in rows.items() if name.startswith("score")
        )

    def test_the_normalisation_barely_matters(self):
        assert the_normalisation_barely_matters()["barely_matters"]

    def test_the_four_span_two_points(self):
        assert the_normalisation_barely_matters()["spread"] < 0.05

    def test_the_normalisations_all_preserve_the_order(self):
        assert the_normalisations_all_preserve_the_order()["all_monotone"]

    def test_each_one_individually(self):
        result = the_normalisations_all_preserve_the_order()
        assert result["maximum"] and result["range"] and result["best"] and result["global"]

    def test_the_best_anchored_normaliser_puts_the_top_at_one(self):
        ranking = a_ranking()
        assert float((normalise_by_the_best(ranking)[:, 0] - 1.0).abs().max()) < 1e-5

    def test_the_range_normaliser_puts_the_top_at_zero(self):
        ranking = a_ranking()
        assert float(normalise_by_range(ranking)[:, 0].abs().max()) < 1e-5

    def test_and_the_bottom_at_one(self):
        ranking = a_ranking()
        assert float((normalise_by_range(ranking)[:, -1] - 1.0).abs().max()) < 1e-5

    def test_the_maximum_normaliser_puts_the_bottom_at_one(self):
        ranking = a_ranking()
        assert float((normalise_by_maximum(ranking)[:, -1] - 1.0).abs().max()) < 1e-5

    def test_the_global_normaliser_uses_one_divisor(self):
        ranking = a_ranking()
        moved = normalise_globally(ranking)
        ratio = moved / ranking.scores
        assert float(ratio.std()) < 1e-5

    def test_a_range_normalisation_of_one_result_is_refused(self):
        assert a_range_normalisation_of_one_result_is_refused()

    def test_an_empty_normalisation_is_refused(self):
        empty = Ranking(
            identifiers=torch.zeros(2, 0, dtype=torch.long),
            scores=torch.zeros(2, 0),
            name="empty",
        )
        with pytest.raises(ConfigError, match="cannot be normalised"):
            normalise_by_maximum(empty)


class TestInvariance:
    def test_rank_fusion_is_exactly_invariant(self):
        assert rank_fusion_is_exactly_invariant()["invariant"]

    def test_giving_one_distinct_result(self):
        assert rank_fusion_is_exactly_invariant()["distinct_rank_results"] == 1

    def test_score_fusion_collapses(self):
        assert rank_fusion_is_exactly_invariant()["score_collapses"]

    def test_by_more_than_half(self):
        result = rank_fusion_is_exactly_invariant()
        assert result["score_at_the_sharpest"] < result["score_at_the_mildest"] * 0.6

    def test_the_score_column_falls_monotonically(self):
        rows = [row["score_fusion"] for row in score_fusion_breaks_under_a_rescaling()]
        assert rows == sorted(rows, reverse=True)

    def test_and_the_rank_column_does_not_move(self):
        rows = [row["rank_fusion"] for row in score_fusion_breaks_under_a_rescaling()]
        assert len(set(rows)) == 1

    def test_an_empty_rescaling_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            score_fusion_breaks_under_a_rescaling(scales=())


class TestDepth:
    def test_depth_is_inert_with_a_small_constant(self):
        assert a_small_constant_makes_depth_free()["depth_is_inert_when_the_constant_is_small"]

    def test_and_harmful_with_a_large_one(self):
        assert a_small_constant_makes_depth_free()["depth_is_harmful_when_it_is_large"]

    def test_they_agree_at_the_shallowest_depth(self):
        assert a_small_constant_makes_depth_free()["they_agree_at_depth_ten"]

    def test_and_diverge_by_a_factor_of_three(self):
        result = a_small_constant_makes_depth_free()
        assert (
            result["small_constant_at_two_hundred"]
            > result["large_constant_at_two_hundred"] * 2.5
        )

    def test_ten_rows_are_measured(self):
        assert len(the_depth_and_the_constant_are_one_parameter()) == 10

    def test_an_empty_depth_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            the_depth_and_the_constant_are_one_parameter(depths=())

    def test_an_empty_constant_list_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            the_depth_and_the_constant_are_one_parameter(constants=())


class TestWeights:
    def test_the_best_weight_is_interior(self):
        assert the_best_weight_is_interior()["interior"]

    def test_and_beats_both_retrievers_alone(self):
        assert the_best_weight_is_interior()["beats_both"]

    def test_a_weight_of_zero_is_the_second_retriever(self):
        rows = {row["weight"]: row for row in weighting_a_better_retriever_helps()}
        result = the_best_weight_is_interior()
        assert rows[0.0]["recall"] == result["second_alone"]

    def test_and_a_weight_of_one_is_the_first(self):
        rows = {row["weight"]: row for row in weighting_a_better_retriever_helps()}
        result = the_best_weight_is_interior()
        assert rows[1.0]["recall"] == result["first_alone"]

    def test_an_empty_weight_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            weighting_a_better_retriever_helps(weights=())

    def test_a_mismatched_weight_list_is_refused(self):
        assert a_mismatched_weight_list_is_refused()

    def test_for_score_fusion_too(self):
        left, right = a_ranking(seed=1), a_ranking(seed=2)
        with pytest.raises(ConfigError, match="weights for"):
            fuse_by_score([left, right], 50, k=3, weights=[1.0, 1.0, 1.0])


class TestMoreRetrievers:
    def test_a_third_retriever_helps(self):
        assert fusing_three_is_better_than_fusing_two()["third_helps"]

    def test_with_diminishing_returns(self):
        assert fusing_three_is_better_than_fusing_two()["diminishing"]

    def test_two_beats_one(self):
        result = fusing_three_is_better_than_fusing_two()
        assert result["two"] > result["one"]

    def test_and_three_beats_two(self):
        result = fusing_three_is_better_than_fusing_two()
        assert result["three"] > result["two"]


class TestDegenerateCases:
    def test_fusing_a_list_with_itself_is_the_identity(self):
        assert fusing_a_list_with_itself_changes_nothing()["rank_is_identity"]

    def test_for_score_fusion_too(self):
        assert fusing_a_list_with_itself_changes_nothing()["score_is_identity"]

    def test_and_the_recall_does_not_move(self):
        assert fusing_a_list_with_itself_changes_nothing()["recall_unchanged"]

    def test_a_useless_retriever_costs_something(self):
        assert fusing_with_a_useless_retriever_costs_something()["both_lose"]

    def test_rank_fusion_is_more_robust_to_it(self):
        assert fusing_with_a_useless_retriever_costs_something()["rank_is_more_robust"]

    def test_by_a_wide_margin(self):
        result = fusing_with_a_useless_retriever_costs_something()
        assert result["fused_by_rank"] > result["fused_by_score"] * 1.4

    def test_an_empty_fusion_is_refused(self):
        assert an_empty_fusion_is_refused()

    def test_an_empty_score_fusion_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to fuse"):
            fuse_by_score([], 100, k=10)

    def test_rankings_of_different_query_counts_are_refused(self):
        assert rankings_of_different_query_counts_are_refused()

    def test_mismatched_queries_are_refused_for_score_fusion_too(self):
        left = a_ranking(queries=4)
        right = a_ranking(queries=2)
        with pytest.raises(DataError, match="cannot fuse with"):
            fuse_by_score([left, right], 50, k=3)


class TestShapes:
    def test_a_ranking_whose_scores_do_not_match_is_refused(self):
        assert a_ranking_whose_scores_do_not_match_is_refused()

    def test_a_rank_one_ranking_is_refused(self):
        assert a_rank_one_ranking_is_refused()

    def test_a_ranking_reports_its_shape(self):
        ranking = a_ranking(queries=5, depth=7)
        assert ranking.queries == 5
        assert ranking.depth == 7

    def test_and_serialises(self):
        assert a_ranking().as_dict()["depth"] == 6

    def test_it_converts_to_neighbours(self):
        ranking = a_ranking()
        assert ranking.as_neighbours().k == ranking.depth

    def test_a_rank_fusion_returns_k_results(self):
        left, right = a_ranking(seed=1), a_ranking(seed=2)
        fused = fuse_by_rank([left, right], 50, k=3)
        assert tuple(fused.identifiers.shape) == (4, 3)

    def test_a_score_fusion_returns_k_results(self):
        left, right = a_ranking(seed=1), a_ranking(seed=2)
        fused = fuse_by_score([left, right], 50, k=3)
        assert tuple(fused.identifiers.shape) == (4, 3)

    def test_a_single_ranking_fuses_to_itself(self):
        left = a_ranking()
        fused = fuse_by_rank([left], 50, k=3)
        assert bool(torch.equal(fused.identifiers, left.identifiers[:, :3]))

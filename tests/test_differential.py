from __future__ import annotations

import pytest
import torch

from vse.errors import ConfigError
from vse.index.base import top_up
from vse.index.flat import FlatIndex
from vse.index.tree import TreeIndex
from vse.vectors.dataset import gaussian
from vse.vectors.exact import Neighbours
from vse.verify.differential import (
    RULES,
    Report,
    Violation,
    a_check_for_no_neighbours_is_refused,
    a_check_with_no_queries_is_refused,
    a_corpus_of_identical_vectors_is_the_hard_one,
    a_deliberately_broken_index_is_caught,
    a_duplicated_corpus_makes_ties_unavoidable,
    a_query_of_the_wrong_width_is_refused,
    a_report_counts_by_rule,
    an_empty_report_is_clean,
    an_index_of_one_vector_works,
    asking_for_more_than_the_corpus_holds_is_refused,
    awkward_corpora,
    different_seeds_check_different_queries,
    every_index,
    every_index_returns_well_formed_results,
    identifiers_are_distinct,
    identifiers_are_in_range,
    results_are_ordered,
    returns_k_results,
    scores_agree_with_identifiers,
    searching_an_unbuilt_index_is_refused,
    searching_for_a_corpus_vector_finds_it,
    sweep,
    the_exact_structures_agree_with_each_other,
    the_nearest_is_at_least_as_near_as_the_worst_exact,
    the_rules_do_not_fire_on_a_correct_result,
    the_sweep_is_deterministic,
)


def a_result(count: int = 256, dimension: int = 8, k: int = 10):
    """An exact result and the corpus and queries it came from."""
    corpus = gaussian(count=count, dimension=dimension).vectors
    probes = corpus[:8].clone()
    index = FlatIndex(dimension)
    index.build(corpus)
    found, _ = index.search(probes, k=k)
    return found, corpus, probes, k


class TestTheRules:
    def test_none_fire_on_a_correct_result(self):
        found, corpus, probes, k = a_result()
        assert all(rule(found, corpus, probes, k) is None for _, rule in RULES)

    def test_each_fires_on_its_own_break(self):
        assert the_rules_do_not_fire_on_a_correct_result()["every_rule_fires_on_its_own_break"]

    def test_the_shape_rule_catches_a_short_result(self):
        found, corpus, probes, k = a_result()
        short = Neighbours(found.identifiers[:, :5], found.scores[:, :5])
        assert returns_k_results(short, corpus, probes, k) is not None

    def test_and_a_wrong_row_count(self):
        found, corpus, probes, k = a_result()
        fewer = Neighbours(found.identifiers[:3], found.scores[:3])
        assert returns_k_results(fewer, corpus, probes, k) is not None

    def test_the_distinctness_rule_catches_a_repeat(self):
        found, corpus, probes, k = a_result()
        repeated = Neighbours(found.identifiers.clone(), found.scores.clone())
        repeated.identifiers[:, 1] = repeated.identifiers[:, 0]
        assert identifiers_are_distinct(repeated, corpus, probes, k) is not None

    def test_the_range_rule_catches_a_high_identifier(self):
        found, corpus, probes, k = a_result()
        broken = Neighbours(found.identifiers.clone(), found.scores.clone())
        broken.identifiers[0, 0] = 9999
        assert identifiers_are_in_range(broken, corpus, probes, k) is not None

    def test_and_a_negative_one(self):
        found, corpus, probes, k = a_result()
        broken = Neighbours(found.identifiers.clone(), found.scores.clone())
        broken.identifiers[0, 0] = -1
        assert identifiers_are_in_range(broken, corpus, probes, k) is not None

    def test_the_score_rule_catches_a_shifted_score(self):
        found, corpus, probes, k = a_result()
        broken = Neighbours(found.identifiers.clone(), found.scores.clone() + 5.0)
        assert scores_agree_with_identifiers(broken, corpus, probes, k) is not None

    def test_the_order_rule_catches_a_reversal(self):
        found, corpus, probes, k = a_result()
        reversed_result = Neighbours(found.identifiers.flip(1), found.scores.flip(1))
        assert results_are_ordered(reversed_result, corpus, probes, k) is not None

    def test_the_relatedness_rule_passes_on_exact_results(self):
        found, corpus, probes, k = a_result()
        assert (
            the_nearest_is_at_least_as_near_as_the_worst_exact(found, corpus, probes, k) is None
        )

    def test_and_fires_on_an_unrelated_one(self):
        _, corpus, probes, k = a_result()
        far = Neighbours(
            identifiers=torch.arange(k).expand(8, k).contiguous(),
            scores=torch.full((8, k), 1000.0),
        )
        assert (
            the_nearest_is_at_least_as_near_as_the_worst_exact(far, corpus, probes, k)
            is not None
        )

    def test_six_rules_are_defined(self):
        assert len(RULES) == 6

    def test_every_rule_has_a_name(self):
        assert all(isinstance(name, str) and name for name, _ in RULES)


class TestTheSweep:
    def test_the_sweep_runs_every_combination(self):
        assert every_index_returns_well_formed_results()["checks"] == 240

    def test_and_finds_only_the_known_four(self):
        assert every_index_returns_well_formed_results()["violations"] == 4

    def test_all_of_one_rule(self):
        assert set(every_index_returns_well_formed_results()["by_rule"]) == {"not unrelated"}

    def test_eight_indexes_are_covered(self):
        assert len(every_index()) == 8

    def test_five_corpora_are_covered(self):
        assert len(awkward_corpora()) == 5

    def test_every_corpus_has_the_right_shape(self):
        for _, corpus in awkward_corpora(dimension=16, count=512):
            assert tuple(corpus.shape) == (512, 16)

    def test_the_sweep_is_deterministic(self):
        assert the_sweep_is_deterministic()["identical"]

    def test_with_matching_check_counts(self):
        assert the_sweep_is_deterministic()["checks_match"]

    def test_different_seeds_pick_different_queries(self):
        assert different_seeds_check_different_queries()["differ"]

    def test_a_sweep_with_no_queries_is_refused(self):
        assert a_check_with_no_queries_is_refused()

    def test_a_sweep_for_no_neighbours_is_refused(self):
        assert a_check_for_no_neighbours_is_refused()

    def test_a_negative_k_is_refused(self):
        with pytest.raises(ConfigError, match="is not a check"):
            sweep(k=-1)


class TestAwkwardCorpora:
    def test_a_corpus_of_identical_vectors_is_clean(self):
        assert a_corpus_of_identical_vectors_is_the_hard_one()["clean"]

    def test_and_every_index_builds_on_it(self):
        assert a_corpus_of_identical_vectors_is_the_hard_one()["indexes_built"] == 8

    def test_a_duplicated_corpus_produces_no_repeats(self):
        assert a_duplicated_corpus_makes_ties_unavoidable()["repeated_identifiers"] == 0

    def test_searching_for_a_corpus_vector_finds_it(self):
        assert searching_for_a_corpus_vector_finds_it()["exact_structures_never_miss"]

    def test_the_approximate_ones_mostly_do_too(self):
        assert searching_for_a_corpus_vector_finds_it()["worst"] > 20

    def test_the_exact_structures_agree(self):
        assert the_exact_structures_agree_with_each_other()["identifiers_identical"]

    def test_to_within_floating_point(self):
        assert the_exact_structures_agree_with_each_other()["scores_identical"]

    def test_with_a_negligible_score_gap(self):
        assert the_exact_structures_agree_with_each_other()["max_score_gap"] < 1e-4


class TestContracts:
    def test_an_index_of_one_vector_works_or_refuses(self):
        assert an_index_of_one_vector_works()["all_built_are_correct"]

    def test_most_structures_refuse_a_corpus_of_one(self):
        assert an_index_of_one_vector_works()["refused"] > 4

    def test_asking_for_more_than_the_corpus_holds_is_refused(self):
        assert asking_for_more_than_the_corpus_holds_is_refused()["all_refuse"]

    def test_searching_an_unbuilt_index_is_refused(self):
        assert searching_an_unbuilt_index_is_refused()["all_refuse"]

    def test_by_all_eight(self):
        assert searching_an_unbuilt_index_is_refused()["checked"] == 8

    def test_a_query_of_the_wrong_width_is_refused(self):
        assert a_query_of_the_wrong_width_is_refused()["all_refuse"]


class TestTopUp:
    def test_a_full_result_is_returned_unchanged(self):
        corpus = gaussian(count=64, dimension=8).vectors
        live = torch.ones(64, dtype=torch.bool)
        found = [(float(row), row) for row in range(5)]
        assert top_up(found, 5, corpus[:1], corpus, live) == found

    def test_a_short_result_is_filled(self):
        corpus = gaussian(count=64, dimension=8).vectors
        live = torch.ones(64, dtype=torch.bool)
        filled = top_up([(0.0, 3)], 5, corpus[:1], corpus, live)
        assert len(filled) == 5

    def test_the_fill_is_distinct(self):
        corpus = gaussian(count=64, dimension=8).vectors
        live = torch.ones(64, dtype=torch.bool)
        filled = top_up([(0.0, 3)], 5, corpus[:1], corpus, live)
        assert len({identifier for _, identifier in filled}) == 5

    def test_and_sorted(self):
        corpus = gaussian(count=64, dimension=8).vectors
        live = torch.ones(64, dtype=torch.bool)
        filled = top_up([(0.0, 3)], 5, corpus[:1], corpus, live)
        scores = [score for score, _ in filled]
        assert scores == sorted(scores)

    def test_and_the_scores_are_real_distances(self):
        corpus = gaussian(count=64, dimension=8).vectors
        live = torch.ones(64, dtype=torch.bool)
        filled = top_up([], 3, corpus[:1], corpus, live)
        for score, identifier in filled:
            wanted = float(((corpus[0] - corpus[identifier]) ** 2).sum())
            assert abs(score - wanted) < 1e-4

    def test_dead_rows_are_not_used(self):
        corpus = gaussian(count=64, dimension=8).vectors
        live = torch.ones(64, dtype=torch.bool)
        live[:60] = False
        filled = top_up([], 4, corpus[:1], corpus, live)
        assert all(identifier >= 60 for _, identifier in filled)

    def test_a_corpus_too_small_is_refused(self):
        corpus = gaussian(count=8, dimension=8).vectors
        live = torch.ones(8, dtype=torch.bool)
        with pytest.raises(ConfigError, match="cannot supply"):
            top_up([], 20, corpus[:1], corpus, live)

    def test_a_zero_width_is_refused(self):
        corpus = gaussian(count=8, dimension=8).vectors
        live = torch.ones(8, dtype=torch.bool)
        with pytest.raises(ConfigError, match="not a result width"):
            top_up([], 0, corpus[:1], corpus, live)


class TestTheTreeScoreFix:
    def test_the_tree_reports_squared_distances(self):
        corpus = gaussian(count=512, dimension=8).vectors
        probes = corpus[:4].clone()
        tree = TreeIndex(8, leaf_size=32)
        tree.build(corpus)
        found, _ = tree.search(probes, k=5)
        wanted = ((probes[0] - corpus[found.identifiers[0]]) ** 2).sum(dim=1)
        assert float((wanted - found.scores[0]).abs().max()) < 1e-3

    def test_and_agrees_with_the_flat_index(self):
        corpus = gaussian(count=512, dimension=8).vectors
        probes = corpus[:4].clone()
        tree = TreeIndex(8, leaf_size=32)
        tree.build(corpus)
        flat = FlatIndex(8)
        flat.build(corpus)
        left, _ = tree.search(probes, k=5)
        right, _ = flat.search(probes, k=5)
        assert bool(torch.allclose(left.scores, right.scores, atol=1e-4))


class TestReports:
    def test_a_report_counts_by_rule(self):
        assert a_report_counts_by_rule()["distinct_count"] == 2

    def test_and_counts_every_check(self):
        assert a_report_counts_by_rule()["checks"] == 4

    def test_an_empty_report_is_clean(self):
        assert an_empty_report_is_clean()["clean"]

    def test_with_no_checks(self):
        assert an_empty_report_is_clean()["checks"] == 0

    def test_a_report_with_a_violation_is_not_clean(self):
        report = Report()
        report.record(Violation("flat", "gaussian", "distinct", "detail"))
        assert not report.clean

    def test_recording_nothing_still_counts_the_check(self):
        report = Report()
        report.record(None)
        assert report.checks == 1 and report.clean

    def test_a_violation_serialises(self):
        row = Violation("flat", "gaussian", "distinct", "detail").as_dict()
        assert row["index"] == "flat" and row["rule"] == "distinct"

    def test_a_report_serialises(self):
        report = Report()
        report.record(Violation("flat", "gaussian", "ordered", "detail"))
        assert report.as_dict()["violations"] == 1

    def test_the_broken_index_check_reports_every_rule(self):
        result = a_deliberately_broken_index_is_caught()
        assert result["clean_input_passes"]
        assert result["distinct_fires"]
        assert result["scores_fire"]

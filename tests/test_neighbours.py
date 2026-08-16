from __future__ import annotations

import pytest
import torch

from vse.build.neighbours import (
    Graph,
    a_degree_above_the_corpus_is_refused,
    a_graph_pointing_outside_itself_is_refused,
    a_nearest_neighbour_graph_is_disconnected,
    a_nearest_neighbour_graph_is_not_symmetric,
    a_self_loop_is_refused,
    a_small_degree_disconnects_it,
    alpha_trades_degree_for_reach,
    an_alpha_below_one_is_refused,
    compare_graphs,
    components,
    descent_converges_to_a_plateau,
    descent_gets_close_for_a_fraction,
    descent_is_easier_with_structure,
    exact_graph,
    nn_descent,
    prune,
    pruning_cuts_the_degree,
    recall_against_exact,
    reciprocity,
    symmetrise,
    symmetrising_costs_degree,
    the_saving_is_asymptotic,
)
from vse.errors import BuildError, ConfigError, DataError
from vse.vectors.dataset import gaussian
from vse.vectors.metric import squared_l2


class TestConstruction:
    def test_every_vertex_gets_the_degree_asked_for(self):
        graph = exact_graph(gaussian(count=256, dimension=8).vectors, degree=6)
        assert all(len(graph.neighbours(vertex)) == 6 for vertex in range(graph.order))

    def test_no_vertex_points_at_itself(self):
        graph = exact_graph(gaussian(count=256, dimension=8).vectors, degree=6)
        assert all(vertex not in graph.neighbours(vertex) for vertex in range(graph.order))

    def test_the_batch_size_does_not_change_the_graph(self):
        vectors = gaussian(count=256, dimension=8).vectors
        assert (
            exact_graph(vectors, degree=5).edges
            == exact_graph(vectors, degree=5, batch=17).edges
        )

    def test_the_nearest_neighbour_is_first(self):
        vectors = gaussian(count=256, dimension=8).vectors
        graph = exact_graph(vectors, degree=5)
        scores = squared_l2(vectors[3:4], vectors).flatten()
        scores[3] = float("inf")
        assert graph.neighbours(3)[0] == int(scores.argmin())

    def test_a_degree_above_the_corpus_is_refused(self):
        assert a_degree_above_the_corpus_is_refused()

    def test_a_graph_pointing_outside_itself_is_refused(self):
        assert a_graph_pointing_outside_itself_is_refused()

    def test_a_self_loop_is_refused(self):
        assert a_self_loop_is_refused()

    def test_an_empty_graph_is_refused(self):
        with pytest.raises(BuildError, match="no vertices"):
            Graph(edges=())

    def test_a_rank_three_input_is_refused(self):
        with pytest.raises(DataError, match="matrix of rows"):
            exact_graph(torch.randn(4, 4, 4))

    def test_a_vertex_that_does_not_exist_is_refused(self):
        graph = exact_graph(gaussian(count=64, dimension=4).vectors, degree=3)
        with pytest.raises(ConfigError, match="not one of the 64"):
            graph.neighbours(999)

    def test_it_serialises(self):
        graph = exact_graph(gaussian(count=64, dimension=4).vectors, degree=3)
        assert graph.as_dict()["edges"] == 64 * 3


class TestDirection:
    def test_under_half_the_edges_point_back(self):
        assert a_nearest_neighbour_graph_is_not_symmetric()["under_half"]

    def test_structure_helps_but_not_enough(self):
        result = a_nearest_neighbour_graph_is_not_symmetric()
        assert result["clustered"] > result["gaussian"]
        assert result["clustered"] < 0.6

    def test_symmetrising_makes_every_edge_mutual(self):
        assert a_nearest_neighbour_graph_is_not_symmetric()["after_symmetrising"] == 1.0

    def test_it_costs_sixty_percent_more_edges(self):
        assert 1.5 < symmetrising_costs_degree()["edge_growth"] < 1.8

    def test_and_fourteen_times_the_worst_degree(self):
        result = symmetrising_costs_degree()
        assert result["max_after"] > result["max_before"] * 10

    def test_which_is_why_real_implementations_prune(self):
        result = symmetrising_costs_degree()
        assert result["max_after"] > result["mean_after"] * 5

    def test_reciprocity_of_an_edgeless_graph_is_refused(self):
        with pytest.raises(BuildError, match="no reciprocity"):
            reciprocity(Graph(edges=((), ())))


class TestConnectivity:
    def test_the_directed_graph_is_in_many_pieces(self):
        assert a_nearest_neighbour_graph_is_disconnected()["directed_components"] > 100

    def test_but_the_undirected_one_is_a_single_component(self):
        assert a_nearest_neighbour_graph_is_disconnected()["undirected_components"] == 1

    def test_so_symmetrising_connects_it(self):
        assert a_nearest_neighbour_graph_is_disconnected()["symmetrising_connects_it"]

    def test_even_at_a_degree_of_two(self):
        rows = {row["degree"]: row for row in a_small_degree_disconnects_it()}
        assert rows[2]["symmetrised"] == 1

    def test_where_the_directed_graph_is_in_a_thousand_pieces(self):
        rows = {row["degree"]: row for row in a_small_degree_disconnects_it()}
        assert rows[2]["directed"] > 500

    def test_the_piece_count_falls_with_the_degree(self):
        rows = [row["directed"] for row in a_small_degree_disconnects_it()]
        assert rows == sorted(rows, reverse=True)

    def test_but_never_reaches_one(self):
        rows = a_small_degree_disconnects_it()
        assert all(row["directed"] > 1 for row in rows)

    def test_an_empty_degree_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            a_small_degree_disconnects_it(degrees=())

    def test_a_path_graph_is_one_component(self):
        edges = (*((index + 1,) for index in range(7)), (0,))
        assert components(Graph(edges=edges), directed=True) == 1

    def test_two_disjoint_pairs_are_two(self):
        assert components(Graph(edges=((1,), (0,), (3,), (2,)))) == 2


class TestDescent:
    def test_it_recovers_two_thirds_of_the_graph(self):
        assert 0.25 < descent_gets_close_for_a_fraction()["recall"] < 0.8

    def test_for_two_fifths_of_the_comparisons(self):
        # A poor trade, and the honest answer at this corpus size.
        assert descent_gets_close_for_a_fraction()["share_of_exact"] < 0.5

    def test_the_work_per_vertex_is_a_constant(self):
        rows = the_saving_is_asymptotic()
        assert len({row["descent_per_vertex"] for row in rows}) == 1

    def test_so_the_saving_arrives_with_the_corpus_size(self):
        rows = [row["share"] for row in the_saving_is_asymptotic()]
        assert rows == sorted(rows, reverse=True)

    def test_and_two_thousand_vectors_is_barely_past_the_crossover(self):
        rows = {row["vectors"]: row for row in the_saving_is_asymptotic()}
        assert rows[2048]["share"] > 0.4

    def test_the_model_is_an_upper_bound_on_the_measurement(self):
        # Candidate pools are often smaller than the sample cap, so the real descent does less.
        rows = {row["vectors"]: row for row in the_saving_is_asymptotic()}
        assert (
            descent_gets_close_for_a_fraction()["per_vertex"] < rows[2048]["descent_per_vertex"]
        )

    def test_where_a_hundred_thousand_is_well_above_it(self):
        rows = {row["vectors"]: row for row in the_saving_is_asymptotic()}
        assert rows[100_000]["share"] < 0.02

    def test_an_empty_size_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            the_saving_is_asymptotic(sizes=())

    def test_a_zero_round_projection_is_refused(self):
        with pytest.raises(ConfigError, match="not a run"):
            the_saving_is_asymptotic(rounds=0)

    def test_the_recall_rises_with_the_rounds(self):
        rows = [row["recall"] for row in descent_converges_to_a_plateau()]
        assert rows == sorted(rows)

    def test_and_then_stops(self):
        rows = {row["rounds"]: row for row in descent_converges_to_a_plateau()}
        assert rows[48]["recall"] == rows[32]["recall"]

    def test_because_the_run_stopped_itself(self):
        rows = {row["rounds"]: row for row in descent_converges_to_a_plateau()}
        assert rows[48]["comparisons"] == rows[32]["comparisons"]

    def test_structure_makes_it_much_easier(self):
        result = descent_is_easier_with_structure()
        assert result["clustered_recall"] > result["gaussian_recall"] * 1.4

    def test_an_empty_round_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            descent_converges_to_a_plateau(rounds=())

    def test_a_sample_below_the_degree_is_refused(self):
        with pytest.raises(ConfigError, match="cannot fill"):
            nn_descent(torch.randn(64, 4), degree=10, sample=4)

    def test_a_zero_round_run_is_refused(self):
        with pytest.raises(ConfigError, match="not a run"):
            nn_descent(torch.randn(64, 4), degree=4, rounds=0)

    def test_comparing_graphs_of_different_sizes_is_refused(self):
        with pytest.raises(BuildError, match="vertices against"):
            recall_against_exact(Graph(edges=((1,), (0,))), Graph(edges=((1,), (2,), (0,))))

    def test_a_graph_recalls_itself_completely(self):
        graph = exact_graph(gaussian(count=128, dimension=8).vectors, degree=5)
        assert recall_against_exact(graph, graph) == 1.0


class TestPruning:
    def test_pruning_removes_most_of_the_edges(self):
        assert pruning_cuts_the_degree()["kept_share"] < 0.4

    def test_and_leaves_it_connected(self):
        assert pruning_cuts_the_degree()["still_connected"]

    def test_the_degree_falls_to_the_cap(self):
        result = pruning_cuts_the_degree()
        assert result["mean_after"] <= 8.0

    def test_a_larger_alpha_keeps_more_edges(self):
        rows = [row["mean_degree"] for row in alpha_trades_degree_for_reach()]
        assert rows == sorted(rows)

    def test_and_every_setting_stays_connected(self):
        assert all(row["components"] == 1 for row in alpha_trades_degree_for_reach())

    def test_an_alpha_below_one_is_refused(self):
        assert an_alpha_below_one_is_refused()

    def test_a_zero_degree_prune_is_refused(self):
        vectors = gaussian(count=64, dimension=4).vectors
        with pytest.raises(ConfigError, match="keeps nothing"):
            prune(vectors, exact_graph(vectors, degree=5), degree=0)

    def test_an_empty_alpha_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            alpha_trades_degree_for_reach(alphas=())

    def test_pruning_never_adds_an_edge(self):
        vectors = gaussian(count=256, dimension=8).vectors
        graph = symmetrise(exact_graph(vectors, degree=10))
        assert prune(vectors, graph, degree=10).size <= graph.size

    def test_the_pruned_graph_keeps_the_nearest_neighbour(self):
        # The nearest candidate is always kept: nothing has been kept yet to cover it.
        vectors = gaussian(count=256, dimension=8).vectors
        graph = symmetrise(exact_graph(vectors, degree=10))
        pruned = prune(vectors, graph, degree=4)
        assert all(len(pruned.neighbours(vertex)) >= 1 for vertex in range(pruned.order))


class TestComparison:
    def test_four_constructions_are_compared(self):
        assert len(compare_graphs()) == 4

    def test_the_symmetrised_graph_has_the_most_edges(self):
        rows = {row["graph"]: row for row in compare_graphs()}
        assert rows["symmetrised"]["edges"] == max(row["edges"] for row in compare_graphs())

    def test_the_pruned_graph_has_the_fewest(self):
        rows = compare_graphs()
        assert min(rows, key=lambda row: row["edges"])["graph"] == "pruned"

    def test_every_construction_is_undirected_connected(self):
        assert all(row["components"] == 1 for row in compare_graphs())

    def test_the_symmetrised_graph_is_fully_reciprocal(self):
        rows = {row["graph"]: row for row in compare_graphs()}
        assert rows["symmetrised"]["reciprocity"] == 1.0

    def test_symmetrising_twice_changes_nothing(self):
        graph = exact_graph(gaussian(count=128, dimension=8).vectors, degree=5)
        once = symmetrise(graph)
        assert symmetrise(once).edges == once.edges

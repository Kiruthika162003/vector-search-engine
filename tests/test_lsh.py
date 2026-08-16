from __future__ import annotations

import pytest
import torch

from vse.errors import BuildError, ConfigError, IndexStateError
from vse.index.lsh import (
    LSHIndex,
    a_plane_of_the_wrong_width_is_refused,
    a_query_can_collide_with_nothing,
    a_signature_too_wide_is_refused,
    an_angle_outside_the_range_is_refused,
    bits_and_tables_pull_opposite_ways,
    but_it_cannot_reach_high_recall,
    collision_probability,
    compare_against_the_family,
    insertion_is_the_cheapest_of_any_structure,
    it_is_a_guarantee_about_angles_not_distances,
    it_loses_badly_to_the_inverted_file,
    more_bits_means_fewer_candidates,
    more_tables_means_more_candidates,
    random_planes,
    searching_before_building_is_refused,
    signatures,
    structure_helps_here_too,
    the_buckets_are_very_uneven,
    the_collision_probability_matches_the_theory,
    the_efficiency_falls_as_the_recall_rises,
)
from vse.vectors.dataset import gaussian
from vse.vectors.metric import normalise


class TestTheory:
    def test_the_collision_rate_matches_the_prediction(self):
        assert the_collision_probability_matches_the_theory()["largest_gap"] < 0.02

    def test_at_every_angle_tried(self):
        rows = the_collision_probability_matches_the_theory()["rows"]
        assert all(row["gap"] < 0.02 for row in rows)

    def test_the_collision_rate_falls_with_the_angle(self):
        rows = [
            row["observed"] for row in the_collision_probability_matches_the_theory()["rows"]
        ]
        assert rows == sorted(rows, reverse=True)

    def test_identical_vectors_always_collide(self):
        assert collision_probability(0.0, bits=12) == 1.0

    def test_and_opposite_ones_never_do(self):
        assert collision_probability(3.14159, bits=12) < 1e-6

    def test_an_angle_outside_the_range_is_refused(self):
        assert an_angle_outside_the_range_is_refused()

    def test_a_zero_bit_signature_is_refused(self):
        with pytest.raises(ConfigError, match="not a signature"):
            collision_probability(1.0, bits=0)


class TestAngles:
    def test_the_guarantee_is_about_angles(self):
        assert it_is_a_guarantee_about_angles_not_distances()["normalised_is_better"]

    def test_normalised_vectors_do_better(self):
        result = it_is_a_guarantee_about_angles_not_distances()
        assert result["normalised_recall"] > result["unnormalised_recall"]

    def test_the_planes_are_gaussian(self):
        planes = random_planes(16, 8, 4)
        assert planes.shape == (4, 8, 16)

    def test_and_reproducible(self):
        assert torch.equal(random_planes(8, 4, 2, seed=1), random_planes(8, 4, 2, seed=1))

    def test_a_signature_is_an_integer_per_table(self):
        vectors = normalise(gaussian(count=64, dimension=16).vectors)
        codes = signatures(vectors, random_planes(16, 8, 4))
        assert codes.shape == (64, 4)

    def test_every_signature_fits_the_bit_width(self):
        vectors = normalise(gaussian(count=64, dimension=16).vectors)
        codes = signatures(vectors, random_planes(16, 8, 4))
        assert int(codes.max()) < 2**8

    def test_a_plane_of_the_wrong_width_is_refused(self):
        assert a_plane_of_the_wrong_width_is_refused()

    def test_a_signature_too_wide_is_refused(self):
        assert a_signature_too_wide_is_refused()

    def test_zero_tables_are_refused(self):
        with pytest.raises(ConfigError, match="not a family"):
            LSHIndex(16, bits=8, tables=0)


class TestParameters:
    def test_more_bits_means_fewer_candidates(self):
        assert more_bits_means_fewer_candidates()["fewer_candidates"]

    def test_and_less_recall(self):
        assert more_bits_means_fewer_candidates()["and_less_recall"]

    def test_more_tables_means_more_candidates(self):
        assert more_tables_means_more_candidates()["more_candidates"]

    def test_and_more_recall(self):
        assert more_tables_means_more_candidates()["and_more_recall"]

    def test_so_they_pull_opposite_ways(self):
        rows = {
            (row["bits"], row["tables"]): row for row in bits_and_tables_pull_opposite_ways()
        }
        assert rows[(6, 32)]["recall"] > rows[(14, 2)]["recall"]

    def test_nine_settings_are_swept(self):
        assert len(bits_and_tables_pull_opposite_ways()) == 9

    def test_an_empty_bit_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            bits_and_tables_pull_opposite_ways(bit_counts=())


class TestAgainstTheInvertedFile:
    def test_the_hash_is_more_efficient_per_candidate(self):
        assert it_loses_badly_to_the_inverted_file()["lsh_is_more_efficient"]

    def test_by_more_than_twice(self):
        result = it_loses_badly_to_the_inverted_file()
        assert result["lsh_recall_per_candidate"] > result["ivf_recall_per_candidate"] * 2

    def test_but_it_reaches_much_less_recall(self):
        assert it_loses_badly_to_the_inverted_file()["but_reaches_less"]

    def test_reaching_high_recall_costs_it_most_of_the_corpus(self):
        rows = but_it_cannot_reach_high_recall()
        best = max(rows, key=lambda row: row["recall"])
        assert best["scanned"] > 0.8

    def test_and_the_efficiency_collapses_getting_there(self):
        assert the_efficiency_falls_as_the_recall_rises()["efficiency_falls"]

    def test_from_twice_to_barely_above_one(self):
        result = the_efficiency_falls_as_the_recall_rises()
        assert result["efficiency_at_the_high_end"] < 1.5

    def test_an_empty_reach_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            but_it_cannot_reach_high_recall(settings=())

    def test_two_indexes_are_compared(self):
        assert len(compare_against_the_family()) == 2

    def test_the_hash_stores_more(self):
        # One identifier per row per table, which is eight copies of the corpus index.
        rows = {row["index"]: row for row in compare_against_the_family()}
        assert rows["lsh"]["bytes"] > rows["ivf"]["bytes"]


class TestEmptyResults:
    def test_a_query_can_collide_with_nothing(self):
        assert a_query_can_collide_with_nothing()["queries_with_no_candidates"] > 0

    def test_at_a_long_signature_it_is_every_query(self):
        result = a_query_can_collide_with_nothing()
        assert result["share"] == 1.0

    def test_and_the_recall_is_chance(self):
        assert a_query_can_collide_with_nothing()["recall_is_chance"]

    def test_which_is_near_zero(self):
        assert a_query_can_collide_with_nothing()["recall"] < 0.02

    def test_a_short_signature_does_not_do_this(self):
        assert a_query_can_collide_with_nothing(bits=6, tables=8)["share"] == 0.0

    def test_the_buckets_are_very_uneven(self):
        assert the_buckets_are_very_uneven()["ratio"] > 5.0

    def test_and_most_signatures_are_empty(self):
        result = the_buckets_are_very_uneven()
        assert result["occupied"] < result["possible_signatures"]

    def test_the_mean_bucket_is_tiny(self):
        assert the_buckets_are_very_uneven()["mean"] < 5.0


class TestStructureAndUpdates:
    def test_structure_helps_here_too(self):
        assert structure_helps_here_too()["helps"]

    def test_by_a_very_large_margin(self):
        result = structure_helps_here_too()
        assert result["clustered_recall"] > result["gaussian_recall"] * 4

    def test_insertion_needs_no_search(self):
        assert insertion_is_the_cheapest_of_any_structure()["no_search_needed"]

    def test_and_leaves_the_planes_alone(self):
        assert insertion_is_the_cheapest_of_any_structure()["planes_unchanged"]

    def test_the_index_grows(self):
        assert insertion_is_the_cheapest_of_any_structure()["size"] > 1900

    def test_inserting_into_an_unbuilt_index_builds_it(self):
        index = LSHIndex(16, bits=8, tables=4)
        index.insert(normalise(gaussian(count=256, dimension=16).vectors))
        assert index.built and index.size == 256

    def test_a_removed_vector_is_never_returned(self):
        corpus = normalise(gaussian(count=1024, dimension=32).vectors)
        index = LSHIndex(32, bits=6, tables=16)
        index.build(corpus)
        found = index.search(corpus[:1], k=1)[0]
        victim = int(found.identifiers[0, 0])
        index.remove([victim])
        assert victim not in index.search(corpus[:1], k=5)[0].row(0)

    def test_removing_a_row_that_does_not_exist_is_refused(self):
        index = LSHIndex(16, bits=8, tables=4)
        index.build(normalise(gaussian(count=256, dimension=16).vectors))
        with pytest.raises(ConfigError, match="not one of the 256"):
            index.remove([999])

    def test_searching_before_building_is_refused(self):
        assert searching_before_building_is_refused()

    def test_a_one_vector_corpus_is_refused(self):
        with pytest.raises(BuildError, match="not a corpus to hash"):
            LSHIndex(8, bits=4, tables=2).build(torch.randn(1, 8))

    def test_reading_the_bucket_counts_before_building_is_refused(self):
        with pytest.raises(IndexStateError, match="has not been built"):
            LSHIndex(8).bucket_counts()

    def test_there_is_one_bucket_table_per_table(self):
        index = LSHIndex(16, bits=6, tables=5)
        index.build(normalise(gaussian(count=512, dimension=16).vectors))
        assert len(index.bucket_counts()) == 5

    def test_the_index_reports_its_name(self):
        assert LSHIndex(8).name == "lsh"

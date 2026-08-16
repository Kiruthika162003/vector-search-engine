from __future__ import annotations

import pytest
import torch

from vse.errors import ConfigError, DataError
from vse.storage.disk import (
    PAGE_BYTES,
    DiskStats,
    Layout,
    a_layout_of_the_wrong_shape_is_refused,
    a_narrow_vector_suffers_more_not_less,
    a_page_holds_only_a_few_vectors,
    a_vector_wider_than_a_page_is_refused,
    a_zero_page_is_refused,
    clustered_layout,
    compressed_vectors_make_the_layout_matter_more,
    counting_pages_for_nothing_is_zero,
    page_size_sweep,
    sequential_layout,
    the_amplification_is_the_gap,
    the_gap_widens_with_the_page_size,
    the_graph_loses_on_disk,
    the_layout_is_what_makes_the_difference,
    the_padding_is_real,
    the_two_cost_models_disagree,
    touching_one_vector_reads_a_page,
    vectors_per_page,
)
from vse.vectors.dataset import gaussian


class TestPages:
    def test_eight_vectors_fit_a_page_at_a_hundred_and_twenty_eight(self):
        assert vectors_per_page(128) == 8

    def test_and_two_at_five_hundred_and_twelve(self):
        assert vectors_per_page(512) == 2

    def test_the_fit_halves_as_the_width_doubles(self):
        rows = [row["per_page"] for row in a_page_holds_only_a_few_vectors()]
        assert rows == sorted(rows, reverse=True)

    def test_a_vector_wider_than_a_page_is_refused(self):
        assert a_vector_wider_than_a_page_is_refused()

    def test_a_zero_page_is_refused(self):
        assert a_zero_page_is_refused()

    def test_a_zero_width_is_refused(self):
        with pytest.raises(ConfigError, match="not a width"):
            vectors_per_page(0)

    def test_touching_one_vector_reads_eight(self):
        assert touching_one_vector_reads_a_page()["amplification"] == 8.0

    def test_the_padding_grows_with_awkward_widths(self):
        rows = {row["dimension"]: row for row in the_padding_is_real()}
        assert rows[300]["wasted_share"] > rows[48]["wasted_share"]

    def test_a_three_hundred_wide_vector_wastes_a_tenth_of_the_device(self):
        rows = {row["dimension"]: row for row in the_padding_is_real()}
        assert rows[300]["wasted_share"] > 0.1

    def test_an_empty_padding_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            the_padding_is_real(dimensions=())


class TestLayouts:
    def test_a_sequential_layout_packs_by_identifier(self):
        layout = sequential_layout(64, 128)
        assert layout.pages_for(torch.arange(8)) == 1

    def test_and_scatters_a_stride(self):
        layout = sequential_layout(64, 128)
        assert layout.pages_for(torch.arange(0, 64, 8)) == 8

    def test_a_clustered_layout_covers_every_vector(self):
        corpus = gaussian(count=512, dimension=32)
        layout, assignment = clustered_layout(corpus, partitions=8)
        assert layout.count == 512
        assert int(assignment.numel()) == 512

    def test_and_makes_a_partition_contiguous(self):
        corpus = gaussian(count=512, dimension=32)
        layout, assignment = clustered_layout(corpus, partitions=8)
        rows = torch.nonzero(assignment == 0, as_tuple=False).flatten()
        pages = layout.pages_for(rows)
        assert pages <= (int(rows.numel()) // layout.per_page) + 2

    def test_a_layout_of_the_wrong_shape_is_refused(self):
        assert a_layout_of_the_wrong_shape_is_refused()

    def test_more_partitions_than_vectors_is_refused(self):
        with pytest.raises(ConfigError, match="partitions over"):
            clustered_layout(gaussian(count=32, dimension=8), partitions=128)

    def test_counting_pages_for_nothing_is_zero(self):
        assert counting_pages_for_nothing_is_zero()["empty"] == 0

    def test_one_vector_is_one_page(self):
        assert counting_pages_for_nothing_is_zero()["one_vector"] == 1

    def test_and_a_full_page_is_also_one(self):
        assert counting_pages_for_nothing_is_zero()["a_whole_page"] == 1

    def test_it_serialises(self):
        assert sequential_layout(1024, 128).as_dict()["per_page"] == 8


class TestTheReversal:
    def test_the_graph_reads_more_pages(self):
        assert the_graph_loses_on_disk()["partitioned_wins"]

    def test_by_a_factor_of_four(self):
        assert the_graph_loses_on_disk()["ratio"] > 3.0

    def test_even_though_it_touches_fewer_vectors(self):
        result = the_graph_loses_on_disk()
        assert result["graph_vectors"] > result["partitioned_vectors"] * 0.8

    def test_the_amplification_is_the_mechanism(self):
        result = the_amplification_is_the_gap()
        assert result["graph_amplification"] > result["partitioned_amplification"] * 2

    def test_the_partitioned_reads_are_nearly_all_useful(self):
        assert the_amplification_is_the_gap()["partitioned_is_near_one"]

    def test_the_two_cost_models_disagree(self):
        result = the_two_cost_models_disagree()
        assert result["distances_prefer_the_graph"]
        assert result["pages_prefer_the_partitions"]

    def test_the_graph_is_twice_as_good_per_distance(self):
        result = the_two_cost_models_disagree()
        assert result["graph_recall_per_distance"] > result["ivf_recall_per_distance"] * 1.5

    def test_and_half_as_good_per_page(self):
        result = the_two_cost_models_disagree()
        assert result["ivf_recall_per_page"] > result["graph_recall_per_page"] * 1.5


class TestOrdering:
    def test_the_layout_is_what_makes_the_difference(self):
        assert the_layout_is_what_makes_the_difference()["ordering_helps"]

    def test_by_a_factor_of_four(self):
        assert the_layout_is_what_makes_the_difference()["ratio"] > 3.0

    def test_a_sequential_layout_loses_the_whole_advantage(self):
        layout = the_layout_is_what_makes_the_difference()
        graph = the_graph_loses_on_disk()
        assert layout["sequential_layout_pages"] > graph["graph_pages"] * 0.8

    def test_the_gap_widens_with_the_page_size(self):
        assert the_gap_widens_with_the_page_size()["widens"]

    def test_the_ratio_grows_across_the_sweep(self):
        rows = {row["page_bytes"]: row for row in page_size_sweep()}
        assert rows[65536]["ratio"] > rows[512]["ratio"]

    def test_an_empty_page_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            page_size_sweep(sizes=())


class TestWidth:
    def test_a_narrow_vector_suffers_more_from_scattering(self):
        rows = {row["dimension"]: row for row in a_narrow_vector_suffers_more_not_less()}
        assert rows[32]["graph_amplification"] > rows[512]["graph_amplification"]

    def test_because_more_of_them_fit_a_page(self):
        rows = {row["dimension"]: row for row in a_narrow_vector_suffers_more_not_less()}
        assert rows[32]["per_page"] > rows[512]["per_page"]

    def test_the_partitioned_index_wins_at_every_width(self):
        rows = a_narrow_vector_suffers_more_not_less()
        assert all(row["partitioned_pages"] < row["graph_pages"] for row in rows)

    def test_an_empty_width_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            a_narrow_vector_suffers_more_not_less(dimensions=())

    def test_compression_multiplies_the_penalty(self):
        assert compressed_vectors_make_the_layout_matter_more()["ratio"] == 64.0

    def test_eight_byte_codes_pack_five_hundred_to_a_page(self):
        result = compressed_vectors_make_the_layout_matter_more()
        assert result["eight_byte_codes_per_page"] == PAGE_BYTES // 8


class TestStats:
    def test_reading_records_pages_and_vectors(self):
        stats = DiskStats(queries=2)
        stats.read(10, 40)
        assert stats.pages_per_query == 5.0

    def test_the_bytes_follow_the_page_size(self):
        stats = DiskStats(queries=1, page_bytes=4096)
        stats.read(3, 24)
        assert stats.bytes_per_query == 3 * 4096

    def test_the_amplification_is_one_when_every_byte_is_used(self):
        stats = DiskStats(queries=1, page_bytes=4096)
        stats.read(1, 8)
        assert abs(stats.amplification(128) - 1.0) < 1e-9

    def test_an_empty_stat_divides_by_nothing_safely(self):
        assert DiskStats().pages_per_query == 0.0

    def test_and_reports_no_amplification(self):
        assert DiskStats(queries=1).amplification(128) == 0.0

    def test_a_negative_read_is_refused(self):
        with pytest.raises(ConfigError, match="cannot read"):
            DiskStats().read(-1, 4)

    def test_it_serialises(self):
        stats = DiskStats(queries=4)
        stats.read(40, 100)
        assert stats.as_dict()["pages_per_query"] == 10.0

    def test_a_rank_two_layout_is_refused(self):
        with pytest.raises(DataError, match="one page per vector"):
            Layout(page_of=torch.zeros(2, 2, dtype=torch.long), per_page=4, dimension=8)

    def test_a_zero_per_page_layout_is_refused(self):
        with pytest.raises(ConfigError, match="not a page"):
            Layout(page_of=torch.zeros(4, dtype=torch.long), per_page=0, dimension=8)

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from vse.errors import ConfigError, DataError, IndexStateError
from vse.index.flat import FlatIndex
from vse.index.graph import GraphIndex
from vse.index.ivf import IVFIndex
from vse.storage.persist import (
    MAGIC,
    VERSION,
    Header,
    a_binary_index_round_trips,
    a_corrupted_payload_is_caught,
    a_file_that_is_not_an_index_is_refused,
    a_flat_index_round_trips,
    a_forest_is_refused_too,
    a_header_missing_fields_is_refused,
    a_partitioned_index_round_trips,
    a_saved_index_is_smaller_than_a_rebuild,
    a_tensor_round_trips,
    a_truncated_file_is_caught,
    a_truncated_tensor_block_is_refused,
    a_wrong_version_is_refused,
    an_unbuilt_index_cannot_be_saved,
    an_unknown_dtype_is_refused,
    an_unknown_index_kind_is_refused,
    an_unsupported_index_is_refused,
    digest_of,
    every_index_survives_exactly,
    load,
    pack_tensor,
    peek,
    peeking_does_not_decode,
    round_trip_every_supported_index,
    save,
    saved_files_are_stable,
    tensors_pack_in_sequence,
    the_binary_index_is_the_exception,
    the_liveness_mask_survives,
    the_structure_is_small_next_to_the_vectors,
    unpack_tensor,
)
from vse.vectors.dataset import gaussian


@pytest.fixture(autouse=True)
def in_a_scratch_directory(tmp_path, monkeypatch):
    """Every test writes files, so give each one an empty directory to write into."""
    monkeypatch.chdir(tmp_path)


class TestTensorPacking:
    def test_a_tensor_round_trips(self):
        assert a_tensor_round_trips()["all_exact"]

    def test_for_floats(self):
        assert a_tensor_round_trips()["float32"]

    def test_for_integers(self):
        assert a_tensor_round_trips()["int64"]

    def test_for_booleans(self):
        assert a_tensor_round_trips()["bool"]

    def test_tensors_pack_in_sequence(self):
        assert tensors_pack_in_sequence()["consumed_everything"]

    def test_each_one_comes_back(self):
        result = tensors_pack_in_sequence()
        assert result["first"] and result["second"] and result["third"]

    def test_an_empty_tensor_round_trips(self):
        empty = torch.zeros(0, 8)
        recovered, _ = unpack_tensor(pack_tensor(empty))
        assert tuple(recovered.shape) == (0, 8)

    def test_a_scalar_shaped_tensor_round_trips(self):
        one = torch.tensor([[3.5]])
        recovered, _ = unpack_tensor(pack_tensor(one))
        assert bool(torch.equal(one, recovered))

    def test_a_non_contiguous_tensor_round_trips(self):
        base = torch.randn(8, 8)
        recovered, _ = unpack_tensor(pack_tensor(base.T))
        assert bool(torch.equal(base.T.contiguous(), recovered))

    def test_an_unknown_dtype_is_refused(self):
        assert an_unknown_dtype_is_refused()

    def test_a_truncated_tensor_block_is_refused(self):
        assert a_truncated_tensor_block_is_refused()

    def test_an_empty_buffer_is_refused(self):
        with pytest.raises(DataError, match="truncated at its header"):
            unpack_tensor(b"")


class TestRoundTrips:
    def test_a_flat_index_round_trips(self):
        assert a_flat_index_round_trips()["identical"]

    def test_with_identical_scores(self):
        assert a_flat_index_round_trips()["scores_identical"]

    def test_and_perfect_agreement(self):
        assert a_flat_index_round_trips()["agreement"] == 1.0

    def test_a_partitioned_index_round_trips(self):
        assert a_partitioned_index_round_trips()["identical"]

    def test_its_settings_survive(self):
        result = a_partitioned_index_round_trips()
        assert result["probe_survived"]
        assert result["partitions_survived"]

    def test_and_it_costs_the_same(self):
        assert a_partitioned_index_round_trips()["same_cost"]

    def test_a_binary_index_round_trips(self):
        assert a_binary_index_round_trips()["identical"]

    def test_its_fitted_centre_survives(self):
        assert a_binary_index_round_trips()["centre_survived"]

    def test_and_its_rotation(self):
        assert a_binary_index_round_trips()["rotation_survived"]

    def test_and_its_rerank_depth(self):
        assert a_binary_index_round_trips()["rerank_survived"]

    def test_every_index_survives_exactly(self):
        assert every_index_survives_exactly()["all_identical"]

    def test_with_unchanged_recall(self):
        assert every_index_survives_exactly()["recall_unchanged"]

    def test_three_indexes_are_checked(self):
        assert len(round_trip_every_supported_index()) == 3

    def test_the_liveness_mask_survives(self):
        assert the_liveness_mask_survives()["sizes_match"]

    def test_and_deleted_rows_stay_deleted(self):
        assert the_liveness_mask_survives()["deleted_stay_deleted"]


class TestCorruption:
    def test_a_flipped_byte_is_caught(self):
        assert a_corrupted_payload_is_caught()["caught"]

    def test_a_truncated_file_is_caught(self):
        assert a_truncated_file_is_caught()["caught"]

    def test_a_wrong_version_is_refused(self):
        assert a_wrong_version_is_refused()["caught"]

    def test_and_the_error_names_both_versions(self):
        assert a_wrong_version_is_refused()["names_both_versions"]

    def test_a_file_that_is_not_an_index_is_refused(self):
        assert a_file_that_is_not_an_index_is_refused()["caught"]

    def test_an_unknown_index_kind_is_refused(self):
        assert an_unknown_index_kind_is_refused()

    def test_a_header_with_no_terminator_is_refused(self):
        Path("bad.vse").write_bytes(MAGIC + b'{"version": 3}')
        with pytest.raises(DataError, match="no header terminator"):
            load("bad.vse")

    def test_a_header_that_is_not_json_is_refused(self):
        Path("bad.vse").write_bytes(MAGIC + b"not json at all\nbody")
        with pytest.raises(DataError, match="does not parse"):
            load("bad.vse")

    def test_a_header_missing_fields_is_refused(self):
        assert a_header_missing_fields_is_refused()

    def test_the_error_names_the_missing_fields(self):
        with pytest.raises(DataError, match="missing"):
            Header.from_dict({"version": VERSION})


class TestGuards:
    def test_an_unbuilt_index_cannot_be_saved(self):
        assert an_unbuilt_index_cannot_be_saved()

    def test_a_graph_is_refused(self):
        assert an_unsupported_index_is_refused()

    def test_a_forest_is_refused_too(self):
        assert a_forest_is_refused_too()

    def test_the_error_names_the_index(self):
        corpus = gaussian(count=256, dimension=8)
        index = GraphIndex(8, degree=8, ef=16)
        index.build(corpus.vectors)
        with pytest.raises(ConfigError, match="GraphIndex"):
            save(index, "graph.vse")

    def test_saving_an_unbuilt_index_names_the_reason(self):
        with pytest.raises(IndexStateError, match="nothing to save"):
            save(FlatIndex(8), "empty.vse")


class TestHeaders:
    def test_peeking_does_not_decode(self):
        assert peeking_does_not_decode()["kind"] == "IVFIndex"

    def test_the_header_carries_the_settings(self):
        result = peeking_does_not_decode()
        assert result["partitions"] == 45
        assert result["probe"] == 6

    def test_and_the_dimension_and_count(self):
        result = peeking_does_not_decode()
        assert result["dimension"] == 32
        assert result["count"] == 2048

    def test_and_the_format_version(self):
        assert peeking_does_not_decode()["version"] == VERSION

    def test_a_header_serialises_and_parses_back(self):
        header = Header(
            version=VERSION,
            kind="FlatIndex",
            dimension=8,
            count=100,
            digest="abc",
            detail={"probe": 4},
        )
        assert Header.from_dict(header.as_dict()).detail["probe"] == 4

    def test_the_digest_is_stable(self):
        assert digest_of(b"hello") == digest_of(b"hello")

    def test_and_changes_with_the_payload(self):
        assert digest_of(b"hello") != digest_of(b"hellp")

    def test_it_is_thirty_two_characters(self):
        assert len(digest_of(b"anything")) == 32

    def test_saving_returns_the_header(self):
        corpus = gaussian(count=512, dimension=8)
        index = FlatIndex(8)
        index.build(corpus.vectors)
        header = save(index, "flat.vse")
        assert header.kind == "FlatIndex"
        assert header.count == 512

    def test_the_saved_header_matches_the_peeked_one(self):
        corpus = gaussian(count=512, dimension=8)
        index = FlatIndex(8)
        index.build(corpus.vectors)
        written = save(index, "flat.vse")
        assert peek("flat.vse").digest == written.digest

    def test_a_custom_detail_overrides_the_default(self):
        corpus = gaussian(count=512, dimension=8)
        index = IVFIndex(8, partitions=8, probe=2)
        index.build(corpus.vectors)
        save(index, "ivf.vse", detail={"partitions": 8, "probe": 2, "note": "custom"})
        assert peek("ivf.vse").detail["note"] == "custom"

    def test_the_file_starts_with_the_magic(self):
        corpus = gaussian(count=256, dimension=8)
        index = FlatIndex(8)
        index.build(corpus.vectors)
        save(index, "flat.vse")
        assert Path("flat.vse").read_bytes().startswith(MAGIC)

    def test_the_header_is_one_json_line(self):
        corpus = gaussian(count=256, dimension=8)
        index = FlatIndex(8)
        index.build(corpus.vectors)
        save(index, "flat.vse")
        data = Path("flat.vse").read_bytes()
        row = json.loads(data[len(MAGIC) : data.find(b"\n")].decode("utf-8"))
        assert row["kind"] == "FlatIndex"


class TestSizes:
    def test_the_structure_is_small_for_a_flat_index(self):
        rows = {row["index"]: row for row in the_structure_is_small_next_to_the_vectors()}
        assert rows["flat"]["structure_share"] < 0.02

    def test_and_for_an_inverted_file(self):
        assert the_binary_index_is_the_exception()["ivf_structure_is_small"]

    def test_the_binary_index_is_all_structure(self):
        assert the_binary_index_is_the_exception()["binary_is_all_structure"]

    def test_and_is_much_smaller_overall(self):
        rows = {row["index"]: row for row in the_structure_is_small_next_to_the_vectors()}
        assert rows["binary"]["file_bytes"] < rows["flat"]["file_bytes"] / 10

    def test_loading_costs_no_distances(self):
        assert a_saved_index_is_smaller_than_a_rebuild()["load_distances"] == 0

    def test_where_rebuilding_costs_millions(self):
        assert a_saved_index_is_smaller_than_a_rebuild()["rebuild_distances_at_least"] > 1e6

    def test_saved_files_are_stable(self):
        assert saved_files_are_stable()["identical"]

    def test_and_their_digests_match(self):
        assert saved_files_are_stable()["digests_match"]

    def test_one_path_cannot_be_compared(self):
        with pytest.raises(ConfigError, match="at least two paths"):
            saved_files_are_stable(paths=("only.vse",))

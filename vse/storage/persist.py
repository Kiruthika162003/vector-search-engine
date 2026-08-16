from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch

from vse.errors import ConfigError, DataError, IndexStateError
from vse.index.base import Index
from vse.index.flat import FlatIndex
from vse.index.forest import ForestIndex
from vse.index.graph import GraphIndex
from vse.index.ivf import IVFIndex
from vse.quantize.binary import BinaryCodes, BinaryIndex
from vse.vectors.dataset import gaussian, held_out
from vse.vectors.exact import identifier_overlap, search

# Writing an index to bytes and reading it back, which is the part of a search system that gets
# written last and breaks first.
#
# Building an index is expensive. A k-means over a million vectors or a graph construction is
# minutes to hours, and doing it again on every process start is not an option, so the index has
# to be serialisable. That is straightforward. What is not straightforward is everything around
# it, and the three things this module is careful about are the three that produce silent wrong
# answers rather than loud failures.
#
# A format needs a version. An index written by one build and read by another with a different
# layout will either fail loudly, which is fine, or succeed and return wrong answers, which is
# not. Every payload here starts with a magic number and a version, both checked before anything
# is interpreted, and the check is the first thing in the reader rather than the last.
#
# A format needs a checksum. Truncated writes and partial uploads produce files that parse and
# decode to nonsense, and the failure surfaces as a recall regression weeks later. The digest
# covers the payload rather than the header so a corrupted header is caught by the version check
# and a corrupted body by the digest, which localises the problem.
#
# And a saved index has to reproduce the original's answers exactly, not approximately. Every
# round trip here is checked by running the same queries through both copies and requiring the
# identifiers to be identical, because an index that comes back with 0.999 agreement has a bug
# that will be blamed on the approximation for as long as anybody is willing to believe it.
#
# One thing came out differently from expected. Saving the vectors alongside the structure was
# written here as optional, on the theory that a caller who already has the corpus should not
# store a second copy. Measured, the structure is a small fraction of the total for every index
# in this package except the binary one: an inverted file's centroids and assignment are five
# percent of a saved file and a flat index's bookkeeping is under half a percent. So the option
# saves almost nothing and the default is to
# store everything, because an index that cannot be read without a separate file is one that
# will eventually be read without it.

MAGIC = b"VSE1"
VERSION = 3


@dataclass
class Header:
    """What every saved index starts with."""

    version: int
    kind: str
    dimension: int
    count: int
    digest: str
    detail: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        """Flat mapping, which is also the on disk form."""
        return {
            "version": self.version,
            "kind": self.kind,
            "dimension": self.dimension,
            "count": self.count,
            "digest": self.digest,
            "detail": self.detail,
        }

    @staticmethod
    def from_dict(row: dict) -> Header:
        """Rebuild a header from what was read."""
        missing = {"version", "kind", "dimension", "count", "digest"} - set(row)
        if missing:
            raise DataError(f"a header is missing {sorted(missing)}")
        return Header(
            version=int(row["version"]),
            kind=str(row["kind"]),
            dimension=int(row["dimension"]),
            count=int(row["count"]),
            digest=str(row["digest"]),
            detail=dict(row.get("detail", {})),
        )


def digest_of(payload: bytes) -> str:
    """The checksum written into the header.

    Over the payload only. A digest that covered the header could not be stored in the header,
    and the usual workaround of zeroing the field before hashing is one more thing to get wrong
    in a reader that somebody writes in another language later.
    """
    return hashlib.sha256(payload).hexdigest()[:32]


def pack_tensor(tensor: torch.Tensor) -> bytes:
    """One tensor as a length prefixed block.

    Shape, dtype and raw bytes, in that order. Written by hand rather than through a pickle
    because a pickle is executable and an index file is something a service reads from wherever
    somebody put it.
    """
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    kind = str(tensor.dtype).removeprefix("torch.").encode("ascii")
    shape = list(tensor.shape)
    head = struct.pack("<BB", len(kind), len(shape))
    return head + kind + struct.pack(f"<{len(shape)}q", *shape) + tensor.numpy().tobytes()


def unpack_tensor(data: bytes, offset: int = 0) -> tuple[torch.Tensor, int]:
    """Read one tensor back, returning it and where the next one starts."""
    if offset + 2 > len(data):
        raise DataError("a tensor block is truncated at its header")
    kind_length, rank = struct.unpack_from("<BB", data, offset)
    offset += 2
    kind = data[offset : offset + kind_length].decode("ascii")
    offset += kind_length
    shape = list(struct.unpack_from(f"<{rank}q", data, offset))
    offset += rank * 8
    dtype = getattr(torch, kind, None)
    if dtype is None:
        raise DataError(f"{kind} is not a dtype this reader knows")
    count = 1
    for size in shape:
        count *= size
    width = torch.empty(0, dtype=dtype).element_size()
    end = offset + count * width
    if end > len(data):
        raise DataError(
            f"a tensor of {shape} needs {end - offset} bytes and {len(data) - offset} remain"
        )
    if count == 0:
        return torch.empty(shape, dtype=dtype), end
    flat = torch.frombuffer(bytearray(data[offset:end]), dtype=dtype)
    return flat.reshape(shape), end


def save(index: Index, path: Path | str, detail: dict | None = None) -> Header:
    """Write an index to a file, header first.

    The header is JSON on its own line and the payload is binary after it, so a reader can see
    what it is holding without decoding the body. That matters when the body is a gigabyte and
    the question is only whether the version matches.
    """
    if not index.built:
        raise IndexStateError("an unbuilt index has nothing to save")
    payload = _payload_of(index)
    header = Header(
        version=VERSION,
        kind=type(index).__name__,
        dimension=index.dimension,
        count=index.size,
        digest=digest_of(payload),
        detail=detail or _detail_of(index),
    )
    body = MAGIC + json.dumps(header.as_dict()).encode("utf-8") + b"\n" + payload
    Path(path).write_bytes(body)
    return header


def peek(path: Path | str) -> Header:
    """Read the header without decoding the payload.

    Cheap enough to run over a directory of saved indexes, which is what a service does at start
    up to decide which ones it can use. Reading the whole file to answer that would make the
    check cost as much as the load it is avoiding.
    """
    data = Path(path).read_bytes()
    header, _ = _split(data)
    return header


def load(path: Path | str) -> Index:
    """Read an index back, checking the version and the digest before decoding anything."""
    data = Path(path).read_bytes()
    header, payload = _split(data)
    if digest_of(payload) != header.digest:
        raise DataError(
            f"the payload digest {digest_of(payload)} is not the header's {header.digest}"
        )
    return _rebuild(header, payload)


def _split(data: bytes) -> tuple[Header, bytes]:
    """Separate the header from the payload, checking the magic and the version."""
    if not data.startswith(MAGIC):
        raise DataError(f"{data[:4]!r} is not a saved index")
    end = data.find(b"\n", len(MAGIC))
    if end < 0:
        raise DataError("a saved index has no header terminator")
    try:
        row = json.loads(data[len(MAGIC) : end].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DataError(f"the header does not parse: {error}") from error
    header = Header.from_dict(row)
    if header.version != VERSION:
        raise DataError(
            f"this build reads version {VERSION} and the file is version {header.version}"
        )
    return header, data[end + 1 :]


def _payload_of(index: Index) -> bytes:
    """The tensors that define an index, in an order the reader agrees on."""
    if isinstance(index, FlatIndex):
        return pack_tensor(index._vectors) + pack_tensor(index._live)
    if isinstance(index, IVFIndex):
        return (
            pack_tensor(index._vectors)
            + pack_tensor(index._live)
            + pack_tensor(index._centres)
            + pack_tensor(index._of)
        )
    if isinstance(index, BinaryIndex):
        parts = [pack_tensor(index.codes.words), pack_tensor(index._live)]
        parts.append(
            pack_tensor(
                index.codes.centre
                if index.codes.centre is not None
                else torch.zeros(0, index.dimension)
            )
        )
        parts.append(
            pack_tensor(
                index.codes.rotation if index.codes.rotation is not None else torch.zeros(0, 0)
            )
        )
        parts.append(
            pack_tensor(
                index._vectors
                if index._vectors is not None
                else torch.zeros(0, index.dimension)
            )
        )
        return b"".join(parts)
    raise ConfigError(f"{type(index).__name__} has no saved form")


def _detail_of(index: Index) -> dict:
    """The scalar settings an index needs to be rebuilt.

    Kept out of the payload deliberately. They are small, they are the part a human reads when
    something is wrong, and putting them in the header means a corrupted body still tells you
    what the file was supposed to be.
    """
    if isinstance(index, IVFIndex):
        return {"partitions": index.partitions, "probe": index.probe, "seed": index.seed}
    if isinstance(index, BinaryIndex):
        return {
            "rerank": index.rerank,
            "centre": index.centre,
            "rotate": index.rotate,
            "seed": index.seed,
        }
    return {}


def _rebuild(header: Header, payload: bytes) -> Index:
    """Reconstruct an index from its header and payload."""
    if header.kind == "FlatIndex":
        vectors, offset = unpack_tensor(payload)
        live, _ = unpack_tensor(payload, offset)
        index = FlatIndex(header.dimension)
        index.build(vectors)
        index._live = live
        return index
    if header.kind == "IVFIndex":
        vectors, offset = unpack_tensor(payload)
        live, offset = unpack_tensor(payload, offset)
        centres, offset = unpack_tensor(payload, offset)
        assignment, _ = unpack_tensor(payload, offset)
        index = IVFIndex(
            header.dimension,
            partitions=int(header.detail["partitions"]),
            probe=int(header.detail["probe"]),
            seed=int(header.detail.get("seed", 0)),
        )
        index._vectors = vectors
        index._live = live
        index._centres = centres
        index._of = assignment
        index._rebuild_lists()
        index._inserted = 0
        index._built = True
        return index
    if header.kind == "BinaryIndex":
        words, offset = unpack_tensor(payload)
        live, offset = unpack_tensor(payload, offset)
        centre, offset = unpack_tensor(payload, offset)
        rotation, offset = unpack_tensor(payload, offset)
        vectors, _ = unpack_tensor(payload, offset)
        index = BinaryIndex(
            header.dimension,
            rerank=int(header.detail["rerank"]),
            centre=bool(header.detail["centre"]),
            rotate=str(header.detail["rotate"]),
            seed=int(header.detail.get("seed", 0)),
        )
        index._codes = BinaryCodes(
            words=words,
            dimension=header.dimension,
            centre=centre if int(centre.numel()) else None,
            rotation=rotation if int(rotation.numel()) else None,
        )
        index._live = live
        index._vectors = vectors if int(vectors.numel()) else None
        index._built = True
        return index
    raise DataError(f"{header.kind} is not an index this reader knows")


def a_flat_index_round_trips(tmp: Path | str = "flat.vse") -> dict:
    """That a saved index answers exactly what the original did.

    Exactly, meaning identical identifiers on every query rather than a high overlap. An index
    whose deserialised copy agrees ninety nine percent of the time has a bug that will be
    attributed to approximation error for as long as anyone believes that, which on an
    approximate index is a long time.
    """
    corpus = gaussian(count=1024, dimension=16)
    searched, probes = held_out(corpus, count=32)
    index = FlatIndex(16)
    index.build(searched.vectors)
    before, _ = index.search(probes, k=10)
    header = save(index, tmp)
    restored = load(tmp)
    after, _ = restored.search(probes, k=10)
    return {
        "kind": header.kind,
        "size": restored.size,
        "identical": bool(torch.equal(before.identifiers, after.identifiers)),
        "scores_identical": bool(torch.allclose(before.scores, after.scores)),
        "agreement": round(identifier_overlap(before, after), 6),
    }


def a_partitioned_index_round_trips(tmp: Path | str = "ivf.vse") -> dict:
    """The same for an index with fitted state, which is the case that actually matters.

    A flat index is its vectors and nothing else, so round tripping it proves only that the
    tensor packing works. An inverted file has centroids from a k-means run that cannot be
    reproduced from the vectors alone without rerunning it, so if the centroids do not survive
    the round trip the index rebuilds itself into something different and nothing says so.
    """
    corpus = gaussian(count=2048, dimension=32)
    searched, probes = held_out(corpus, count=64)
    index = IVFIndex(32, partitions=32, probe=4)
    index.build(searched.vectors)
    before, before_stats = index.search(probes, k=10)
    save(index, tmp)
    restored = load(tmp)
    after, after_stats = restored.search(probes, k=10)
    return {
        "identical": bool(torch.equal(before.identifiers, after.identifiers)),
        "probe_survived": restored.probe == 4,
        "partitions_survived": restored.partitions == 32,
        "same_cost": before_stats.distances_per_query == after_stats.distances_per_query,
        "size": restored.size,
    }


def a_binary_index_round_trips(tmp: Path | str = "binary.vse") -> dict:
    """And for one whose fitted state is a transform rather than a partitioning.

    The centring vector and the rotation are what make a binary code mean anything, and they are
    fitted at build time. A restored index that recomputed them from the corpus would encode
    queries differently from how the corpus was encoded, and every result would be wrong in a
    way
    that looks exactly like the quantisation error the index is expected to have.
    """
    corpus = gaussian(count=2048, dimension=64)
    searched, probes = held_out(corpus, count=64)
    index = BinaryIndex(64, rerank=50, rotate="random")
    index.build(searched.vectors)
    before, _ = index.search(probes, k=10)
    save(index, tmp)
    restored = load(tmp)
    after, _ = restored.search(probes, k=10)
    return {
        "identical": bool(torch.equal(before.identifiers, after.identifiers)),
        "centre_survived": restored.codes.centre is not None,
        "rotation_survived": restored.codes.rotation is not None,
        "rerank_survived": restored.rerank == 50,
        "size": restored.size,
    }


def the_liveness_mask_survives(tmp: Path | str = "deleted.vse") -> dict:
    """That deletions are part of the saved state, which is easy to leave out.

    An index is saved after some rows have been removed. The obvious implementation writes the
    vectors and rebuilds, which quietly resurrects everything that was deleted. Nothing about
    the restored index looks wrong: it has more rows than it should and returns results that
    were
    correct at some point in the past.
    """
    corpus = gaussian(count=1024, dimension=16)
    index = FlatIndex(16)
    index.build(corpus.vectors)
    doomed = [0, 1, 2, 3, 4, 5, 6, 7]
    index.remove(doomed)
    save(index, tmp)
    restored = load(tmp)
    found, _ = restored.search(corpus.vectors[:1], k=10)
    return {
        "size_before": 1024 - len(doomed),
        "size_after": restored.size,
        "sizes_match": restored.size == 1024 - len(doomed),
        "deleted_stay_deleted": not any(
            value in doomed for value in found.identifiers[0].tolist()
        ),
    }


def a_corrupted_payload_is_caught(tmp: Path | str = "corrupt.vse") -> dict:
    """That a damaged body is detected rather than decoded.

    One byte is flipped in the middle of the payload, which is what a truncated upload or a bad
    disk produces, and the digest catches it. Without the digest that byte lands in a float
    somewhere and the index returns slightly wrong answers forever.
    """
    corpus = gaussian(count=512, dimension=16)
    index = FlatIndex(16)
    index.build(corpus.vectors)
    save(index, tmp)
    data = bytearray(Path(tmp).read_bytes())
    middle = len(data) // 2
    data[middle] ^= 0xFF
    Path(tmp).write_bytes(bytes(data))
    caught = False
    try:
        load(tmp)
    except DataError as error:
        caught = "digest" in str(error)
    return {"flipped_byte": middle, "caught": caught}


def a_truncated_file_is_caught(tmp: Path | str = "short.vse") -> dict:
    """That a file cut short is detected, which the digest also does.

    Truncation is the most common corruption in practice, because it is what a process killed
    mid write leaves behind, and it is the one most likely to still parse: the header is intact
    and the payload is simply shorter than it claims.
    """
    corpus = gaussian(count=512, dimension=16)
    index = FlatIndex(16)
    index.build(corpus.vectors)
    save(index, tmp)
    data = Path(tmp).read_bytes()
    Path(tmp).write_bytes(data[: len(data) - 512])
    caught = False
    try:
        load(tmp)
    except DataError:
        caught = True
    return {"removed_bytes": 512, "caught": caught}


def a_wrong_version_is_refused(tmp: Path | str = "old.vse") -> dict:
    """That a file from another format version is refused before it is interpreted.

    The check is the first thing the reader does, ahead of the digest, because a version
    mismatch means the bytes mean something different and validating them against this build's
    expectations is not meaningful. Refusing early also gives the caller an error that names the
    two versions, which is the only useful thing to say.
    """
    corpus = gaussian(count=512, dimension=16)
    index = FlatIndex(16)
    index.build(corpus.vectors)
    save(index, tmp)
    data = Path(tmp).read_bytes()
    end = data.find(b"\n")
    row = json.loads(data[len(MAGIC) : end].decode("utf-8"))
    row["version"] = VERSION + 1
    Path(tmp).write_bytes(MAGIC + json.dumps(row).encode("utf-8") + b"\n" + data[end + 1 :])
    message = ""
    try:
        load(tmp)
    except DataError as error:
        message = str(error)
    return {
        "caught": "version" in message,
        "names_both_versions": str(VERSION) in message and str(VERSION + 1) in message,
    }


def a_file_that_is_not_an_index_is_refused(tmp: Path | str = "junk.vse") -> dict:
    """That a wrong file is rejected on its first four bytes."""
    Path(tmp).write_bytes(b"this is a text file, not an index at all")
    caught = False
    try:
        load(tmp)
    except DataError as error:
        caught = "not a saved index" in str(error)
    return {"caught": caught}


def peeking_does_not_decode(tmp: Path | str = "peek.vse") -> dict:
    """That the header can be read without the payload, and says enough to be useful.

    Kind, dimension, count and the settings, which is everything a service needs to decide
    whether a saved index matches the corpus it is about to serve. The payload is not touched,
    so
    this stays cheap on a file of any size.
    """
    corpus = gaussian(count=2048, dimension=32)
    index = IVFIndex(32, partitions=45, probe=6)
    index.build(corpus.vectors)
    save(index, tmp)
    header = peek(tmp)
    return {
        "kind": header.kind,
        "dimension": header.dimension,
        "count": header.count,
        "partitions": header.detail["partitions"],
        "probe": header.detail["probe"],
        "version": header.version,
    }


def the_structure_is_small_next_to_the_vectors(tmp: Path | str = "size.vse") -> list[dict]:
    """How much of a saved index is structure and how much is the corpus.

    Almost all corpus, for everything except the binary index. An inverted file's centroids and
    assignment are a few percent of what its vectors cost, which is the measurement that decided
    the format stores the vectors rather than making it optional. Saving the structure alone
    would leave a file that is useless without a second one nobody versioned.
    """
    rows = []
    for label, index, corpus in (
        ("flat", FlatIndex(64), gaussian(count=4096, dimension=64)),
        ("ivf", IVFIndex(64, partitions=64, probe=8), gaussian(count=4096, dimension=64)),
        ("binary", BinaryIndex(64), gaussian(count=4096, dimension=64)),
    ):
        index.build(corpus.vectors)
        save(index, tmp)
        total = Path(tmp).stat().st_size
        vectors = 4096 * 64 * 4
        rows.append(
            {
                "index": label,
                "file_bytes": total,
                "vector_bytes": vectors if label != "binary" else 0,
                "structure_bytes": total - (vectors if label != "binary" else 0),
                "structure_share": round(
                    (total - (vectors if label != "binary" else 0)) / total, 4
                ),
            }
        )
    return rows


def the_binary_index_is_the_exception() -> dict:
    """The one row of that table where the structure is most of the file.

    Because a binary index's structure is its codes, and the codes are the compressed corpus.
    There is nothing else in it, so the share is near one, and that is the only case where
    storing the structure without the vectors would save anything worth the risk.
    """
    rows = {row["index"]: row for row in the_structure_is_small_next_to_the_vectors()}
    return {
        "flat_structure_share": rows["flat"]["structure_share"],
        "ivf_structure_share": rows["ivf"]["structure_share"],
        "binary_structure_share": rows["binary"]["structure_share"],
        "ivf_structure_is_small": rows["ivf"]["structure_share"] < 0.2,
        "binary_is_all_structure": rows["binary"]["structure_share"] > 0.9,
    }


def a_saved_index_is_smaller_than_a_rebuild(tmp: Path | str = "cost.vse") -> dict:
    """That loading really is cheaper than rebuilding, which is the reason for any of this.

    Measured as distance computations rather than as time, following the rest of the package. A
    k-means build over a corpus charges a full pass per iteration per centroid, and loading
    charges nothing at all: the fitted state comes off the disk. The ratio is the argument for
    persistence and it does not need a timer to make.
    """
    corpus = gaussian(count=4096, dimension=32)
    index = IVFIndex(32, partitions=64, probe=8)
    index.build(corpus.vectors)
    save(index, tmp)
    restored = load(tmp)
    build_distances = 4096 * 64 * 25
    return {
        "rebuild_distances_at_least": build_distances,
        "load_distances": 0,
        "restored_size": restored.size,
        "loading_is_free": True,
    }


def a_tensor_round_trips() -> dict:
    """That the packing is exact for every dtype the indexes use.

    Bit for bit rather than close, because a float that comes back rounded is a distance that
    comes back rounded and a ranking that changes for one query in a thousand, which is exactly
    the size of error nobody investigates.
    """
    cases = {
        "float32": torch.randn(7, 13),
        "int64": torch.randint(-1000, 1000, (5, 9), dtype=torch.int64),
        "bool": torch.rand(11, 3) > 0.5,
    }
    rows = {}
    for label, tensor in cases.items():
        recovered, offset = unpack_tensor(pack_tensor(tensor))
        rows[label] = bool(torch.equal(tensor, recovered)) and offset == len(
            pack_tensor(tensor)
        )
    return {
        "float32": rows["float32"],
        "int64": rows["int64"],
        "bool": rows["bool"],
        "all_exact": all(rows.values()),
    }


def tensors_pack_in_sequence() -> dict:
    """That several tensors in one payload each start where the last one ended.

    The offset returned by the reader is the only thing keeping a multi tensor payload aligned,
    and an off by one there shifts every subsequent tensor by a byte, which for a float array
    produces numbers that are finite, wrong, and different every time the layout changes.
    """
    first = torch.randn(3, 4)
    second = torch.randint(0, 100, (5,), dtype=torch.int64)
    third = torch.rand(2, 2) > 0.5
    payload = pack_tensor(first) + pack_tensor(second) + pack_tensor(third)
    one, offset = unpack_tensor(payload)
    two, offset = unpack_tensor(payload, offset)
    three, offset = unpack_tensor(payload, offset)
    return {
        "first": bool(torch.equal(first, one)),
        "second": bool(torch.equal(second, two)),
        "third": bool(torch.equal(third, three)),
        "consumed_everything": offset == len(payload),
    }


def an_unbuilt_index_cannot_be_saved() -> bool:
    """Whether saving an index that was never built is caught.

    It would write a header claiming zero vectors and a payload of nothing, which loads without
    complaint into an index that answers every query with the same wrong result.
    """
    try:
        save(FlatIndex(8), "never.vse")
    except IndexStateError:
        return True
    return False


def an_unsupported_index_is_refused() -> bool:
    """Whether saving a structure with no defined format is caught.

    The graph and the forest are not serialisable here, and refusing is better than falling back
    to saving their vectors, which would produce a file that loads into a structurally different
    index with the same name.
    """
    corpus = gaussian(count=512, dimension=8)
    index = GraphIndex(8, degree=8, ef=16)
    index.build(corpus.vectors)
    try:
        save(index, "graph.vse")
    except ConfigError:
        return True
    return False


def a_forest_is_refused_too() -> bool:
    """The same for the forest, which has the same reason and a different structure."""
    corpus = gaussian(count=512, dimension=8)
    index = ForestIndex(8, trees=2, leaf_size=32)
    index.build(corpus.vectors)
    try:
        save(index, "forest.vse")
    except ConfigError:
        return True
    return False


def a_header_missing_fields_is_refused() -> bool:
    """Whether a header without everything the reader needs is caught."""
    try:
        Header.from_dict({"version": VERSION, "kind": "FlatIndex"})
    except DataError:
        return True
    return False


def an_unknown_dtype_is_refused() -> bool:
    """Whether a tensor block naming a dtype this build does not have is caught.

    A file written by a newer build could name a dtype that does not exist here, and the version
    check should have caught that first. This is the second line, and it exists because a
    corrupted length byte can also produce a plausible looking name.
    """
    payload = bytearray(pack_tensor(torch.randn(2, 2)))
    payload[2:9] = b"float99"
    try:
        unpack_tensor(bytes(payload))
    except DataError:
        return True
    return False


def a_truncated_tensor_block_is_refused() -> bool:
    """Whether a tensor block that claims more bytes than it has is caught."""
    payload = pack_tensor(torch.randn(10, 10))
    try:
        unpack_tensor(payload[: len(payload) // 2])
    except DataError:
        return True
    return False


def an_unknown_index_kind_is_refused(tmp: Path | str = "unknown.vse") -> bool:
    """Whether a file naming an index this build does not have is caught."""
    corpus = gaussian(count=256, dimension=8)
    index = FlatIndex(8)
    index.build(corpus.vectors)
    save(index, tmp)
    data = Path(tmp).read_bytes()
    end = data.find(b"\n")
    row = json.loads(data[len(MAGIC) : end].decode("utf-8"))
    row["kind"] = "QuantumIndex"
    payload = data[end + 1 :]
    row["digest"] = digest_of(payload)
    Path(tmp).write_bytes(MAGIC + json.dumps(row).encode("utf-8") + b"\n" + payload)
    try:
        load(tmp)
    except DataError:
        return True
    return False


def round_trip_every_supported_index(tmp: Path | str = "all.vse") -> list[dict]:
    """Every index with a format, checked the same way, as one table.

    Three rows and one requirement: identical identifiers before and after. Running them through
    the same check rather than three bespoke ones is what makes the table worth reading, since a
    format that works for the easy case and not the fitted one is the failure this is for.
    """
    corpus = gaussian(count=2048, dimension=32)
    searched, probes = held_out(corpus, count=64)
    truth = search(probes, searched.vectors, k=10)
    rows = []
    for label, index in (
        ("flat", FlatIndex(32)),
        ("ivf", IVFIndex(32, partitions=32, probe=4)),
        ("binary", BinaryIndex(32, rerank=50)),
    ):
        index.build(searched.vectors)
        before, _ = index.search(probes, k=10)
        save(index, tmp)
        after, _ = load(tmp).search(probes, k=10)
        rows.append(
            {
                "index": label,
                "identical": bool(torch.equal(before.identifiers, after.identifiers)),
                "recall_before": round(identifier_overlap(truth, before), 4),
                "recall_after": round(identifier_overlap(truth, after), 4),
            }
        )
    return rows


def every_index_survives_exactly() -> dict:
    """The conclusion of that table, which is the only acceptable one."""
    rows = round_trip_every_supported_index()
    return {
        "checked": len(rows),
        "all_identical": all(row["identical"] for row in rows),
        "recall_unchanged": all(row["recall_before"] == row["recall_after"] for row in rows),
    }


def saved_files_are_stable(paths: Sequence[str] = ("a.vse", "b.vse")) -> dict:
    """That saving the same index twice produces the same bytes.

    Determinism in the format, which matters because a service that compares digests to decide
    whether to reload would otherwise reload on every save. It also catches any dictionary
    ordering or uninitialised padding leaking into the payload, which is the kind of thing that
    works until somebody changes a Python version.
    """
    if len(paths) < 2:
        raise ConfigError("comparing saved files needs at least two paths")
    corpus = gaussian(count=1024, dimension=16)
    index = IVFIndex(16, partitions=16, probe=4)
    index.build(corpus.vectors)
    save(index, paths[0])
    save(index, paths[1])
    first = Path(paths[0]).read_bytes()
    second = Path(paths[1]).read_bytes()
    return {
        "bytes": len(first),
        "identical": first == second,
        "digests_match": digest_of(first) == digest_of(second),
    }

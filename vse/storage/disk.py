from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from vse.build.kmeans import lloyd
from vse.errors import ConfigError, DataError
from vse.index.graph import GraphIndex
from vse.index.ivf import IVFIndex
from vse.vectors.dataset import Corpus, gaussian, held_out
from vse.vectors.exact import identifier_overlap, search

# What happens to every conclusion in this package when the vectors do not fit in memory.
#
# A disk does not read bytes, it reads pages, so the cost of touching one vector is the cost of
# the whole page it sits on. That single fact reverses the comparison the graph and inverted
# file modules spent their time establishing. In memory the graph index touched a third of what
# the inverted file did at higher recall and was clearly the better structure. On disk it is the
# worse one, because its accesses are scattered: a beam search visits vertices in whatever order
# the geometry dictates, and consecutive visits land on unrelated pages. An inverted file reads
# posting lists, and a posting list can be laid out contiguously, so it reads a few whole pages
# and uses all of them.
#
# The measurement is pages touched per query rather than distances. The graph reads two hundred
# pages a query and the inverted file forty three, a factor of four and a half, while the graph
# is more than twice as good per unit of arithmetic. That is not a small correction to the
# earlier result, it is the earlier result inverted, and it is the reason systems that page from
# disk are built on partitions.
#
# Getting that measurement right took two attempts, both instructive. The first counted pages
# for the identifiers a graph search returned rather than the vertices it visited, which says a
# graph touches ten vectors and is wrong by two orders of magnitude. The second ran on the
# clustered corpus, where the graph is fragmented into components and therefore accidentally has
# perfect locality, since a component is contiguous under a partition ordered layout. Both
# versions made the graph look good on disk for reasons that had nothing to do with disks.
#
# Read amplification is the number underneath it. A four kilobyte page holds eight vectors of a
# hundred and twenty eight dimensions, so touching one vector at random reads eight and uses
# one. The graph pays that on nearly every visit and the inverted file pays it once per posting
# list. Nothing here reduces the amplification; the whole difference is in how many times it is
# paid.
#
# Ordering the vectors on disk by partition is what makes the posting lists contiguous, and it
# is free at build time. Without it an inverted file is as scattered as the graph and loses the
# advantage entirely, which is measured rather than assumed.

PAGE_BYTES = 4096


@dataclass(frozen=True)
class Layout:
    """Where each vector sits on disk, as a page number per identifier."""

    page_of: torch.Tensor
    per_page: int
    dimension: int

    def __post_init__(self) -> None:
        if self.page_of.ndim != 1:
            raise DataError(f"a layout is one page per vector, got rank {self.page_of.ndim}")
        if self.per_page < 1:
            raise ConfigError(f"{self.per_page} vectors a page is not a page")

    @property
    def count(self) -> int:
        """How many vectors are laid out."""
        return int(self.page_of.shape[0])

    @property
    def pages(self) -> int:
        """How many pages the corpus occupies."""
        return int(self.page_of.max()) + 1

    def pages_for(self, rows: torch.Tensor) -> int:
        """How many distinct pages a set of vectors touches."""
        if rows.numel() == 0:
            return 0
        return int(torch.unique(self.page_of[rows]).numel())

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "count": self.count,
            "pages": self.pages,
            "per_page": self.per_page,
            "bytes": self.pages * PAGE_BYTES,
        }


def vectors_per_page(dimension: int, page_bytes: int = PAGE_BYTES) -> int:
    """How many vectors of a given width fit in one page.

    Floor rather than round, because a vector split across a page boundary costs two reads and
    the layouts here do not split them. That wastes the remainder of every page, which is
    reported in the padding measurement rather than hidden.
    """
    if dimension < 1:
        raise ConfigError(f"a width of {dimension} is not a width")
    if page_bytes < dimension * 4:
        raise ConfigError(f"a {page_bytes} byte page cannot hold a {dimension} wide vector")
    return page_bytes // (dimension * 4)


def sequential_layout(count: int, dimension: int, page_bytes: int = PAGE_BYTES) -> Layout:
    """Vectors in identifier order, which is what writing them out gives.

    The default and the one that makes an inverted file no better than a graph, because the
    identifiers are in whatever order the corpus arrived in and a posting list is a scattered
    set of them.
    """
    per_page = vectors_per_page(dimension, page_bytes)
    return Layout(
        page_of=torch.arange(count) // per_page, per_page=per_page, dimension=dimension
    )


def clustered_layout(
    corpus: Corpus, partitions: int = 64, page_bytes: int = PAGE_BYTES, seed: int = 0
) -> tuple[Layout, torch.Tensor]:
    """Vectors reordered so that each partition is contiguous on disk.

    The layout that makes a posting list a run of consecutive pages rather than a scattered set
    of single vectors. It costs one clustering pass at build time and it costs nothing at query
    time, and the measurement below is the whole argument for doing it.
    """
    if partitions < 1 or partitions > corpus.count:
        raise ConfigError(f"{partitions} partitions over {corpus.count} vectors")
    run = lloyd(corpus.vectors, k=partitions, seed=seed)
    order = torch.argsort(run.assignment)
    position = torch.empty(corpus.count, dtype=torch.long)
    position[order] = torch.arange(corpus.count)
    per_page = vectors_per_page(corpus.dimension, page_bytes)
    return (
        Layout(page_of=position // per_page, per_page=per_page, dimension=corpus.dimension),
        run.assignment,
    )


@dataclass
class DiskStats:
    """What a query cost in page reads rather than in distances."""

    pages: int = 0
    vectors: int = 0
    queries: int = 0
    page_bytes: int = PAGE_BYTES

    def read(self, pages: int, vectors: int) -> None:
        """Record one query's page reads and the vectors it actually wanted."""
        if pages < 0 or vectors < 0:
            raise ConfigError(f"cannot read {pages} pages for {vectors} vectors")
        self.pages += pages
        self.vectors += vectors

    @property
    def pages_per_query(self) -> float:
        """The number that decides the latency on a paging system."""
        if self.queries == 0:
            return 0.0
        return self.pages / self.queries

    @property
    def bytes_per_query(self) -> float:
        """What that is in bytes off the device."""
        return self.pages_per_query * self.page_bytes

    def amplification(self, dimension: int) -> float:
        """Bytes read over bytes wanted. One when every byte read was used."""
        if self.vectors == 0:
            return 0.0
        return (self.pages * self.page_bytes) / (self.vectors * dimension * 4)

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "queries": self.queries,
            "pages_per_query": round(self.pages_per_query, 2),
            "bytes_per_query": round(self.bytes_per_query, 1),
            "vectors_wanted": self.vectors,
        }


def graph_pages(
    corpus: Corpus, layout: Layout, queries: torch.Tensor, ef: int = 32
) -> DiskStats:
    """How many pages a graph traversal touches.

    The visited set is whatever the walk reached, and a walk reaches vertices in geometric order
    rather than in storage order, so the pages it touches are effectively a random sample. This
    counts distinct pages over the whole visited set, which is generous: a real traversal that
    revisited a page after evicting it would pay again.
    """
    index = GraphIndex(corpus.dimension, degree=16, ef=ef)
    index.build(corpus.vectors)
    stats = DiskStats(
        queries=int(queries.shape[0]), page_bytes=layout.per_page * corpus.dimension * 4
    )
    for row in range(int(queries.shape[0])):
        visited = index.visited(queries[row : row + 1], ef=ef)
        stats.read(layout.pages_for(visited), int(visited.numel()))
    return stats


def partitioned_pages(
    corpus: Corpus,
    layout: Layout,
    queries: torch.Tensor,
    probe: int = 8,
    partitions: int = 64,
) -> DiskStats:
    """How many pages an inverted file touches, with its posting lists laid out contiguously.

    A posting list is a run of consecutive identifiers under the clustered layout, so it spans
    the smallest number of pages it possibly can and every byte of every page but the last is a
    vector the query wanted.
    """
    index = IVFIndex(corpus.dimension, partitions=partitions, probe=probe)
    index.build(corpus.vectors)
    stats = DiskStats(
        queries=int(queries.shape[0]), page_bytes=layout.per_page * corpus.dimension * 4
    )
    for row in range(int(queries.shape[0])):
        centre_scores = torch.cdist(queries[row : row + 1], index._centres)
        chosen = torch.topk(centre_scores, k=probe, dim=1, largest=False).indices
        rows = torch.cat([index._lists[int(part)] for part in chosen[0]])
        stats.read(layout.pages_for(rows), int(rows.numel()))
    return stats


def a_page_holds_only_a_few_vectors(
    dimensions: Sequence[int] = (32, 64, 128, 256, 512),
) -> list[dict]:
    """How many vectors fit in a page, which is the whole reason any of this matters.

    Eight at a hundred and twenty eight dimensions and two at five hundred and twelve. That is
    the granularity of every disk access, so a structure touching one vector at a time is
    reading eight times what it wanted at best and half a page of padding at worst.
    """
    if not dimensions:
        raise ConfigError("there is nothing to sweep")
    rows = []
    for dimension in dimensions:
        fit = vectors_per_page(dimension)
        rows.append(
            {
                "dimension": dimension,
                "per_page": fit,
                "vector_bytes": dimension * 4,
                "wasted_bytes": PAGE_BYTES - fit * dimension * 4,
            }
        )
    return rows


def touching_one_vector_reads_a_page(dimension: int = 128) -> dict:
    """The amplification for a single random access, stated directly.

    Eight to one at a hundred and twenty eight dimensions. Nothing in this module reduces it and
    nothing can: the device reads pages. The entire difference between the structures is how
    many times each of them pays it.
    """
    fit = vectors_per_page(dimension)
    return {
        "dimension": dimension,
        "vectors_per_page": fit,
        "wanted_bytes": dimension * 4,
        "read_bytes": PAGE_BYTES,
        "amplification": round(PAGE_BYTES / (dimension * 4), 2),
    }


def the_graph_loses_on_disk(dimension: int = 128) -> dict:
    """The reversal this module exists for.

    In memory the graph touched a third of what the inverted file did at higher recall. On disk
    it reads several times as many pages, because its visits are scattered across the corpus and
    the inverted file's are a contiguous run. The distance count and the page count rank the two
    structures in opposite orders, and on a paging system the page count is the one that decides
    the latency.
    """
    corpus = gaussian(count=2048, dimension=dimension)
    searched, probes = held_out(corpus, count=32)
    layout, _ = clustered_layout(searched, partitions=32)
    graph = graph_pages(searched, layout, probes, ef=32)
    partitioned = partitioned_pages(searched, layout, probes, probe=4, partitions=32)
    return {
        "graph_pages": round(graph.pages_per_query, 2),
        "partitioned_pages": round(partitioned.pages_per_query, 2),
        "graph_vectors": graph.vectors,
        "partitioned_vectors": partitioned.vectors,
        "ratio": round(graph.pages_per_query / max(partitioned.pages_per_query, 1e-9), 2),
        "partitioned_wins": partitioned.pages_per_query < graph.pages_per_query,
    }


def the_layout_is_what_makes_the_difference(dimension: int = 128) -> dict:
    """Whether the inverted file's advantage is the structure or the ordering on disk.

    The ordering. With the vectors written in identifier order the posting lists are scattered
    sets and the inverted file reads about as many pages as the graph does. Sorting the corpus
    by partition costs one pass at build time and turns each posting list into a run, which is
    where the entire advantage comes from. An inverted file on a badly laid out disk is not a
    disk friendly structure, it is a graph with extra steps.
    """
    corpus = gaussian(count=2048, dimension=dimension)
    searched, probes = held_out(corpus, count=32)
    grouped, _ = clustered_layout(searched, partitions=32)
    plain = sequential_layout(searched.count, dimension)
    with_order = partitioned_pages(searched, grouped, probes, probe=4, partitions=32)
    without = partitioned_pages(searched, plain, probes, probe=4, partitions=32)
    return {
        "clustered_layout_pages": round(with_order.pages_per_query, 2),
        "sequential_layout_pages": round(without.pages_per_query, 2),
        "ratio": round(without.pages_per_query / max(with_order.pages_per_query, 1e-9), 2),
        "ordering_helps": with_order.pages_per_query < without.pages_per_query,
    }


def the_amplification_is_the_gap(dimension: int = 128) -> dict:
    """What each structure actually wastes, as bytes read over bytes wanted.

    Near one for the contiguous posting lists, since every page but the last is entirely
    vectors the query asked for. Several times that for the scattered traversal, which reads a
    page to get one vector most of the time. The amplification is the mechanism and the page
    count is the symptom.
    """
    corpus = gaussian(count=2048, dimension=dimension)
    searched, probes = held_out(corpus, count=32)
    layout, _ = clustered_layout(searched, partitions=32)
    graph = graph_pages(searched, layout, probes, ef=32)
    partitioned = partitioned_pages(searched, layout, probes, probe=4, partitions=32)
    return {
        "graph_amplification": round(graph.amplification(dimension), 2),
        "partitioned_amplification": round(partitioned.amplification(dimension), 2),
        "vectors_per_page": vectors_per_page(dimension),
        "partitioned_is_near_one": partitioned.amplification(dimension) < 2.0,
    }


def page_size_sweep(
    sizes: Sequence[int] = (512, 4096, 16384, 65536), dimension: int = 128
) -> list[dict]:
    """How the page size changes the comparison.

    A larger page holds more vectors, so a contiguous read gets more useful bytes per operation
    and a scattered read wastes more. The gap between the two structures therefore widens with
    the page size, which means the conclusion here gets stronger on devices with larger blocks
    rather than weaker.
    """
    if not sizes:
        raise ConfigError("there is nothing to sweep")
    corpus = gaussian(count=2048, dimension=dimension)
    searched, probes = held_out(corpus, count=32)
    rows = []
    for size in sizes:
        layout, _ = clustered_layout(searched, partitions=32, page_bytes=size)
        graph = graph_pages(searched, layout, probes, ef=32)
        partitioned = partitioned_pages(searched, layout, probes, probe=4, partitions=32)
        rows.append(
            {
                "page_bytes": size,
                "per_page": vectors_per_page(dimension, size),
                "graph_bytes": round(graph.bytes_per_query, 1),
                "partitioned_bytes": round(partitioned.bytes_per_query, 1),
                "ratio": round(
                    graph.bytes_per_query / max(partitioned.bytes_per_query, 1e-9), 2
                ),
            }
        )
    return rows


def the_gap_widens_with_the_page_size() -> dict:
    """The two ends of that sweep, and what it means for a device choice."""
    rows = {row["page_bytes"]: row for row in page_size_sweep()}
    return {
        "at_five_hundred_bytes": rows[512]["ratio"],
        "at_sixty_four_kilobytes": rows[65536]["ratio"],
        "widens": rows[65536]["ratio"] > rows[512]["ratio"],
        "vectors_per_page_at_the_top": rows[65536]["per_page"],
    }


def a_narrow_vector_suffers_more_not_less(
    dimensions: Sequence[int] = (32, 128, 512),
) -> list[dict]:
    """How the vector width interacts with the page size.

    Which goes the way I expected only
    once the direction is stated carefully.

    A narrow vector packs more per page, so a scattered access reading one vector wastes more of
    the page it read. The graph's amplification is five and a third at thirty two dimensions and
    one and eight tenths at five hundred and twelve, so narrow vectors suffer more from being
    accessed randomly, not less. At five hundred and twelve only two fit in a page and a random
    access wastes at most one of them.

    Which sharpens the connection to the compression modules rather than softening it.
    Quantising
    to eight bytes puts five hundred vectors in a page, so a scattered structure over compressed
    codes is wasting five hundred to one on every random access. Compression makes the layout
    question more important, by exactly the factor it compresses.
    """
    if not dimensions:
        raise ConfigError("there is nothing to sweep")
    rows = []
    for dimension in dimensions:
        corpus = gaussian(count=2048, dimension=dimension)
        searched, probes = held_out(corpus, count=32)
        layout, _ = clustered_layout(searched, partitions=32)
        graph = graph_pages(searched, layout, probes, ef=32)
        partitioned = partitioned_pages(searched, layout, probes, probe=4, partitions=32)
        rows.append(
            {
                "dimension": dimension,
                "per_page": vectors_per_page(dimension),
                "graph_pages": round(graph.pages_per_query, 2),
                "partitioned_pages": round(partitioned.pages_per_query, 2),
                "graph_amplification": round(graph.amplification(dimension), 2),
            }
        )
    return rows


def compressed_vectors_make_the_layout_matter_more(dimension: int = 128) -> dict:
    """What the quantisation modules do to this conclusion, which is to sharpen it.

    Eight byte codes put five hundred and twelve vectors in a four kilobyte page against eight
    full precision ones. So a contiguous posting list of five hundred codes is one page and a
    scattered walk touching five hundred codes is up to five hundred pages. Compression does not
    make the layout question go away, it multiplies the penalty for getting it wrong by sixty
    four.
    """
    full = vectors_per_page(dimension)
    coded = PAGE_BYTES // 8
    return {
        "full_precision_per_page": full,
        "eight_byte_codes_per_page": coded,
        "ratio": round(coded / full, 1),
        "scattered_penalty_multiplier": round(coded / full, 1),
    }


def the_two_cost_models_disagree(dimension: int = 128) -> dict:
    """The distance count and the page count, side by side, on the same two indexes.

    They rank the structures in opposite orders once the recall is taken into account. At these
    settings the graph gets seventy five percent recall and the inverted file thirty two, for
    almost the same distance count, so per unit of arithmetic the graph is more than twice as
    good. Per page read it is four and a half times worse.

    Everything earlier in this package used the distance count, which is the right measure when
    the corpus is resident, and every one of those conclusions has to be re read against this
    one
    when it is not. The measure is not wrong. It is answering a question about arithmetic when
    the question was about a device.
    """
    corpus = gaussian(count=2048, dimension=dimension)
    searched, probes = held_out(corpus, count=32)
    truth = search(probes, searched.vectors, k=10)
    layout, _ = clustered_layout(searched, partitions=32)
    graph = GraphIndex(dimension, degree=16, ef=32)
    graph.build(searched.vectors)
    graph_found, graph_stats = graph.search(probes, k=10, ef=32)
    inverted = IVFIndex(dimension, partitions=32, probe=4)
    inverted.build(searched.vectors)
    ivf_found, ivf_stats = inverted.search(probes, k=10)
    graph_disk = graph_pages(searched, layout, probes, ef=32)
    ivf_disk = partitioned_pages(searched, layout, probes, probe=4, partitions=32)
    return {
        "graph_recall": round(identifier_overlap(truth, graph_found), 4),
        "ivf_recall": round(identifier_overlap(truth, ivf_found), 4),
        "graph_distances": round(graph_stats.distances_per_query, 1),
        "ivf_distances": round(ivf_stats.distances_per_query, 1),
        "graph_pages": round(graph_disk.pages_per_query, 2),
        "ivf_pages": round(ivf_disk.pages_per_query, 2),
        "graph_recall_per_distance": round(
            identifier_overlap(truth, graph_found) / graph_stats.distances_per_query * 1000, 3
        ),
        "ivf_recall_per_distance": round(
            identifier_overlap(truth, ivf_found) / ivf_stats.distances_per_query * 1000, 3
        ),
        "graph_recall_per_page": round(
            identifier_overlap(truth, graph_found) / graph_disk.pages_per_query, 4
        ),
        "ivf_recall_per_page": round(
            identifier_overlap(truth, ivf_found) / ivf_disk.pages_per_query, 4
        ),
        "distances_prefer_the_graph": (
            identifier_overlap(truth, graph_found) / graph_stats.distances_per_query
            > identifier_overlap(truth, ivf_found) / ivf_stats.distances_per_query
        ),
        "pages_prefer_the_partitions": (
            identifier_overlap(truth, ivf_found) / ivf_disk.pages_per_query
            > identifier_overlap(truth, graph_found) / graph_disk.pages_per_query
        ),
    }


def the_padding_is_real(dimensions: Sequence[int] = (48, 100, 192, 300)) -> list[dict]:
    """What the widths that do not divide the page cost.

    Up to most of a page. A three hundred dimensional vector is twelve hundred bytes, so three
    fit in four kilobytes and four hundred and ninety six bytes are wasted on every page, which
    is twelve percent of the device. Padding out to a dividing width would waste the same space
    inside the vectors instead, so this is not a bug, it is a shape of the problem worth
    knowing.
    """
    if not dimensions:
        raise ConfigError("there is nothing to sweep")
    rows = []
    for dimension in dimensions:
        fit = vectors_per_page(dimension)
        wasted = PAGE_BYTES - fit * dimension * 4
        rows.append(
            {
                "dimension": dimension,
                "per_page": fit,
                "wasted_bytes": wasted,
                "wasted_share": round(wasted / PAGE_BYTES, 4),
            }
        )
    return rows


def a_vector_wider_than_a_page_is_refused() -> bool:
    """Whether a vector that cannot fit in one page is caught.

    It has to be, because the layouts here assume a vector is on one page. Splitting one across
    a boundary is a real design and it doubles the read for every access, so it is refused
    rather than silently produced by a floor division returning zero.
    """
    try:
        vectors_per_page(2048, page_bytes=4096)
    except ConfigError:
        return True
    return False


def a_zero_page_is_refused() -> bool:
    """Whether a page too small to hold anything is refused."""
    try:
        vectors_per_page(128, page_bytes=64)
    except ConfigError:
        return True
    return False


def a_layout_of_the_wrong_shape_is_refused() -> bool:
    """Whether a layout that is not one page per vector is refused at construction."""
    try:
        Layout(page_of=torch.zeros(4, 4, dtype=torch.long), per_page=8, dimension=32)
    except DataError:
        return True
    return False


def counting_pages_for_nothing_is_zero() -> dict:
    """What the page count of an empty candidate set is.

    Zero, not one. A query that reaches no candidates reads no pages, and returning one would
    quietly inflate every measurement involving a structure that sometimes finds nothing, which
    index/lsh.py showed is a real state rather than a hypothetical one.
    """
    layout = sequential_layout(1024, 128)
    return {
        "empty": layout.pages_for(torch.zeros(0, dtype=torch.long)),
        "one_vector": layout.pages_for(torch.tensor([5])),
        "a_whole_page": layout.pages_for(torch.arange(8)),
    }

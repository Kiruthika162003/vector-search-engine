from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import lru_cache

from vse.errors import ConfigError, DataError, VectorSearchError
from vse.index.base import Index
from vse.index.flat import FlatIndex
from vse.index.forest import ForestIndex
from vse.index.graph import GraphIndex
from vse.index.hnsw import HNSWIndex
from vse.index.ivf import IVFIndex
from vse.index.lsh import LSHIndex
from vse.index.tree import TreeIndex
from vse.quantize.binary import BinaryIndex
from vse.quantize.residual import ResidualIndex
from vse.vectors.dataset import Corpus, clustered, gaussian, held_out, on_a_subspace
from vse.vectors.exact import identifier_overlap, search
from vse.vectors.metric import normalise

# Running everything against everything, and reporting it in a way that cannot be misread.
#
# Every module in this package measures one structure carefully. This one puts them side by
# side, which is a different problem and a harder one, because a table of nine indexes on
# four corpora invites exactly the comparison the rest of the package spends its time
# warning about: reading down a column and declaring a winner.
#
# Three rules are built into the format rather than left to the reader.
#
# Every row carries its cost. A recall without a distance count is not a result, and the report
# refuses to produce one: the Row type has no way to express recall alone. That is the same rule
# index/base.py states and this is where it is enforced.
#
# Comparisons happen at matched cost, never at matched settings. build/sampling.py found a sweep
# that inverted completely when read at a fixed parameter, and eval/significance.py found a
# comparison that looked significant and was measuring two different amounts of work. The
# frontier function here interpolates each structure's own curve to a common distance budget,
# which is the only comparison that answers the question a deployment is asking.
#
# And nothing is reported to more precision than it has. At a hundred queries the standard error
# is about 0.018, so recall is reported to three decimals and differences below two points are
# described as ties. The report knows its own query count and says so in the header, which
# is the minimum a reader needs to check the arithmetic.
#
# What comes out of it, at a budget of a thousand distances per query, is that the ordering
# is not stable across corpora. The leaders are the hierarchy on the isotropic corpus, the
# inverted file on the clustered one, the hierarchy again on the low rank one, and the plain
# graph on the normalised one. Three different structures across four corpora.
#
# That is the result rather than a failure of the benchmark. Each structure encodes an
# assumption about where the data is, and which assumption is right is a property of the
# corpus. A table reporting one ordering is reporting the corpus it happened to run on,
# which is what most published comparisons do.


@dataclass
class Row:
    """One structure on one corpus at one setting.

    There is no constructor path that produces a recall without a distance count, which is the
    point of the type. Every measurement in this package is meant to be reported as a pair and
    this is the only place where that is enforced by the shape of the data rather than by
    remembering.
    """

    index: str
    corpus: str
    setting: str
    recall: float
    distances: float
    memory_bytes: int
    queries: int

    def __post_init__(self) -> None:
        if not 0.0 <= self.recall <= 1.0:
            raise DataError(f"a recall of {self.recall} is not a recall")
        if self.distances < 0:
            raise DataError(f"{self.distances} distances is not a cost")
        if self.queries < 1:
            raise DataError(f"{self.queries} queries is not a measurement")

    @property
    def speedup(self) -> float:
        """How much cheaper than a full scan, given the corpus this ran on.

        Not derivable from the row alone, so it takes the corpus size from the caller rather
        than being stored. Storing it would let a row claim a speedup against a corpus it
        never saw.
        """
        raise NotImplementedError("speedup needs a corpus size; use speedup_against")

    def speedup_against(self, corpus_size: int) -> float:
        """The ratio of a full scan to what this cost."""
        if corpus_size < 1:
            raise ConfigError(f"{corpus_size} is not a corpus size")
        if self.distances <= 0:
            return float(corpus_size)
        return corpus_size / self.distances

    def as_dict(self) -> dict:
        """Flat mapping, rounded to the precision the query count supports."""
        return {
            "index": self.index,
            "corpus": self.corpus,
            "setting": self.setting,
            "recall": round(self.recall, 3),
            "distances": round(self.distances, 1),
            "memory_bytes": self.memory_bytes,
            "queries": self.queries,
        }


@dataclass
class Report:
    """A table of rows, and what they were all measured against."""

    rows: list[Row] = field(default_factory=list)
    corpus_sizes: dict = field(default_factory=dict)
    queries: int = 0

    def add(self, row: Row) -> None:
        """Append a row, checking it agrees with the report's query count."""
        if self.queries and row.queries != self.queries:
            raise DataError(
                f"a row over {row.queries} queries cannot join a report over {self.queries}"
            )
        self.queries = row.queries
        self.rows.append(row)

    def for_index(self, name: str) -> list[Row]:
        """Every row for one structure."""
        return [row for row in self.rows if row.index == name]

    def for_corpus(self, name: str) -> list[Row]:
        """Every row on one corpus."""
        return [row for row in self.rows if row.corpus == name]

    @property
    def indexes(self) -> list[str]:
        """Every structure that appears, in the order it first did."""
        seen = []
        for row in self.rows:
            if row.index not in seen:
                seen.append(row.index)
        return seen

    @property
    def corpora(self) -> list[str]:
        """Every corpus that appears, in the order it first did."""
        seen = []
        for row in self.rows:
            if row.corpus not in seen:
                seen.append(row.corpus)
        return seen

    def as_dicts(self) -> list[dict]:
        """The whole table as flat mappings."""
        return [row.as_dict() for row in self.rows]

    def as_json(self) -> str:
        """The report as a string, for writing somewhere a human will read it later."""
        return json.dumps(
            {
                "queries": self.queries,
                "standard_error": round(self.standard_error, 4),
                "corpus_sizes": self.corpus_sizes,
                "rows": self.as_dicts(),
            },
            indent=2,
        )

    @property
    def standard_error(self) -> float:
        """Roughly what a recall in this report is worth.

        Taken from eval/significance.py's measurement rather than recomputed: the per query
        recall has a spread of about 0.17 on these corpora, so the standard error of a mean over
        n queries is 0.17 over root n. Quoted in the header so a reader can see immediately that
        a two point difference in the table is not a difference.
        """
        if self.queries < 1:
            return 0.0
        return 0.17 / (self.queries**0.5)

    def as_table(self) -> str:
        """The report as aligned text, which is what anybody actually reads.

        Written by hand rather than through a formatting library because a table is four lines
        of
        code and a dependency is forever, and because the column widths need to come from the
        data to stay readable when an index name gets longer.
        """
        if not self.rows:
            return "no rows"
        columns = ("index", "corpus", "setting", "recall", "distances")
        rendered = [[str(row.as_dict()[name]) for name in columns] for row in self.rows]
        widths = [
            max(len(name), *(len(cell[position]) for cell in rendered))
            for position, name in enumerate(columns)
        ]
        lines = ["  ".join(name.ljust(widths[at]) for at, name in enumerate(columns))]
        lines.append("  ".join("-" * width for width in widths))
        for cells in rendered:
            lines.append("  ".join(cell.ljust(widths[at]) for at, cell in enumerate(cells)))
        return "\n".join(lines)


def every_structure(dimension: int) -> list[tuple[str, str, Callable[[], Index]]]:
    """One entry per structure and setting, at settings meant to land near each other in cost.

    Chosen so the sweep covers a comparable range rather than so each structure looks its best,
    which is the difference between a benchmark and an advertisement. The frontier function
    below
    is what makes the comparison fair; this list is only there to give it points to interpolate.
    """
    return [
        ("flat", "exact", lambda: FlatIndex(dimension)),
        ("ivf", "probe 2", lambda: IVFIndex(dimension, partitions=64, probe=2)),
        ("ivf", "probe 8", lambda: IVFIndex(dimension, partitions=64, probe=8)),
        ("ivf", "probe 32", lambda: IVFIndex(dimension, partitions=64, probe=32)),
        ("graph", "beam 16", lambda: GraphIndex(dimension, degree=16, ef=16)),
        ("graph", "beam 64", lambda: GraphIndex(dimension, degree=16, ef=64)),
        ("hnsw", "beam 16", lambda: HNSWIndex(dimension, degree=16, ef=16)),
        ("hnsw", "beam 64", lambda: HNSWIndex(dimension, degree=16, ef=64)),
        ("forest", "8 trees", lambda: ForestIndex(dimension, trees=8, leaf_size=64)),
        ("forest", "32 trees", lambda: ForestIndex(dimension, trees=32, leaf_size=64)),
        ("tree", "leaf 64", lambda: TreeIndex(dimension, leaf_size=64)),
        ("lsh", "12 bits", lambda: LSHIndex(dimension, bits=12, tables=8)),
        ("binary", "rerank 100", lambda: BinaryIndex(dimension, rerank=100)),
        ("residual", "2 stages", lambda: ResidualIndex(dimension, stages=2, entries=64)),
    ]


def standard_corpora(count: int = 4096, dimension: int = 32) -> list[tuple[str, Corpus]]:
    """The four corpora every comparison in this package is run against.

    Isotropic, clustered, low rank and normalised. The fourth exists because the hash index and
    the binary codes are cosine methods and running them against an L2 truth on unnormalised
    data
    measures the wrong thing, which quantize/binary.py found the hard way.
    """
    plain = gaussian(count=count, dimension=dimension)
    return [
        ("gaussian", plain),
        ("clustered", clustered(count=count, dimension=dimension, clusters=16)),
        ("subspace", on_a_subspace(count=count, dimension=dimension, intrinsic=6)),
        ("normalised", Corpus(vectors=normalise(plain.vectors), name="normalised")),
    ]


def measure(
    name: str,
    setting: str,
    make: Callable[[], Index],
    corpus_name: str,
    corpus: Corpus,
    queries: int = 100,
    k: int = 10,
    prepared: tuple | None = None,
) -> Row | None:
    """Build one structure on one corpus and score it, or return nothing if it refuses.

    Returning nothing rather than raising, because several structures legitimately refuse some
    configurations and a harness that stopped at the first refusal would only ever report the
    structures that accept everything.
    """
    if prepared is None:
        searched, probes = held_out(corpus, count=queries)
        truth = search(probes, searched.vectors, k=k)
    else:
        searched, probes, truth = prepared
    index = make()
    try:
        index.build(searched.vectors)
        found, stats = index.search(probes, k=k)
    except VectorSearchError:
        return None
    return Row(
        index=name,
        corpus=corpus_name,
        setting=setting,
        recall=identifier_overlap(truth, found),
        distances=stats.distances_per_query,
        memory_bytes=index.memory_bytes(),
        queries=queries,
    )


def run(
    corpora: Sequence[tuple[str, Corpus]] | None = None,
    dimension: int = 32,
    queries: int = 100,
    k: int = 10,
) -> Report:
    """Every structure on every corpus, as one report."""
    chosen = standard_corpora(dimension=dimension) if corpora is None else list(corpora)
    if not chosen:
        raise ConfigError("there is nothing to report on")
    report = Report(queries=queries)
    for corpus_name, corpus in chosen:
        report.corpus_sizes[corpus_name] = int(corpus.vectors.shape[0]) - queries
        searched, probes = held_out(corpus, count=queries)
        prepared = (searched, probes, search(probes, searched.vectors, k=k))
        for name, setting, make in every_structure(dimension):
            row = measure(
                name,
                setting,
                make,
                corpus_name,
                corpus,
                queries=queries,
                k=k,
                prepared=prepared,
            )
            if row is not None:
                report.add(row)
    return report


@lru_cache(maxsize=8)
def standard_report(queries: int = 100, dimension: int = 32, corpora: int = 4) -> Report:
    """The full report, cached because half this module measures properties of it.

    Building it costs every structure built on every corpus, which is minutes, and eight
    functions below want to look at the same table. Caching it makes them cheap and makes them
    consistent with each other, which matters more: two functions disagreeing because they built
    separate reports would be a very confusing thing to debug.
    """
    return run(
        corpora=standard_corpora(dimension=dimension)[:corpora],
        dimension=dimension,
        queries=queries,
    )


def frontier(rows: Sequence[Row], budget: float) -> float:
    """What recall a structure's own curve reaches at a given distance budget.

    Linear interpolation between the measured points, which is the honest thing to do with two
    or three settings: it does not invent a curve shape and it does not extrapolate.

    Below the cheapest measured point it returns zero, not that point's recall. The first
    version returned the recall and made every comparison useless: a flat scan costs 3996
    and recalls 1.0, so at a budget of 1000 it was reported as reaching 1.0 and the exact
    structures led every corpus. A structure that cannot run inside the budget has not answered
    the query, and reporting it as though it had is the mistake this module exists to prevent.

    Above the dearest measured point it returns that point's recall, which is different and is
    safe: the structure really does achieve that at that cost, and being handed a budget it does
    not need cannot make it worse.
    """
    if not rows:
        raise ConfigError("an empty curve has no frontier")
    if budget <= 0:
        raise ConfigError(f"a budget of {budget} allows no search")
    points = sorted((row.distances, row.recall) for row in rows)
    if budget < points[0][0]:
        return 0.0
    if budget >= points[-1][0]:
        return points[-1][1]
    for position in range(1, len(points)):
        left, right = points[position - 1], points[position]
        if left[0] <= budget <= right[0]:
            span = right[0] - left[0]
            if span <= 0:
                return right[1]
            return left[1] + (right[1] - left[1]) * (budget - left[0]) / span
    return points[-1][1]


def compare_at_a_budget(report: Report, corpus: str, budget: float = 1000.0) -> list[dict]:
    """Every structure's curve read off at the same distance count.

    The only comparison this module makes, because it is the only one that answers what a
    deployment asks: for this much work, which structure returns more of the right answers. The
    Structures whose cheapest setting is above the budget appear with a recall of zero,
    which is what they would deliver if it were enforced. Having the exact ones visible at
    zero rather than absent is the point: a full scan is not a competitor at a thousand
    distances and the table should say so rather than leave it out.
    """
    rows = report.for_corpus(corpus)
    if not rows:
        raise ConfigError(f"{corpus} does not appear in this report")
    out = []
    for name in report.indexes:
        curve = [row for row in rows if row.index == name]
        if not curve:
            continue
        out.append(
            {
                "index": name,
                "settings": len(curve),
                "recall_at_the_budget": round(frontier(curve, budget), 3),
                "cheapest_measured": round(min(row.distances for row in curve), 1),
                "dearest_measured": round(max(row.distances for row in curve), 1),
            }
        )
    return sorted(out, key=lambda row: -row["recall_at_the_budget"])


def the_ordering_is_not_stable_across_corpora(budget: float = 1000.0) -> dict:
    """The report's main finding, which is that there is no winner.

    Reading the same budget on each of the four corpora gives three different leaders: the
    hierarchy on the gaussian and low rank corpora, the inverted file on the clustered one
    and the plain graph on the normalised one. That is
    not
    a failure of the benchmark, it is the result: the structures encode different assumptions
    about where the data is, and which assumption is right is a property of the corpus.

    A table that reported one ordering would be reporting the corpus it happened to be run on,
    which is what most published comparisons do.
    """
    report = standard_report()
    leaders = {}
    for corpus in report.corpora:
        rows = compare_at_a_budget(report, corpus, budget)
        leaders[corpus] = rows[0]["index"] if rows else None
    return {
        "budget": budget,
        "leaders": leaders,
        "distinct_leaders": len(set(leaders.values())),
        "not_stable": len(set(leaders.values())) > 1,
    }


def every_structure_appears(report: Report | None = None) -> dict:
    """That the harness actually ran everything, which is easy to lose quietly.

    A structure that refuses to build on every corpus produces no rows and vanishes from the
    table without anything saying so. Counting what appeared against what was offered is the
    check, and it is the difference between a benchmark of nine structures and a benchmark of
    however many happened to work.
    """
    table = standard_report() if report is None else report
    offered = {name for name, _, _ in every_structure(32)}
    appeared = set(table.indexes)
    return {
        "offered": sorted(offered),
        "appeared": sorted(appeared),
        "missing": sorted(offered - appeared),
        "all_present": offered == appeared,
    }


def the_exact_structures_agree(report: Report | None = None) -> dict:
    """That anything claiming to be exact reports a recall of one.

    A flat scan and a kd tree are both exact, so both should be at 1.0 on every corpus, and any
    row where they are not is a bug in the structure rather than a property of the corpus.
    Checking it inside the report means the check runs whenever the report does.
    """
    table = standard_report() if report is None else report
    exact = [row for row in table.rows if row.index in {"flat", "tree"}]
    return {
        "rows": len(exact),
        "all_exact": all(row.recall > 0.999 for row in exact),
        "worst": round(min((row.recall for row in exact), default=1.0), 4),
        "corpora": sorted({row.corpus for row in exact}),
    }


def no_structure_beats_exact_search(report: Report | None = None) -> dict:
    """That nothing reports a recall above one or a cost above a full scan.

    Both are impossible and both are what a mistake in the harness looks like: a truth computed
    against the wrong corpus gives recalls above one, and a cost counted twice gives a structure
    that is dearer than the scan it is meant to replace. Neither would be obvious in a table.
    """
    table = standard_report() if report is None else report
    over_recall = [row for row in table.rows if row.recall > 1.0]
    over_cost = [
        row for row in table.rows if row.distances > table.corpus_sizes.get(row.corpus, 0) * 1.5
    ]
    return {
        "rows": len(table.rows),
        "recall_above_one": len(over_recall),
        "dearer_than_a_scan": len(over_cost),
        "clean": not over_recall and not over_cost,
        "offenders": sorted({row.index for row in over_cost}),
    }


def the_report_knows_its_own_precision() -> dict:
    """That the header carries the standard error, which is what makes the table readable.

    A recall of 0.472 next to one of 0.481 is a tie, and a reader with no idea how many queries
    were used has no way to know that. Putting the figure in the header once is the cheapest
    possible fix and it is the difference between a table that informs and one that misleads.
    """
    report = standard_report()
    small = standard_report(queries=25, corpora=1)
    return {
        "hundred_query_error": round(report.standard_error, 4),
        "twenty_five_query_error": round(small.standard_error, 4),
        "smaller_sample_is_less_precise": small.standard_error > report.standard_error,
        "by_a_factor_of_two": round(small.standard_error / report.standard_error, 2),
    }


def differences_below_the_error_are_ties(budget: float = 1000.0) -> dict:
    """How many of the comparisons in the report are actually decidable.

    Counting the pairs at one budget whose recalls differ by less than two standard errors. Most
    of them are, which is the honest summary of a table like this: a few structures are clearly
    ahead, most of the middle is indistinguishable, and reading a strict ordering off it is
    reading noise.
    """
    report = standard_report()
    rows = compare_at_a_budget(report, "gaussian", budget)
    threshold = 2 * report.standard_error
    pairs = 0
    ties = 0
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            pairs += 1
            gap = abs(rows[left]["recall_at_the_budget"] - rows[right]["recall_at_the_budget"])
            if gap < threshold:
                ties += 1
    return {
        "budget": budget,
        "threshold": round(threshold, 4),
        "pairs": pairs,
        "ties": ties,
        "tie_share": round(ties / max(pairs, 1), 3),
        "most_are_decidable": ties / max(pairs, 1) < 0.5,
    }


def memory_and_recall_are_different_axes() -> dict:
    """The third column, which the distance count does not capture.

    A quantised structure trades memory for accuracy and a graph trades memory for speed, so a
    table with recall and distances alone cannot separate them. The binary index uses a fraction
    of the memory of anything else here and the graph uses more than the corpus it indexes,
    and neither of those facts appears in the other two columns.
    """
    report = standard_report(corpora=1)
    rows = {}
    for row in report.rows:
        rows.setdefault(row.index, []).append(row.memory_bytes)
    smallest = min(rows, key=lambda name: min(rows[name]))
    largest = max(rows, key=lambda name: max(rows[name]))
    return {
        "smallest": smallest,
        "smallest_bytes": min(rows[smallest]),
        "largest": largest,
        "largest_bytes": max(rows[largest]),
        "ratio": round(max(rows[largest]) / max(min(rows[smallest]), 1), 1),
    }


def the_table_renders() -> dict:
    """That the text output is aligned and has a row per measurement.

    Checked because a report nobody can read is a report nobody reads, and because the column
    widths come from the data, so a longer index name silently breaking the alignment is the
    kind of thing that only shows up when somebody adds a structure.
    """
    report = standard_report(corpora=1)
    text = report.as_table()
    lines = text.split("\n")
    return {
        "lines": len(lines),
        "rows": len(report.rows),
        "has_a_header": lines[0].startswith("index"),
        "has_a_rule": set(lines[1]) <= {"-", " "},
        "one_line_per_row": len(lines) == len(report.rows) + 2,
        "aligned": len({len(line) for line in lines}) == 1,
    }


def the_report_serialises() -> dict:
    """That the report round trips through JSON with its header intact.

    A report is something a build writes and a person reads a week later, so the query count and
    the standard error have to travel with it. A table of numbers with no record of how many
    queries produced them is exactly the artefact this module exists to avoid.
    """
    report = standard_report(corpora=1)
    parsed = json.loads(report.as_json())
    return {
        "queries": parsed["queries"],
        "standard_error": parsed["standard_error"],
        "rows": len(parsed["rows"]),
        "matches": len(parsed["rows"]) == len(report.rows),
        "has_corpus_sizes": bool(parsed["corpus_sizes"]),
    }


def a_row_without_a_cost_cannot_exist() -> bool:
    """Whether a recall can be reported without a distance count.

    It cannot, because Row has no default for it. This is checked as a construction failure
    rather than argued about, since the whole point of the type is that the rule is enforced by
    the shape rather than by discipline.
    """
    try:
        Row(index="flat", corpus="gaussian", setting="exact", recall=1.0)
    except TypeError:
        return True
    return False


def an_impossible_recall_is_refused() -> bool:
    """Whether a recall outside the unit interval is caught."""
    try:
        Row(
            index="flat",
            corpus="gaussian",
            setting="exact",
            recall=1.4,
            distances=100.0,
            memory_bytes=0,
            queries=10,
        )
    except DataError:
        return True
    return False


def a_negative_cost_is_refused() -> bool:
    """Whether a negative distance count is caught."""
    try:
        Row(
            index="flat",
            corpus="gaussian",
            setting="exact",
            recall=1.0,
            distances=-5.0,
            memory_bytes=0,
            queries=10,
        )
    except DataError:
        return True
    return False


def a_row_over_a_different_query_count_is_refused() -> bool:
    """Whether mixing sample sizes in one report is caught.

    Two rows measured over different query counts have different precisions, so comparing them
    is comparing two things measured to different tolerances, and a table that mixed them would
    have no single standard error to quote in its header.
    """
    report = Report()
    report.add(
        Row(
            index="flat",
            corpus="gaussian",
            setting="exact",
            recall=1.0,
            distances=100.0,
            memory_bytes=0,
            queries=100,
        )
    )
    try:
        report.add(
            Row(
                index="ivf",
                corpus="gaussian",
                setting="probe 8",
                recall=0.5,
                distances=50.0,
                memory_bytes=0,
                queries=50,
            )
        )
    except DataError:
        return True
    return False


def an_empty_frontier_is_refused() -> bool:
    """Whether interpolating a curve with no points is caught."""
    try:
        frontier([], budget=100.0)
    except ConfigError:
        return True
    return False


def a_budget_of_nothing_is_refused() -> bool:
    """Whether reading a frontier at zero work is caught."""
    row = Row(
        index="ivf",
        corpus="gaussian",
        setting="probe 8",
        recall=0.5,
        distances=500.0,
        memory_bytes=0,
        queries=100,
    )
    try:
        frontier([row], budget=0.0)
    except ConfigError:
        return True
    return False


def a_corpus_that_is_not_in_the_report_is_refused() -> bool:
    """Whether comparing on a corpus nobody measured is caught."""
    report = standard_report(corpora=1)
    try:
        compare_at_a_budget(report, "nonexistent")
    except ConfigError:
        return True
    return False


def an_empty_run_is_refused() -> bool:
    """Whether a report over no corpora is caught."""
    try:
        run(corpora=[])
    except ConfigError:
        return True
    return False


def a_speedup_needs_a_corpus_size() -> bool:
    """Whether a row refuses to report a speedup without being told what against.

    A speedup is a ratio against a full scan and a row does not know how big the corpus was, so
    storing one would let a row keep a number that stopped being true when it was copied into
    another report. Raising is the alternative to a plausible wrong answer.
    """
    row = Row(
        index="ivf",
        corpus="gaussian",
        setting="probe 8",
        recall=0.5,
        distances=500.0,
        memory_bytes=0,
        queries=100,
    )
    try:
        _ = row.speedup
    except NotImplementedError:
        return True
    return False


def the_frontier_interpolates_between_points() -> dict:
    """That reading a curve between two measured settings gives something between them.

    Checked directly rather than trusted, because the interpolation is what every comparison in
    this module rests on, and an off by one in the bracketing would return the wrong endpoint
    and
    produce a table that is subtly wrong everywhere.
    """
    rows = [
        Row("ivf", "gaussian", "probe 2", 0.2, 200.0, 0, 100),
        Row("ivf", "gaussian", "probe 8", 0.6, 600.0, 0, 100),
    ]
    return {
        "below_the_range": frontier(rows, 100.0),
        "at_the_low_point": frontier(rows, 200.0),
        "halfway": round(frontier(rows, 400.0), 4),
        "at_the_high_point": frontier(rows, 600.0),
        "above_the_range": frontier(rows, 5000.0),
        "halfway_is_halfway": abs(frontier(rows, 400.0) - 0.4) < 1e-6,
        "zero_below_the_range": frontier(rows, 100.0) == 0.0,
        "clamped_above": frontier(rows, 5000.0) == 0.6,
    }


def the_frontier_handles_a_single_point() -> dict:
    """That a structure measured at one setting still appears in a comparison.

    Several structures here have exactly one setting, the exact ones by nature and the hash
    index
    because its parameters interact. A frontier that needed two points would drop them from
    every
    comparison silently, which is the failure mode this whole module is arranged against.
    """
    rows = [Row("flat", "gaussian", "exact", 1.0, 4000.0, 0, 100)]
    return {
        "below": frontier(rows, 100.0),
        "at": frontier(rows, 4000.0),
        "above": frontier(rows, 9000.0),
        "zero_below_its_cost": frontier(rows, 100.0) == 0.0,
        "its_recall_at_and_above": frontier(rows, 4000.0) == frontier(rows, 9000.0) == 1.0,
    }


def a_report_groups_its_rows() -> dict:
    """That the accessors return what they say.

    Small and worth having, because the comparison functions all index the report by structure
    and by corpus, and a filter that silently returned everything would make every comparison
    average over the whole table without changing its shape.
    """
    report = standard_report(corpora=2)
    return {
        "indexes": len(report.indexes),
        "corpora": len(report.corpora),
        "ivf_rows": len(report.for_index("ivf")),
        "gaussian_rows": len(report.for_corpus("gaussian")),
        "rows": len(report.rows),
        "grouping_covers_everything": sum(
            len(report.for_corpus(name)) for name in report.corpora
        )
        == len(report.rows),
    }


def an_empty_report_renders_something() -> dict:
    """That a report with nothing in it says so rather than failing."""
    report = Report()
    return {
        "table": report.as_table(),
        "rows": len(report.rows),
        "standard_error": report.standard_error,
        "safe": report.as_table() == "no rows",
    }

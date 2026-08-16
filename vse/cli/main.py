from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from vse.errors import ConfigError, VectorSearchError
from vse.eval.report import every_structure, standard_corpora
from vse.eval.report import run as run_report
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
from vse.storage.persist import peek, save
from vse.vectors.dataset import clustered, gaussian, on_a_subspace
from vse.vectors.exact import identifier_overlap, search
from vse.verify.differential import sweep

# A command line for the package, which exists so the measurements can be run without writing
# Python and so a saved index can be inspected without loading it into a session.
#
# Six subcommands, all of them thin. Nothing here computes anything the library does not already
# compute, and nothing here interprets a result: the report command prints the table and the
# reader draws the conclusion. A command line that summarised its own output into a
# recommendation would be doing the thing the report module is arranged to prevent.
#
# The design decisions worth stating are about failure. Every command returns an exit code, zero
# for success and one for a refusal the user can fix, because a script calling this needs to
# branch on something better than parsing the output. Errors go to standard error and results go
# to standard output, so piping the results somewhere does not silently swallow the reason a
# command did nothing. And a bad argument is reported by name rather than as a stack trace: the
# library raises exceptions with useful messages and the top level turns them into one line.
#
# The corpora are generated rather than loaded, which is a real limitation and is deliberate. A
# corpus loader is a file format problem and this package does not have opinions about file
# formats; the corpora here are the same synthetic ones every measurement in the package uses,
# which is what makes the numbers reproducible against the module docstrings.


@dataclass
class Outcome:
    """What a command produced and whether it worked."""

    code: int
    output: str
    error: str = ""

    @property
    def ok(self) -> bool:
        """Whether it succeeded."""
        return self.code == 0

    def emit(self, out=None, err=None) -> int:
        """Write the streams and return the exit code."""
        if self.output:
            print(self.output, file=out or sys.stdout)
        if self.error:
            print(self.error, file=err or sys.stderr)
        return self.code


CORPORA = {
    "gaussian": lambda count, dimension: gaussian(count=count, dimension=dimension),
    "clustered": lambda count, dimension: clustered(
        count=count, dimension=dimension, clusters=16
    ),
    "subspace": lambda count, dimension: on_a_subspace(
        count=count, dimension=dimension, intrinsic=6
    ),
}

INDEXES = {
    # Every entry takes the same two arguments so the dispatch is a lookup rather than a
    # conditional; the exact structures ignore the settings because they have none.
    "flat": lambda dimension, args: FlatIndex(dimension),  # noqa: ARG005
    "ivf": lambda dimension, args: IVFIndex(
        dimension, partitions=args.partitions, probe=args.probe
    ),
    "graph": lambda dimension, args: GraphIndex(dimension, degree=args.degree, ef=args.beam),
    "hnsw": lambda dimension, args: HNSWIndex(dimension, degree=args.degree, ef=args.beam),
    "forest": lambda dimension, args: ForestIndex(
        dimension, trees=args.trees, leaf_size=args.leaf
    ),
    "tree": lambda dimension, args: TreeIndex(dimension, leaf_size=args.leaf),
    "lsh": lambda dimension, args: LSHIndex(dimension, bits=args.bits, tables=args.tables),
    "binary": lambda dimension, args: BinaryIndex(dimension, rerank=args.rerank),
    "residual": lambda dimension, args: ResidualIndex(
        dimension, stages=args.stages, entries=args.entries, rerank=args.rerank
    ),
}


def build_corpus(name: str, count: int, dimension: int) -> torch.Tensor:
    """Generate one of the standard corpora."""
    if name not in CORPORA:
        raise ConfigError(f"{name} is not a corpus; try one of {sorted(CORPORA)}")
    if count < 2:
        raise ConfigError(f"{count} vectors is not a corpus")
    if dimension < 1:
        raise ConfigError(f"{dimension} is not a dimension")
    return CORPORA[name](count, dimension).vectors


def build_index(name: str, dimension: int, args) -> Index:
    """Construct one of the indexes from parsed arguments."""
    if name not in INDEXES:
        raise ConfigError(f"{name} is not an index; try one of {sorted(INDEXES)}")
    return INDEXES[name](dimension, args)


def parser() -> argparse.ArgumentParser:
    """The whole argument surface, in one place.

    Every subcommand shares the corpus arguments, which is why they are added by a helper rather
    than repeated: three copies of the same four options is where a default drifts and one
    command starts measuring a different corpus from the others without anybody noticing.
    """
    top = argparse.ArgumentParser(prog="vse", description="vector search measurements")
    commands = top.add_subparsers(dest="command", required=True)

    def with_corpus(sub):
        sub.add_argument("--corpus", default="gaussian", choices=sorted(CORPORA))
        sub.add_argument("--count", type=int, default=4096)
        sub.add_argument("--dimension", type=int, default=32)
        sub.add_argument("--queries", type=int, default=100)
        return sub

    def with_index(sub):
        sub.add_argument("--index", default="ivf", choices=sorted(INDEXES))
        sub.add_argument("--partitions", type=int, default=64)
        sub.add_argument("--probe", type=int, default=8)
        sub.add_argument("--degree", type=int, default=16)
        sub.add_argument("--beam", type=int, default=32)
        sub.add_argument("--trees", type=int, default=8)
        sub.add_argument("--leaf", type=int, default=64)
        sub.add_argument("--bits", type=int, default=12)
        sub.add_argument("--tables", type=int, default=8)
        sub.add_argument("--rerank", type=int, default=0)
        sub.add_argument("--stages", type=int, default=2)
        sub.add_argument("--entries", type=int, default=64)
        return sub

    with_index(with_corpus(commands.add_parser("measure", help="score one index")))
    with_corpus(commands.add_parser("report", help="score every index"))
    with_index(with_corpus(commands.add_parser("sweep", help="sweep one search parameter")))
    build = with_index(with_corpus(commands.add_parser("build", help="build and save")))
    build.add_argument("--out", required=True)
    inspect = commands.add_parser("inspect", help="read a saved index header")
    inspect.add_argument("--path", required=True)
    verify = commands.add_parser("verify", help="run the invariant checks")
    verify.add_argument("--dimension", type=int, default=16)
    verify.add_argument("--count", type=int, default=512)

    for sub in commands.choices.values():
        sub.add_argument("--json", action="store_true", help="print machine readable output")
    return top


def measure(args) -> Outcome:
    """Score one index on one corpus, and say what it cost.

    The smallest useful thing the command line does, and the one every other command is built
    out of. Prints recall and distances together because the report module refuses to print one
    without the other and this is the same rule at the command line.
    """
    corpus = build_corpus(args.corpus, args.count, args.dimension)
    searched, probes = corpus[: -args.queries], corpus[-args.queries :]
    truth = search(probes, searched, k=10)
    index = build_index(args.index, args.dimension, args)
    index.build(searched)
    found, stats = index.search(probes, k=10)
    row = {
        "index": args.index,
        "corpus": args.corpus,
        "vectors": int(searched.shape[0]),
        "queries": int(probes.shape[0]),
        "recall": round(identifier_overlap(truth, found), 3),
        "distances": round(stats.distances_per_query, 1),
        "speedup": round(int(searched.shape[0]) / max(stats.distances_per_query, 1e-9), 1),
        "memory_bytes": index.memory_bytes(),
    }
    if args.json:
        return Outcome(0, json.dumps(row, indent=2))
    return Outcome(
        0,
        f"{row['index']} on {row['corpus']}: recall {row['recall']} "
        f"for {row['distances']} distances per query, "
        f"{row['speedup']} times cheaper than a scan",
    )


def report(args) -> Outcome:
    """Every index on every corpus, as the aligned table from eval/report.py."""
    table = run_report(
        corpora=standard_corpora(count=args.count, dimension=args.dimension),
        dimension=args.dimension,
        queries=args.queries,
    )
    if args.json:
        return Outcome(0, table.as_json())
    lines = [
        f"{table.queries} queries, standard error about "
        f"{round(table.standard_error, 3)}, so differences below "
        f"{round(4 * table.standard_error, 2)} are ties",
        "",
        table.as_table(),
    ]
    return Outcome(0, "\n".join(lines))


def sweep_parameter(args) -> Outcome:
    """Sweep whichever knob the chosen index has, and print the curve.

    Which knob depends on the structure, which is the whole point of
    eval/autotune.py's finding that probe eight and beam eight are not comparable
    quantities. The command picks the right one rather than making the caller know.
    """
    corpus = build_corpus(args.corpus, args.count, args.dimension)
    searched, probes = corpus[: -args.queries], corpus[-args.queries :]
    truth = search(probes, searched, k=10)
    index = build_index(args.index, args.dimension, args)
    index.build(searched)
    knob, values = _knob_for(args.index, args)
    if knob is None:
        return Outcome(1, "", f"{args.index} has no search time parameter to sweep")
    rows = []
    for value in values:
        setattr(index, knob, value)
        try:
            found, stats = index.search(probes, k=10)
        except VectorSearchError:
            continue
        rows.append(
            {
                knob: value,
                "recall": round(identifier_overlap(truth, found), 3),
                "distances": round(stats.distances_per_query, 1),
            }
        )
    if args.json:
        return Outcome(
            0, json.dumps({"index": args.index, "knob": knob, "rows": rows}, indent=2)
        )
    lines = [f"{args.index} on {args.corpus}, sweeping {knob}", ""]
    lines.append(f"{knob:>8}  {'recall':>7}  {'distances':>10}")
    for row in rows:
        lines.append(f"{row[knob]:>8}  {row['recall']:>7}  {row['distances']:>10}")
    return Outcome(0, "\n".join(lines))


def _knob_for(name: str, args) -> tuple[str | None, Sequence[int]]:
    """Which attribute a structure's accuracy knob lives on, and a sensible range.

    The beam ranges start at ten because a beam below k cannot return k neighbours, which is a
    constraint the inverted file does not have. Encoding that here rather than letting the
    search
    refuse is the difference between a sweep with a hole in it and a sweep that stops.
    """
    if name == "ivf":
        return "probe", [1, 2, 4, 8, 16, 32, min(64, args.partitions)]
    if name in {"graph", "hnsw"}:
        return "ef", [10, 16, 24, 32, 48, 64, 96]
    if name == "forest":
        return "trees", [1, 2, 4, 8, 16, 32]
    if name in {"binary", "residual"}:
        return "rerank", [0, 20, 50, 100, 200, 400]
    return None, ()


def build_and_save(args) -> Outcome:
    """Build an index and write it to a file, then say what went in it."""
    corpus = build_corpus(args.corpus, args.count, args.dimension)
    index = build_index(args.index, args.dimension, args)
    index.build(corpus)
    header = save(index, args.out)
    row = {
        "path": str(args.out),
        "kind": header.kind,
        "dimension": header.dimension,
        "count": header.count,
        "bytes": Path(args.out).stat().st_size,
        "digest": header.digest,
    }
    if args.json:
        return Outcome(0, json.dumps(row, indent=2))
    return Outcome(
        0,
        f"wrote {row['kind']} over {row['count']} vectors to {row['path']}, "
        f"{row['bytes']} bytes, digest {row['digest']}",
    )


def inspect(args) -> Outcome:
    """Read a saved index's header without decoding its payload.

    Cheap on a file of any size, which is the reason storage/persist.py separates the two, and
    the reason this is a command rather than something a caller does by loading the index and
    printing its attributes.
    """
    header = peek(args.path)
    row = header.as_dict()
    row["bytes"] = Path(args.path).stat().st_size
    if args.json:
        return Outcome(0, json.dumps(row, indent=2))
    detail = ", ".join(f"{key} {value}" for key, value in sorted(header.detail.items()))
    return Outcome(
        0,
        f"{header.kind} version {header.version}, {header.count} vectors "
        f"of {header.dimension} dimensions, {row['bytes']} bytes"
        + (f" ({detail})" if detail else ""),
    )


def verify(args) -> Outcome:
    """Run the invariant sweep from verify/differential.py and report what it found.

    Exits non zero when anything violates anything, so this is usable in a build. The four
    violations that module documents as real index behaviour rather than format errors are still
    reported and still fail, because a command that knew which failures to ignore would be a
    command nobody could trust.
    """
    result = sweep(dimension=args.dimension, count=args.count)
    row = result.as_dict()
    row["violations_detail"] = [violation.as_dict() for violation in result.violations]
    if args.json:
        return Outcome(0 if result.clean else 1, json.dumps(row, indent=2))
    lines = [f"{row['checks']} checks, {row['violations']} violations"]
    for violation in result.violations:
        lines.append(
            f"  {violation.index} on {violation.corpus}: {violation.rule}, {violation.detail}"
        )
    return Outcome(0 if result.clean else 1, "\n".join(lines))


COMMANDS = {
    "measure": measure,
    "report": report,
    "sweep": sweep_parameter,
    "build": build_and_save,
    "inspect": inspect,
    "verify": verify,
}


def dispatch(argv: Sequence[str] | None = None) -> Outcome:
    """Parse arguments and run the chosen command, turning library errors into one line.

    Every exception the library raises inherits from VectorSearchError and carries a message
    written to be read by a person, so the top level prints it rather than a traceback. Anything
    else propagates, because an unexpected exception is a bug and hiding it behind a tidy
    message
    is how a bug survives to a release.
    """
    args = parser().parse_args(argv)
    try:
        return COMMANDS[args.command](args)
    except VectorSearchError as error:
        return Outcome(1, "", f"{type(error).__name__}: {error}")
    except FileNotFoundError as error:
        return Outcome(1, "", f"no such file: {error.filename}")


def main(argv: Sequence[str] | None = None) -> int:
    """The entry point, which writes the streams and returns an exit code."""
    return dispatch(argv).emit()


def measuring_one_index_works() -> dict:
    """That the smallest command produces a recall and a cost together."""
    result = dispatch(["measure", "--count", "1024", "--dimension", "16", "--json"])
    row = json.loads(result.output)
    return {
        "code": result.code,
        "recall": row["recall"],
        "distances": row["distances"],
        "has_both": "recall" in row and "distances" in row,
        "has_a_speedup": row["speedup"] > 1.0,
    }


def the_text_output_says_both_numbers() -> dict:
    """That the human readable form does not drop the cost.

    A command that printed only the recall would be the thing every module here argues against,
    and it is exactly the shortcut a command line invites, so it is checked rather than assumed.
    """
    result = dispatch(["measure", "--count", "1024", "--dimension", "16"])
    return {
        "code": result.code,
        "output": result.output,
        "mentions_recall": "recall" in result.output,
        "mentions_distances": "distances" in result.output,
        "mentions_a_speedup": "cheaper" in result.output,
    }


def every_index_can_be_measured() -> dict:
    """That every structure in the package is reachable from the command line.

    A structure missing from the dispatch table is a structure nobody runs from a script, which
    is how a measurement drifts out of date without failing. Checked by name against the report
    module's own list.
    """
    offered = {name for name, _, _ in every_structure(16)}
    reachable = set(INDEXES)
    return {
        "reachable": sorted(reachable),
        "in_the_report": sorted(offered),
        "missing_from_the_cli": sorted(offered - reachable),
        "all_reachable": offered <= reachable,
    }


def a_sweep_produces_a_curve() -> dict:
    """That sweeping a parameter gives back a monotone cost curve."""
    result = dispatch(
        ["sweep", "--index", "ivf", "--count", "2048", "--dimension", "16", "--json"]
    )
    rows = json.loads(result.output)["rows"]
    return {
        "code": result.code,
        "points": len(rows),
        "knob": json.loads(result.output)["knob"],
        "cost_rises": [row["distances"] for row in rows]
        == sorted(row["distances"] for row in rows),
        "recall_rises": [row["recall"] for row in rows]
        == sorted(row["recall"] for row in rows),
    }


def the_sweep_picks_the_right_knob() -> dict:
    """That each structure is swept on its own parameter rather than on a shared name.

    eval/autotune.py found that probe and beam are not comparable quantities, so a command that
    swept a parameter called probe on every index would be sweeping nothing on most of them.
    """
    knobs = {}
    for name in ("ivf", "graph", "hnsw", "forest", "binary"):
        args = parser().parse_args(["sweep", "--index", name])
        knob, values = _knob_for(name, args)
        knobs[name] = (knob, len(values))
    return {
        "ivf": knobs["ivf"][0],
        "graph": knobs["graph"][0],
        "forest": knobs["forest"][0],
        "binary": knobs["binary"][0],
        "all_have_one": all(knob is not None for knob, _ in knobs.values()),
        "they_differ": len({knob for knob, _ in knobs.values()}) > 1,
    }


def a_structure_with_no_knob_says_so() -> dict:
    """That an exact structure refuses a sweep rather than producing one point.

    A flat scan has no accuracy parameter, so a sweep of it is a single row repeated, and
    printing that would look like a structure whose recall does not respond to its settings
    rather than one that has none.
    """
    result = dispatch(["sweep", "--index", "flat", "--count", "512", "--dimension", "8"])
    return {
        "code": result.code,
        "error": result.error,
        "refused": result.code == 1,
        "says_why": "no search time parameter" in result.error,
    }


def a_bad_corpus_is_reported_by_name() -> dict:
    """That an unknown corpus gives one readable line rather than a traceback."""
    top = parser()
    caught = ""
    try:
        top.parse_args(["measure", "--corpus", "spirals"])
    except SystemExit:
        caught = "argparse refused it"
    return {
        "argparse_refuses_it": bool(caught),
        "known": sorted(CORPORA),
        "and_the_library_would_too": _library_refuses_a_bad_corpus(),
    }


def _library_refuses_a_bad_corpus() -> bool:
    """That build_corpus checks its own argument rather than relying on the parser."""
    try:
        build_corpus("spirals", 100, 8)
    except ConfigError:
        return True
    return False


def a_bad_setting_is_reported_as_one_line() -> dict:
    """That a library refusal becomes a message rather than a stack trace.

    Asking for sixty four vectors and holding a hundred out as queries leaves the index nothing
    to build on, and the library says so: DataError, corpus is empty. The command line prints
    that message, names the error type, exits one and writes nothing to standard output.

    That the failure comes from an interaction between two arguments rather than from one bad
    value is why it is worth having here. Neither sixty four nor a hundred is wrong on its own.
    """
    result = dispatch(
        [
            "measure",
            "--index",
            "ivf",
            "--count",
            "64",
            "--partitions",
            "512",
            "--dimension",
            "8",
        ]
    )
    return {
        "code": result.code,
        "error": result.error,
        "one_line": "\n" not in result.error.strip(),
        "names_the_error": "Error" in result.error,
        "no_output": result.output == "",
    }


def errors_go_to_the_error_stream() -> dict:
    """That a failure does not appear on standard output.

    A script piping the results somewhere would otherwise get the error message as data, which
    is
    the difference between a pipeline that fails loudly and one that writes a plausible looking
    file containing an apology.
    """
    result = dispatch(["measure", "--index", "ivf", "--count", "64", "--partitions", "512"])
    return {
        "output_is_empty": result.output == "",
        "error_is_not": result.error != "",
        "code": result.code,
    }


def a_missing_file_is_reported() -> dict:
    """That inspecting a file that is not there says so."""
    result = dispatch(["inspect", "--path", "does-not-exist.vse"])
    return {
        "code": result.code,
        "error": result.error,
        "names_the_file": "does-not-exist.vse" in result.error,
        "no_traceback": "Traceback" not in result.error,
    }


def building_and_inspecting_round_trip(tmp: str = "cli.vse") -> dict:
    """That what the build command writes, the inspect command can read.

    The two commands are the only pair here that talk to each other, and a format change that
    broke the pairing would leave both of them working in isolation and useless together.
    """
    built = dispatch(
        [
            "build",
            "--index",
            "ivf",
            "--count",
            "1024",
            "--dimension",
            "16",
            "--partitions",
            "32",
            "--out",
            tmp,
            "--json",
        ]
    )
    read = dispatch(["inspect", "--path", tmp, "--json"])
    written = json.loads(built.output)
    seen = json.loads(read.output)
    return {
        "build_code": built.code,
        "inspect_code": read.code,
        "kind_matches": written["kind"] == seen["kind"],
        "count_matches": written["count"] == seen["count"],
        "digest_matches": written["digest"] == seen["digest"],
        "settings_survived": seen["detail"]["partitions"] == 32,
    }


def a_structure_with_no_format_refuses_to_build(tmp: str = "graph.vse") -> dict:
    """That building a graph to a file fails rather than writing something unreadable."""
    result = dispatch(
        [
            "build",
            "--index",
            "graph",
            "--count",
            "512",
            "--dimension",
            "8",
            "--out",
            tmp,
        ]
    )
    return {
        "code": result.code,
        "error": result.error,
        "refused": result.code == 1,
        "names_the_index": "GraphIndex" in result.error,
    }


def the_report_command_prints_its_precision() -> dict:
    """That the report's header travels to the command line.

    The table is only readable next to the standard error, and a command that printed the rows
    without it would be exactly the misleading artefact eval/report.py exists to prevent.
    """
    result = dispatch(["report", "--count", "1024", "--dimension", "16", "--queries", "40"])
    first = result.output.split("\n")[0]
    return {
        "code": result.code,
        "header": first,
        "mentions_queries": "queries" in first,
        "mentions_the_error": "standard error" in first,
        "mentions_ties": "ties" in first,
    }


def the_json_output_parses() -> dict:
    """That every command's machine readable form is valid JSON.

    Checked across all of them rather than one, because the flag is added to every subparser in
    one loop and a command that formatted its own output by hand would pass the flag and print
    something else.
    """
    rows = {}
    for argv in (
        ["measure", "--count", "512", "--dimension", "8", "--json"],
        ["sweep", "--index", "ivf", "--count", "512", "--dimension", "8", "--json"],
        ["verify", "--count", "256", "--dimension", "8", "--json"],
    ):
        result = dispatch(argv)
        try:
            json.loads(result.output)
            rows[argv[0]] = True
        except json.JSONDecodeError:
            rows[argv[0]] = False
    return {
        "measure": rows["measure"],
        "sweep": rows["sweep"],
        "verify": rows["verify"],
        "all_parse": all(rows.values()),
    }


def the_verify_command_fails_when_it_finds_something() -> dict:
    """That the invariant sweep's exit code is usable in a build.

    verify/differential.py documents four violations that are real index behaviour rather than
    format errors, and this command reports them and exits one anyway. A command that suppressed
    known failures would be a command that suppresses unknown ones the first time somebody adds
    to the list.
    """
    result = dispatch(["verify", "--count", "512", "--dimension", "16"])
    return {
        "code": result.code,
        "output": result.output.split("\n")[0],
        "exits_non_zero_on_violations": result.code == 1,
        "lists_them": len(result.output.split("\n")) > 1,
    }


def an_outcome_reports_success() -> dict:
    """That the outcome type says whether it worked."""
    return {
        "ok": Outcome(0, "fine").ok,
        "not_ok": Outcome(1, "", "broken").ok,
        "code": Outcome(1, "", "broken").code,
    }


def every_command_is_reachable() -> dict:
    """That the parser and the dispatch table agree.

    A subcommand in the parser with no entry in the table raises a KeyError at dispatch, and one
    in the table with no parser entry is unreachable. Both are silent until somebody runs the
    command, so they are checked here.
    """
    top = parser()
    subparsers = [
        action for action in top._actions if isinstance(action, argparse._SubParsersAction)
    ]
    declared = set(subparsers[0].choices) if subparsers else set()
    return {
        "declared": sorted(declared),
        "dispatched": sorted(COMMANDS),
        "unreachable": sorted(set(COMMANDS) - declared),
        "undispatched": sorted(declared - set(COMMANDS)),
        "they_agree": declared == set(COMMANDS),
    }


def a_zero_count_corpus_is_refused() -> bool:
    """Whether a corpus of nothing is caught."""
    try:
        build_corpus("gaussian", 1, 8)
    except ConfigError:
        return True
    return False


def a_zero_dimension_is_refused() -> bool:
    """Whether a corpus of zero width is caught."""
    try:
        build_corpus("gaussian", 100, 0)
    except ConfigError:
        return True
    return False


def an_unknown_index_is_refused() -> bool:
    """Whether an index nobody implemented is caught by name."""

    class Bare:
        pass

    try:
        build_index("quantum", 8, Bare())
    except ConfigError:
        return True
    return False


def the_error_names_what_is_available() -> dict:
    """That a refusal lists the valid options rather than only saying no.

    A message saying spirals is not a corpus is half an error. One that also lists the three
    that exist saves the reader a trip to the source, and it costs one formatted list.
    """
    corpus_error = ""
    index_error = ""
    try:
        build_corpus("spirals", 100, 8)
    except ConfigError as error:
        corpus_error = str(error)

    class Bare:
        pass

    try:
        build_index("quantum", 8, Bare())
    except ConfigError as error:
        index_error = str(error)
    return {
        "corpus_error": corpus_error,
        "index_error": index_error,
        "corpus_lists_options": "gaussian" in corpus_error,
        "index_lists_options": "ivf" in index_error,
    }

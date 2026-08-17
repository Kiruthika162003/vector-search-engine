from __future__ import annotations

import ast
import importlib
import pathlib

import pytest

from vse import errors

DASHES = (chr(8212), chr(8211))
ROOT = pathlib.Path(__file__).resolve().parent.parent
SOURCES = sorted(
    path
    for path in (ROOT / "vse").rglob("*.py")
    if "__pycache__" not in path.parts and path.name != "__init__.py"
)
TESTS = sorted(path for path in (ROOT / "tests").glob("test_*.py"))
EVERYTHING = sorted(
    path
    for folder in ("vse", "tests", "examples")
    for path in (ROOT / folder).rglob("*.py")
    if "__pycache__" not in path.parts
)


def parsed(path: pathlib.Path) -> ast.Module:
    """The syntax tree of one file."""
    return ast.parse(path.read_text(encoding="utf-8"))


def module_name(path: pathlib.Path) -> str:
    """The importable name of a source file."""
    return ".".join(path.relative_to(ROOT).with_suffix("").parts)


def public_functions(tree: ast.Module):
    """Top level functions that are not private helpers."""
    return [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
    ]


def public_classes(tree: ast.Module):
    """Top level classes that are not private helpers."""
    return [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_")
    ]


class TestTheShape:
    def test_there_are_modules_to_check(self):
        assert len(SOURCES) >= 30

    def test_most_modules_have_a_test_file_of_their_own(self):
        covered = {path.stem.removeprefix("test_") for path in TESTS}
        missing = sorted({path.stem for path in SOURCES} - covered)
        assert len(missing) <= 4, missing

    def test_the_uncovered_ones_are_tested_under_another_name(self):
        """Four modules have no test file named after them, and each is covered elsewhere.

        The exception list is written out rather than computed, so adding a module without a
        test file fails here instead of silently widening the exemption.
        """
        elsewhere = {
            "errors": "raised and caught in every other test file",
            "graph": "tests/test_graph_index.py",
            "main": "tests/test_cli.py",
            "recall": "tests/test_measures.py",
        }
        covered = {path.stem.removeprefix("test_") for path in TESTS}
        missing = {path.stem for path in SOURCES} - covered
        assert missing == set(elsewhere), sorted(missing)

    def test_every_test_file_reaches_the_package(self):
        """By import, or by loading a script that imports it.

        The examples are not a package, so their tests load them by path and never name vse
        directly. That is the one exception and it is checked rather than exempted.
        """
        for path in TESTS:
            names = {
                node.module
                for node in ast.walk(parsed(path))
                if isinstance(node, ast.ImportFrom) and node.module
            }
            if any(name.startswith("vse") for name in names):
                continue
            assert "examples" in path.read_text(encoding="utf-8"), path.name

    def test_no_module_is_empty(self):
        for path in SOURCES:
            assert len(parsed(path).body) > 2, path.name


class TestTheStyle:
    @pytest.mark.parametrize("path", EVERYTHING, ids=lambda path: path.name)
    def test_it_has_no_em_dashes(self, path):
        text = path.read_text(encoding="utf-8")
        assert not any(dash in text for dash in DASHES)

    @pytest.mark.parametrize("path", EVERYTHING, ids=lambda path: path.name)
    def test_it_is_plain_ascii(self, path):
        """No emoji, no smart quotes, no stray unicode punctuation."""
        text = path.read_text(encoding="utf-8")
        offenders = sorted({character for character in text if ord(character) > 127})
        assert offenders == [], f"{path.name} has {offenders}"

    @pytest.mark.parametrize("path", EVERYTHING, ids=lambda path: path.name)
    def test_it_has_no_trailing_whitespace(self, path):
        for number, line in enumerate(path.read_text(encoding="utf-8").split("\n"), start=1):
            assert line == line.rstrip(), f"{path.name}:{number}"

    @pytest.mark.parametrize("path", EVERYTHING, ids=lambda path: path.name)
    def test_its_lines_fit(self, path):
        """Ninety six columns, except where a line carries an explicit exemption."""
        for number, line in enumerate(path.read_text(encoding="utf-8").split("\n"), start=1):
            if "noqa" in line:
                continue
            assert len(line) <= 96, f"{path.name}:{number}"

    @pytest.mark.parametrize("path", EVERYTHING, ids=lambda path: path.name)
    def test_it_ends_with_one_newline(self, path):
        text = path.read_text(encoding="utf-8")
        if not text:
            return
        assert text.endswith("\n") and not text.endswith("\n\n")


class TestTheDocumentation:
    @pytest.mark.parametrize("path", SOURCES, ids=lambda path: path.name)
    def test_every_public_function_is_documented(self, path):
        for node in public_functions(parsed(path)):
            assert ast.get_docstring(node), f"{path.name}:{node.name}"

    @pytest.mark.parametrize("path", SOURCES, ids=lambda path: path.name)
    def test_every_public_class_is_documented(self, path):
        for node in public_classes(parsed(path)):
            assert ast.get_docstring(node), f"{path.name}:{node.name}"

    @pytest.mark.parametrize("path", SOURCES, ids=lambda path: path.name)
    def test_the_docstrings_are_sentences(self, path):
        """A one word docstring is a placeholder rather than a description."""
        for node in public_functions(parsed(path)):
            first = ast.get_docstring(node).split("\n")[0]
            assert len(first.split()) >= 2, f"{path.name}:{node.name}"
            assert first.endswith("."), f"{path.name}:{node.name}"

    @pytest.mark.parametrize("path", SOURCES, ids=lambda path: path.name)
    def test_no_module_uses_a_module_docstring_where_a_comment_belongs(self, path):
        """The package puts its long explanations in a comment block above the code.

        A module docstring would be picked up by help() and by documentation tools, which turns
        a piece of reasoning meant for a reader of the source into an interface description.
        Every module here uses a comment block instead, and this checks the convention holds.
        """
        tree = parsed(path)
        docstring = ast.get_docstring(tree)
        assert docstring is None or len(docstring) < 200, path.name


class TestTheMeasurements:
    @pytest.mark.parametrize("path", SOURCES, ids=lambda path: path.name)
    def test_a_refusal_check_can_fail(self, path):
        """A check named for a refusal must have a path that returns False.

        The failure mode this catches is a check that swallows its own exception and returns
        True whatever happens, which passes forever and tests nothing. Every one of them here
        returns True from inside the except and False after it, so both constants appear.
        """
        for node in public_functions(parsed(path)):
            if not node.name.endswith("_is_refused"):
                continue
            values = {
                child.value.value
                for child in ast.walk(node)
                if isinstance(child, ast.Return) and isinstance(child.value, ast.Constant)
            }
            if not values:
                continue
            assert values == {True, False}, f"{path.name}:{node.name}"

    @pytest.mark.parametrize("path", SOURCES, ids=lambda path: path.name)
    def test_nothing_is_left_marked_unfinished(self, path):
        text = path.read_text(encoding="utf-8").lower()
        for marker in ("todo", "fixme", "xxx", "hack:"):
            assert marker not in text, f"{path.name} mentions {marker}"

    @pytest.mark.parametrize("path", SOURCES, ids=lambda path: path.name)
    def test_no_module_prints_from_the_library(self, path):
        """Only the command line and the examples write to a stream.

        A library that prints is a library that cannot be used inside anything else, and the
        temptation is strongest in exactly the measurement functions this package is made of.
        """
        if path.parts[-2] == "cli":
            return
        calls = [
            node
            for node in ast.walk(parsed(path))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "print"
        ]
        assert calls == [], path.name

    @pytest.mark.parametrize("path", SOURCES, ids=lambda path: path.name)
    def test_it_imports(self, path):
        assert importlib.import_module(module_name(path)) is not None


class TestTheErrors:
    def test_every_error_derives_from_one_base(self):
        base = errors.VectorSearchError
        found = [
            value
            for name, value in vars(errors).items()
            if isinstance(value, type)
            and issubclass(value, Exception)
            and not name.startswith("_")
        ]
        assert len(found) >= 3
        assert all(issubclass(one, base) for one in found)

    @pytest.mark.parametrize("path", SOURCES, ids=lambda path: path.name)
    def test_nothing_raises_a_bare_exception(self, path):
        """Every refusal names a type the caller can catch."""
        for node in ast.walk(parsed(path)):
            if not isinstance(node, ast.Raise) or node.exc is None:
                continue
            raised = node.exc
            if isinstance(raised, ast.Call) and isinstance(raised.func, ast.Name):
                assert raised.func.id not in {"Exception", "BaseException"}, path.name

    @pytest.mark.parametrize("path", SOURCES, ids=lambda path: path.name)
    def test_every_refusal_carries_a_message(self, path):
        for node in ast.walk(parsed(path)):
            if not isinstance(node, ast.Raise) or node.exc is None:
                continue
            raised = node.exc
            if isinstance(raised, ast.Call):
                assert raised.args, f"{path.name} raises without a message"

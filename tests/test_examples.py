from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

EXAMPLES = pathlib.Path(__file__).resolve().parent.parent / "examples"
NAMES = sorted(path.stem for path in EXAMPLES.glob("*.py"))


def load(name: str):
    """Import an example by path, since the directory is not a package."""
    if name in sys.modules:
        return sys.modules[name]
    specification = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


class TestEveryExample:
    def test_the_directory_is_not_empty(self):
        assert len(NAMES) >= 7

    @pytest.mark.parametrize("name", NAMES)
    def test_it_imports(self, name):
        assert load(name) is not None

    @pytest.mark.parametrize("name", NAMES)
    def test_it_has_a_main(self, name):
        assert callable(load(name).main)

    @pytest.mark.parametrize("name", NAMES)
    def test_it_documents_itself(self, name):
        assert load(name).__doc__ and len(load(name).__doc__) > 200

    @pytest.mark.parametrize("name", NAMES)
    def test_its_docstring_shows_how_to_run_it(self, name):
        assert f"python examples/{name}.py" in load(name).__doc__

    @pytest.mark.parametrize("name", NAMES)
    def test_it_rejects_an_unknown_argument(self, name):
        with pytest.raises(SystemExit):
            load(name).main(["--not-an-argument"])


class TestWhenTheQueriesMove:
    def test_it_runs(self, capsys):
        module = load("when_the_queries_move")
        assert module.main(["--count", "512", "--queries", "32", "--partitions", "16"]) == 0
        assert "what it cost" in capsys.readouterr().out

    def test_a_shrinking_drift_costs_recall(self):
        module = load("when_the_queries_move")
        result = module.run(1024, 16, 64, 0.25, 4, 32)
        assert result["drifted"] < result["baseline"]

    def test_an_identity_drift_costs_nothing(self):
        module = load("when_the_queries_move")
        result = module.run(1024, 16, 64, 1.0, 4, 32)
        assert result["drifted"] == result["baseline"]

    def test_the_old_truth_understates_the_index(self):
        module = load("when_the_queries_move")
        result = module.run(1024, 16, 64, 0.25, 4, 32)
        assert result["against_the_old_truth"] < result["drifted"]

    def test_the_repair_stops_at_the_baseline(self):
        module = load("when_the_queries_move")
        result = module.run(1024, 16, 64, 0.25, 4, 32)
        rows = module.repair(result, 16, 32, 4)
        assert rows[-1][1] >= result["baseline"] or rows[-1][0] >= 32

    def test_the_repair_costs_more_each_step(self):
        module = load("when_the_queries_move")
        result = module.run(1024, 16, 64, 0.25, 4, 32)
        costs = [row[2] for row in module.repair(result, 16, 32, 4)]
        assert costs == sorted(costs)

    def test_the_report_names_the_rebuild_trap(self):
        module = load("when_the_queries_move")
        result = module.run(1024, 16, 64, 0.25, 4, 32)
        text = module.report(result, module.repair(result, 16, 32, 4), 0.25)
        assert "a rebuild would not have helped" in text


class TestIsThisDifferenceReal:
    def test_it_runs(self, capsys):
        module = load("is_this_difference_real")
        arguments = ["--seeds", "3", "--count", "512", "--queries", "32"]
        assert module.main(arguments) == 0
        assert "paired gap" in capsys.readouterr().out

    def test_a_setting_parses(self):
        module = load("is_this_difference_real")
        assert module.parse("ivf:32:4", 8, 0) is not None

    def test_every_kind_parses(self):
        module = load("is_this_difference_real")
        for setting in ("ivf:16:2", "forest:4:16", "lsh:8:4"):
            assert module.parse(setting, 8, 0) is not None

    def test_a_malformed_setting_is_refused(self, capsys):
        module = load("is_this_difference_real")
        assert module.main(["--left", "ivf:32", "--seeds", "2", "--count", "512"]) == 1
        assert "is not a setting" in capsys.readouterr().out

    def test_an_unknown_kind_is_refused(self, capsys):
        module = load("is_this_difference_real")
        assert module.main(["--left", "bloom:8:4", "--seeds", "2", "--count", "512"]) == 1
        assert "is not an index kind" in capsys.readouterr().out

    def test_a_setting_against_itself_is_undecided(self):
        module = load("is_this_difference_real")
        call = module.verdict([0.5, 0.6, 0.55], [0.5, 0.6, 0.55])
        assert not call["decided"] and call["gap"] == 0.0

    def test_a_clear_win_is_decided(self):
        module = load("is_this_difference_real")
        call = module.verdict([0.9, 0.9, 0.9], [0.5, 0.5, 0.5])
        assert call["decided"] and call["wins"] == 3

    def test_the_report_warns_about_unmatched_costs(self):
        module = load("is_this_difference_real")
        call = module.verdict([0.9, 0.9], [0.5, 0.5])
        text = module.report(
            "a", "b", [0.9, 0.9], [0.5, 0.5], [100.0, 100.0], [20.0, 20.0], call
        )
        assert "not a" in text and "matched cost comparison" in text


class TestSizingAShortlist:
    def test_it_runs(self, capsys):
        module = load("sizing_a_shortlist")
        arguments = ["--count", "512", "--queries", "32", "--dimension", "16"]
        assert module.main(arguments) == 0
        assert "distance equivalents" in capsys.readouterr().out

    def test_the_stages_cover_the_dimension(self):
        module = load("sizing_a_shortlist")
        names = [name for name, _, _ in module.stages(32)]
        assert names[0] == "sign" and "rank 32" in names

    def test_an_unreachable_target_is_reported(self, capsys):
        module = load("sizing_a_shortlist")
        arguments = [
            "--count",
            "512",
            "--queries",
            "32",
            "--target",
            "1.0",
            "--dimension",
            "16",
        ]
        assert module.main(arguments) == 0
        assert "never reached the target" in capsys.readouterr().out

    def test_a_report_with_nothing_priced_says_so(self):
        module = load("sizing_a_shortlist")
        rows = [
            {
                "stage": "sign",
                "description": "one bit",
                "depth": None,
                "recall": None,
                "scan": None,
                "rerank": None,
                "total": None,
            }
        ]
        assert "nothing cleared the target" in module.report(rows, 0.9, 1000)

    def test_the_cheapest_row_leads_the_table(self):
        module = load("sizing_a_shortlist")
        rows = [
            {
                "stage": "alpha",
                "description": "",
                "depth": 10,
                "recall": 0.9,
                "scan": 100.0,
                "rerank": 10.0,
                "total": 110.0,
            },
            {
                "stage": "beta",
                "description": "",
                "depth": 10,
                "recall": 0.9,
                "scan": 10.0,
                "rerank": 10.0,
                "total": 20.0,
            },
        ]
        text = module.report(rows, 0.9, 1000)
        assert text.index("beta") < text.index("alpha")

    def test_it_says_which_stage_to_improve(self):
        module = load("sizing_a_shortlist")
        rows = [
            {
                "stage": "a",
                "description": "",
                "depth": 10,
                "recall": 0.9,
                "scan": 10.0,
                "rerank": 100.0,
                "total": 110.0,
            }
        ]
        assert "a finer first stage" in module.report(rows, 0.9, 1000)

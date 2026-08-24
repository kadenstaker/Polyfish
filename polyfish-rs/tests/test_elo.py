#!/usr/bin/env python3
"""Tests for elo.py - the joint Bradley-Terry fit over the strength ladder.

The fit was orphaned for its whole life (nothing in the repo ran it), so its
one consumer is now run_training_loop.sh's gauge block and these tests: the
shell case below runs the exact command line the loop builds, the way
tests/test_ladder.py does for ladder.py. Stdlib `unittest`, no pytest.

    python3 -m unittest discover -s tests -p 'test_*.py'
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOOP_SCRIPT = os.path.join(ROOT, "run_training_loop.sh")
sys.path.insert(0, ROOT)


def budget(mcts=64, gumbel_k=16, max_turns=45, **extra):
    return {"mcts": mcts, "gumbel_k": gumbel_k, "max_turns": max_turns, **extra}


def reading(iteration, opponent="greedy", kind="gauge", wins=40, losses=20, draws=4,
            run_id="r1", **extra):
    r = {
        "run_id": run_id,
        "iteration": iteration,
        "kind": kind,
        "model": f"model@iter{iteration}",
        "opponent": opponent,
        "wins": wins,
        "losses": losses,
        "draws": draws,
    }
    r.update(extra)
    return r


def write_ladder(path, readings, anchors=None):
    data = {
        "anchors": anchors or [{"name": "greedy", "path": "", "elo": 0.0}],
        "readings": readings,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return path


class LadderExpansionTest(unittest.TestCase):
    def setUp(self):
        import elo

        self.elo = elo
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "ladder.json")

    def test_a_reading_expands_into_one_row_per_game(self):
        write_ladder(self.path, [reading(1, wins=3, losses=2, draws=1, budget=budget())])
        games = self.elo.load_ladder_games(self.path)
        self.assertEqual(len(games), 6)
        scores = sorted(s for _, _, s, _ in games)
        self.assertEqual(scores, [0.0, 0.0, 0.5, 1.0, 1.0, 1.0])
        self.assertEqual({b for _, b, _, _ in games}, {"greedy"})

    def test_a_tribe_audit_contributes_no_rows(self):
        write_ladder(self.path, [
            reading(1, budget=budget()),
            reading(1, kind="tribe_audit", budget=budget()),
        ])
        pinned = write_ladder(
            os.path.join(self.tmp.name, "pinned.json"), [reading(1, budget=budget())]
        )
        self.assertEqual(
            self.elo.load_ladder_games(self.path), self.elo.load_ladder_games(pinned)
        )

    def test_the_node_name_is_stable_across_calls(self):
        write_ladder(self.path, [reading(1, budget=budget())])
        self.assertEqual(
            self.elo.load_ladder_games(self.path), self.elo.load_ladder_games(self.path)
        )

    def test_the_latest_rating_bearing_reading_names_the_focus_node(self):
        write_ladder(self.path, [
            reading(1, budget=budget()),
            reading(2, budget=budget()),
            reading(2, kind="tribe_audit", budget=budget()),
        ])
        self.assertEqual(
            self.elo.latest_ladder_node(self.path), "r1/model@iter2#m64k16t45"
        )

    def test_each_node_carries_the_reading_it_came_from(self):
        """The dashboard places the fit on a campaign by run and iteration; it
        must not have to re-derive the node-naming rule in JavaScript."""
        write_ladder(self.path, [reading(4, budget=budget())])
        self.assertEqual(
            self.elo.ladder_node_meta(self.path),
            {"r1/model@iter4#m64k16t45": {"run_id": "r1", "iteration": 4}},
        )

    def test_an_empty_ladder_has_no_focus_node(self):
        write_ladder(self.path, [])
        self.assertIsNone(self.elo.latest_ladder_node(self.path))


class BudgetNodeTest(unittest.TestCase):
    """A ladder rating is a function of (weights x sims x turn cap); ladder.py's
    plateau window already keys on it, and the fit has to agree or a 16-sim
    stint chains onto a 64-sim one as if it measured weights alone."""

    def setUp(self):
        import elo

        self.elo = elo
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "ladder.json")

    def _nodes(self, readings, anchors=None):
        write_ladder(self.path, readings, anchors)
        return {a for a, _, _, _ in self.elo.load_ladder_games(self.path)}

    def test_two_budgets_at_one_iteration_are_two_players(self):
        nodes = self._nodes([
            reading(7, budget=budget(mcts=16)),
            reading(7, budget=budget(mcts=64)),
        ])
        self.assertEqual(nodes, {"r1/model@iter7#m16k16t45", "r1/model@iter7#m64k16t45"})

    def test_the_turn_cap_forks_the_node_like_the_sim_count(self):
        nodes = self._nodes([
            reading(7, budget=budget(max_turns=10)),
            reading(7, budget=budget(max_turns=45)),
        ])
        self.assertEqual(len(nodes), 2)

    def test_the_ramped_search_knobs_do_not_fork_the_node(self):
        """prior_heuristic_w and q_weight move every iteration by design, so
        keying on them would give every reading a private player."""
        nodes = self._nodes([
            reading(7, budget=budget(prior_heuristic_w=0.5, q_weight=0.0)),
            reading(7, budget=budget(prior_heuristic_w=0.1, q_weight=1.0)),
        ])
        self.assertEqual(nodes, {"r1/model@iter7#m64k16t45"})

    def test_a_gauge_and_its_audit_row_share_a_node(self):
        """The audit cross-check is played at the same budget, so its games must
        pool with the gauge's instead of forking a second player at that
        iteration - which is why the loop records max_turns on the audit row."""
        nodes = self._nodes([
            reading(7, budget=budget()),
            reading(7, kind="audit", opponent="anchor_x", budget=budget()),
        ])
        self.assertEqual(nodes, {"r1/model@iter7#m64k16t45"})

    def test_a_legacy_reading_without_a_budget_keeps_its_bare_name(self):
        self.assertEqual(self._nodes([reading(7)]), {"r1/model@iter7"})

    def test_an_anchor_is_aliased_to_its_untagged_link_node(self):
        """The link match defines a frozen anchor's identity. Tagging it would
        strand every later reading taken at another budget in its own
        disconnected component, which is worse than pooling the anchor."""
        anchors = [
            {"name": "greedy", "path": "", "elo": 0.0},
            {"name": "anchor_iter3", "path": "p", "elo": 250.0, "frozen_iteration": 3},
        ]
        readings = [
            reading(3, budget=budget(max_turns=20)),
            reading(3, kind="link", opponent="greedy", wins=90, losses=38, draws=0),
            reading(4, opponent="anchor_iter3", budget=budget(mcts=16)),
        ]
        write_ladder(self.path, readings, anchors)
        games = self.elo.load_ladder_games(self.path)
        opponents = {b for _, b, _, _ in games}
        self.assertIn("r1/model@iter3", opponents)
        self.assertNotIn("anchor_iter3", opponents)
        # The point of the alias: a freeze must not cut the graph in two, even
        # when the budget moves right after it.
        reachable = self.elo.connected_to_anchor(
            [(a, b, s) for a, b, s, _ in games], "greedy"
        )
        self.assertIn("r1/model@iter4#m16k16t45", reachable)


class FitTest(unittest.TestCase):
    def setUp(self):
        import elo

        self.elo = elo

    def test_the_anchor_stays_pinned_at_zero(self):
        games = [("a", "greedy", 1.0), ("a", "greedy", 0.0), ("b", "a", 1.0)]
        ratings = self.elo.fit_ratings(games, anchor="greedy")
        self.assertEqual(ratings["greedy"], 0.0)
        self.assertGreater(ratings["b"], ratings["a"])

    def test_a_dropped_ledger_row_is_refused(self):
        """arena writes `dropped` for a panicked game; scoring it as a loss
        would silently shrink n and unbalance the side-swap pairing."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "matches.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"player1": "a", "player2": "b", "result": "dropped"}))
                f.write("\n")
            with self.assertRaises(ValueError):
                self.elo.load_games(path)


def run_elo(args, cwd=None):
    return subprocess.run(
        [sys.executable, os.path.join(ROOT, "elo.py")] + args,
        capture_output=True, text=True, cwd=cwd or ROOT,
    )


class CliTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ladder = os.path.join(self.tmp.name, "ladder.json")
        self.out = os.path.join(self.tmp.name, "elo_ratings.json")
        write_ladder(self.ladder, [
            reading(1, wins=20, losses=40, draws=4, budget=budget()),
            reading(2, wins=40, losses=20, draws=4, budget=budget()),
        ])

    def _fit(self, *extra):
        proc = run_elo(["fit", "--source", "ladder", "--ladder", self.ladder,
                        "--out", self.out, "--bootstrap", "20", *extra])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        return proc

    def test_fit_writes_a_table_the_report_can_read_back(self):
        self._fit()
        with open(self.out, encoding="utf-8") as f:
            ratings = json.load(f)
        self.assertEqual(ratings["greedy"]["elo"], 0.0)
        self.assertGreater(ratings["r1/model@iter2#m64k16t45"]["elo"],
                           ratings["r1/model@iter1#m64k16t45"]["elo"])
        fitted = ratings["r1/model@iter2#m64k16t45"]
        self.assertEqual(fitted["budget"], "m64k16t45")
        self.assertEqual((fitted["run_id"], fitted["iteration"]), ("r1", 2))
        report = run_elo(["report", "--ratings", self.out, "--anchor", "greedy"])
        self.assertEqual(report.returncode, 0, report.stderr)
        self.assertIn("greedy", report.stdout)

    def test_quiet_prints_one_line_naming_the_newest_node(self):
        line = self._fit("--quiet").stdout.strip()
        self.assertEqual(len(line.splitlines()), 1)
        self.assertTrue(line.startswith("r1/model@iter2#m64k16t45 "), line)

    def test_a_ladder_spanning_budgets_says_so(self):
        write_ladder(self.ladder, [
            reading(1, budget=budget(mcts=16)),
            reading(2, budget=budget(mcts=64)),
        ])
        self.assertIn("spans 2 search budgets", self._fit("--quiet").stderr)

    def test_a_missing_ladder_fails_loudly(self):
        proc = run_elo(["fit", "--source", "ladder",
                        "--ladder", os.path.join(self.tmp.name, "nope.json")])
        self.assertNotEqual(proc.returncode, 0)


def _contract_module():
    """scripts/check_cli_contract.py, which already knows how to read a shell
    script's flags. Importing it keeps this command line tied to the loop
    instead of to a copy of it that can drift."""
    import importlib.util

    path = os.path.join(ROOT, "scripts", "check_cli_contract.py")
    spec = importlib.util.spec_from_file_location("check_cli_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# One representative value per flag run_training_loop.sh passes to `elo.py fit`.
# A flag the loop gains and this table has not is a hard failure: the point is
# that the command line the loop builds is the one that gets run.
SHELL_FLAG_VALUES = {"--source": "ladder", "--ladder": None, "--out": None, "--quiet": ""}


class ShellCommandLineTest(unittest.TestCase):
    """elo.py had no caller at all until the gauge block gained one, so nothing
    would have noticed the loop and argparse disagreeing until a campaign was
    already running."""

    @classmethod
    def setUpClass(cls):
        contract = _contract_module()
        # check_cli_contract.py does not yet list elo.py among the python CLIs
        # it resolves; the extractor is reused here rather than copied.
        contract.PY_SCRIPTS = set(contract.PY_SCRIPTS) | {"elo.py"}
        with open(LOOP_SCRIPT, encoding="utf-8") as fh:
            lines = contract.logical_lines(fh.read())
        _, flag_vars = contract.collect_assignments(lines, contract.known_binaries())
        usage = contract.collect_python_usage(lines, flag_vars, LOOP_SCRIPT)
        cls.shell_usage = {
            target[len("elo.py "):]: flags
            for target, flags in usage.items()
            if target.startswith("elo.py ")
        }

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_the_loop_still_runs_the_joint_fit(self):
        self.assertIn("fit", self.shell_usage,
                      "run_training_loop.sh no longer invokes `elo.py fit` - "
                      "the joint Elo fit is orphaned again (#8)")

    def test_the_loop_s_fit_command_line_produces_a_rating_table(self):
        ladder = write_ladder(os.path.join(self.tmp.name, "ladder.json"),
                              [reading(1, budget=budget())])
        out = os.path.join(self.tmp.name, "elo_ratings.json")
        values = dict(SHELL_FLAG_VALUES, **{"--ladder": ladder, "--out": out})
        argv = ["fit"]
        for flag in sorted(self.shell_usage["fit"]):
            self.assertIn(flag, values,
                          f"run_training_loop.sh passes {flag} to `elo.py fit` "
                          "but this test has no value for it")
            argv.append(flag)
            if values[flag]:
                argv.append(values[flag])
        proc = run_elo(argv, cwd=self.tmp.name)
        self.assertEqual(proc.returncode, 0, f"{' '.join(argv)}\n{proc.stderr}")
        with open(out, encoding="utf-8") as f:
            self.assertEqual(json.load(f)["greedy"]["elo"], 0.0)


if __name__ == "__main__":
    unittest.main()

"""auto_train.sh contract.

The idle babysitter is the one automation wrapper nothing else checks:
scripts/check_cli_contract.py resolves no target inside it, and it drives
training on a desktop nobody watches. It once pointed at run-server.sh (the
simulator, not training) and halted with SIGKILL on the wrapper pid alone, so
every halt orphaned the loop's children. These are pure text assertions - no
process is started.
"""

import os
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKSPACE = os.path.dirname(ROOT)
AUTO_TRAIN = os.path.join(WORKSPACE, "auto_train.sh")

# The chain deleted in #56: start-hidden.vbs -> auto_train.ps1 ->
# run-training.bat -> run_training_loop.ps1, which self-played and trained with
# no gauge, ladder, anchor or plateau stop of any kind.
WINDOWS_CHAIN = [
    "start-hidden.vbs",
    "auto_train.ps1",
    "run-training.bat",
    "run_training_loop.ps1",
]


def read_auto_train():
    with open(AUTO_TRAIN, encoding="utf-8") as fh:
        return fh.read()


class AutoTrainContractTest(unittest.TestCase):
    def test_drives_the_real_training_loop(self):
        text = read_auto_train()
        line = next(
            (ln for ln in text.splitlines() if ln.startswith("TRAIN_SCRIPT=")), None
        )
        self.assertIsNotNone(line, "auto_train.sh has no TRAIN_SCRIPT assignment")
        script = line.split("=", 1)[1].strip().strip('"').strip("'")
        self.assertTrue(
            script.endswith("run_training_loop.sh"),
            f"TRAIN_SCRIPT is {script!r}, not the training loop",
        )
        self.assertTrue(
            os.path.isfile(os.path.join(WORKSPACE, script)),
            f"TRAIN_SCRIPT points at {script!r}, which does not exist",
        )
        self.assertNotIn(
            "run-server.sh",
            text,
            "auto_train.sh is babysitting the simulator server again",
        )

    def test_passes_resume(self):
        text = read_auto_train()
        line = next(
            (ln for ln in text.splitlines() if ln.startswith("TRAIN_ARGS=")), None
        )
        self.assertIsNotNone(line, "auto_train.sh has no TRAIN_ARGS assignment")
        self.assertIn(
            "--resume",
            line,
            "run_training_loop.sh refuses a bare launch once model.safetensors "
            "and training_log.csv history both exist (#37), so an auto-restart "
            "without --resume dies at startup",
        )

    def test_halts_the_process_group(self):
        text = read_auto_train()
        self.assertIn(
            "kill -TERM -- -",
            text,
            "the halt must signal the process group, or the loop's children "
            "(self_play, train.py, the polyfish server on port 3000) survive it",
        )
        self.assertNotIn(
            'kill -9 "$TRAIN_PID"',
            text,
            "SIGKILL on the wrapper pid skips run_training_loop.sh's EXIT trap",
        )
        self.assertIn("setsid", text, "the loop must start in its own session")

    def test_no_inline_analysis_prompt(self):
        text = read_auto_train()
        self.assertNotIn(
            "agy ",
            text,
            "the inline report prompt restated the curriculum and read "
            "session.log from the wrong directory; run_analysis_now.js is the "
            "CSV-driven replacement",
        )


class WindowsChainRetiredTest(unittest.TestCase):
    def test_chain_is_gone(self):
        for name in WINDOWS_CHAIN:
            for directory in (WORKSPACE, ROOT):
                path = os.path.join(directory, name)
                self.assertFalse(
                    os.path.exists(path),
                    f"{path} is back: the PowerShell training chain has no "
                    "gauge, ladder, anchor freeze or plateau stop, and is "
                    "outside check_cli_contract.py. Train under WSL2 with "
                    "polyfish-rs/run_training_loop.sh instead (#56).",
                )


if __name__ == "__main__":
    unittest.main()

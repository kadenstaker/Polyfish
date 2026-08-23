#!/usr/bin/env python3
"""Tests for scripts/backup_experiment_record.sh and its wiring into the loop.

The experiment record (training_log.csv, ladder.json, checkpoints/) is the only
part of a campaign a rerun cannot reproduce, and audit T3 filed it as living on
one disk. The script that copies it off-box existed but nothing called it, so
neither the copy nor its failure policy had ever run. These drive the script
end to end and then check the loop's side of the contract: that it invokes it,
and that a failing backup cannot take down a `set -e` training run.

Stdlib `unittest` on purpose (see tests/test_ladder.py):

    python3 -m unittest discover -s tests -p 'test_*.py'
"""
import json
import os
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "scripts", "backup_experiment_record.sh")
LOOP_SCRIPT = os.path.join(ROOT, "run_training_loop.sh")

if shutil.which("sha256sum"):
    SUM_CMD = ["sha256sum"]
elif shutil.which("shasum"):
    SUM_CMD = ["shasum", "-a", "256"]
else:  # the script itself refuses to run without one
    SUM_CMD = None


@unittest.skipIf(SUM_CMD is None, "no sha256sum/shasum on PATH")
class BackupScriptTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.src = os.path.join(self.tmp.name, "run")
        self.dest = os.path.join(self.tmp.name, "backup")
        os.makedirs(self.src)

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self, rel, text):
        path = os.path.join(self.src, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return path

    def _record(self):
        self._write("training_log.csv", "run_id,iteration,loss\n1,1,0.5\n1,2,0.4\n")
        self._write("ladder.json", json.dumps({"anchors": [], "readings": []}))
        self._write("checkpoints/model_checkpoint_iter1.safetensors", "weights-1")

    def _run(self, dest=None, env=None):
        return subprocess.run(
            [SCRIPT] + ([dest] if dest is not None else []),
            capture_output=True,
            text=True,
            env=dict(os.environ, POLYFISH_RUN_DIR=self.src, **(env or {})),
        )

    def _latest(self):
        with open(os.path.join(self.dest, "LATEST"), encoding="utf-8") as f:
            return f.read().strip()

    def _manifest(self, stamp):
        with open(os.path.join(self.dest, stamp, "MANIFEST"), encoding="utf-8") as f:
            return f.read()

    def _item_note(self, stamp, rel):
        for line in self._manifest(stamp).splitlines():
            parts = line.split("\t")
            if parts[0] == "item" and parts[1] == rel:
                return parts[4]
        self.fail(f"MANIFEST has no item line for {rel}")

    def test_a_snapshot_is_published_with_a_manifest_and_a_latest_pointer(self):
        self._record()
        proc = self._run(self.dest)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        stamp = self._latest()
        snap = os.path.join(self.dest, stamp)
        for rel in ("MANIFEST", "SHA256SUMS", "training_log.csv", "ladder.json",
                    "checkpoints/model_checkpoint_iter1.safetensors"):
            self.assertTrue(os.path.exists(os.path.join(snap, rel)), rel)
        self.assertIn("status=complete", self._manifest(stamp))
        self.assertFalse([n for n in os.listdir(self.dest) if n.startswith(".incoming-")])

    def test_the_destination_can_come_from_the_environment(self):
        self._record()
        proc = self._run(env={"POLYFISH_BACKUP_DIR": self.dest})
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertTrue(os.path.isdir(os.path.join(self.dest, self._latest())))

    def test_the_published_checksums_verify(self):
        self._record()
        self.assertEqual(self._run(self.dest).returncode, 0)
        snap = os.path.join(self.dest, self._latest())
        check = subprocess.run(SUM_CMD + ["-c", "SHA256SUMS"], cwd=snap,
                               capture_output=True, text=True)
        self.assertEqual(check.returncode, 0, check.stdout + check.stderr)

    def test_a_second_snapshot_advances_latest_and_reuses_checkpoints(self):
        self._record()
        self.assertEqual(self._run(self.dest).returncode, 0)
        first = self._latest()
        self._write("training_log.csv", "run_id,iteration,loss\n1,1,0.5\n1,2,0.4\n1,3,0.3\n")
        self.assertEqual(self._run(self.dest).returncode, 0)
        second = self._latest()
        self.assertNotEqual(first, second)

        rel = "checkpoints/model_checkpoint_iter1.safetensors"
        old = os.stat(os.path.join(self.dest, first, rel))
        new = os.stat(os.path.join(self.dest, second, rel))
        self.assertEqual(old.st_ino, new.st_ino, "unchanged checkpoint was re-copied")
        self.assertGreaterEqual(new.st_nlink, 2)
        with open(os.path.join(self.dest, second, "training_log.csv"), encoding="utf-8") as f:
            self.assertEqual(len(f.read().splitlines()), 4, "csv did not advance")

    def test_a_torn_final_csv_line_is_trimmed_and_recorded(self):
        self._record()
        self._write("training_log.csv", "run_id,iteration,loss\n1,1,0.5\n1,2,0.")
        proc = self._run(self.dest)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        stamp = self._latest()
        self.assertIn("trimmed_torn_line", self._item_note(stamp, "training_log.csv"))
        with open(os.path.join(self.dest, stamp, "training_log.csv"), encoding="utf-8") as f:
            self.assertEqual(f.read(), "run_id,iteration,loss\n1,1,0.5\n")

    def test_unparseable_json_publishes_but_reports_a_suspect_item(self):
        self._record()
        self._write("ladder.json", '{"anchors": [')
        proc = self._run(self.dest)
        self.assertEqual(proc.returncode, 3, proc.stdout + proc.stderr)
        stamp = self._latest()
        self.assertEqual(self._item_note(stamp, "ladder.json"), "invalid_json")
        self.assertIn("status=complete_with_suspect_items", self._manifest(stamp))

    def test_a_source_with_no_record_fails_rather_than_publishing_nothing(self):
        for extra in (None, "checkpoints"):
            with self.subTest(extra=extra):
                if extra:
                    os.makedirs(os.path.join(self.src, extra), exist_ok=True)
                proc = self._run(self.dest)
                self.assertEqual(proc.returncode, 1, proc.stdout)
                self.assertFalse(os.path.exists(os.path.join(self.dest, "LATEST")))

    def test_no_destination_is_a_usage_error(self):
        env = dict(os.environ)
        env.pop("POLYFISH_BACKUP_DIR", None)
        env["POLYFISH_RUN_DIR"] = self.src
        proc = subprocess.run([SCRIPT], capture_output=True, text=True, env=env)
        self.assertEqual(proc.returncode, 2)
        self.assertIn("Usage:", proc.stderr)

    def test_the_backed_up_items_cover_what_a_resume_needs(self):
        self._record()
        self._write(".current_run", "1755900000\n")
        self.assertEqual(self._run(self.dest).returncode, 0)
        snap = os.path.join(self.dest, self._latest())
        self.assertTrue(os.path.exists(os.path.join(snap, ".current_run")))


class LoopWiringTest(unittest.TestCase):
    """The bug audit T3 filed was not that the script was wrong, it was that
    nothing ran it. These pin the loop's side: it calls the script, on a
    cadence and on exit, and a failed snapshot does not end the campaign."""

    @classmethod
    def setUpClass(cls):
        with open(LOOP_SCRIPT, encoding="utf-8") as f:
            cls.loop = f.read()

    def test_the_loop_invokes_the_backup_script(self):
        self.assertIn("scripts/backup_experiment_record.sh", self.loop)

    def test_the_snapshot_is_taken_on_a_cadence_and_on_exit(self):
        calls = re.findall(r"^\s*snapshot_record .*$", self.loop, re.M)
        self.assertGreaterEqual(len(calls), 2, f"only found {calls}")
        self.assertRegex(self.loop, r"i % BACKUP_EVERY")
        cleanup = self.loop.split("cleanup() {", 1)[1].split("\n}", 1)[0]
        self.assertIn("snapshot_record", cleanup, "no snapshot on the way out")

    def test_an_unset_backup_dir_is_announced_rather_than_silent(self):
        self.assertIn("POLYFISH_BACKUP_DIR is unset", self.loop)

    def _snapshot_record_source(self):
        match = re.search(r"^snapshot_record \(\) \{.*?^\}", self.loop, re.M | re.S)
        self.assertIsNotNone(match, "run_training_loop.sh defines no snapshot_record")
        return match.group(0)

    def _drive(self, stub_exit):
        """Run the loop's own snapshot_record under `set -e`, against a backup
        script that exits with the given status."""
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "scripts"))
            stub = os.path.join(tmp, "scripts", "backup_experiment_record.sh")
            with open(stub, "w", encoding="utf-8") as f:
                f.write(f"#!/bin/sh\nexit {stub_exit}\n")
            os.chmod(stub, 0o755)
            script = (
                "set -e\n"
                "RECORD_DIRTY=1\n"
                + self._snapshot_record_source()
                + '\nsnapshot_record "test"\n'
                'echo "dirty=$RECORD_DIRTY"\n'
                'echo SURVIVED\n'
            )
            return subprocess.run(
                ["bash", "-c", script], cwd=tmp, capture_output=True, text=True,
                env=dict(os.environ, POLYFISH_BACKUP_DIR=os.path.join(tmp, "dest")),
            )

    def test_a_failed_snapshot_does_not_end_the_run(self):
        proc = self._drive(1)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("SURVIVED", proc.stdout)
        self.assertIn("BACKUP: snapshot failed", proc.stderr)
        self.assertIn("dirty=1", proc.stdout, "a failed snapshot must stay pending")

    def test_a_published_snapshot_clears_the_pending_flag(self):
        for status in (0, 3):
            with self.subTest(status=status):
                proc = self._drive(status)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertIn("dirty=0", proc.stdout)

    def test_no_backup_dir_makes_the_snapshot_a_no_op(self):
        source = self._snapshot_record_source()
        script = "set -e\nRECORD_DIRTY=1\n" + source + '\nsnapshot_record "test"\necho SURVIVED\n'
        env = dict(os.environ)
        env.pop("POLYFISH_BACKUP_DIR", None)
        proc = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("SURVIVED", proc.stdout)


if __name__ == "__main__":
    unittest.main()

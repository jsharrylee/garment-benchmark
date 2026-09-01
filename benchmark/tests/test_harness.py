import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from benchmark.harness.artifact_registry import ArtifactRegistry
from benchmark.harness.executor import Executor, now
from benchmark.harness.repair_policy import next_failure_state
from benchmark.harness.schemas import ActionReceipt, Stage
from benchmark.harness.state import RunState, TransitionError, atomic_json_write
from benchmark.harness.validators import coherent_views, finite_nonempty, nonempty_mask, provenance_matches, valid_references


class HarnessTests(unittest.TestCase):
    def receipt(self, action_id, next_stage):
        return ActionReceipt(action_id, 0, now(), "logs/out.log", "logs/err.log", next_stage=next_stage)

    def test_forced_failure_single_repair_and_idempotent_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root, calls = Path(directory), []
            runner = Executor(root)
            runner.run("sample", Stage.DISCOVER, "forced failure", lambda: self.receipt("sample:DISCOVER:1", Stage.ACQUIRE))
            runner.run("sample", Stage.ACQUIRE, "repair download", lambda: calls.append("repair") or self.receipt("sample:ACQUIRE:2", Stage.VALIDATE_RAW), attempt=2, repair_hypothesis="retry interrupted official download")
            resumed = Executor(root)
            self.assertIsNone(resumed.run("sample", Stage.ACQUIRE, "repair download", lambda: calls.append("duplicate") or self.receipt("x", Stage.VALIDATE_RAW), attempt=2))
            events = [json.loads(line) for line in (root / "state/events.jsonl").read_text().splitlines()]
            self.assertEqual(calls, ["repair"])
            self.assertEqual([event["type"] for event in events].count("intent"), 2)
            self.assertEqual(next_failure_state(1), None)
            self.assertEqual(next_failure_state(2), Stage.FAILED_VALIDATION)

    def test_transition_validation_and_atomic_write(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            atomic_json_write(path, {"safe": True})
            self.assertEqual(json.loads(path.read_text()), {"safe": True})
            state = RunState(path)
            with self.assertRaises(TransitionError):
                state.transition("j", Stage.ACQUIRE, "a")

    def test_checksum_mismatch_and_output_validators(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); thing = root / "artifact.txt"; thing.write_text("one")
            record = ArtifactRegistry(root / "index.jsonl").register(thing, "j", "input")
            thing.write_text("two")
            self.assertFalse(ArtifactRegistry.verify(record))
        self.assertEqual(finite_nonempty([]), (False, "EMPTY_OUTPUT"))
        self.assertEqual(finite_nonempty([1.0, float("nan")]), (False, "NONFINITE_OUTPUT"))
        self.assertEqual(valid_references([0, 2], 2), (False, "INVALID_PATTERN_STRUCTURE"))

    def test_failed_archive_integrity_and_invalid_multiview_group(self):
        with tempfile.TemporaryDirectory() as directory:
            broken = Path(directory) / "broken.zip"
            broken.write_bytes(b"not-a-zip")
            with self.assertRaises(zipfile.BadZipFile):
                with zipfile.ZipFile(broken) as archive:
                    archive.testzip()
        self.assertEqual(coherent_views(["front", "left", "front", "back"]), (False, "NO_COHERENT_MULTIVIEW_GROUP"))

    def test_empty_mask_and_provenance_mismatch(self):
        self.assertEqual(nonempty_mask(0), (False, "MASK_FAILURE"))
        self.assertEqual(provenance_matches("input-a", "input-b"), (False, "OUTPUT_NOT_BOUND_TO_INPUT"))

    def test_resume_after_interruption_does_not_duplicate_intent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            interrupted = Executor(root)
            interrupted._event({"type": "intent", "action_id": "j:DISCOVER:1"})
            interrupted.state.transition("j", Stage.DISCOVER, "j:DISCOVER:1")
            resumed = Executor(root)
            # An intent without a receipt may be safely resumed exactly once.
            resumed.run("j", Stage.DISCOVER, "retry after interruption", lambda: self.receipt("j:DISCOVER:1", Stage.ACQUIRE))
            events = [json.loads(line) for line in (root / "state/events.jsonl").read_text().splitlines()]
            self.assertEqual([e["type"] for e in events].count("receipt"), 1)

"""Resumability tests.

These cover the file-based state that lets a run resume after a crash or
quota hit: scenes.json (hot-written per scene), veo_progress.json (per
segment), and batch_jobs.json (batch registry). We test the data formats
and the batch registry directly, and use lightweight simulations for the
main.py loops.
"""
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import batch_stylizer
from models import QuotaExceededError


class TestScenesJsonHotWrite(unittest.TestCase):
    """Simulate main.py's hot-write: after each scene, flush scenes[] to disk
    so a mid-loop QuotaExceededError leaves a valid partial file."""

    def test_partial_file_is_valid_json_after_quota_mid_loop(self):
        with tempfile.TemporaryDirectory() as d:
            scenes_file = os.path.join(d, "scenes.json")

            scenes = [
                {"scene_index": 0, "description": "first", "stylized_frames": [{"path": "/s0.png", "original_frame_index": 0}]},
                {"scene_index": 1, "description": "second", "stylized_frames": [{"path": "/s1.png", "original_frame_index": 1}]},
                {"scene_index": 2},  # not yet processed
            ]

            def save(upto: int):
                with open(scenes_file, "w") as f:
                    json.dump({"fps": 24.0, "scenes": scenes[: upto + 1]}, f, indent=2)

            # Simulate two scenes completing, then quota on scene 3.
            save(0)
            save(1)
            try:
                raise QuotaExceededError("Stylizer daily quota exceeded.")
            except QuotaExceededError:
                pass  # main.py would exit here

            # File is valid and contains exactly the completed scenes.
            with open(scenes_file) as f:
                data = json.load(f)
            self.assertEqual(len(data["scenes"]), 2)
            self.assertEqual(data["scenes"][0]["description"], "first")
            self.assertEqual(data["scenes"][1]["description"], "second")


class TestVeoProgressFile(unittest.TestCase):
    """veo_progress.json tracks per-segment state. Resume should skip any
    segment already in state 'synced'."""

    def _load(self, path):
        with open(path) as f:
            return json.load(f)

    def _save(self, path, data):
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def test_synced_entries_are_skipped_on_resume(self):
        with tempfile.TemporaryDirectory() as d:
            progress = os.path.join(d, "veo_progress.json")
            self._save(progress, {
                "seg_a": {"state": "synced", "sync_path": "/a.mp4"},
                "seg_b": {"state": "pending", "dur_str": "4"},
            })
            data = self._load(progress)
            self.assertEqual(data["seg_a"]["state"], "synced")
            # main.py's resume logic: skip synced, re-run pending.
            to_run = [k for k, v in data.items() if v["state"] != "synced"]
            self.assertEqual(to_run, ["seg_b"])

    def test_failed_state_records_reason_for_debugging(self):
        with tempfile.TemporaryDirectory() as d:
            progress = os.path.join(d, "veo_progress.json")
            self._save(progress, {
                "seg_x": {"state": "failed", "reason": "ffmpeg exit 1"},
            })
            data = self._load(progress)
            self.assertEqual(data["seg_x"]["state"], "failed")
            self.assertIn("ffmpeg", data["seg_x"]["reason"])


class TestBatchRegistryResume(unittest.TestCase):
    """batch_jobs.json short-circuits re-submission on rerun. This test
    asserts the full resume path: prior RUNNING job → poll directly, no
    re-upload, no re-create."""

    def test_rerun_uses_existing_job_name_and_skips_resubmission(self):
        with tempfile.TemporaryDirectory() as session_dir:
            items = [
                {"key": "frame_000001", "frame_path": "/doesnt/matter/1.jpg",
                 "original_frame_index": 1, "prompt": "p"},
            ]
            registry_key = batch_stylizer.compute_batch_key("flash-img", items)
            batch_stylizer.save_registry(session_dir, {
                registry_key: {
                    "job_name": "batches/already-running",
                    "state": "JOB_STATE_RUNNING",
                    "submitted_at": 0,
                    "n_requests": 1,
                    "model_id": "flash-img",
                },
            })

            client = MagicMock()
            job_name, resumed = batch_stylizer.get_or_submit_job(
                client, items, "flash-img", session_dir,
                temperature=0.7, top_p=0.95, top_k=40,
            )
            self.assertEqual(job_name, "batches/already-running")
            self.assertTrue(resumed)
            # Critical: did NOT touch files.upload or batches.create.
            client.files.upload.assert_not_called()
            client.batches.create.assert_not_called()

    def test_rerun_after_success_still_skips_resubmission(self):
        """If a prior run submitted a batch and the job succeeded, but the
        process died before processing results, the next run should pick up
        the same SUCCEEDED job rather than submitting fresh."""
        with tempfile.TemporaryDirectory() as session_dir:
            items = [{"key": "f1", "frame_path": "/a/1.jpg",
                      "original_frame_index": 1, "prompt": "p"}]
            reg_key = batch_stylizer.compute_batch_key("m", items)
            batch_stylizer.save_registry(session_dir, {
                reg_key: {"job_name": "batches/done", "state": "JOB_STATE_SUCCEEDED"},
            })

            client = MagicMock()
            name, resumed = batch_stylizer.get_or_submit_job(
                client, items, "m", session_dir,
                temperature=0.7, top_p=0.95, top_k=40,
            )
            self.assertEqual(name, "batches/done")
            self.assertTrue(resumed)
            client.batches.create.assert_not_called()

    def test_expired_prior_job_triggers_resubmission(self):
        with tempfile.TemporaryDirectory() as session_dir:
            items = [{"key": "f1", "frame_path": "/a/1.jpg",
                      "original_frame_index": 1, "prompt": "p"}]
            reg_key = batch_stylizer.compute_batch_key("m", items)
            batch_stylizer.save_registry(session_dir, {
                reg_key: {"job_name": "batches/old", "state": "JOB_STATE_EXPIRED"},
            })

            client = MagicMock()
            client.files.upload.return_value.name = "files/fresh-input"
            client.batches.create.return_value.name = "batches/fresh"

            with patch.object(batch_stylizer, "_write_jsonl", return_value=1):
                name, resumed = batch_stylizer.get_or_submit_job(
                    client, items, "m", session_dir,
                    temperature=0.7, top_p=0.95, top_k=40,
                )
            self.assertEqual(name, "batches/fresh")
            self.assertFalse(resumed)


if __name__ == "__main__":
    unittest.main()

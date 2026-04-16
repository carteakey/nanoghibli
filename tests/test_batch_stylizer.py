import base64
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import batch_stylizer
from batch_stylizer import (
    BatchJobFailed,
    REGISTRY_FILENAME,
    compute_batch_key,
    get_or_submit_job,
    load_registry,
    poll_batch_job,
    process_batch_results,
    save_registry,
    _frame_to_request,
)
from models import QuotaExceededError


def _fake_job(state: str, file_name: str = "files/result-123", name: str = "batches/abc"):
    """Build a MagicMock that mimics the SDK's job object shape."""
    job = MagicMock()
    job.name = name
    job.state.name = state
    job.dest.file_name = file_name
    job.error = None if state == "JOB_STATE_SUCCEEDED" else {"message": "something"}
    return job


class TestBatchKeyAndRegistry(unittest.TestCase):
    def test_compute_key_stable_for_same_inputs(self):
        items_a = [{"key": "f1", "frame_path": "/a/1.jpg", "prompt": "p1"},
                   {"key": "f2", "frame_path": "/a/2.jpg", "prompt": "p2"}]
        items_b = list(reversed(items_a))  # same set, different order
        self.assertEqual(
            compute_batch_key("m", items_a),
            compute_batch_key("m", items_b),
        )

    def test_compute_key_differs_on_model_change(self):
        items = [{"key": "f1", "frame_path": "/a/1.jpg", "prompt": "p1"}]
        self.assertNotEqual(
            compute_batch_key("flash", items),
            compute_batch_key("pro", items),
        )

    def test_registry_roundtrip(self):
        with tempfile.TemporaryDirectory() as session_dir:
            self.assertEqual(load_registry(session_dir), {})
            save_registry(session_dir, {"k": {"job_name": "batches/1"}})
            self.assertEqual(
                load_registry(session_dir),
                {"k": {"job_name": "batches/1"}},
            )
            # File was written at the expected path.
            self.assertTrue(os.path.exists(os.path.join(session_dir, REGISTRY_FILENAME)))


class TestJSONLSerialization(unittest.TestCase):
    def test_frame_to_request_includes_image_and_prompt(self):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
            tf.write(b"fake-image-bytes")
            frame_path = tf.name
        try:
            item = {
                "key": "frame_42",
                "frame_path": frame_path,
                "original_frame_index": 42,
                "prompt": "stylize it",
            }
            req = _frame_to_request(item, temperature=0.5, top_p=0.9, top_k=40)

            self.assertEqual(req["key"], "frame_42")
            parts = req["request"]["contents"][0]["parts"]
            self.assertEqual(parts[0]["text"], "stylize it")
            # Image payload is base64 of the file bytes.
            decoded = base64.b64decode(parts[1]["inlineData"]["data"])
            self.assertEqual(decoded, b"fake-image-bytes")
            # Image generation requires responseModalities.
            self.assertIn("IMAGE", req["request"]["generation_config"]["responseModalities"])
            self.assertEqual(req["request"]["generation_config"]["temperature"], 0.5)
        finally:
            os.unlink(frame_path)


class TestGetOrSubmitJob(unittest.TestCase):
    def test_resumes_existing_running_job_without_resubmission(self):
        with tempfile.TemporaryDirectory() as session_dir:
            items = [{"key": "f1", "frame_path": "/a/1.jpg", "prompt": "p"}]
            key = compute_batch_key("flash-img", items)
            save_registry(session_dir, {
                key: {"job_name": "batches/existing", "state": "JOB_STATE_RUNNING"},
            })
            client = MagicMock()
            name, resumed = get_or_submit_job(
                client, items, "flash-img", session_dir,
                temperature=0.7, top_p=0.95, top_k=40,
            )
            self.assertEqual(name, "batches/existing")
            self.assertTrue(resumed)
            # No submission attempted.
            client.files.upload.assert_not_called()
            client.batches.create.assert_not_called()

    def test_resubmits_if_prior_job_failed(self):
        with tempfile.TemporaryDirectory() as session_dir:
            items = [{"key": "f1", "frame_path": "/a/1.jpg", "prompt": "p"}]
            key = compute_batch_key("flash-img", items)
            save_registry(session_dir, {
                key: {"job_name": "batches/old", "state": "JOB_STATE_FAILED"},
            })
            client = MagicMock()
            client.files.upload.return_value = MagicMock(name="files/new-input")
            client.files.upload.return_value.name = "files/new-input"
            client.batches.create.return_value = MagicMock(name="batches/fresh")
            client.batches.create.return_value.name = "batches/fresh"

            with patch.object(batch_stylizer, "_write_jsonl", return_value=1):
                name, resumed = get_or_submit_job(
                    client, items, "flash-img", session_dir,
                    temperature=0.7, top_p=0.95, top_k=40,
                )
            self.assertEqual(name, "batches/fresh")
            self.assertFalse(resumed)
            client.batches.create.assert_called_once()


class TestPolling(unittest.TestCase):
    def test_returns_job_on_success(self):
        client = MagicMock()
        client.batches.get.side_effect = [
            _fake_job("JOB_STATE_RUNNING"),
            _fake_job("JOB_STATE_SUCCEEDED"),
        ]
        with patch("time.sleep"):
            job = poll_batch_job(client, "batches/abc", poll_interval=0, max_wait_hours=1)
        self.assertEqual(job.state.name, "JOB_STATE_SUCCEEDED")

    def test_raises_on_failed_terminal_state(self):
        client = MagicMock()
        client.batches.get.return_value = _fake_job("JOB_STATE_FAILED")
        with patch("time.sleep"), self.assertRaises(BatchJobFailed):
            poll_batch_job(client, "batches/abc", poll_interval=0, max_wait_hours=1)

    def test_quota_error_translates_to_QuotaExceededError(self):
        client = MagicMock()
        client.batches.get.side_effect = RuntimeError("RESOURCE_EXHAUSTED: quota exceeded")
        with patch("time.sleep"), self.assertRaises(QuotaExceededError):
            poll_batch_job(client, "batches/abc", poll_interval=0, max_wait_hours=1)

    def test_registry_updated_on_each_poll(self):
        with tempfile.TemporaryDirectory() as session_dir:
            save_registry(session_dir, {
                "k": {"job_name": "batches/abc", "state": "JOB_STATE_PENDING"},
            })
            client = MagicMock()
            client.batches.get.side_effect = [
                _fake_job("JOB_STATE_RUNNING"),
                _fake_job("JOB_STATE_SUCCEEDED"),
            ]
            with patch("time.sleep"):
                poll_batch_job(
                    client, "batches/abc",
                    session_dir=session_dir, poll_interval=0, max_wait_hours=1,
                )
            reg = load_registry(session_dir)
            self.assertEqual(reg["k"]["state"], "JOB_STATE_SUCCEEDED")


class TestProcessResults(unittest.TestCase):
    def _write_png(self, path):
        # Tiny valid PNG (1x1 white). Not strictly required to be valid for
        # these tests, but makes debug output sensible.
        with open(path, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\nfake-image")

    def test_processes_jsonl_and_writes_output_plus_cache(self):
        with tempfile.TemporaryDirectory() as workdir:
            frame_path = os.path.join(workdir, "f1.jpg")
            self._write_png(frame_path)

            stylize_items = [{
                "key": "frame_000042",
                "frame_path": frame_path,
                "original_frame_index": 42,
                "prompt": "stylize",
            }]

            # Fake result JSONL (one line per request, camelCase inlineData).
            image_bytes = b"\x89PNG\r\n\x1a\n-result-bytes-"
            result_line = json.dumps({
                "key": "frame_000042",
                "response": {
                    "candidates": [{
                        "content": {"parts": [
                            {"text": "ok"},
                            {"inlineData": {"mimeType": "image/png",
                                             "data": base64.b64encode(image_bytes).decode()}},
                        ]},
                    }],
                },
            })

            client = MagicMock()
            client.files.download.return_value = (result_line + "\n").encode("utf-8")
            job = _fake_job("JOB_STATE_SUCCEEDED", file_name="files/result-xyz")

            output_dir = os.path.join(workdir, "out")
            cache_dir = os.path.join(workdir, "cache")
            results = process_batch_results(
                client, job, stylize_items, output_dir, cache_dir,
                model_id="gemini-3.1-flash-image-preview",
            )
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["original_frame_index"], 42)
            # Output file exists with the expected naming.
            expected = os.path.join(output_dir, "stylized_000042.png")
            self.assertTrue(os.path.exists(expected))
            with open(expected, "rb") as f:
                self.assertEqual(f.read(), image_bytes)
            # Cache populated under MD5 + model slug.
            cache_files = os.listdir(cache_dir)
            self.assertEqual(len(cache_files), 1)
            self.assertTrue(cache_files[0].endswith("_flash.png"))

    def test_raises_if_job_has_no_dest_file(self):
        with tempfile.TemporaryDirectory() as workdir:
            client = MagicMock()
            job = MagicMock()
            job.name = "batches/x"
            job.dest = None
            with self.assertRaises(BatchJobFailed):
                process_batch_results(
                    client, job, [], os.path.join(workdir, "o"),
                    os.path.join(workdir, "c"), "gemini-3.1-flash-image-preview",
                )


if __name__ == "__main__":
    unittest.main()

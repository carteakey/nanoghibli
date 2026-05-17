import json
import os
import sys
import tempfile
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from manual_chatgpt import MANIFEST_FILENAME, manual_upload_path, queue_manual_frames


class TestManualChatGPTFallback(unittest.TestCase):
    def test_queue_writes_manifest_and_readme_without_ready_upload(self):
        with tempfile.TemporaryDirectory() as d:
            frame_path = os.path.join(d, "frame.jpg")
            with open(frame_path, "wb") as f:
                f.write(b"source-frame")

            ready = queue_manual_frames(
                [{"path": frame_path, "original_frame_index": 7}],
                "stylize this",
                session_dir=d,
                output_dir=os.path.join(d, "stylized"),
                cache_dir=os.path.join(d, "cache"),
            )

            self.assertEqual(ready, [])
            manifest_path = os.path.join(d, "manual_chatgpt", MANIFEST_FILENAME)
            self.assertTrue(os.path.exists(manifest_path))
            with open(manifest_path) as f:
                manifest = json.load(f)
            self.assertEqual(manifest["items"][0]["original_frame_index"], 7)
            self.assertIn("stylize this", manifest["items"][0]["prompt"])
            self.assertTrue(os.path.exists(os.path.join(d, "manual_chatgpt", "README.md")))

    def test_existing_upload_is_imported_to_output_and_cache(self):
        with tempfile.TemporaryDirectory() as d:
            frame_path = os.path.join(d, "frame.jpg")
            with open(frame_path, "wb") as f:
                f.write(b"source-frame")

            upload_path = manual_upload_path(d, 12)
            os.makedirs(os.path.dirname(upload_path), exist_ok=True)
            with open(upload_path, "wb") as f:
                f.write(b"manual-result")

            output_dir = os.path.join(d, "stylized")
            cache_dir = os.path.join(d, "cache")
            ready = queue_manual_frames(
                [{"path": frame_path, "original_frame_index": 12}],
                "stylize this",
                session_dir=d,
                output_dir=output_dir,
                cache_dir=cache_dir,
            )

            self.assertEqual(
                ready,
                [{"path": os.path.join(output_dir, "stylized_000012.png"), "original_frame_index": 12}],
            )
            with open(ready[0]["path"], "rb") as f:
                self.assertEqual(f.read(), b"manual-result")
            cache_files = os.listdir(cache_dir)
            self.assertEqual(len(cache_files), 1)
            self.assertTrue(cache_files[0].endswith("_chatgpt_manual.png"))


if __name__ == "__main__":
    unittest.main()

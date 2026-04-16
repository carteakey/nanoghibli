import os
import sys
import threading
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from models import UsageMetrics


class _Meta:
    def __init__(self, prompt=0, candidates=0, thoughts=0, cached=0):
        self.prompt_token_count = prompt
        self.candidates_token_count = candidates
        self.thoughts_token_count = thoughts
        self.cached_content_token_count = cached


class _Resp:
    def __init__(self, **kw):
        self.usage_metadata = _Meta(**kw)


class TestUsageMetrics(unittest.TestCase):
    def test_flash_stylization_cost(self):
        m = UsageMetrics(model_tier="flash")
        # 5 frames: 500 in, 300 out tokens each, on flash-image
        for _ in range(5):
            m.add_usage(_Resp(prompt=500, candidates=300), "gemini-3.1-flash-image-preview")
            m.add_image("gemini-3.1-flash-image-preview")
        # tokens: 2500 in * 0.50/1M + 1500 out * 3.00/1M = 0.00125 + 0.0045 = 0.00575
        # images: 5 * 0.067 = 0.335
        self.assertAlmostEqual(m._compute_cost(), 0.00575 + 0.335, places=6)

    def test_pro_stylization_cost(self):
        m = UsageMetrics(model_tier="pro")
        for _ in range(10):
            m.add_usage(_Resp(prompt=1000, candidates=500, thoughts=200), "gemini-3-pro-image-preview")
            m.add_image("gemini-3-pro-image-preview")
        # tokens: 10000 * 2/1M + 7000 * 12/1M = 0.02 + 0.084 = 0.104
        # images: 10 * 0.134 = 1.34
        self.assertAlmostEqual(m._compute_cost(), 0.104 + 1.34, places=6)

    def test_veo_priced_by_seconds(self):
        m = UsageMetrics()
        m.add_video("veo-3.1-fast-generate-preview", 4)
        m.add_video("veo-3.1-fast-generate-preview", 8)
        # 12s * $0.15 = $1.80
        self.assertAlmostEqual(m._compute_cost(), 1.80, places=6)
        self.assertEqual(m.total_videos, 2)
        self.assertAlmostEqual(m.total_video_seconds, 12.0)

    def test_unknown_model_excluded(self):
        m = UsageMetrics()
        m.add_usage(_Resp(prompt=1000, candidates=1000), "unknown-model-x")
        m.add_image("unknown-model-x")
        m.add_video("unknown-veo", 10)
        # All unknown — cost should be 0, not a crash.
        self.assertEqual(m._compute_cost(), 0.0)

    def test_no_usage_metadata_does_not_crash(self):
        class Bare:
            pass
        m = UsageMetrics()
        m.add_usage(Bare(), "gemini-3.1-flash-lite-preview")
        self.assertEqual(m.total_input_tokens, 0)

    def test_batch_halves_token_cost(self):
        """Batch API calls bill at 50% of sync. Verify per-token rate."""
        sync = UsageMetrics()
        batch = UsageMetrics()
        # Same usage routed through sync vs batch.
        for _ in range(5):
            sync.add_usage(_Resp(prompt=1000, candidates=1000), "gemini-3.1-flash-image-preview")
            batch.add_usage(_Resp(prompt=1000, candidates=1000), "gemini-3.1-flash-image-preview", is_batch=True)
        # sync: 5000 * 0.50/1M + 5000 * 3/1M = 0.0025 + 0.015 = 0.0175
        # batch: 0.5 * 0.0175 = 0.00875
        self.assertAlmostEqual(sync._compute_cost(), 0.0175, places=6)
        self.assertAlmostEqual(batch._compute_cost(), 0.00875, places=6)

    def test_batch_halves_image_cost(self):
        sync = UsageMetrics()
        batch = UsageMetrics()
        for _ in range(10):
            sync.add_image("gemini-3-pro-image-preview")
            batch.add_image("gemini-3-pro-image-preview", is_batch=True)
        # sync: 10 * 0.134 = 1.34; batch: 0.67
        self.assertAlmostEqual(sync._compute_cost(), 1.34, places=6)
        self.assertAlmostEqual(batch._compute_cost(), 0.67, places=6)

    def test_batch_and_sync_keep_separate_buckets(self):
        """A single session can mix sync + batch for the same model. Each
        should stay in its own bucket and display with a [batch 50%] tag."""
        m = UsageMetrics()
        m.add_usage(_Resp(prompt=500, candidates=500), "gemini-3.1-flash-image-preview")
        m.add_usage(_Resp(prompt=500, candidates=500), "gemini-3.1-flash-image-preview", is_batch=True)
        m.add_image("gemini-3.1-flash-image-preview")
        m.add_image("gemini-3.1-flash-image-preview", is_batch=True)

        self.assertIn("gemini-3.1-flash-image-preview", m.tokens)
        self.assertIn("gemini-3.1-flash-image-preview:batch", m.tokens)
        self.assertEqual(m.total_images, 2)

        # Cost: sync = 500/1M*0.50 + 500/1M*3 + 0.067 = 0.00025+0.0015+0.067 = 0.06875
        #       batch = half of sync = 0.034375
        #       total = 0.103125
        self.assertAlmostEqual(m._compute_cost(), 0.103125, places=6)

        # Display marks batch entries.
        rendered = str(m)
        self.assertIn("[batch 50%]", rendered)

    def test_thread_safety(self):
        m = UsageMetrics()

        def worker():
            for _ in range(1000):
                m.add_usage(_Resp(prompt=1, candidates=1), "gemini-3.1-flash-lite-preview")
                m.add_image("gemini-3.1-flash-image-preview")
                m.add_description()

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads: t.start()
        for t in threads: t.join()

        self.assertEqual(m.total_input_tokens, 8000)
        self.assertEqual(m.total_output_tokens, 8000)
        self.assertEqual(m.total_images, 8000)
        self.assertEqual(m.descriptions_generated, 8000)


if __name__ == "__main__":
    unittest.main()

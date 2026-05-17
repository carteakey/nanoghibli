import os
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from sequence_trailer import (  # noqa: E402
    _apply_sequence_preset,
    _crop_black_borders,
    _write_normalized_frame,
)


class TestSequenceTrailer(unittest.TestCase):
    def test_clean_still_trailer_preset_preserves_source_black_and_ranges(self):
        values = _apply_sequence_preset(
            "clean_still_trailer",
            source_ranges="",
            title_source_ranges="49-63",
            tail_source_range="1130-1289",
            source_black_mean_threshold=None,
            source_black_max_threshold=45,
            crop_source_bars=False,
            source_fit_mode="contain",
            generated_fit_mode="contain",
            overlay_letterbox_pixels=129,
        )

        self.assertEqual(values["source_ranges"], "49-63,1130-1289")
        self.assertEqual(values["source_black_mean_threshold"], 5.0)
        self.assertEqual(values["source_black_max_threshold"], 45)
        self.assertTrue(values["crop_source_bars"])
        self.assertEqual(values["source_fit_mode"], "cover")
        self.assertEqual(values["generated_fit_mode"], "cover")
        self.assertEqual(values["overlay_letterbox_pixels"], 129)

    def test_crop_black_borders_handles_all_black_frames(self):
        frame = Image.new("RGB", (64, 36), "black")
        cropped = _crop_black_borders(frame)
        self.assertEqual(cropped.size, frame.size)

    def test_fixed_letterbox_overlay_is_applied_after_cover_fit(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src.png"
            out = Path(tmp) / "out.png"
            Image.new("RGB", (100, 100), (200, 80, 40)).save(src)

            _write_normalized_frame(
                src,
                out,
                width=160,
                height=90,
                fit_mode="cover",
                overlay_letterbox_pixels=10,
            )

            result = Image.open(out).convert("RGB")
            self.assertEqual(result.size, (160, 90))
            self.assertEqual(result.getpixel((80, 5)), (0, 0, 0))
            self.assertEqual(result.getpixel((80, 84)), (0, 0, 0))
            self.assertEqual(result.getpixel((80, 45)), (200, 80, 40))


if __name__ == "__main__":
    unittest.main()

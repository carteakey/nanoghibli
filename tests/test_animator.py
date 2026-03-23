import os
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import cv2
import sys
from PIL import Image

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from animator import create_video_from_frames

class TestAnimator(unittest.TestCase):
    def setUp(self):
        self.output_dir = "tests/test_output"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Create some dummy images
        self.dummy_frames = []
        for i in range(3):
            path = os.path.join(self.output_dir, f"frame_{i}.png")
            img = Image.new("RGB", (100, 100), color=(i*50, 0, 0))
            img.save(path)
            self.dummy_frames.append({"path": path, "original_frame_index": i})

    def tearDown(self):
        # Cleanup test output
        if os.path.exists(self.output_dir):
            import shutil
            shutil.rmtree(self.output_dir)

    @patch("cv2.VideoWriter")
    def test_create_video(self, mock_videowriter):
        mock_out = MagicMock()
        mock_videowriter.return_value = mock_out
        
        output_path = os.path.join(self.output_dir, "test_video.mp4")
        create_video_from_frames(self.dummy_frames, output_path, fps=30.0)
        
        mock_videowriter.assert_called()
        self.assertEqual(mock_out.write.call_count, 3)
        mock_out.release.assert_called()

    @patch("PIL.Image.Image.save")
    def test_create_gif(self, mock_save):
        output_path = os.path.join(self.output_dir, "test_video.gif")
        create_video_from_frames(self.dummy_frames, output_path, fps=10.0)
        
        # PIL's save is called once for the first image with append_images
        mock_save.assert_called()
        # Verify duration is set correctly (1000/10 = 100)
        args, kwargs = mock_save.call_args
        self.assertEqual(kwargs["duration"], 100)

if __name__ == "__main__":
    unittest.main()

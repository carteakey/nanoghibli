import os
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import cv2
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from extractor import extract_scenes_from_video

class TestExtractor(unittest.TestCase):
    def setUp(self):
        self.output_dir = "tests/test_output"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def tearDown(self):
        # Cleanup test output
        if os.path.exists(self.output_dir):
            import shutil
            shutil.rmtree(self.output_dir)

    @patch("cv2.VideoCapture")
    @patch("extractor.detect")
    @patch("cv2.imwrite")
    def test_extract_scenes(self, mock_imwrite, mock_detect, mock_video_capture):
        # Mock VideoCapture
        mock_cap = MagicMock()
        mock_video_capture.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_COUNT: 10,
            cv2.CAP_PROP_POS_FRAMES: 0
        }.get(prop, 0)
        
        # Mock frame reading
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_cap.read.side_effect = [(True, dummy_frame)] * 10 + [(False, None)]
        
        # Mock scene detection
        mock_scene_start = MagicMock()
        mock_scene_start.get_frames.return_value = 0
        mock_scene_end = MagicMock()
        mock_scene_end.get_frames.return_value = 10
        mock_detect.return_value = [(mock_scene_start, mock_scene_end)]

        scenes, fps = extract_scenes_from_video("dummy_video.mp4", self.output_dir)
        
        self.assertEqual(fps, 30.0)
        self.assertEqual(len(scenes), 1)
        self.assertEqual(scenes[0]["start_frame"], 0)
        self.assertEqual(scenes[0]["end_frame"], 10)
        # First frame is always a keyframe
        self.assertGreater(len(scenes[0]["frames"]), 0)
        mock_imwrite.assert_called()

    @patch("cv2.VideoCapture")
    @patch("extractor.detect")
    @patch("cv2.imwrite")
    @patch("cv2.resize")
    def test_resize_on_extraction(self, mock_resize, mock_imwrite, mock_detect, mock_video_capture):
        # Mock VideoCapture
        mock_cap = MagicMock()
        mock_video_capture.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_COUNT: 5,
        }.get(prop, 0)
        
        # Mock a large frame (2000x2000)
        large_frame = np.zeros((2000, 2000, 3), dtype=np.uint8)
        mock_cap.read.side_effect = [(True, large_frame)] * 5 + [(False, None)]
        
        # Mock scene detection (one scene, all 5 frames)
        mock_scene_start = MagicMock()
        mock_scene_start.get_frames.return_value = 0
        mock_scene_end = MagicMock()
        mock_scene_end.get_frames.return_value = 5
        mock_detect.return_value = [(mock_scene_start, mock_scene_end)]

        # Mock resize to return a smaller frame
        mock_resize.return_value = np.zeros((1080, 1080, 3), dtype=np.uint8)

        extract_scenes_from_video("dummy_video.mp4", self.output_dir)
        
        # Check if resize was called for the large frame (ignoring the internal 320x180 resize for motion detection)
        # The internal resize is called for EVERY frame.
        # The quality resize is called only for keyframes if they are too large.
        
        # How to distinguish between them? 
        # The internal one is (320, 180). The quality one is (1080, 1080).
        resize_calls = [call for call in mock_resize.call_args_list if call[0][1] == (1080, 1080)]
        self.assertGreater(len(resize_calls), 0)

if __name__ == "__main__":
    unittest.main()

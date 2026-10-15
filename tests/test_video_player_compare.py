import unittest
from types import SimpleNamespace

import cv2
import numpy as np


class CompareFrameIndexTests(unittest.TestCase):
    def test_wall_clock_playback_skips_to_real_time_frame(self):
        from app.video_player import playback_frame_index

        self.assertEqual(playback_frame_index(0, 0.100, 60.0), 6)
        self.assertEqual(playback_frame_index(120, 0.050, 60.0), 123)

    def test_default_compare_keeps_frame_index_behavior(self):
        from app.video_player import compare_frame_index

        self.assertEqual(compare_frame_index(17, 60.0, 24.0, False), 17)

    def test_time_synced_compare_maps_output_frame_to_source_time(self):
        from app.video_player import compare_frame_index

        mapped = [compare_frame_index(i, 48.0, 24.0, True)
                  for i in range(6)]
        self.assertEqual(mapped, [0, 0, 1, 1, 2, 2])

    def test_time_synced_compare_handles_invalid_fps_safely(self):
        from app.video_player import compare_frame_index

        self.assertEqual(compare_frame_index(9, 0.0, 24.0, True), 9)
        self.assertEqual(compare_frame_index(9, 60.0, 0.0, True), 9)

    def test_time_synced_compare_reuses_same_source_frame(self):
        from app.video_player import VideoPlayer

        class FakeCapture:
            def __init__(self):
                self.position = 0
                self.read_count = 0
            def get(self, prop):
                if prop == cv2.CAP_PROP_POS_FRAMES:
                    return self.position
                if prop == cv2.CAP_PROP_FRAME_COUNT:
                    return 30
                return 0
            def set(self, prop, value):
                if prop == cv2.CAP_PROP_POS_FRAMES:
                    self.position = int(value)
                return True
            def read(self):
                self.read_count += 1
                self.position += 1
                return True, np.full((2, 2, 3), 32, np.uint8)

        capture = FakeCapture()
        player = SimpleNamespace(
            _before_cap=capture,
            _before_fps=30.0,
            _before_sync_by_time=True,
            _before_frame_cache=None,
            _before_frame_index=-1,
            video_info=SimpleNamespace(fps=60.0),
            frame_view=SimpleNamespace(wipe_pos=lambda: 0.5),
        )
        after = np.zeros((2, 2, 3), np.uint8)
        VideoPlayer._composite_with_before(player, after, 0)
        VideoPlayer._composite_with_before(player, after, 1)
        self.assertEqual(capture.read_count, 1)

    def test_all_after_wipe_does_not_decode_before_stream(self):
        from app.video_player import VideoPlayer

        class FakeCapture:
            def __init__(self): self.read_count = 0
            def get(self, _prop): return 0
            def set(self, _prop, _value): return True
            def read(self):
                self.read_count += 1
                return True, np.zeros((2, 2, 3), np.uint8)

        capture = FakeCapture()
        player = SimpleNamespace(
            _before_cap=capture,
            _before_fps=30.0,
            _before_sync_by_time=True,
            _before_frame_cache=None,
            _before_frame_index=-1,
            video_info=SimpleNamespace(fps=60.0),
            frame_view=SimpleNamespace(wipe_pos=lambda: 0.0),
        )
        after = np.zeros((2, 2, 3), np.uint8)
        VideoPlayer._composite_with_before(player, after, 0)
        self.assertEqual(capture.read_count, 0)


if __name__ == "__main__":
    unittest.main()

"""Long-video inpaint chunking invariants."""

import unittest

from vdcore.inpaint_chunking import iter_context_windows, resolve_window_size


class InpaintChunkingTests(unittest.TestCase):
    @staticmethod
    def _reader(values):
        iterator = iter(values)
        return lambda: next(iterator, None)

    def test_auto_mode_is_bounded_for_long_inputs(self):
        windows = list(iter_context_windows(
            self._reader(range(10_000)), window_size=0, overlap=10))
        self.assertGreater(len(windows), 1)
        self.assertTrue(all(len(frames) <= 80
                            for frames, _, _ in windows))

    def test_every_frame_is_emitted_once_in_order(self):
        emitted = []
        for frames, keep_start, keep_end in iter_context_windows(
                self._reader(range(257)), window_size=80, overlap=10):
            emitted.extend(frames[keep_start:keep_end])
        self.assertEqual(emitted, list(range(257)))

    def test_neighboring_windows_include_temporal_context(self):
        windows = list(iter_context_windows(
            self._reader(range(150)), window_size=80, overlap=10))
        second_frames, second_keep_start, _ = windows[1]
        self.assertEqual(second_keep_start, 10)
        self.assertEqual(second_frames[:second_keep_start], list(range(50, 60)))
        self.assertEqual(second_frames[second_keep_start], 60)

    def test_short_clip_and_frame_limit(self):
        short = list(iter_context_windows(
            self._reader(range(3)), window_size=0, overlap=10))
        self.assertEqual(short, [([0, 1, 2], 0, 3)])

        emitted = []
        for frames, start, end in iter_context_windows(
                self._reader(range(100)), window_size=32, overlap=5,
                frame_limit=17):
            emitted.extend(frames[start:end])
        self.assertEqual(emitted, list(range(17)))

    def test_tiny_manual_window_clamps_overlap_safely(self):
        self.assertEqual(resolve_window_size(1), 1)
        emitted = []
        windows = list(iter_context_windows(
            self._reader(range(5)), window_size=1, overlap=20))
        for frames, start, end in windows:
            self.assertLessEqual(len(frames), 1)
            emitted.extend(frames[start:end])
        self.assertEqual(emitted, list(range(5)))


if __name__ == "__main__":
    unittest.main()

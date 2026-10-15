import unittest

import numpy as np


class InterpolationSceneCutTests(unittest.TestCase):
    def test_default_matches_flow_denoise_mse_threshold(self):
        from app.interpolation_worker import DEFAULT_SCENE_CUT_THRESHOLD

        self.assertAlmostEqual(DEFAULT_SCENE_CUT_THRESHOLD, 0.030, places=6)

    def test_default_protection_catches_moderate_hard_cut(self):
        from app.interpolation_worker import (
            DEFAULT_SCENE_CUT_THRESHOLD, _scene_cut)

        before = np.zeros((90, 160, 3), dtype=np.uint8)
        after = np.full((90, 160, 3), 64, dtype=np.uint8)
        self.assertTrue(_scene_cut(
            before, after, DEFAULT_SCENE_CUT_THRESHOLD))

    def test_default_protection_ignores_small_exposure_change(self):
        from app.interpolation_worker import (
            DEFAULT_SCENE_CUT_THRESHOLD, _scene_cut)

        before = np.full((90, 160, 3), 100, dtype=np.uint8)
        after = np.full((90, 160, 3), 116, dtype=np.uint8)
        self.assertFalse(_scene_cut(
            before, after, DEFAULT_SCENE_CUT_THRESHOLD))

    def test_disabled_protection_never_marks_cut(self):
        from app.interpolation_worker import _scene_cut

        before = np.zeros((90, 160, 3), dtype=np.uint8)
        after = np.full((90, 160, 3), 255, dtype=np.uint8)
        self.assertFalse(_scene_cut(before, after, 0.0))

    def test_mse_detector_catches_localized_large_change(self):
        from app.interpolation_worker import (
            DEFAULT_SCENE_CUT_THRESHOLD, _scene_cut)

        before = np.zeros((100, 100, 3), dtype=np.uint8)
        after = before.copy()
        after[:, :10] = 255
        self.assertTrue(_scene_cut(
            before, after, DEFAULT_SCENE_CUT_THRESHOLD))


if __name__ == "__main__":
    unittest.main()

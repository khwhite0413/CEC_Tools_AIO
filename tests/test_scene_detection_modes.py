"""Contracts for selectable scene-change detection across temporal tools."""

import os
import unittest
from pathlib import Path

import numpy as np


class SceneDetectionCoreTests(unittest.TestCase):
    def test_default_mode_and_legacy_threshold_remain_unchanged(self):
        from fdcore.scene_detection import (
            DEFAULT_SCENE_DETECTION_MODE, DEFAULT_SCENE_MSE_THRESHOLD)

        self.assertEqual(DEFAULT_SCENE_DETECTION_MODE, "mse")
        self.assertAlmostEqual(DEFAULT_SCENE_MSE_THRESHOLD, 0.030, places=6)

    def test_legacy_mse_stream_detects_the_current_pair_without_delay(self):
        from fdcore.scene_detection import SceneCutStream

        detector = SceneCutStream("mse", mse_threshold=0.030,
                                  mse_size=(160, 90))
        before = np.zeros((90, 160, 3), dtype=np.uint8)
        after = np.full((90, 160, 3), 64, dtype=np.uint8)

        self.assertEqual(detector.feed(0, before), (0, set()))
        self.assertEqual(detector.feed(1, after), (1, {1}))

    def test_pyscenedetect_stream_delays_until_adaptive_result_is_confirmed(self):
        from fdcore.scene_detection import SceneCutStream

        class FakeAdaptive:
            event_buffer_length = 2

            def process_frame(self, frame_num, _frame):
                number = int(frame_num)
                return [number - 2] if number == 3 else []

        detector = SceneCutStream(
            "pyscenedetect", adaptive_detector=FakeAdaptive())
        frame = np.zeros((12, 16, 3), dtype=np.uint8)

        self.assertEqual(detector.feed(0, frame), (-1, set()))
        self.assertEqual(detector.feed(1, frame), (-1, set()))
        self.assertEqual(detector.feed(2, frame), (0, set()))
        self.assertEqual(detector.feed(3, frame), (1, {1}))

    def test_inpaint_window_split_never_crosses_a_scene_boundary(self):
        from vdcore.inpaint_chunking import split_window_on_boundaries

        parts = list(split_window_on_boundaries(
            list(range(10)), keep_start=2, keep_end=8,
            boundaries={5}))

        self.assertEqual(parts, [
            (list(range(5)), 2, 5),
            (list(range(5, 10)), 0, 3),
        ])

    def test_real_adaptive_detector_finds_a_hard_cut(self):
        from fdcore.scene_detection import SceneCutStream

        detector = SceneCutStream("pyscenedetect")
        cuts = set()
        frames = ([np.zeros((90, 160, 3), dtype=np.uint8)] * 5
                  + [np.full((90, 160, 3), 255, dtype=np.uint8)] * 5)
        for index, frame in enumerate(frames):
            _confirmed, new_cuts = detector.feed(index, frame)
            cuts.update(new_cuts)
        _confirmed, new_cuts = detector.flush()
        cuts.update(new_cuts)
        self.assertEqual(cuts, {5})

    def test_flow_denoise_routes_adaptive_boundaries(self):
        import torch
        from fdcore.temporal_average import _detect_scene_changes

        frames = torch.cat((
            torch.zeros(5, 3, 12, 16),
            torch.ones(5, 3, 12, 16),
        ), dim=0)
        self.assertEqual(
            _detect_scene_changes(frames, 0.030, "pyscenedetect"), {5})


class SceneDetectionSetupTests(unittest.TestCase):
    def test_setup_installs_pinned_pyscenedetect(self):
        requirements = (Path(__file__).parents[1] / "requirements.txt").read_text(
            encoding="utf-8")
        self.assertIn("scenedetect==", requirements)


class SceneDetectionUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        import torch  # noqa: F401 - application imports torch before Qt.
        from PyQt5.QtCore import QCoreApplication
        from PyQt5.QtWidgets import QApplication

        existing = QCoreApplication.instance()
        if existing is not None and not isinstance(existing, QApplication):
            raise unittest.SkipTest("Widget tests require QApplication.")
        cls.app = QApplication.instance() or QApplication([])

    def test_denoise_defaults_to_legacy_mse_and_routes_setting(self):
        from app.parameter_panel import ParameterPanel

        panel = ParameterPanel()
        self.assertEqual(panel.scene_detection_mode.currentData(), "mse")
        self.assertEqual(panel.settings()["scene_detection_mode"], "mse")
        self.assertAlmostEqual(panel.settings()["scene_threshold"], 0.030)

    def test_interpolation_defaults_to_legacy_mse_and_routes_setting(self):
        from app.interpolation_page import InterpolationPage

        page = InterpolationPage()
        self.assertEqual(page.scene_detection_mode.currentData(), "mse")
        page.video_path = "input.mp4"
        settings = page._gather_settings()
        self.assertEqual(settings["scene_detection_mode"], "mse")

    def test_inpaint_defaults_to_legacy_mse_and_routes_setting(self):
        from app.inpaint_page import InpaintPage

        page = InpaintPage()
        self.assertEqual(page.scene_detection_mode.currentData(), "mse")
        self.assertEqual(page._gather_settings()["scene_detection_mode"], "mse")

    def test_all_single_tabs_route_adaptive_selection(self):
        from app.inpaint_page import InpaintPage
        from app.interpolation_page import InterpolationPage
        from app.parameter_panel import ParameterPanel

        denoise = ParameterPanel()
        denoise.scene_detection_mode.setCurrentIndex(1)
        self.assertEqual(
            denoise.settings()["scene_detection_mode"], "pyscenedetect")

        interpolation = InterpolationPage()
        interpolation.scene_detection_mode.setCurrentIndex(1)
        self.assertEqual(
            interpolation._gather_settings()["scene_detection_mode"],
            "pyscenedetect")

        inpaint = InpaintPage()
        inpaint.scene_detection_mode.setCurrentIndex(1)
        self.assertEqual(
            inpaint._gather_settings()["scene_detection_mode"],
            "pyscenedetect")

    def test_batch_routes_independent_inpaint_and_denoise_modes(self):
        from app.batch_page import BatchPage

        page = BatchPage()
        page.inp_scene_detection_mode.setCurrentIndex(1)
        page.den_scene_detection_mode.setCurrentIndex(0)
        inpaint, denoise, _interpolation, _upscale, _final = page._gather_settings()
        self.assertEqual(inpaint["scene_detection_mode"], "pyscenedetect")
        self.assertEqual(denoise["scene_detection_mode"], "mse")
        self.assertAlmostEqual(inpaint["scene_threshold"], 0.030)
        self.assertAlmostEqual(denoise["scene_threshold"], 0.030)


if __name__ == "__main__":
    unittest.main()

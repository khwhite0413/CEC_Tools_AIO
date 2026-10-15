"""Batch frame-interpolation stage contracts."""

import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import torch  # Import before PyQt5, matching the application startup order.
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtWidgets import QApplication


class BatchInterpolationUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        existing = QCoreApplication.instance()
        if existing is not None and not isinstance(existing, QApplication):
            raise unittest.SkipTest("Widget tests require QApplication.")
        cls.app = QApplication.instance() or QApplication([])

    def test_all_batch_processing_stages_default_off(self):
        from app.batch_page import BatchPage

        page = BatchPage()
        self.assertFalse(page.inp_box.isChecked())
        self.assertFalse(page.den_box.isChecked())
        self.assertFalse(page.int_box.isChecked())
        self.assertFalse(page.ups_box.isChecked())
        inpaint, denoise, interpolation, upscale, final = page._gather_settings()
        self.assertFalse(inpaint["enabled"])
        self.assertFalse(denoise["enabled"])
        self.assertFalse(interpolation["enabled"])
        self.assertFalse(upscale["enabled"])
        self.assertEqual(interpolation["model"], "rife_426")
        self.assertEqual(interpolation["multiplier"], 2)
        self.assertIsNone(interpolation["output_fps"])
        self.assertTrue(interpolation["fp16"])
        self.assertEqual(interpolation["scene_detection_mode"], "mse")
        self.assertAlmostEqual(interpolation["scene_cut_threshold"], 0.030)

    def test_output_fps_drawer_routes_preset_custom_and_off(self):
        from app.batch_page import BatchPage

        page = BatchPage()
        off_index = page.int_output_fps.findData(None)
        preset_index = page.int_output_fps.findData(60.0)
        custom_index = page.int_output_fps.findData(-1.0)

        page.int_output_fps.setCurrentIndex(off_index)
        self.assertIsNone(page._gather_settings()[2]["output_fps"])
        page.int_output_fps.setCurrentIndex(preset_index)
        self.assertEqual(page._gather_settings()[2]["output_fps"], 60.0)
        page.int_output_fps.setCurrentIndex(custom_index)
        page.int_custom_fps.setValue(72.0)
        self.assertEqual(page._gather_settings()[2]["output_fps"], 72.0)


class BatchInterpolationOrchestratorTests(unittest.TestCase):
    def test_stage_plan_places_interpolation_between_denoise_and_upscale(self):
        from app.batch_orchestrator import (
            BatchOrchestrator, STAGE_DENOISE, STAGE_INPAINT,
            STAGE_INTERPOLATION, STAGE_UPSCALE)

        orchestrator = BatchOrchestrator(
            ["clip.mp4"],
            {"enabled": True}, {"enabled": True}, {"enabled": True},
            interpolation_settings={"enabled": True, "model": "rife_426"})
        stages = orchestrator._build_stage_plan()
        self.assertEqual(
            [stage["name"] for stage in stages],
            [STAGE_INPAINT, STAGE_DENOISE,
             STAGE_INTERPOLATION, STAGE_UPSCALE])

    def test_interpolation_worker_reuses_batch_stage_settings(self):
        from app.batch_orchestrator import BatchOrchestrator
        from app.interpolation_worker import InterpolationWorker

        settings = {
            "enabled": True,
            "model": "rife_426",
            "multiplier": 4,
            "output_fps": 60.0,
            "scene_detection_mode": "pyscenedetect",
            "scene_cut_threshold": 0.030,
        }
        orchestrator = BatchOrchestrator(
            ["clip.mp4"], {"enabled": False}, {"enabled": False},
            {"enabled": False}, interpolation_settings=settings)
        orchestrator._device = torch.device("cpu")
        worker = orchestrator._build_interpolation_worker(
            "input.mp4", "output.mp4")
        self.assertIsInstance(worker, InterpolationWorker)
        self.assertEqual(worker.settings["multiplier"], 4)
        self.assertEqual(worker.settings["output_fps"], 60.0)
        self.assertEqual(
            worker.settings["scene_detection_mode"], "pyscenedetect")

    def test_manual_final_fps_keeps_terminal_transcode_after_interpolation(self):
        from app.batch_orchestrator import (
            BatchOrchestrator, STAGE_INTERPOLATION, STAGE_TRANSCODE)

        orchestrator = BatchOrchestrator(
            ["clip.mp4"], {"enabled": False}, {"enabled": False},
            {"enabled": False},
            final_settings={"fps_override": 24.0, "output_format": "h264"},
            interpolation_settings={
                "enabled": True, "model": "rife_426",
                "output_format": "h264"})
        # The original itself is already 24 fps, but interpolation changes
        # the timeline. Final 24 fps must therefore still be applied.
        with patch("app.batch_orchestrator._final_is_noop", return_value=True):
            stages = orchestrator._build_stage_plan()
        self.assertEqual(
            [stage["name"] for stage in stages],
            [STAGE_INTERPOLATION, STAGE_TRANSCODE])

    def test_final_codec_is_applied_after_interpolation_codec_differs(self):
        from app.batch_orchestrator import (
            BatchOrchestrator, STAGE_INTERPOLATION, STAGE_TRANSCODE)

        orchestrator = BatchOrchestrator(
            ["clip.mp4"], {"enabled": False}, {"enabled": False},
            {"enabled": False},
            final_settings={"fps_override": 0.0, "output_format": "h264"},
            interpolation_settings={
                "enabled": True, "model": "rife_426",
                "output_format": "prores"})
        with patch("app.batch_orchestrator._final_is_noop", return_value=True):
            stages = orchestrator._build_stage_plan()
        self.assertEqual(
            [stage["name"] for stage in stages],
            [STAGE_INTERPOLATION, STAGE_TRANSCODE])

    def test_cleanup_includes_interpolation_cache(self):
        from app.batch_orchestrator import (
            BATCH_DENOISE_CACHE, BATCH_INPAINT_CACHE,
            BATCH_INTERPOLATION_CACHE, BatchOrchestrator)

        orchestrator = BatchOrchestrator(
            ["clip.mp4"], {}, {}, {}, interpolation_settings={})
        with patch("app.batch_orchestrator._wipe_dir") as wipe:
            orchestrator._cleanup_stage_caches()
        self.assertEqual(
            [call.args[0] for call in wipe.call_args_list],
            [BATCH_INPAINT_CACHE, BATCH_DENOISE_CACHE,
             BATCH_INTERPOLATION_CACHE])

    def test_matching_interpolation_and_final_fps_skips_reencode(self):
        from app.batch_orchestrator import (
            BatchOrchestrator, STAGE_INTERPOLATION)

        orchestrator = BatchOrchestrator(
            ["clip.mp4"], {"enabled": False}, {"enabled": False},
            {"enabled": False},
            final_settings={
                "fps_override": 60.0, "output_format": "h264",
                "aspect_mode": "none", "target_long_edge": 0},
            interpolation_settings={
                "enabled": True, "model": "rife_426",
                "output_fps": 60.0, "output_format": "h264"})
        stages = orchestrator._build_stage_plan()
        self.assertEqual(
            [stage["name"] for stage in stages], [STAGE_INTERPOLATION])


if __name__ == "__main__":
    unittest.main()

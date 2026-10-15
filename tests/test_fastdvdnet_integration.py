"""Contract tests for the FastDVDnet denoise-model integration."""

import os
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import torch  # Import before PyQt5 on Windows, matching the application.
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QCoreApplication

from app.denoise_models import DEFAULT_DENOISE_MODEL, DENOISE_MODELS
from app.parameter_panel import ParameterPanel
from app.batch_page import BatchPage
from app.workers import (DenoiseWorker, create_denoise_worker)
from app.fastdvdnet_worker import FastDVDnetWorker
from fdcore.fastdvdnet_model import infer_frame
from fdcore.fastdvdnet_pipeline import _reflect_index


class DenoiseModelRegistryTests(unittest.TestCase):
    def test_fastdvdnet_is_default_and_flow_is_second(self):
        self.assertEqual(DEFAULT_DENOISE_MODEL, "fastdvdnet")
        self.assertEqual(list(DENOISE_MODELS), ["fastdvdnet", "flow_denoise"])


class DenoiseUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        existing = QCoreApplication.instance()
        if existing is not None and not isinstance(existing, QApplication):
            raise unittest.SkipTest(
                "A QCoreApplication was created by an earlier test module; "
                "run this module alone for widget assertions.")
        cls.app = QApplication.instance() or QApplication([])

    def test_single_tab_defaults_to_fastdvdnet(self):
        panel = ParameterPanel()
        self.assertEqual(panel.model_combo.currentData(), "fastdvdnet")
        self.assertEqual(panel.model_combo.itemData(1), "flow_denoise")
        self.assertFalse(panel.fastdvdnet_params.isHidden())
        self.assertTrue(panel.flow_params.isHidden())
        settings = panel.settings()
        self.assertEqual(settings["denoise_model"], "fastdvdnet")
        self.assertEqual(settings["noise_sigma"], 25)

    def test_selecting_flow_preserves_existing_flow_settings(self):
        panel = ParameterPanel()
        panel.model_combo.setCurrentIndex(1)
        self.assertEqual(panel.model_combo.currentData(), "flow_denoise")
        self.assertTrue(panel.fastdvdnet_params.isHidden())
        self.assertFalse(panel.flow_params.isHidden())
        settings = panel.settings()
        self.assertEqual(settings["denoise_model"], "flow_denoise")
        self.assertEqual(settings["flow_model"], "memfof")
        self.assertEqual(settings["window_size"], 3)

    def test_batch_tab_defaults_to_fastdvdnet_and_routes_settings(self):
        page = BatchPage()
        self.assertEqual(page.den_model.currentData(), "fastdvdnet")
        self.assertEqual(page.den_model.itemData(1), "flow_denoise")
        _, denoise, _, _, _ = page._gather_settings()
        self.assertEqual(denoise["denoise_model"], "fastdvdnet")
        self.assertEqual(denoise["noise_sigma"], 25)


class DenoiseWorkerRoutingTests(unittest.TestCase):
    def test_fastdvdnet_setting_creates_fastdvdnet_worker(self):
        settings = {"denoise_model": "fastdvdnet", "noise_sigma": 25}
        worker = create_denoise_worker("in.mp4", settings, "out.mp4")
        self.assertIsInstance(worker, FastDVDnetWorker)

    def test_flow_setting_keeps_legacy_worker(self):
        settings = {"denoise_model": "flow_denoise"}
        worker = create_denoise_worker("in.mp4", settings, "out.mp4")
        self.assertIsInstance(worker, DenoiseWorker)

    def test_missing_model_key_is_backward_compatible_flow(self):
        worker = create_denoise_worker("in.mp4", {}, "out.mp4")
        self.assertIsInstance(worker, DenoiseWorker)


class SetupIntegrationTests(unittest.TestCase):
    def test_setup_prefetches_fastdvdnet_weights(self):
        setup = (Path(__file__).parents[1] / "setup.bat").read_text(
            encoding="utf-8")
        self.assertIn("download_fastdvdnet_weights", setup)


class FastDVDnetBackendTests(unittest.TestCase):
    def test_reflection_indices_cover_short_and_tail_boundaries(self):
        self.assertEqual([_reflect_index(i, 1) for i in range(-3, 4)],
                         [0, 0, 0, 0, 0, 0, 0])
        self.assertEqual([_reflect_index(i, 3) for i in range(7)],
                         [0, 1, 2, 1, 0, 1, 2])

    def test_infer_frame_pads_and_crops_non_multiple_of_four(self):
        class CenterFrame(torch.nn.Module):
            def forward(self, frames, _sigma):
                return frames[:, 6:9]

        frames = torch.rand(1, 15, 7, 9)
        sigma = torch.full((1, 1, 7, 9), 25.0 / 255.0)
        output = infer_frame(CenterFrame(), frames, sigma)
        self.assertEqual(tuple(output.shape), (1, 3, 7, 9))
        self.assertTrue(torch.equal(output, frames[:, 6:9]))


if __name__ == "__main__":
    unittest.main()

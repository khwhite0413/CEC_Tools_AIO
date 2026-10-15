import math
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch


class InterpolationContractTests(unittest.TestCase):
    def test_registry_contains_only_ranked_commercially_safe_models(self):
        from app.interpolation_models import INTERPOLATION_MODELS

        self.assertEqual(
            list(INTERPOLATION_MODELS),
            ["rife_426", "atm_vfi_lite", "sgm_vfi", "ema_vfi"],
        )
        self.assertTrue(all(item["commercial_use"] for item in INTERPOLATION_MODELS.values()))
        self.assertEqual(
            [item["license"] for item in INTERPOLATION_MODELS.values()],
            ["MIT", "Apache-2.0", "Apache-2.0", "Apache-2.0"],
        )

    def test_setup_wires_the_pinned_interpolation_installer(self):
        import os
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        with open(os.path.join(root, "setup.bat"), "r", encoding="utf-8") as handle:
            setup = handle.read()
        self.assertIn("-m app.interpolation_setup", setup)
        self.assertIn("[10/12] Fetching frame interpolation models", setup)

        from app.interpolation_setup import SOURCES
        self.assertTrue(all(len(source) == 4 for source in SOURCES))
        self.assertTrue(all(len(source[3]) == 64 for source in SOURCES))

    def test_native_output_fps_when_final_override_is_off(self):
        from app.interpolation_fps import build_fps_plan

        plan = build_fps_plan(23.976, 2, None)
        self.assertEqual(plan.interpolation_multiplier, 2)
        self.assertAlmostEqual(plan.generated_fps, 47.952, places=3)
        self.assertAlmostEqual(plan.output_fps, 47.952, places=3)

    def test_exact_target_fps_preserves_duration_and_oversamples_when_needed(self):
        from app.interpolation_fps import build_fps_plan, output_frame_count

        plan = build_fps_plan(24.0, 2, 60.0)
        self.assertEqual(plan.interpolation_multiplier, 4)
        self.assertEqual(plan.generated_fps, 96.0)
        self.assertEqual(plan.output_fps, 60.0)
        self.assertEqual(output_frame_count(240, 24.0, plan.output_fps), 600)

    def test_output_index_mapping_keeps_first_last_and_monotonic_order(self):
        from app.interpolation_fps import selected_generated_indices

        indices = selected_generated_indices(
            source_frame_count=24,
            source_fps=24.0,
            interpolation_multiplier=4,
            output_fps=60.0,
        )
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 94)
        self.assertEqual(len(indices), 60)
        self.assertTrue(all(a < b for a, b in zip(indices, indices[1:])))

    def test_invalid_fps_values_are_rejected(self):
        from app.interpolation_fps import build_fps_plan

        for source, multiplier, target in ((0, 2, None), (24, 3, None), (24, 2, 0)):
            with self.subTest(source=source, multiplier=multiplier, target=target):
                with self.assertRaises(ValueError):
                    build_fps_plan(source, multiplier, target)

    def test_worker_streams_two_x_and_preserves_endpoints(self):
        from app import interpolation_worker as module

        frames = [np.full((4, 6, 3), value, np.uint8)
                  for value in (0, 100, 200)]
        written = []
        writer_args = {}

        class Info:
            effective_fps = 24.0
            fps = 24.0
            def __init__(self, _path): pass
            def color_tags(self): return {}

        class Reader:
            # Deliberately wrong metadata count: worker must finish from the
            # actual decoded EOF so VFR/CFR normalization cannot shorten it.
            width, height, frame_count = 6, 4, 999
            def __init__(self, _path, fps=None): self.pos = 0
            def read(self):
                if self.pos >= len(frames): return None
                frame = frames[self.pos]
                self.pos += 1
                return frame
            def release(self): pass

        class Writer:
            def __init__(self, _path, fps, width, height, **kwargs):
                writer_args.update(fps=fps, width=width, height=height)
            def write(self, frame): written.append(frame.copy())
            def release(self): pass

        class Backend:
            def __init__(self, *_args, **_kwargs): pass
            def midpoint(self, a, b): return (a + b) / 2
            def close(self): pass

        with mock.patch.object(module, "VideoInfo", Info), \
             mock.patch.object(module, "StreamingFrameReader", Reader), \
             mock.patch.object(module, "FFmpegWriter", Writer), \
             mock.patch.object(module, "InterpolationBackend", Backend):
            worker = module.InterpolationWorker(
                "input.mp4", "output.mp4",
                {"model": "rife_426", "multiplier": 2,
                 "output_fps": None, "scene_cut_threshold": 0},
                torch.device("cpu"))
            errors = []
            progress = []
            worker.error_signal.connect(errors.append)
            worker.progress.connect(lambda done, total: progress.append((done, total)))
            worker.run()

        self.assertEqual(errors, [])
        self.assertEqual(writer_args["fps"], 48.0)
        self.assertEqual(len(written), 6)
        self.assertEqual(int(written[0][0, 0, 0]), 0)
        self.assertGreater(int(written[-1][0, 0, 0]), 50000)
        self.assertEqual(progress[-1], (3, 3))

    def test_rife_loader_scopes_and_restores_comfy_compatibility_stub(self):
        from app import interpolation_backend as module

        backend = module.InterpolationBackend.__new__(module.InterpolationBackend)
        backend.device = torch.device("cpu")
        backend.dtype = torch.float32
        old_comfy = object()
        old_mm = object()
        seen = {}

        class FakeModel:
            def load_state_dict(self, _state, strict=False):
                return type("Result", (), {"missing_keys": []})()
            def to(self, *_args, **_kwargs): return self
            def eval(self): return self

        class FakeModule:
            IFNet = staticmethod(lambda **_kwargs: FakeModel())

        def fake_load_module(*_args, **_kwargs):
            seen["parent"] = sys.modules.get("comfy")
            seen["child"] = sys.modules.get("comfy.model_management")
            return FakeModule

        with mock.patch.dict(sys.modules, {
                "comfy": old_comfy,
                "comfy.model_management": old_mm,
             }), mock.patch.object(module, "_load_module", fake_load_module), \
             mock.patch.object(module.torch, "load", return_value={}):
            backend._load_rife("unused.pth")
            self.assertIs(sys.modules["comfy"], old_comfy)
            self.assertIs(sys.modules["comfy.model_management"], old_mm)

        self.assertIs(getattr(seen["parent"], "model_management"), seen["child"])

    def test_page_defaults_to_rife_and_native_interpolated_fps(self):
        from PyQt5.QtCore import QCoreApplication
        from PyQt5.QtWidgets import QApplication
        from app.interpolation_page import InterpolationPage

        existing = QCoreApplication.instance()
        if existing is not None and not isinstance(existing, QApplication):
            self.skipTest("Earlier tests created a non-widget QCoreApplication")
        app = QApplication.instance() or QApplication([])
        page = InterpolationPage()
        try:
            self.assertEqual(page.model_combo.currentData(), "rife_426")
            self.assertEqual(page.multiplier_combo.currentData(), 2)
            self.assertIsNone(page.output_fps_combo.currentData())
        finally:
            page.shutdown()
            page.deleteLater()

    def test_completed_interpolation_enables_time_synced_wipe(self):
        from PyQt5.QtCore import QCoreApplication
        from PyQt5.QtWidgets import QApplication
        from app import interpolation_page as module

        existing = QCoreApplication.instance()
        if existing is not None and not isinstance(existing, QApplication):
            self.skipTest("Earlier tests created a non-widget QCoreApplication")
        app = QApplication.instance() or QApplication([])
        page = module.InterpolationPage()

        class PlayerSpy:
            def __init__(self): self.calls = []
            def disable_compare(self): self.calls.append(("disable",))
            def load(self, path): self.calls.append(("load", path))
            def enable_compare(self, path, sync_by_time=False):
                self.calls.append(("enable", path, sync_by_time))
                return True

        try:
            with tempfile.TemporaryDirectory() as temp:
                source = os.path.join(temp, "source.mp4")
                output = os.path.join(temp, "result.mp4")
                open(source, "wb").close()
                open(output, "wb").close()
                spy = PlayerSpy()
                page.player = spy
                page.video_path = source
                page._started = 0.0
                with mock.patch.object(module, "_CACHE", temp):
                    page._on_finished(output)
                self.assertIn(("enable", source, True), spy.calls)
        finally:
            page.shutdown()
            page.deleteLater()

    def test_worker_reference_lives_until_qthread_finished_and_stale_progress_is_ignored(self):
        from PyQt5.QtCore import QCoreApplication, QObject, pyqtSignal
        from PyQt5.QtWidgets import QApplication
        from app.interpolation_page import InterpolationPage

        existing = QCoreApplication.instance()
        if existing is not None and not isinstance(existing, QApplication):
            self.skipTest("Earlier tests created a non-widget QCoreApplication")
        app = QApplication.instance() or QApplication([])
        page = InterpolationPage()

        class WorkerSignals(QObject):
            progress = pyqtSignal(int, int)
            cancelled_signal = pyqtSignal()
            finished = pyqtSignal()

        old_worker = WorkerSignals()
        current_worker = WorkerSignals()
        old_worker.progress.connect(page._on_progress)
        current_worker.cancelled_signal.connect(page._on_cancelled)
        current_worker.finished.connect(page._on_worker_thread_finished)
        page.worker = current_worker
        page._last_done, page._last_total = 0, 960
        try:
            old_worker.progress.emit(100, 1200)
            self.assertEqual((page._last_done, page._last_total), (0, 960))
            current_worker.cancelled_signal.emit()
            self.assertIs(page.worker, current_worker)
            current_worker.finished.emit()
            self.assertIsNone(page.worker)
        finally:
            page.worker = None
            page.shutdown()
            page.deleteLater()

    def test_new_run_progress_label_replaces_previous_total_immediately(self):
        from PyQt5.QtCore import QCoreApplication
        from PyQt5.QtWidgets import QApplication
        from app.interpolation_page import InterpolationPage

        existing = QCoreApplication.instance()
        if existing is not None and not isinstance(existing, QApplication):
            self.skipTest("Earlier tests created a non-widget QCoreApplication")
        app = QApplication.instance() or QApplication([])
        page = InterpolationPage()
        try:
            page.eta_label.setText("12/1200 | 이전 작업")
            page._reset_run_progress(960)
            self.assertEqual(page.eta_label.text(), "0/960 | 준비 중...")
            self.assertEqual((page._last_done, page._last_total), (0, 960))
        finally:
            page.shutdown()
            page.deleteLater()

    def test_new_video_waits_for_cancelled_worker_to_fully_stop(self):
        from PyQt5.QtCore import QCoreApplication
        from PyQt5.QtWidgets import QApplication
        from app import interpolation_page as module

        existing = QCoreApplication.instance()
        if existing is not None and not isinstance(existing, QApplication):
            self.skipTest("Earlier tests created a non-widget QCoreApplication")
        app = QApplication.instance() or QApplication([])
        page = module.InterpolationPage()

        class ActiveWorker:
            def __init__(self): self.cancelled = False
            def cancel(self): self.cancelled = True

        worker = ActiveWorker()
        page.worker = worker
        page.video_path = "first.mp4"
        page.player = type("Player", (), {"load": lambda self, _path: None,
                                          "clear": lambda self: None})()

        class Info:
            codec_name = "h264"
            width, height, fps, frame_count, duration = 64, 64, 24.0, 960, 40.0
            effective_fps = 24.0
            def __init__(self, _path): pass
        try:
            with mock.patch.object(module.os.path, "isfile", return_value=True), \
                 mock.patch.object(module, "VideoInfo", Info):
                page._set_video("second.mp4")
            self.assertEqual(page.video_path, "first.mp4")
            self.assertEqual(page._pending_video_path, "second.mp4")
            self.assertTrue(worker.cancelled)
        finally:
            page.worker = None
            page.shutdown()
            page.deleteLater()


if __name__ == "__main__":
    unittest.main()

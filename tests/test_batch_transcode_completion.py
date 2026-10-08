"""Regression tests for the final-output-only batch path."""

import io
import subprocess
import unittest
from unittest.mock import MagicMock, patch

import torch  # Must precede PyQt5 on Windows in this application.
from PyQt5.QtCore import QCoreApplication, QThread, pyqtSignal

from app.batch_orchestrator import BatchOrchestrator
from fdcore import transcode


class _ImmediateWorker(QThread):
    progress = pyqtSignal(int, int)
    finished_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)
    cancelled_signal = pyqtSignal()

    def run(self):
        self.finished_signal.emit("out.mp4")


class _WorkerDriver(QThread):
    def __init__(self, orchestrator):
        super().__init__()
        self.orchestrator = orchestrator

    def run(self):
        self.orchestrator._run_worker(
            _ImmediateWorker(), "transcode", 1, 1)


class BatchWorkerCompletionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QCoreApplication.instance() or QCoreApplication([])

    def test_immediate_worker_cannot_finish_before_wait_loop_is_armed(self):
        orchestrator = BatchOrchestrator(
            [], {"enabled": False}, {"enabled": False},
            {"enabled": False})
        driver = _WorkerDriver(orchestrator)
        driver.start()
        completed = driver.wait(1500)
        if not completed:
            # Let the old broken implementation unwind so the test process
            # itself does not retain a stuck QThread after asserting RED.
            loop = orchestrator._cur_loop
            if loop is not None:
                loop.quit()
            driver.wait(1500)
        self.assertTrue(completed)


class TranscodeStderrTests(unittest.TestCase):
    @staticmethod
    def _video_info():
        info = MagicMock()
        info.width = 64
        info.height = 48
        info.effective_fps = 24.0
        info.fps = 24.0
        info.duration = 1.0
        info.frame_count = 24
        info.has_audio = False
        return info

    def test_ffmpeg_stderr_is_not_a_blocking_pipe(self):
        info = self._video_info()
        captured = {}

        class FakeProcess:
            stdout = io.BytesIO(b"frame=24\n")
            stderr = io.BytesIO()

            def poll(self):
                return 0

            def wait(self, timeout=None):
                return 0

            def kill(self):
                pass

        def fake_popen(*args, **kwargs):
            captured.update(kwargs)
            return FakeProcess()

        with patch.object(transcode, "VideoInfo", return_value=info), \
             patch.object(transcode, "_get_ffmpeg", return_value="ffmpeg"), \
             patch.object(transcode.subprocess, "Popen", side_effect=fake_popen):
            transcode.run_transcode_only(
                "input.mp4", "output.mp4", {"output_format": "h264"})

        self.assertIsNot(captured["stderr"], subprocess.PIPE)

    def test_cancelled_ffmpeg_is_killed_and_waited(self):
        process = MagicMock()
        process.stdout = iter([b"frame=1\n"])
        process.poll.return_value = None
        process.stderr = None

        with patch.object(transcode, "VideoInfo",
                          return_value=self._video_info()), \
             patch.object(transcode, "_get_ffmpeg", return_value="ffmpeg"), \
             patch.object(transcode.subprocess, "Popen",
                          return_value=process):
            with self.assertRaisesRegex(RuntimeError, "취소"):
                transcode.run_transcode_only(
                    "input.mp4", "output.mp4",
                    {"output_format": "h264"}, cancel_cb=lambda: True)

        process.kill.assert_called()
        process.wait.assert_called()


if __name__ == "__main__":
    unittest.main()

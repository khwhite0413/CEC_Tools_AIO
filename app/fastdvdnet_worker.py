"""QThread adapters for the standalone FastDVDnet pipeline."""

import time
import traceback

from PyQt5.QtCore import QThread, pyqtSignal

from fdcore import fastdvdnet_pipeline
from fdcore.progress import CancelledError, ProgressReporter


class FastDVDnetWorker(QThread):
    progress = pyqtSignal(int, int)
    chunk_progress = pyqtSignal(int, int)
    finished_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)
    noise_ready = pyqtSignal(str)
    phase_changed = pyqtSignal(str)
    cancelled_signal = pyqtSignal()

    def __init__(self, video_path, settings, output_path, frame_cap=-1,
                 noise_path=None, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.settings = dict(settings)
        self.output_path = output_path
        self.frame_cap = int(frame_cap)
        self.noise_path = noise_path
        self.reporter = ProgressReporter(
            lambda done, total: self.progress.emit(done, total),
            lambda done, total: self.chunk_progress.emit(done, total))

    def cancel(self):
        self.reporter.request_cancel()

    def run(self):
        started = time.time()
        try:
            result = fastdvdnet_pipeline.run_on_file(
                self.video_path, self.output_path, self.settings,
                reporter=self.reporter, frame_cap=self.frame_cap,
                noise_path=self.noise_path,
                phase_cb=self.phase_changed.emit)
            print(f"[FastDVDnet] done in {time.time() - started:.1f}s "
                  f"-> {self.output_path}")
            if result.get("noise"):
                self.noise_ready.emit(result["noise"])
            self.finished_signal.emit(self.output_path)
        except CancelledError:
            self.cancelled_signal.emit()
        except Exception as exc:
            self.error_signal.emit(f"{exc}\n\n{traceback.format_exc()}")


class FastDVDnetPreviewWorker(FastDVDnetWorker):
    def __init__(self, video_path, settings, output_path, noise_path=None,
                 parent=None):
        super().__init__(video_path, settings, output_path, frame_cap=80,
                         noise_path=noise_path, parent=parent)

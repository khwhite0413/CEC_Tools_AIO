"""QThread workers for full denoise and 80-frame preview.

Signal pattern (회고 §10):
    progress(int done, int total)
    finished_signal(str output_path)
    error_signal(str message)
"""

import os
import time
import traceback

from PyQt5.QtCore import QThread, pyqtSignal

from fdcore import pipeline, video_io
from fdcore.progress import ProgressReporter, CancelledError


PREVIEW_FRAMES = 80


class _Worker(QThread):
    progress = pyqtSignal(int, int)
    # Chunk-scoped progress within a single temporal_flow_average call.
    # In streaming mode this is emitted many times per run (once per
    # inner batch inside a chunk); the UI wires it to a secondary bar
    # that oscillates while the main `progress` bar climbs monotonically.
    chunk_progress = pyqtSignal(int, int)
    finished_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)
    noise_ready = pyqtSignal(str)
    phase_changed = pyqtSignal(str)  # "processing" | "encoding" | "encoding_noise" | "done"
    cancelled_signal = pyqtSignal()  # user pressed Cancel and the worker bailed

    def __init__(self, video_path: str, settings: dict, output_path: str,
                 frame_cap: int = -1, noise_path: str = None, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.settings = settings
        self.output_path = output_path
        self.frame_cap = frame_cap
        self.noise_path = noise_path
        self.reporter = ProgressReporter(self._emit_progress,
                                         self._emit_chunk_progress)

    def _emit_progress(self, done: int, total: int):
        self.progress.emit(int(done), int(total))

    def _emit_chunk_progress(self, done: int, chunk_size: int):
        self.chunk_progress.emit(int(done), int(chunk_size))

    def _emit_phase(self, phase: str):
        self.phase_changed.emit(phase)

    def cancel(self):
        self.reporter.request_cancel()

    def run(self):
        t0 = time.time()
        try:
            result = pipeline.run_on_file(
                self.video_path, self.output_path, self.settings,
                reporter=self.reporter, frame_cap=self.frame_cap,
                noise_path=self.noise_path,
                phase_cb=self._emit_phase)
            elapsed = time.time() - t0
            print(f"[FlowDenoise] done in {elapsed:.1f}s -> {self.output_path}")
            if isinstance(result, dict) and result.get("noise"):
                self.noise_ready.emit(result["noise"])
            self.finished_signal.emit(self.output_path)
        except CancelledError:
            self.cancelled_signal.emit()
        except Exception as e:
            self.error_signal.emit(f"{e}\n\n{traceback.format_exc()}")


class DenoiseWorker(_Worker):
    pass


class PreviewWorker(_Worker):
    """Process only PREVIEW_FRAMES frames (회고 §6)."""

    def __init__(self, video_path: str, settings: dict, output_path: str,
                 noise_path: str = None, parent=None):
        super().__init__(video_path, settings, output_path,
                         frame_cap=PREVIEW_FRAMES, noise_path=noise_path,
                         parent=parent)


def derive_output_path(input_path: str, suffix: str = "_denoised") -> str:
    base, ext = os.path.splitext(input_path)
    if not ext:
        ext = ".mp4"
    return base + suffix + ext


def probe_video(path: str):
    """Returns VideoInfo or raises."""
    return video_io.VideoInfo(path)

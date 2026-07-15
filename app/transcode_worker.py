"""QThread wrapper around fdcore.transcode.run_transcode_only.

Signal pattern mirrors InpaintWorker / UpscaleWorker / DenoiseWorker so
BatchOrchestrator can drive it through the same _run_worker eventloop
without special-casing:

    progress(int done, int total)
    finished_signal(str output_path)
    error_signal(str message)
    cancelled_signal()
    phase_changed(str)         # "processing" | "done"

Used by BatchOrchestrator when either
  (a) all three processing stages are disabled but the Final Output
      group has non-default params — "AME substitute" pass, or
  (b) upscale is disabled but Final Output requires a resize / aspect
      change that inpaint or denoise workers don't apply themselves —
      appended as a final polish step after the last processing stage.
"""

from __future__ import annotations

import traceback

from PyQt5.QtCore import QThread, pyqtSignal

from fdcore.transcode import run_transcode_only


class TranscodeWorker(QThread):
    """Stand-alone ffmpeg pass. No GPU, no model load — cheap to spin up."""

    progress = pyqtSignal(int, int)
    finished_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)
    cancelled_signal = pyqtSignal()
    phase_changed = pyqtSignal(str)

    def __init__(self, video_path: str, output_path: str, settings: dict,
                 parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.output_path = output_path
        self.settings = dict(settings or {})
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True
        self.requestInterruption()

    def _is_cancelled(self) -> bool:
        return bool(self._cancelled or self.isInterruptionRequested())

    def _emit_progress(self, done: int, total: int) -> None:
        self.progress.emit(int(done), int(total))

    def run(self) -> None:
        try:
            self.phase_changed.emit("processing")
            try:
                run_transcode_only(
                    self.video_path,
                    self.output_path,
                    self.settings,
                    progress_cb=self._emit_progress,
                    cancel_cb=self._is_cancelled,
                )
            except RuntimeError as e:
                # run_transcode_only raises RuntimeError("사용자가 취소했습니다.")
                # on cancel — treat that as a clean cancel rather than an
                # error message the user would misread as ffmpeg failure.
                if "취소" in str(e):
                    self.cancelled_signal.emit()
                    return
                raise
            if self._is_cancelled():
                self.cancelled_signal.emit()
                return
            self.phase_changed.emit("done")
            self.finished_signal.emit(self.output_path)
        except Exception as e:
            self.error_signal.emit(f"{e}\n\n{traceback.format_exc()}")

"""QThread wrapper around vdcore.InpaintEngine.

Signal pattern mirrors fdcore workers so the MainWindow can reuse the
same progress / cancel / error UI for both tabs:
    progress(int done, int total)
    finished_signal(str output_path)
    error_signal(str message)
    cancelled_signal()
    phase_changed(str)  # "processing" | "encoding" | "done"
"""

import traceback

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

# FlowDenoise's VideoInfo is reused so the Inpaint path inherits its
# untagged-passthrough + non-integer fps snapping (24.04 → 24.0). Frame
# reading + writing stays on vdcore (bgr24, simpler — ProPainter is
# uint8 internally so the fdcore 16-bit pipe would be wasted bandwidth).
from fdcore.video_io import VideoInfo
from vdcore.video_io import read_frames_batch, FFmpegWriter
from vdcore.mask_manager import mask_to_binary
from vdcore.inpaint_engine import InpaintEngine


def compute_crop_box(mask, padding=50):
    """Square crop region around mask, divisible by 8, with padding.

    Copied from VDInpaint inpaint_worker — keeps the model input small
    by feeding only the masked region.
    """
    ys, xs = np.where(mask > 127)
    if len(ys) == 0:
        return None
    y1, y2 = int(ys.min()), int(ys.max())
    x1, x2 = int(xs.min()), int(xs.max())
    h, w = mask.shape[:2]
    y1 = max(0, y1 - padding); y2 = min(h, y2 + padding)
    x1 = max(0, x1 - padding); x2 = min(w, x2 + padding)
    crop_h = ((y2 - y1 + 7) // 8) * 8
    crop_w = ((x2 - x1 + 7) // 8) * 8
    cy, cx = (y1 + y2) // 2, (x1 + x2) // 2
    y1 = max(0, cy - crop_h // 2); x1 = max(0, cx - crop_w // 2)
    y2 = min(h, y1 + crop_h);      x2 = min(w, x1 + crop_w)
    if y2 - y1 < crop_h: y1 = max(0, y2 - crop_h)
    if x2 - x1 < crop_w: x1 = max(0, x2 - crop_w)
    return x1, y1, x2, y2


class InpaintWorker(QThread):
    progress = pyqtSignal(int, int)
    finished_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)
    cancelled_signal = pyqtSignal()
    phase_changed = pyqtSignal(str)

    def __init__(self, video_path, mask_full, settings, output_path, device,
                 frame_cap=-1, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.mask_full = mask_full       # uint8 H x W, full-frame
        self.settings = settings or {}
        self.output_path = output_path
        self.device = device
        # >0 caps the number of frames read & processed. Used by the
        # Preview button to inpaint just the first ~80 frames so the
        # user can iterate on parameters without paying for the full
        # clip every time.
        self.frame_cap = int(frame_cap) if frame_cap else -1
        self._cancelled = False

    def cancel(self):
        self._cancelled = True
        self.requestInterruption()

    def run(self):
        try:
            info = VideoInfo(self.video_path)
            total_frames = info.frame_count
            if self.frame_cap > 0:
                total_frames = min(total_frames, self.frame_cap)

            mask_binary = mask_to_binary(self.mask_full)
            if mask_binary.shape[:2] != (info.height, info.width):
                mask_binary = cv2.resize(
                    mask_binary, (info.width, info.height),
                    interpolation=cv2.INTER_NEAREST)

            box = compute_crop_box(mask_binary, padding=50)
            if box is None:
                self.error_signal.emit(
                    "마스크 영역이 비어 있습니다. 프리셋을 다시 적용하세요.")
                return
            x1, y1, x2, y2 = box
            cropped_mask = mask_binary[y1:y2, x1:x2].copy()

            max_batch = int(self.settings.get("max_batch_frames", 0) or 0)
            if max_batch <= 0 or max_batch >= total_frames:
                max_batch = total_frames
                num_batches = 1
            else:
                num_batches = (total_frames + max_batch - 1) // max_batch

            # effective_fps: 24.04-style noisy source → snapped to nearest
            # standard (24.000). color_tags: forwarded so untagged sources
            # stay untagged on output (matches Nuke heuristic on both files).
            # output_format: "h264" (mp4) or "prores" (mov); resolved by the
            # page before the worker starts.
            writer = FFmpegWriter(
                self.output_path,
                info.effective_fps,
                info.width, info.height,
                source_path=self.video_path,
                color_tags=info.color_tags(),
                output_format=self.settings.get("output_format", "h264"),
                prores_profile=self.settings.get("prores_profile", "4444"))

            engine = InpaintEngine(
                self.device, fp16=bool(self.settings.get("fp16", True)))

            self.phase_changed.emit("processing")
            processed = 0
            for batch_idx in range(num_batches):
                if self._cancelled or self.isInterruptionRequested():
                    break

                batch_start = batch_idx * max_batch
                batch_end = min(batch_start + max_batch, total_frames)
                batch_count = batch_end - batch_start

                frames = read_frames_batch(
                    self.video_path, batch_start, batch_count)
                if not frames:
                    break

                cropped_frames = [f[y1:y2, x1:x2].copy() for f in frames]
                crop_h, crop_w = cropped_mask.shape[:2]

                # ProPainter operates well near 720p — upscale crop if the
                # source resolution is much larger, run inference, then
                # downscale back.
                mask_scale = max(
                    info.height / 720.0, info.width / 1280.0, 1.0)
                if mask_scale > 1.01:
                    proc_w = ((int(crop_w * mask_scale) + 7) // 8) * 8
                    proc_h = ((int(crop_h * mask_scale) + 7) // 8) * 8
                    proc_frames = [cv2.resize(f, (proc_w, proc_h),
                                              interpolation=cv2.INTER_CUBIC)
                                   for f in cropped_frames]
                    proc_mask = cv2.resize(cropped_mask, (proc_w, proc_h),
                                           interpolation=cv2.INTER_NEAREST)
                else:
                    proc_frames = cropped_frames
                    proc_mask = cropped_mask
                    proc_w, proc_h = crop_w, crop_h

                offset = processed

                def on_progress(cur, total, _o=offset):
                    if self._cancelled or self.isInterruptionRequested():
                        return
                    self.progress.emit(_o + cur, total_frames)

                inpainted = engine.inpaint_frames(
                    proc_frames, proc_mask, self.settings,
                    progress_callback=on_progress)

                if proc_w != crop_w or proc_h != crop_h:
                    inpainted = [cv2.resize(r, (crop_w, crop_h),
                                            interpolation=cv2.INTER_CUBIC)
                                 for r in inpainted]

                for i, frame in enumerate(frames):
                    if self._cancelled or self.isInterruptionRequested():
                        break
                    result = frame.copy()
                    if i < len(inpainted):
                        result[y1:y2, x1:x2] = inpainted[i]
                    writer.write(result)

                processed += batch_count
                self.progress.emit(processed, total_frames)

                del frames, cropped_frames, proc_frames, inpainted

            engine.release()

            if self._cancelled or self.isInterruptionRequested():
                try:
                    writer.release()
                except Exception:
                    pass
                self.cancelled_signal.emit()
                return

            self.phase_changed.emit("encoding")
            writer.release()
            self.phase_changed.emit("done")
            self.finished_signal.emit(self.output_path)

        except Exception as e:
            self.error_signal.emit(f"{e}\n\n{traceback.format_exc()}")

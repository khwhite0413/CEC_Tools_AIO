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

# All four tabs share fdcore's reader/writer now: VideoInfo for input
# probing (untagged-passthrough + 24.04 -> 24.0 fps snap), FFmpegWriter
# for output (bgr48le pipe + colr atom patching + rigorous untagged
# passthrough). uint8 frames from read_frames_batch get widened to
# uint16 via bgr24_to_bgr48le right before writing — the upcast is
# lossless and gives ProPainter / OpenProteus tabs the same color
# fidelity FlowDenoise has had since v1.0.0.
from fdcore.video_io import (VideoInfo, FFmpegWriter, SingleImageWriter,
                             is_image_file, bgr24_to_bgr48le)
from vdcore.video_io import SequentialFrameReader
from vdcore.inpaint_chunking import (iter_context_windows, resolve_window_size,
                                     split_window_on_boundaries)
from vdcore.mask_manager import mask_to_binary
from vdcore.inpaint_engine import InpaintEngine
from fdcore.scene_detection import (DEFAULT_SCENE_MSE_THRESHOLD,
                                    detect_scene_boundaries)


class _InpaintCancelled(Exception):
    """Internal control-flow signal for prompt cancellation."""


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
        reader = None
        engine = None
        writer = None
        try:
            info = VideoInfo(self.video_path)
            # effective_fps: user override (Dev-mode FPS dropdown) when
            # positive, otherwise VideoInfo's snapped value (24.04 →
            # 24.000 etc.). Drives BOTH the reader's -vf fps= and the
            # writer's -r so the resampled CFR timeline is consistent
            # across read -> process -> write. Without this matching,
            # VFR sources lose frames at batch boundaries (the "프레임
            # 드롭" symptom users hit).
            fps_override_setting = float(
                self.settings.get("fps_override", 0.0) or 0.0)
            effective_fps = (fps_override_setting if fps_override_setting > 0
                             else info.effective_fps)
            if not effective_fps or effective_fps <= 0:
                effective_fps = info.fps or 30.0
            # Recompute total frame count at the resampled rate so batch
            # math doesn't overshoot or undershoot the actual yield.
            # info.frame_count is the container's stsz count at native
            # fps; ffmpeg with -vf fps= will emit `duration * fps` frames.
            if info.duration and info.duration > 0:
                total_frames = int(round(info.duration * effective_fps))
            else:
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

            max_batch = resolve_window_size(
                self.settings.get("max_batch_frames", 0))
            temporal_overlap = max(
                int(self.settings.get("neighbor_length", 10) or 10),
                int(self.settings.get("ref_stride", 10) or 10))

            is_img_out = (is_image_file(self.output_path) or
                          self.settings.get("output_format") in ("png", "jpg", "jpeg"))
            if is_img_out:
                writer = SingleImageWriter(
                    self.output_path,
                    effective_fps,
                    info.width, info.height,
                    source_path=self.video_path)
            else:
                writer = FFmpegWriter(
                    self.output_path,
                    effective_fps,
                    info.width, info.height,
                    source_path=self.video_path,
                    color_tags=info.color_tags(),
                    output_format=self.settings.get("output_format", "h264"),
                    prores_profile=self.settings.get("prores_profile", "4444"))

            engine = InpaintEngine(
                self.device, fp16=bool(self.settings.get("fp16", True)))
            reader = SequentialFrameReader(
                self.video_path, fps=effective_fps)

            self.phase_changed.emit("processing")
            processed = 0
            frame_limit = self.frame_cap if self.frame_cap > 0 else -1
            windows = iter_context_windows(
                reader.read, window_size=max_batch,
                overlap=temporal_overlap, frame_limit=frame_limit)
            for frames, keep_start, keep_end in windows:
                if self._cancelled or self.isInterruptionRequested():
                    break
                boundaries = detect_scene_boundaries(
                    frames,
                    mode=self.settings.get("scene_detection_mode", "mse"),
                    mse_threshold=float(self.settings.get(
                        "scene_threshold", DEFAULT_SCENE_MSE_THRESHOLD)),
                    mse_size=(160, 90))
                scene_windows = split_window_on_boundaries(
                    frames, keep_start, keep_end, boundaries)
                cancelled_window = False
                for scene_frames, scene_keep_start, scene_keep_end in scene_windows:
                    output_count = scene_keep_end - scene_keep_start
                    cropped_frames = [
                        f[y1:y2, x1:x2].copy() for f in scene_frames]
                    crop_h, crop_w = cropped_mask.shape[:2]

                    # ProPainter operates well near 720p — upscale crop if
                    # the source resolution is much larger, run inference,
                    # then downscale back.
                    mask_scale = max(
                        info.height / 720.0, info.width / 1280.0, 1.0)
                    if mask_scale > 1.01:
                        proc_w = ((int(crop_w * mask_scale) + 7) // 8) * 8
                        proc_h = ((int(crop_h * mask_scale) + 7) // 8) * 8
                        proc_frames = [
                            cv2.resize(f, (proc_w, proc_h),
                                       interpolation=cv2.INTER_CUBIC)
                            for f in cropped_frames]
                        proc_mask = cv2.resize(
                            cropped_mask, (proc_w, proc_h),
                            interpolation=cv2.INTER_NEAREST)
                    else:
                        proc_frames = cropped_frames
                        proc_mask = cropped_mask
                        proc_w, proc_h = crop_w, crop_h

                    offset = processed

                    def on_progress(cur, total, _o=offset, _n=output_count):
                        if self._cancelled or self.isInterruptionRequested():
                            raise _InpaintCancelled()
                        ratio = min(1.0, max(0.0, cur / max(1, total)))
                        done = _o + int(round(ratio * _n))
                        self.progress.emit(done, max(total_frames, done))

                    try:
                        # ProPainter's RAFT path needs at least one frame
                        # pair. A cut can legitimately isolate a one-frame
                        # scene, so duplicate that frame only inside the
                        # model call and discard the duplicate afterwards.
                        model_frames = (proc_frames if len(proc_frames) > 1
                                        else proc_frames + [proc_frames[0].copy()])
                        inpainted = engine.inpaint_frames(
                            model_frames, proc_mask, self.settings,
                            progress_callback=on_progress)
                    except _InpaintCancelled:
                        cancelled_window = True
                        break

                    inpainted = inpainted[:len(scene_frames)]

                    if len(inpainted) != len(scene_frames):
                        raise RuntimeError(
                            "인페인트 결과 프레임 수가 입력과 다릅니다: "
                            f"입력 {len(scene_frames)}, 결과 {len(inpainted)}")

                    if proc_w != crop_w or proc_h != crop_h:
                        inpainted = [
                            cv2.resize(r, (crop_w, crop_h),
                                       interpolation=cv2.INTER_CUBIC)
                            for r in inpainted]

                    written_this_window = 0
                    for i in range(scene_keep_start, scene_keep_end):
                        if self._cancelled or self.isInterruptionRequested():
                            cancelled_window = True
                            break
                        frame = scene_frames[i]
                        result = frame.copy()
                        result[y1:y2, x1:x2] = inpainted[i]
                        if is_img_out:
                            writer.write(result)
                        else:
                            writer.write(bgr24_to_bgr48le(result))
                        written_this_window += 1

                    processed += written_this_window
                    self.progress.emit(processed, max(total_frames, processed))
                    del cropped_frames, proc_frames, model_frames, inpainted
                    if cancelled_window:
                        break
                if cancelled_window:
                    break

            if self._cancelled or self.isInterruptionRequested():
                reader.release()
                reader = None
                engine.release()
                engine = None
                try:
                    writer.release()
                except Exception:
                    pass
                writer = None
                self.cancelled_signal.emit()
                return

            if self.frame_cap <= 0:
                reader.verify_eof()
            reader.release()
            reader = None
            engine.release()
            engine = None

            self.phase_changed.emit("encoding")
            writer.release()
            writer = None
            self.progress.emit(processed, processed)
            self.phase_changed.emit("done")
            self.finished_signal.emit(self.output_path)

        except Exception as e:
            for resource in (reader, engine, writer):
                if resource is None:
                    continue
                try:
                    resource.release()
                except Exception:
                    pass
            self.error_signal.emit(f"{e}\n\n{traceback.format_exc()}")

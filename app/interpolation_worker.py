"""Streaming frame-interpolation worker using the common AIO video I/O."""

import os
import traceback

import cv2
import numpy as np
import torch
from PyQt5.QtCore import QThread, pyqtSignal

from app.interpolation_backend import InterpolationBackend, recursive_interpolate
from app.interpolation_fps import build_fps_plan, output_frame_count
from fdcore.video_io import VideoInfo, StreamingFrameReader, FFmpegWriter
from fdcore.scene_detection import SceneCutStream


# Keep the same normalized frame-MSE threshold used by Flow Denoise.  The
# earlier mean-absolute-difference threshold missed localized hard cuts and
# produced a single blended frame at edit boundaries.
DEFAULT_SCENE_CUT_THRESHOLD = 0.030


def _frame_to_tensor(frame, device):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    scale = 65535.0 if rgb.dtype == np.uint16 else 255.0
    return (torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
            .float().div_(scale).to(device))


def _tensor_to_bgr48(tensor):
    rgb = (tensor.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
           * 65535.0 + 0.5).astype(np.uint16)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _scene_cut(a, b, threshold):
    if threshold <= 0:
        return False
    aa = cv2.resize(a, (160, 90), interpolation=cv2.INTER_AREA).astype(np.float32)
    bb = cv2.resize(b, (160, 90), interpolation=cv2.INTER_AREA).astype(np.float32)
    scale = 65535.0 if a.dtype == np.uint16 else 255.0
    difference = (aa - bb) / scale
    return float(np.mean(difference * difference)) > threshold


class InterpolationWorker(QThread):
    progress = pyqtSignal(int, int)
    finished_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)
    cancelled_signal = pyqtSignal()
    phase_changed = pyqtSignal(str)

    def __init__(self, video_path, output_path, settings, device, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.output_path = output_path
        self.settings = dict(settings or {})
        self.device = device
        self._cancelled = False
        self._writer = None

    def cancel(self):
        self._cancelled = True
        self.requestInterruption()
        # release() may be waiting on ffmpeg's final mux. Terminating the
        # child makes cancellation immediate instead of blocking up to 300s.
        proc = getattr(getattr(self, "_writer", None), "proc", None)
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass

    def _stopped(self):
        return self._cancelled or self.isInterruptionRequested()

    def run(self):
        reader = writer = backend = None
        try:
            info = VideoInfo(self.video_path)
            source_fps = float(getattr(info, "effective_fps", None) or info.fps or 30.0)
            target = self.settings.get("output_fps")
            plan = build_fps_plan(source_fps,
                                  int(self.settings.get("multiplier", 2)),
                                  float(target) if target is not None else None)
            reader = StreamingFrameReader(self.video_path, fps=source_fps)
            total = max(0, int(reader.frame_count))
            backend = InterpolationBackend(
                self.settings.get("model", "rife_426"), self.device,
                bool(self.settings.get("fp16", True)))
            writer = FFmpegWriter(
                self.output_path, plan.output_fps, reader.width, reader.height,
                source_path=self.video_path, color_tags=info.color_tags(),
                output_format=self.settings.get("output_format", "h264"),
                prores_profile=self.settings.get("prores_profile", "hq"),
            )
            self._writer = writer
            self.phase_changed.emit("processing")
            prev_bgr = reader.read()
            if prev_bgr is None:
                raise ValueError("입력 영상에서 프레임을 읽지 못했습니다.")
            prev = _frame_to_tensor(prev_bgr, self.device)
            output_index = 0
            threshold = float(self.settings.get(
                "scene_cut_threshold", DEFAULT_SCENE_CUT_THRESHOLD))
            detector = SceneCutStream(
                self.settings.get("scene_detection_mode", "mse"),
                mse_threshold=threshold, mse_size=(160, 90))
            detector.feed(0, prev_bgr)

            buffered = {0: (prev_bgr, prev)}
            boundaries = set()
            next_pair_right = 1

            def emit_pair(left_bgr, left, right_bgr, right, pair_no,
                          out_index, is_cut):
                if is_cut:
                    generated = [
                        left if i < plan.interpolation_multiplier / 2 else right
                        for i in range(plan.interpolation_multiplier)
                    ] + [right]
                else:
                    generated = recursive_interpolate(
                        backend, left, right,
                        plan.interpolation_multiplier)
                for local_index, frame in enumerate(generated[:-1]):
                    absolute = pair_no * plan.interpolation_multiplier + local_index
                    wanted_absolute = int(round(
                        out_index * plan.generated_fps / plan.output_fps))
                    if absolute == wanted_absolute:
                        writer.write(_tensor_to_bgr48(frame))
                        out_index += 1
                return out_index

            pair_index = 0
            actual_frames = 1
            while True:
                if self._stopped():
                    break
                current_bgr = reader.read()
                if current_bgr is None:
                    break
                actual_frames += 1
                current = _frame_to_tensor(current_bgr, self.device)
                current_index = actual_frames - 1
                buffered[current_index] = (current_bgr, current)
                confirmed, new_boundaries = detector.feed(
                    current_index, current_bgr)
                boundaries.update(new_boundaries)
                while next_pair_right <= confirmed:
                    left_bgr, left = buffered[next_pair_right - 1]
                    right_bgr, right = buffered[next_pair_right]
                    output_index = emit_pair(
                        left_bgr, left, right_bgr, right,
                        next_pair_right - 1, output_index,
                        next_pair_right in boundaries)
                    buffered.pop(next_pair_right - 1, None)
                    next_pair_right += 1
                prev_bgr, prev = current_bgr, current
                pair_index = next_pair_right - 1
                self.progress.emit(actual_frames, max(actual_frames, total))

            if not self._stopped():
                confirmed, new_boundaries = detector.flush()
                boundaries.update(new_boundaries)
                while next_pair_right <= confirmed:
                    left_bgr, left = buffered[next_pair_right - 1]
                    right_bgr, right = buffered[next_pair_right]
                    output_index = emit_pair(
                        left_bgr, left, right_bgr, right,
                        next_pair_right - 1, output_index,
                        next_pair_right in boundaries)
                    buffered.pop(next_pair_right - 1, None)
                    next_pair_right += 1
                pair_index = next_pair_right - 1

            if self._stopped():
                try:
                    reader.release()
                finally:
                    reader = None
                try:
                    writer.release()
                except Exception:
                    pass
                finally:
                    writer = None
                    self._writer = None
                try:
                    backend.close()
                finally:
                    backend = None
                try:
                    if os.path.isfile(self.output_path):
                        os.remove(self.output_path)
                except Exception:
                    pass
                self.cancelled_signal.emit()
                return

            if actual_frames < 2:
                raise ValueError("프레임 보간에는 두 프레임 이상의 영상이 필요합니다.")

            # Metadata counts can be wrong after VFR→CFR normalization.
            # Close the progress contract with the actual decoded count.
            self.progress.emit(actual_frames, actual_frames)

            # A source frame occupies one full source-frame interval.  Hold
            # the final source frame for that last interval as well; ending
            # at only (N-1)/fps would shorten the video and trim its audio.
            # Use the actual decoded count, not cv2/ffprobe metadata. VFR
            # sources can legitimately differ after FFmpeg's CFR filter.
            tail_start = (actual_frames - 1) * plan.interpolation_multiplier
            desired_outputs = output_frame_count(
                actual_frames, source_fps, plan.output_fps)
            final_frame = _tensor_to_bgr48(prev)
            for local_index in range(plan.interpolation_multiplier):
                absolute = tail_start + local_index
                if output_index >= desired_outputs:
                    break
                wanted_absolute = int(round(
                    output_index * plan.generated_fps / plan.output_fps))
                if absolute == wanted_absolute:
                    writer.write(final_frame)
                    output_index += 1
            self.phase_changed.emit("encoding")
            reader.release()
            reader = None
            writer.release()
            writer = None
            self._writer = None
            backend.close()
            backend = None
            self.phase_changed.emit("done")
            self.finished_signal.emit(self.output_path)
        except Exception as exc:
            if self._stopped():
                try:
                    if os.path.isfile(self.output_path):
                        os.remove(self.output_path)
                except Exception:
                    pass
                self.cancelled_signal.emit()
            else:
                self.error_signal.emit(f"{exc}\n\n{traceback.format_exc()}")
        finally:
            self._writer = None
            for item in (reader, writer):
                if item is not None:
                    try:
                        item.release()
                    except Exception:
                        pass
            if backend is not None:
                try:
                    backend.close()
                except Exception:
                    pass

"""QThread wrapper around spandrel-loaded SR models (OpenProteus etc.).

Signal pattern mirrors fdcore / inpaint workers so MainWindow can reuse
the same progress / cancel / error UI for every tab:
    progress(int done, int total)
    finished_signal(str output_path)
    error_signal(str message)
    cancelled_signal()
    phase_changed(str)  # "processing" | "encoding" | "done"
"""

import os
import traceback

import cv2
import numpy as np
import torch
from PyQt5.QtCore import QThread, pyqtSignal

# fdcore VideoInfo so the Upscale tab inherits the same untagged-passthrough
# + non-integer fps snapping (24.04 -> 24.0) the other two tabs use.
# Unified writer path: see inpaint_worker.py for the rationale. uint8
# upscaler output is widened to uint16 via bgr24_to_bgr48le right
# before each writer.write so the fdcore writer's bgr48le pipe gets
# the byte count it expects.
from fdcore.video_io import VideoInfo, FFmpegWriter, bgr24_to_bgr48le
from vdcore.video_io import read_frames_batch


_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_MODULE_DIR, ".."))
_WEIGHTS_DIR = os.path.join(_PROJECT_ROOT, "weights", "openproteus")


# Model registry. Extend here when adding more upscale models — the
# UpscalePage's model dropdown pulls labels from this table via i18n keys.
MODELS = {
    "openproteus_2x": {
        "filename": "2x_OpenProteus_Compact_i2_70K.pth",
        "label_key": "upscale_model_openproteus",
        "scale_fallback": 2,
    },
}


def model_path(model_key: str) -> str:
    info = MODELS.get(model_key)
    if not info:
        return ""
    return os.path.join(_WEIGHTS_DIR, info["filename"])


def model_available(model_key: str) -> bool:
    p = model_path(model_key)
    return bool(p) and os.path.isfile(p) and os.path.getsize(p) > 1_000_000


class UpscaleWorker(QThread):
    progress = pyqtSignal(int, int)
    finished_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)
    cancelled_signal = pyqtSignal()
    phase_changed = pyqtSignal(str)

    def __init__(self, video_path, output_path, settings, device, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.output_path = output_path
        self.settings = settings or {}
        self.device = device
        self._cancelled = False

    def cancel(self):
        self._cancelled = True
        self.requestInterruption()

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------
    def _load_model(self, model_key: str):
        """spandrel.ModelLoader.load_from_file → ImageModelDescriptor."""
        from spandrel import ModelLoader
        path = model_path(model_key)
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(
                f"업스케일 가중치를 찾을 수 없습니다: {path}\n"
                "setup.bat을 다시 실행하거나 weights/openproteus/ 폴더를 확인하세요.")
        loader = ModelLoader()
        desc = loader.load_from_file(path)
        return desc

    def _upscale_frame(self, model, frame_bgr, use_half):
        """One frame through the model. Returns BGR uint8 at scale*input."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # [H, W, C] uint8 → [1, C, H, W] float
        tensor = torch.from_numpy(rgb).to(self.device)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).float() / 255.0
        if use_half:
            tensor = tensor.half()
        with torch.no_grad():
            out = model(tensor)
        out = out.float().clamp_(0.0, 1.0)
        out = out.squeeze(0).permute(1, 2, 0).contiguous().cpu().numpy()
        out = (out * 255.0 + 0.5).astype(np.uint8)
        return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------
    def run(self):
        try:
            info = VideoInfo(self.video_path)
            total_frames = info.frame_count
            if total_frames <= 0:
                self.error_signal.emit("입력 영상에 프레임이 없습니다.")
                return

            model_key = self.settings.get("model", "openproteus_2x")
            max_batch = int(self.settings.get("max_batch_frames", 0) or 0)
            use_fp16 = bool(self.settings.get("fp16", True))

            desc = self._load_model(model_key)
            scale = int(getattr(desc, "scale", None)
                        or MODELS[model_key]["scale_fallback"])
            model = desc.model.to(self.device).eval()
            use_half = use_fp16 and self.device.type == "cuda"
            if use_half:
                try:
                    model = model.half()
                except Exception:
                    use_half = False

            # Model always outputs at native scale (2x for OpenProteus).
            # If the user picked a specific target long-edge, cv2.resize
            # the model output to that resolution before piping to ffmpeg.
            # OpenCV INTER_LANCZOS4 for upscale, INTER_AREA for downscale —
            # standard combo, matches what most NLE workflows ship.
            model_w = info.width * scale
            model_h = info.height * scale
            target_long = int(self.settings.get("target_long_edge", 0) or 0)
            if target_long > 0:
                model_long = max(model_w, model_h)
                if model_long != target_long:
                    ratio = target_long / float(model_long)
                    out_w = max(2, int(round(model_w * ratio / 2.0)) * 2)
                    out_h = max(2, int(round(model_h * ratio / 2.0)) * 2)
                else:
                    out_w, out_h = model_w, model_h
            else:
                out_w, out_h = model_w, model_h

            post_resize = (out_w, out_h) != (model_w, model_h)
            post_interp = (cv2.INTER_LANCZOS4
                           if out_w > model_w else cv2.INTER_AREA)

            # User-set fps override (Dev mode → Auto unchecked → drop-
            # down value) wins over the snapped source fps. 0 = use
            # whatever VideoInfo.effective_fps reports.
            fps_override = float(
                self.settings.get("fps_override", 0.0) or 0.0)
            effective_fps = (fps_override if fps_override > 0
                             else info.effective_fps)
            writer = FFmpegWriter(
                self.output_path,
                effective_fps,
                out_w, out_h,
                source_path=self.video_path,
                color_tags=info.color_tags(),
                output_format=self.settings.get("output_format", "h264"),
                prores_profile=self.settings.get("prores_profile", "4444"))

            self.phase_changed.emit("processing")

            if max_batch <= 0 or max_batch >= total_frames:
                max_batch = total_frames
                num_batches = 1
            else:
                num_batches = (total_frames + max_batch - 1) // max_batch

            processed = 0
            for b in range(num_batches):
                if self._cancelled or self.isInterruptionRequested():
                    break
                bs = b * max_batch
                be = min(bs + max_batch, total_frames)
                frames = read_frames_batch(self.video_path, bs, be - bs)
                if not frames:
                    break
                for i, f in enumerate(frames):
                    if self._cancelled or self.isInterruptionRequested():
                        break
                    up = self._upscale_frame(model, f, use_half)
                    if post_resize:
                        up = cv2.resize(up, (out_w, out_h),
                                        interpolation=post_interp)
                    writer.write(bgr24_to_bgr48le(up))
                    if (i + 1) % 4 == 0:
                        # batch-internal progress so the UI doesn't sit
                        # silent for long batches.
                        self.progress.emit(processed + i + 1, total_frames)
                processed += len(frames)
                self.progress.emit(processed, total_frames)
                del frames

            # Free GPU memory before the (potentially long) ffmpeg flush.
            del model, desc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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

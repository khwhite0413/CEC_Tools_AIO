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
from fdcore.video_io import (VideoInfo, FFmpegWriter, SingleImageWriter,
                             is_image_file, bgr24_to_bgr48le)
from vdcore.video_io import read_frames_batch


_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_MODULE_DIR, ".."))
_WEIGHTS_ROOT = os.path.join(_PROJECT_ROOT, "weights")


# ---------------------------------------------------------------------
# Two-level model registry
# ---------------------------------------------------------------------
# MODEL_FAMILIES groups variants under a "family" (e.g. AniToon has three
# size tiers: Small/Normal/Large). The UI shows a two-step cascading
# dropdown — user picks family, then a variant within that family —
# with a per-variant tooltip explaining trade-offs.
#
# `variant_key` is the identifier that flows through `settings["model"]`
# into UpscaleWorker.  For backwards compatibility the flat `MODELS`
# dict below (built at import time) still maps variant_key → info the
# way earlier callers expected.
#
# Adding a new family / variant: append here.  UI + setup.bat pick it
# up automatically. spandrel handles every arch listed below out of
# the box (PLKSRArch understands both PLKSR and RealPLKSR state dicts).
# ---------------------------------------------------------------------
MODEL_FAMILIES = {
    "openproteus": {
        "label_key": "upscale_family_openproteus",
        "download_base": (
            "https://github.com/Sirosky/Upscale-Hub/releases/download/"
            "OpenProteus/"),
        "weights_subdir": "openproteus",
        "variants": {
            "openproteus_2x": {
                "filename": "2x_OpenProteus_Compact_i2_70K.pth",
                "label_key": "upscale_variant_openproteus_2x",
                "tooltip_key": "upscale_variant_openproteus_2x_tt",
                "scale_fallback": 2,
            },
        },
    },
    "anitoon": {
        "label_key": "upscale_family_anitoon",
        "download_base": (
            "https://github.com/Sirosky/Upscale-Hub/releases/download/"
            "AniToon/"),
        "weights_subdir": "anitoon",
        "variants": {
            "anitoon_small": {
                "filename": "2x_AniToon_RPLKSRS_242500.pth",
                "label_key": "upscale_variant_anitoon_small",
                "tooltip_key": "upscale_variant_anitoon_small_tt",
                "scale_fallback": 2,
            },
            "anitoon_normal": {
                "filename": "2x_AniToon_RPLKSR_197500.pth",
                "label_key": "upscale_variant_anitoon_normal",
                "tooltip_key": "upscale_variant_anitoon_normal_tt",
                "scale_fallback": 2,
            },
            "anitoon_large": {
                "filename": "2x_AniToon_RPLKSRL_280K.pth",
                "label_key": "upscale_variant_anitoon_large",
                "tooltip_key": "upscale_variant_anitoon_large_tt",
                "scale_fallback": 2,
            },
        },
    },
    "anisd": {
        "label_key": "upscale_family_anisd",
        "download_base": (
            "https://github.com/Sirosky/Upscale-Hub/releases/download/"
            "AniSD-RealPLKSR/"),
        "weights_subdir": "anisd",
        "variants": {
            "anisd_base": {
                "filename": "2x_AniSD_RealPLKSR_140K.pth",
                "label_key": "upscale_variant_anisd_base",
                "tooltip_key": "upscale_variant_anisd_base_tt",
                "scale_fallback": 2,
            },
            "anisd_ac": {
                "filename": "2x_AniSD_AC_RealPLKSR_127500.pth",
                "label_key": "upscale_variant_anisd_ac",
                "tooltip_key": "upscale_variant_anisd_ac_tt",
                "scale_fallback": 2,
            },
            "anisd_dc": {
                "filename": "2x_AniSD_DC_RealPLKSR_115K.pth",
                "label_key": "upscale_variant_anisd_dc",
                "tooltip_key": "upscale_variant_anisd_dc_tt",
                "scale_fallback": 2,
            },
        },
    },
    "ani4k_v2": {
        "label_key": "upscale_family_ani4k_v2",
        "download_base": (
            "https://github.com/Sirosky/Upscale-Hub/releases/download/"
            "Ani4K-v2/"),
        "weights_subdir": "ani4k_v2",
        "variants": {
            "ani4k_v2_compact": {
                "filename": "2x_Ani4Kv2_G6i2_Compact_107500.pth",
                "label_key": "upscale_variant_ani4k_v2_compact",
                "tooltip_key": "upscale_variant_ani4k_v2_compact_tt",
                "scale_fallback": 2,
            },
            "ani4k_v2_ultracompact": {
                "filename": "2x_Ani4Kv2_G6i2_UltraCompact_105K.pth",
                "label_key": "upscale_variant_ani4k_v2_ultracompact",
                "tooltip_key": "upscale_variant_ani4k_v2_ultracompact_tt",
                "scale_fallback": 2,
            },
        },
    },
    "aniscale2": {
        "label_key": "upscale_family_aniscale2",
        "download_base": (
            "https://github.com/Sirosky/Upscale-Hub/releases/download/"
            "AniScale2/"),
        "weights_subdir": "aniscale2",
        "variants": {
            "aniscale2_compact_s": {
                "filename": "2x_AniScale2S_Compact_i8_60K.pth",
                "label_key": "upscale_variant_aniscale2_compact_s",
                "tooltip_key": "upscale_variant_aniscale2_compact_s_tt",
                "scale_fallback": 2,
            },
            "aniscale2_omni": {
                "filename": "2x_AniScale2_Omni_i16_40K.pth",
                "label_key": "upscale_variant_aniscale2_omni",
                "tooltip_key": "upscale_variant_aniscale2_omni_tt",
                "scale_fallback": 2,
            },
            "aniscale2_ditn": {
                "filename": "2x_AniScale2_DITN_i16_75K.pth",
                "label_key": "upscale_variant_aniscale2_ditn",
                "tooltip_key": "upscale_variant_aniscale2_ditn_tt",
                "scale_fallback": 2,
            },
            "aniscale2_esrgan_lite": {
                "filename": "2x_AniScale2_ESRGAN_Lite_i16_165K.pth",
                "label_key": "upscale_variant_aniscale2_esrgan_lite",
                "tooltip_key": "upscale_variant_aniscale2_esrgan_lite_tt",
                "scale_fallback": 2,
            },
            "aniscale2_esrgan": {
                "filename": "2x_AniScale2_ESRGAN_i16_110K.pth",
                "label_key": "upscale_variant_aniscale2_esrgan",
                "tooltip_key": "upscale_variant_aniscale2_esrgan_tt",
                "scale_fallback": 2,
            },
            "aniscale2_swinir": {
                "filename": "2x_AniScale2_SwinIR_i16_265K.pth",
                "label_key": "upscale_variant_aniscale2_swinir",
                "tooltip_key": "upscale_variant_aniscale2_swinir_tt",
                "scale_fallback": 2,
            },
            "aniscale2_refiner": {
                "filename": "1x_AniScale2_Refiner_10K.pth",
                "label_key": "upscale_variant_aniscale2_refiner",
                "tooltip_key": "upscale_variant_aniscale2_refiner_tt",
                # 1x refiner — no size change, just line-art / sharpness
                # cleanup. Included per user request; hover tooltip warns
                # the user that this is meant to compose with another
                # upscaler rather than stand alone.
                "scale_fallback": 1,
            },
        },
    },
}


def _build_flat_models() -> dict:
    """Flatten MODEL_FAMILIES into a variant_key → info dict.

    Each entry inherits `family` (family_key), `download_url`, and
    `weights_dir` fields on top of the variant's own metadata so
    callers can look up everything without walking two levels.
    """
    flat = {}
    for fam_key, fam in MODEL_FAMILIES.items():
        weights_dir = os.path.join(_WEIGHTS_ROOT, fam["weights_subdir"])
        for var_key, var in fam["variants"].items():
            flat[var_key] = {
                **var,
                "family": fam_key,
                "download_url": fam["download_base"] + var["filename"],
                "weights_dir": weights_dir,
            }
    return flat


# Flat lookup for backward compat (older callers used MODELS[key] directly).
MODELS = _build_flat_models()


def variants_for_family(family_key: str):
    """Yield (variant_key, variant_info) pairs for a family, preserving
    the registry declaration order — so 'Small → Normal → Large'-style
    ordering that the author intended stays visible in the dropdown."""
    fam = MODEL_FAMILIES.get(family_key)
    if not fam:
        return
    weights_dir = os.path.join(_WEIGHTS_ROOT, fam["weights_subdir"])
    for var_key, var in fam["variants"].items():
        yield var_key, {
            **var,
            "family": family_key,
            "download_url": fam["download_base"] + var["filename"],
            "weights_dir": weights_dir,
        }


def family_of(model_key: str) -> str:
    """Return the family_key that owns `model_key`, or '' if unknown."""
    info = MODELS.get(model_key)
    return info["family"] if info else ""


def model_path(model_key: str) -> str:
    info = MODELS.get(model_key)
    if not info:
        return ""
    return os.path.join(info["weights_dir"], info["filename"])


def model_available(model_key: str) -> bool:
    p = model_path(model_key)
    return bool(p) and os.path.isfile(p) and os.path.getsize(p) > 1_000_000


def download_missing_weights(progress=print) -> None:
    """Fetch every registered weight that isn't already on disk.

    Called by setup.bat step 8. Idempotent — files already present and
    non-truncated are skipped.  Uses urllib so it doesn't depend on
    spandrel / torch having imported successfully, which keeps the
    download step usable even when a later dependency install failed.
    """
    import urllib.request
    for model_key, info in MODELS.items():
        dst = model_path(model_key)
        if os.path.isfile(dst) and os.path.getsize(dst) > 1_000_000:
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        url = info["download_url"]
        progress(f"  {model_key}: downloading {os.path.basename(dst)} ...")
        try:
            urllib.request.urlretrieve(url, dst)
            size_kb = os.path.getsize(dst) // 1024
            progress(f"    OK ({size_kb} KB)")
        except Exception as e:
            progress(f"    FAIL: {e}")
            # Delete partial download so the next run retries cleanly.
            try:
                if os.path.exists(dst):
                    os.remove(dst)
            except Exception:
                pass


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
            info = MODELS.get(model_key, {})
            fam = info.get("family", "")
            raise FileNotFoundError(
                f"업스케일 가중치를 찾을 수 없습니다: {path}\n"
                f"setup.bat 을 다시 실행하거나 weights/{fam}/ 폴더를 "
                "확인하세요.")
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
            # See inpaint_worker.run() for the full rationale on why
            # the reader's -vf fps= and the writer's -r have to be
            # locked to the same effective_fps. Short version: VFR
            # sources drop frames at batch boundaries otherwise.
            fps_override_setting = float(
                self.settings.get("fps_override", 0.0) or 0.0)
            effective_fps = (fps_override_setting if fps_override_setting > 0
                             else info.effective_fps)
            if not effective_fps or effective_fps <= 0:
                effective_fps = info.fps or 30.0
            if info.duration and info.duration > 0:
                total_frames = int(round(info.duration * effective_fps))
            else:
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

            # Aspect + post-resize planning. Two paths depending on mode:
            #
            #   fill / auto — Premiere "Fill Frame" semantics: scale model
            #     output to COVER the target frame in both dims (Lanczos),
            #     then center-crop excess. Reaches exact target frame with
            #     no letterbox. Supersedes the plain long-edge resize.
            #
            #   crop / none — plain long-edge resize of model output; crop
            #     mode additionally trims to target aspect (may not reach
            #     exact target frame). Legacy `force_16_9` bool (from the
            #     single Upscale tab) reads as crop-only for back-compat.
            model_w = info.width * scale
            model_h = info.height * scale
            target_long = int(self.settings.get("target_long_edge", 0) or 0)
            aspect_mode = str(
                self.settings.get("aspect_mode") or
                ("crop" if self.settings.get("force_16_9") else "none")
            ).lower()

            from fdcore.transcode import _resolve_aspect_geometry
            out_w, out_h = model_w, model_h
            final_w, final_h = model_w, model_h
            crop_x, crop_y = 0, 0
            need_crop = False

            geom = _resolve_aspect_geometry(
                model_w, model_h, aspect_mode, target_long)
            if geom is not None and geom[0] == "cover":
                # Fill / auto: scale model output to (out_w, out_h) that
                # covers target, then crop to (final_w, final_h).
                _, final_w, final_h, out_w, out_h = geom
                crop_x = ((out_w - final_w) // 2 // 2) * 2
                crop_y = ((out_h - final_h) // 2 // 2) * 2
                need_crop = True
            else:
                # crop / none: honour target_long_edge via a plain resize.
                if target_long > 0:
                    model_long = max(model_w, model_h)
                    if model_long != target_long:
                        ratio = target_long / float(model_long)
                        out_w = max(2, int(round(model_w * ratio / 2.0)) * 2)
                        out_h = max(2, int(round(model_h * ratio / 2.0)) * 2)
                # Crop mode then trims to target aspect at post-resize dims.
                if geom is not None and geom[0] == "crop":
                    geom2 = _resolve_aspect_geometry(out_w, out_h, "crop")
                    if geom2 is not None:
                        _, gw, gh = geom2
                        final_w, final_h = gw, gh
                        crop_x = (out_w - final_w) // 2
                        crop_y = (out_h - final_h) // 2
                        need_crop = True
                    else:
                        final_w, final_h = out_w, out_h
                else:
                    final_w, final_h = out_w, out_h

            post_resize = (out_w, out_h) != (model_w, model_h)
            post_interp = (cv2.INTER_LANCZOS4
                           if out_w > model_w else cv2.INTER_AREA)

            is_img_out = (is_image_file(self.output_path) or
                          self.settings.get("output_format") in ("png", "jpg", "jpeg"))
            if is_img_out:
                writer = SingleImageWriter(
                    self.output_path,
                    effective_fps,
                    final_w, final_h,
                    source_path=self.video_path)
            else:
                writer = FFmpegWriter(
                    self.output_path,
                    effective_fps,
                    final_w, final_h,
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
                frames = read_frames_batch(self.video_path, bs, be - bs,
                                            fps=effective_fps)
                if not frames:
                    break
                for i, f in enumerate(frames):
                    if self._cancelled or self.isInterruptionRequested():
                        break
                    up = self._upscale_frame(model, f, use_half)
                    if post_resize:
                        up = cv2.resize(up, (out_w, out_h),
                                        interpolation=post_interp)
                    if need_crop:
                        up = up[crop_y:crop_y + final_h,
                                crop_x:crop_x + final_w]
                    if is_img_out:
                        writer.write(up)
                    else:
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

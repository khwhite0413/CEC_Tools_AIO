"""High-level pipeline: settings dict -> denoised video file.

End-to-end: load video -> tensor -> temporal flow average -> (optional)
extract noise visualization -> write video with audio.

The settings dict pattern (회고 §10) lets us add stages without changing
function signatures.

Color management policy: stay in the source's encoded sRGB-ish domain
end-to-end. video_io.FFmpegReader/Writer pin the same matrix/range so the
read->write round-trip preserves the source's colorspace; an 8-bit (h264)
output path adds TPDF dither just before quantization to avoid banding.
The earlier IDT/ODT / linear-working-space machinery was removed because
the denoise modules (MEMFOF, RAFT, color/scene thresholds) are tuned for
encoded values — running them in linear silently changed denoise behavior.
"""

from typing import Optional, Callable

from . import video_io
from .temporal_average import temporal_flow_average
from .extract_noise import extract_noise
from .progress import ProgressReporter


DEFAULT_SETTINGS = {
    # TemporalFlowAverage  (matches the upstream node defaults from the screenshot)
    "window_size": 3,
    "weight_decay": 0.80,
    "flow_model": "memfof",   # memfof | raft_small | raft_large
    "flow_iterations": 4,
    "color_threshold": 0.020,
    "scene_threshold": 0.030,
    "batch_size": 24,
    "precision": "bf16",          # bf16 | fp32
    "flow_scale": 0.5,            # 0.3..1.0 — flow res relative to full
    # ExtractNoise (visualization only)
    "noise_amplify": 5.0,
    "noise_color_space": "YCbCr",
    "noise_preview": "heatmap",
    "compute_noise_preview": False,
    # Output clamp: keep tensor values in [0,1]. The video writer will still
    # clip to uint8 either way, so this only matters for downstream tensor
    # consumers / debugging.
    "clamp_output": True,
    # Output format (full-run only — preview always overrides to h264)
    "output_format": "h264",      # h264 | prores  (default = small, broadly compatible)
    "prores_profile": "4444",     # proxy | lt | standard | hq | 4444 | 4444xq
    # Diagnostic short-circuit. When True, the pipeline skips denoise
    # entirely and pipes decoded BGR frames straight back to the writer.
    # Used to verify the encode/decode chain itself is color-stable, with
    # no float round-trip and no dither in the way. UI exposes it under
    # Dev mode only.
    "passthrough": False,
    # User-forced fps. 0 = auto (use VideoInfo.fps as detected). >0
    # overrides for both the FFmpegReader's -vf fps=X (input resampling
    # nominal) and the FFmpegWriter's -r X (output timebase). Useful when
    # the source's metadata reports a fractional fps (e.g. Seedance AI
    # mp4 reporting 24.04 for what is really 24.000 average) and the
    # auto-detected value produces stutter. UI exposes under Dev mode.
    "fps_override": 0.0,
}


def merged_settings(overrides: Optional[dict] = None) -> dict:
    out = dict(DEFAULT_SETTINGS)
    if overrides:
        out.update(overrides)
    return out


def run_on_tensor(images_bhwc, settings: dict,
                  progress_cb: Optional[Callable[[int, int], None]] = None,
                  reporter: Optional[ProgressReporter] = None) -> dict:
    """Run the full denoise pipeline on a BHWC float [0,1] RGB tensor.

    Returns dict with keys: denoised, clean, weight_map,
    noise_total / noise_chroma / noise_luma (only if compute_noise_preview).
    """
    if reporter is None:
        reporter = ProgressReporter(progress_cb)

    settings = merged_settings(settings)
    clean, weight_map = temporal_flow_average(images_bhwc, settings, reporter)

    out = {"clean": clean, "weight_map": weight_map}

    if settings.get("compute_noise_preview", False):
        n_total, n_chroma, n_luma = extract_noise(images_bhwc, clean, settings)
        out["noise_total"] = n_total
        out["noise_chroma"] = n_chroma
        out["noise_luma"] = n_luma

    # The temporally-averaged result IS the denoised output — no selective
    # blending. Optionally clamp into the standard [0,1] range.
    out["denoised"] = (clean.clamp(0, 1)
                       if settings.get("clamp_output", True)
                       else clean)
    return out


def run_on_file(input_path: str, output_path: str, settings: dict,
                progress_cb: Optional[Callable[[int, int], None]] = None,
                reporter: Optional[ProgressReporter] = None,
                frame_cap: int = -1,
                noise_path: Optional[str] = None,
                phase_cb: Optional[Callable[[str], None]] = None) -> dict:
    """Read input video, denoise, write output with original audio.

    Args:
        frame_cap: -1 = full video, otherwise process only first N frames
            (used for fast preview, 회고 §6).
        noise_path: when set AND settings['compute_noise_preview'] is True,
            also write the noise_total visualization to this path. The file
            is intended as an ephemeral preview, not a deliverable.

    Returns:
        {"denoised": output_path, "noise": noise_path or None}
    """
    if reporter is None:
        reporter = ProgressReporter(progress_cb)

    def _phase(p: str):
        if phase_cb is not None:
            try:
                phase_cb(p)
            except Exception:
                pass

    info = video_io.probe(input_path)
    settings = merged_settings(settings)

    fmt = settings.get("output_format", "h264")
    passthrough = bool(settings.get("passthrough", False))
    src_tags = info.color_tags() if hasattr(info, "color_tags") else {}

    # Effective fps: user override (Dev-mode FPS dropdown) if positive,
    # else the standard-rate-snapped VideoInfo.effective_fps (24.04
    # source → 24.0 standard, etc.). Forwarded to BOTH the reader's
    # -vf fps= (input resampling target) AND the writer's -r (output
    # timebase) so the encoded mp4 plays back at the chosen rate.
    fps_override = float(settings.get("fps_override", 0.0) or 0.0)
    effective_fps = (fps_override if fps_override > 0
                     else getattr(info, "effective_fps", info.fps))

    # Fast path: pure passthrough on a stream-copy-compatible target. No
    # decode, no bgr24 round-trip, no libx264 re-encode — bit-exact remux.
    # Eliminates the 8-bit YUV→RGB→YUV asymmetric ±1 LSB drift that the
    # transcode passthrough still incurred.
    if passthrough and video_io.can_stream_copy(input_path, fmt):
        _phase("processing")
        _phase("encoding")
        video_io.copy_stream(input_path, output_path, frame_cap=frame_cap)
        _phase("done")
        return {"denoised": output_path, "noise": None}

    bgr = video_io.read_frames_bgr(input_path, 0, frame_cap, fps=effective_fps)
    if not bgr:
        raise RuntimeError(f"No frames read from {input_path}")

    if passthrough:
        # Fallback passthrough: source codec / output format ruled out
        # stream-copy (e.g. PNG sequence input or ProRes output requested).
        # Decoded BGR bytes go straight to the writer.
        _phase("processing")
        _phase("encoding")
        out_frames = bgr
        result = {"denoised": None}
    else:
        images = video_io.frames_bgr_to_tensor(bgr)

        _phase("processing")
        result = run_on_tensor(images, settings, reporter=reporter)

        _phase("encoding")
        denoised = result["denoised"]

        # Apply TPDF dither only on the 8-bit h264 path. ProRes (10-bit) and
        # PNG sequence skip it.
        if fmt == "h264":
            denoised = video_io.tpdf_dither(denoised).clamp(0, 1)

        out_frames = video_io.tensor_to_frames_bgr(denoised)

    h, w = out_frames[0].shape[:2]
    if fmt == "image_seq":
        # PNG sequence in a folder + meta.json sidecar (no ffmpeg).
        writer = video_io.ImageSequenceWriter(
            output_path, effective_fps, w, h, source_path=input_path)
    else:
        writer = video_io.FFmpegWriter(
            output_path, effective_fps, w, h,
            source_path=input_path,
            output_format=fmt,
            prores_profile=settings.get("prores_profile", "hq"),
            color_tags=src_tags)
    try:
        for f in out_frames:
            writer.write(f)
    finally:
        writer.release()

    out = {"denoised": output_path, "noise": None}

    if noise_path and result.get("noise_total") is not None:
        _phase("encoding_noise")
        n_frames = video_io.tensor_to_frames_bgr(result["noise_total"])
        nh, nw = n_frames[0].shape[:2]
        # No source_path = no audio mux; visualization-only preview at
        # the same effective fps as the main output so the wipe lines up.
        nwriter = video_io.FFmpegWriter(noise_path, effective_fps, nw, nh,
                                        source_path=None,
                                        color_tags=src_tags)
        try:
            for f in n_frames:
                nwriter.write(f)
        finally:
            nwriter.release()
        out["noise"] = noise_path

    _phase("done")
    return out

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

import torch

from . import video_io
from .temporal_average import temporal_flow_average
from .extract_noise import extract_noise
from .progress import ProgressReporter


# Streaming-loop chunk size: number of frames to commit per iteration. Each
# iteration also holds `window_size` frames of look-behind + `window_size` of
# look-ahead, so the buffer under denoise is `chunk + 2*window` frames.
#
# 200 was picked so the peak per-chunk CPU footprint (buffer + BHWC tensor +
# BCHW copy + zeros_like output + single-channel weight accumulator) fits
# comfortably in ~10 GB at 720p on the desktop's default settings, leaving
# headroom for the ffmpeg reader/writer subprocesses and the model on GPU.
# Larger chunks = fewer Python-loop iterations, smaller = tighter memory
# ceiling. Not currently exposed to the UI; tune here if hitting new limits.
_STREAM_CHUNK_SIZE = 200


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

    Returns dict with keys: denoised, clean, and (only if
    compute_noise_preview) noise_total / noise_chroma / noise_luma.

    The `weight_map` key used to be part of this dict — a normalized 3-channel
    visualization of temporal_flow_average's per-pixel weight accumulator.
    Nothing in `run_on_file` ever read it and on long clips its derivation
    alone allocated tens of GB on CPU, tripping DefaultCPUAllocator. The
    derivation + key have been removed.
    """
    if reporter is None:
        reporter = ProgressReporter(progress_cb)

    settings = merged_settings(settings)
    clean, _ = temporal_flow_average(images_bhwc, settings, reporter)

    out = {"clean": clean}

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
    """Streaming denoise pipeline: read/denoise/encode in overlapping chunks
    so CPU RAM stays bounded regardless of clip length.

    The previous implementation loaded the whole video into a single BHWC
    float32 CPU tensor before running temporal_flow_average on it, then
    converted the entire denoised tensor back to a list of numpy frames
    for the writer. For an 8000-frame 720p clip that meant ~290 GB of
    virtual memory held resident on the CPU side and reliable
    DefaultCPUAllocator OOMs on any halfway-serious source.

    This version keeps only `_STREAM_CHUNK_SIZE + 2*window_size` frames in
    memory at once. Each chunk is denoised with `window_size` frames of
    look-behind and `window_size` frames of look-ahead present in the
    buffer, then only the middle frames are committed to the writer.
    Every emitted frame therefore sees the same ±window_size context it
    would have seen in a full-video batch run, so per-frame output is
    numerically identical to the old batch pipeline. tpdf_dither is
    applied per chunk (its output is statistically indistinguishable from
    a full-tensor dither — the noise is per-pixel-independent by design).

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
    #
    # Skipped when the user explicitly set fps_override to a value other
    # than the source's actual rate: `ffmpeg -c copy` can't rewrite the
    # container timebase, so a user who picks 24 on a 24.04 source would
    # otherwise see 24.04 still in the output. Falling through to the
    # decode/re-encode passthrough path makes the new -r value actually
    # land in the file.
    src_fps = float(getattr(info, "fps", 0.0) or 0.0)
    fps_change_requested = (
        fps_override > 0 and abs(fps_override - src_fps) > 0.001)
    if (passthrough
            and not fps_change_requested
            and video_io.can_stream_copy(input_path, fmt)):
        _phase("processing")
        _phase("encoding")
        video_io.copy_stream(input_path, output_path, frame_cap=frame_cap)
        _phase("done")
        return {"denoised": output_path, "noise": None}

    # Open streaming reader + peek first frame to lock dimensions before
    # constructing writers. Every downstream frame arrives one at a time.
    reader = video_io.StreamingFrameReader(input_path, fps=effective_fps)
    try:
        first_frame = reader.read()
        if first_frame is None:
            raise RuntimeError(f"No frames read from {input_path}")
        h, w = first_frame.shape[:2]

        # Total frame count for progress + frame_cap enforcement. The
        # reader's frame_count comes from the source metadata; frame_cap
        # (set for the 80-frame quick preview path) tightens it further.
        total_frames = reader.frame_count
        if frame_cap > 0:
            total_frames = min(total_frames, frame_cap)

        # Main writer (denoised output).
        is_img_out = video_io.is_image_file(output_path) or fmt in ("png", "jpg", "jpeg")
        if is_img_out and total_frames <= 1:
            writer = video_io.SingleImageWriter(
                output_path, effective_fps, w, h, source_path=input_path)
        elif fmt == "image_seq":
            writer = video_io.ImageSequenceWriter(
                output_path, effective_fps, w, h, source_path=input_path)
        else:
            writer = video_io.FFmpegWriter(
                output_path, effective_fps, w, h,
                source_path=input_path,
                output_format=fmt,
                prores_profile=settings.get("prores_profile", "hq"),
                color_tags=src_tags)

        # Optional noise-preview writer, interleaved with main writer so
        # noise frames land at the same effective fps and can be A/B
        # wiped against the denoised output. Passthrough skips denoise
        # entirely, so an accompanying noise preview would be zeros —
        # skip creating the writer to avoid emitting an empty mp4.
        want_noise = bool(noise_path
                          and settings.get("compute_noise_preview", False)
                          and not passthrough)
        noise_writer = None
        if want_noise:
            # No source_path = no audio mux; visualization-only preview.
            noise_writer = video_io.FFmpegWriter(
                noise_path, effective_fps, w, h,
                source_path=None, color_tags=src_tags)

        try:
            _phase("processing")
            if passthrough:
                # Fallback passthrough (stream_copy path already handled
                # above): decode → immediate re-encode, no denoise, no
                # dither, no tensor round-trip. Also streams the writer.
                _stream_passthrough(reader, writer, first_frame,
                                    total_frames, reporter, frame_cap)
            else:
                _stream_denoise(reader, writer, noise_writer, first_frame,
                                settings, total_frames, reporter, fmt,
                                frame_cap)
            _phase("encoding")
        finally:
            try:
                writer.release()
            finally:
                if noise_writer is not None:
                    noise_writer.release()
    finally:
        reader.release()

    _phase("done")
    return {"denoised": output_path,
            "noise": noise_path if want_noise else None}


def _stream_passthrough(reader, writer, first_frame, total_frames,
                        reporter, frame_cap):
    """Passthrough loop — decode + re-encode with no denoise. Used when the
    source codec / output format combination rules out `ffmpeg -c copy`."""
    writer.write(first_frame)
    written = 1
    reporter.update(written, total_frames)
    while frame_cap <= 0 or written < frame_cap:
        f = reader.read()
        if f is None:
            break
        writer.write(f)
        written += 1
        reporter.update(written, total_frames)


def _stream_denoise(reader, writer, noise_writer, first_frame, settings,
                    total_frames, reporter, fmt, frame_cap):
    """Rolling-buffer streaming denoise. See run_on_file docstring for the
    memory rationale and the ±window_size overlap that makes output
    numerically identical to the old full-tensor path."""
    window = int(settings.get("window_size", 3))
    chunk = _STREAM_CHUNK_SIZE
    clamp_output = bool(settings.get("clamp_output", True))
    compute_noise = (noise_writer is not None
                     and bool(settings.get("compute_noise_preview", False)))

    # Rolling buffer of raw BGR frames straight from the reader.
    buffer = [first_frame]
    reads = 1                     # first_frame is already in `buffer`
    emitted = 0                   # frames committed to the writer so far
    carry_over = 0                # leading `carry_over` frames in `buffer`
                                  # are the previous chunk's look-behind

    def _refill(target: int) -> bool:
        """Read frames until `buffer` has `target` entries or we hit EOF /
        frame_cap. Returns True if the source may still have more; False
        once we've exhausted it."""
        nonlocal reads
        while len(buffer) < target:
            if frame_cap > 0 and reads >= frame_cap:
                return False
            f = reader.read()
            if f is None:
                return False
            buffer.append(f)
            reads += 1
        return True

    has_more = _refill(chunk + window)

    while buffer:
        # Denoise the current buffer as a single mini-batch.
        tensor = video_io.frames_bgr_to_tensor(buffer)
        clean, _ = temporal_flow_average(tensor, settings, reporter)
        clean_out = clean.clamp(0, 1) if clamp_output else clean

        # Which subset of the buffer do we commit to the writer? Skip the
        # look-behind at the front; if more source frames still exist,
        # also hold back a `window` tail as look-ahead for the next chunk.
        emit_start = carry_over
        remaining = len(buffer) - emit_start
        if has_more:
            emit_count = max(0, min(chunk, remaining - window))
        else:
            emit_count = remaining
        emit_end = emit_start + emit_count

        committed = None
        if emit_count > 0:
            committed = clean_out[emit_start:emit_end]
            if fmt == "h264":
                # Per-chunk TPDF dither — the noise is per-pixel-independent
                # so this is statistically identical to a whole-tensor pass.
                committed = video_io.tpdf_dither(committed).clamp(0, 1)
            for f_out in video_io.tensor_to_frames_bgr(committed):
                writer.write(f_out)

            if compute_noise:
                # Match the writer's committed range on both sides of the
                # noise diff so `original - clean` lines up per frame.
                orig_slice = tensor[emit_start:emit_end]
                clean_slice = clean[emit_start:emit_end]
                n_total, _, _ = extract_noise(
                    orig_slice, clean_slice, settings)
                for f_n in video_io.tensor_to_frames_bgr(n_total):
                    noise_writer.write(f_n)

            emitted += emit_count
            reporter.update(emitted, total_frames)

        # Drop everything except the tail we need as look-behind for the
        # next chunk. Free intermediates aggressively — the whole point
        # of streaming is to keep the working set small.
        del tensor, clean, clean_out
        if committed is not None:
            del committed
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if not has_more:
            break

        keep_from = max(0, emit_end - window)
        buffer = buffer[keep_from:]
        carry_over = min(window, len(buffer))

        has_more = _refill(carry_over + chunk + window)

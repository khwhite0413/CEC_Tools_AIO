"""Streaming FastDVDnet adapter for CEC's existing video I/O contract."""

from __future__ import annotations

import os
from typing import Callable, Optional

import torch

from . import video_io
from .extract_noise import extract_noise
from .fastdvdnet_model import FastDVDnet, infer_frame
from .fastdvdnet_weights import WEIGHT_PATH, weights_available
from .progress import CancelledError, ProgressReporter


def _load_model(device: torch.device) -> FastDVDnet:
    if not weights_available():
        raise RuntimeError(
            "FastDVDnet 모델 파일이 없거나 손상되었습니다. "
            "setup.bat을 다시 실행해 주세요.")
    model = FastDVDnet(5)
    try:
        state = torch.load(WEIGHT_PATH, map_location="cpu", weights_only=True)
    except TypeError:  # Older torch fallback.
        state = torch.load(WEIGHT_PATH, map_location="cpu")
    if not isinstance(state, dict):
        raise RuntimeError("FastDVDnet 체크포인트 형식이 올바르지 않습니다.")
    normalized = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state.items()
    }
    model.load_state_dict(normalized, strict=True)
    return model.eval().to(device)


def _reflect_index(index: int, length: int) -> int:
    if length <= 1:
        return 0
    period = 2 * (length - 1)
    index %= period
    return index if index < length else period - index


def _make_writer(output_path, input_path, info, fps, width, height, settings):
    fmt = settings.get("output_format", "h264")
    if (video_io.is_image_file(output_path)
            and int(getattr(info, "frame_count", 1)) <= 1):
        return video_io.SingleImageWriter(
            output_path, fps, width, height, source_path=input_path)
    if fmt == "image_seq":
        return video_io.ImageSequenceWriter(
            output_path, fps, width, height, source_path=input_path)
    tags = info.color_tags() if hasattr(info, "color_tags") else {}
    return video_io.FFmpegWriter(
        output_path, fps, width, height, source_path=input_path,
        output_format=fmt,
        prores_profile=settings.get("prores_profile", "hq"),
        color_tags=tags)


def run_on_file(input_path: str, output_path: str, settings: dict,
                progress_cb: Optional[Callable[[int, int], None]] = None,
                reporter: Optional[ProgressReporter] = None,
                frame_cap: int = -1, noise_path: Optional[str] = None,
                phase_cb: Optional[Callable[[str], None]] = None) -> dict:
    """Denoise a video with bounded RAM using a five-frame rolling window."""
    reporter = reporter or ProgressReporter(progress_cb)

    def phase(value: str) -> None:
        if phase_cb:
            phase_cb(value)

    info = video_io.probe(input_path)
    fps_override = float(settings.get("fps_override", 0.0) or 0.0)
    fps = (fps_override if fps_override > 0
           else float(getattr(info, "effective_fps", info.fps)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Validate/load the checkpoint before spawning the ffmpeg reader.
    # If setup was skipped or the file is corrupt, fail without leaving a
    # child decoder process or an open stdout pipe behind on Windows.
    model = _load_model(device)
    reader = video_io.StreamingFrameReader(input_path, fps=fps)
    sigma = max(5.0, min(55.0, float(settings.get("noise_sigma", 25.0))))
    total = int(reader.frame_count)
    if frame_cap > 0:
        total = min(total, frame_cap)

    writer = None
    noise_writer = None
    try:
        first = reader.read()
        if first is None:
            raise RuntimeError(f"No frames read from {input_path}")
        height, width = first.shape[:2]
        writer = _make_writer(
            output_path, input_path, info, fps, width, height, settings)
        if noise_path and settings.get("compute_noise_preview", False):
            tags = info.color_tags() if hasattr(info, "color_tags") else {}
            noise_writer = video_io.FFmpegWriter(
                noise_path, fps, width, height, source_path=None,
                color_tags=tags)

        frames = {0: first}
        next_index = 1
        eof = False

        def read_to(target_index: int) -> None:
            nonlocal next_index, eof
            while not eof and next_index <= target_index:
                if frame_cap > 0 and next_index >= frame_cap:
                    eof = True
                    break
                frame = reader.read()
                if frame is None:
                    eof = True
                    break
                frames[next_index] = frame
                next_index += 1

        center = 0
        phase("processing")
        with torch.inference_mode():
            while True:
                if reporter.cancelled:
                    raise CancelledError("FastDVDnet processing cancelled")
                read_to(center + 2)
                if eof and center >= next_index:
                    break

                indices = []
                for offset in (-2, -1, 0, 1, 2):
                    raw_index = center + offset
                    if raw_index < 0:
                        raw_index = -raw_index
                    if eof:
                        raw_index = _reflect_index(raw_index, next_index)
                    indices.append(raw_index)

                source_frames = [frames[index] for index in indices]
                seq_bhwc = video_io.frames_bgr_to_tensor(source_frames)
                seq = seq_bhwc.permute(0, 3, 1, 2).contiguous()
                model_input = seq.reshape(1, 15, height, width).to(device)
                sigma_map = torch.full(
                    (1, 1, height, width), sigma / 255.0,
                    dtype=model_input.dtype, device=device)
                output = infer_frame(model, model_input, sigma_map)
                output_bhwc = output.permute(0, 2, 3, 1).float().cpu()

                frame_for_write = output_bhwc
                if settings.get("output_format", "h264") == "h264":
                    frame_for_write = video_io.tpdf_dither(
                        frame_for_write).clamp(0, 1)
                writer.write(video_io.tensor_to_frames_bgr(frame_for_write)[0])

                if noise_writer is not None:
                    original = seq_bhwc[2:3]
                    noise_vis, _, _ = extract_noise(
                        original, output_bhwc, settings)
                    noise_writer.write(
                        video_io.tensor_to_frames_bgr(noise_vis)[0])

                center += 1
                reporter.update(center, max(total, center))
                reporter.update_chunk(1, 1)
                # Next center needs at minimum center-2. Everything older
                # can be released regardless of eventual tail reflection.
                minimum = max(0, center - 2)
                for old_index in [key for key in frames if key < minimum]:
                    del frames[old_index]

        phase("encoding")
    finally:
        reader.release()
        try:
            if writer is not None:
                writer.release()
        finally:
            if noise_writer is not None:
                noise_writer.release()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    phase("done")
    return {"denoised": output_path,
            "noise": noise_path if noise_writer is not None else None}

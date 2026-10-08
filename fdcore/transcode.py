"""Stand-alone ffmpeg pass for the Batch tab's Final Output group.

Used when NONE of the three processing stages (inpaint / denoise / upscale)
is enabled but the user still wants CEC to normalize codec, fps, aspect,
or resolution — the "Adobe Media Encoder subset" path. Also drives the
final polish step when the terminal processing stage isn't upscale
(inpaint or denoise last) but the Final Output group requests a resize
or aspect change those workers don't apply themselves.

Design notes
------------
- Single ffmpeg subprocess. No decode-into-Python, no tensor round-trip.
  fps re-timing uses ffmpeg's `fps=` filter which drops / duplicates
  frames — clean 30.14 → 30 for Topaz outputs, exactly what the user
  asked for.
- Aspect modes match `AspectRatioMode`:
    none — no aspect filter
    crop — center-crop to 16:9 (landscape) / 9:16 (portrait); no-op if
           source is already the target aspect or narrower
    fill — letterbox to 16:9 / 9:16; no-op if source is already the
           target aspect or wider
    auto — pick crop or fill per clip based on source dimensions
- Progress: ffmpeg's `-progress pipe:1` stream is parsed for `frame=N`
  ticks. Reported via `progress_cb(done, total)` where total is
  estimated from source duration × target fps.
- Cancel: `cancel_cb()` returning True kills the ffmpeg subprocess.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import time
from typing import Callable, Optional

import cv2
from fdcore.video_io import (VideoInfo, _get_ffmpeg, _snap_fps,
                             is_image_file, read_frames_bgr)


# ProRes profile → ffmpeg profile int, mirrors fdcore.video_io's mapping
# so the two writers pick the same encoder settings.
_PRORES_PROFILE_MAP = {
    "proxy":     ("0", "yuv422p10le"),
    "lt":        ("1", "yuv422p10le"),
    "standard":  ("2", "yuv422p10le"),
    "hq":        ("3", "yuv422p10le"),
    "4444":      ("4", "yuv444p10le"),
    "4444xq":    ("5", "yuv444p10le"),
}


def _resolve_aspect_geometry(src_w: int, src_h: int, mode: str,
                              target_long_edge: int = 0):
    """Given source w/h, mode, and target long edge, return one of:
        None                                   — no aspect change
        ("crop", w, h)                         — center-crop source to w × h
                                                 (no scaling; keeps source
                                                  resolution minus trim)
        ("cover", tw, th, sw, sh)              — scale source to (sw, sh)
                                                 via Lanczos, then center-
                                                 crop to (tw, th). Reaches
                                                 EXACT target frame with no
                                                 letterbox — matches
                                                 Premiere "Fill Frame".

    Modes:
      * crop  — trim source to target aspect at current source scale.
                Output may NOT match target_long_edge × (target_long_edge /
                target_ar); preserves pixels within the trim region.
      * fill  — scale source to cover target frame (both dims meet or
                exceed target), then crop excess. Output = EXACT target
                frame size. No bars. Sacrifices edge pixels.
      * auto  — same as fill (always reaches exact target, most predictable).

    Orientation is auto-detected from source: landscape → 16:9,
    portrait → 9:16.
    """
    if mode not in ("crop", "fill", "auto") or src_w <= 0 or src_h <= 0:
        return None
    target_ar = (16.0 / 9.0) if src_w >= src_h else (9.0 / 16.0)
    cur_ar = src_w / src_h

    if mode == "crop":
        # Classical trim: no scaling, just cut to target aspect.
        if abs(cur_ar - target_ar) < 1e-6:
            return None
        too_wide = cur_ar > target_ar
        if too_wide:
            new_w = max(2, int(round(src_h * target_ar / 2.0)) * 2)
            return ("crop", new_w, src_h)
        new_h = max(2, int(round(src_w / target_ar / 2.0)) * 2)
        return ("crop", src_w, new_h)

    # fill / auto — scale-to-cover semantics (Premiere Fill Frame).
    # Compute target frame from target_long_edge (or fall back to the
    # source's own long edge when Native is selected — in that case
    # fill still enforces aspect, just at source resolution).
    tle = target_long_edge if target_long_edge > 0 else max(src_w, src_h)
    if src_w >= src_h:                 # landscape target frame 16:9
        tw = tle
        th = max(2, int(round(tle / target_ar / 2.0)) * 2)
    else:                              # portrait target frame 9:16
        th = tle
        tw = max(2, int(round(tle * target_ar / 2.0)) * 2)
    # Already exactly at target — no work.
    if src_w == tw and src_h == th:
        return None
    # Scale so BOTH dims cover target. max() picks the bigger scale
    # factor so the shorter dim reaches target; the longer dim then
    # overshoots and gets center-cropped below.
    scale = max(tw / src_w, th / src_h)
    sw = max(2, int(round(src_w * scale / 2.0)) * 2)
    sh = max(2, int(round(src_h * scale / 2.0)) * 2)
    # Even-rounding can knock one dim below target by 1 px — clamp so
    # the crop below never tries to keep more pixels than exist.
    if sw < tw:
        sw = tw
    if sh < th:
        sh = th
    return ("cover", tw, th, sw, sh)


def _resolve_resize(src_w: int, src_h: int, target_long_edge: int):
    """Compute (w, h) after resizing so the long edge matches
    `target_long_edge`. Both dims kept even for encoder friendliness.
    Returns None when no resize is needed (Native selected or already
    at target)."""
    if target_long_edge <= 0 or src_w <= 0 or src_h <= 0:
        return None
    cur_long = max(src_w, src_h)
    if cur_long == target_long_edge:
        return None
    ratio = target_long_edge / float(cur_long)
    new_w = max(2, int(round(src_w * ratio / 2.0)) * 2)
    new_h = max(2, int(round(src_h * ratio / 2.0)) * 2)
    return (new_w, new_h)


def is_noop(source_path: str, settings: dict) -> bool:
    """Return True iff every Final Output param is at its widget default.

    Widget defaults treat as "user didn't ask for anything" — no aspect
    fix, no resize, Auto fps, H.264 codec. When ALL processing stages
    are also off, this state means "nothing to do" and BatchPage._run
    surfaces the `batch_no_changes` error instead of spawning an ffmpeg
    pass that would just re-encode the source into h264 for no reason.
    """
    del source_path  # placeholder for future match-source semantics
    aspect_mode = (settings.get("aspect_mode") or "none").lower()
    target_long = int(settings.get("target_long_edge", 0) or 0)
    fps_override = float(settings.get("fps_override", 0.0) or 0.0)
    final_codec = (settings.get("output_format") or "h264").lower()
    return (aspect_mode == "none"
            and target_long == 0
            and fps_override == 0.0
            and final_codec == "h264")


def run_transcode_only(
    input_path: str,
    output_path: str,
    settings: dict,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    cancel_cb: Optional[Callable[[], bool]] = None,
    log_cb: Optional[Callable[[str], None]] = None,
) -> None:
    """Run one ffmpeg pass that applies the Final Output group's params.

    settings keys read:
        aspect_mode         : "none" | "crop" | "fill" | "auto"
        target_long_edge    : int (0 = Native = no resize)
        fps_override        : float (0.0 = follow snapped source fps)
        output_format       : "h264" | "prores" | "image_seq"
        prores_profile      : "proxy" | "lt" | "standard" | "hq" |
                              "4444" | "4444xq" (used when output_format=prores)

    Raises RuntimeError with a Korean message on ffmpeg failure so the
    caller (TranscodeWorker) can surface it into the batch UI.
    """
    log = log_cb or (lambda _m: None)

    info = VideoInfo(input_path)
    src_w, src_h = int(info.width), int(info.height)
    if src_w <= 0 or src_h <= 0:
        raise RuntimeError(
            f"입력 영상 해상도를 읽을 수 없습니다: {input_path}")

    aspect_mode = (settings.get("aspect_mode") or "none").lower()
    target_long = int(settings.get("target_long_edge", 0) or 0)
    fps_override = float(settings.get("fps_override", 0.0) or 0.0)
    final_codec = (settings.get("output_format") or "h264").lower()
    prores_profile = settings.get("prores_profile", "4444")

    # Effective output fps: user override wins; else snapped source fps.
    if fps_override > 0.0:
        out_fps = fps_override
    else:
        out_fps = _snap_fps(info.effective_fps or info.fps or 30.0)

    # Direct image-to-image or video-to-image single frame output path
    is_img_out = is_image_file(output_path) or final_codec in ("png", "jpg", "jpeg")
    if is_img_out:
        frames = read_frames_bgr(input_path, 0, 1)
        if not frames:
            raise RuntimeError(f"프레임을 읽을 수 없습니다: {input_path}")
        img = frames[0]
        geom = _resolve_aspect_geometry(src_w, src_h, aspect_mode, target_long)
        if geom is not None and geom[0] == "cover":
            _, tw, th, sw, sh = geom
            img = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_LANCZOS4)
            cx = max(0, (sw - tw) // 2)
            cy = max(0, (sh - th) // 2)
            img = img[cy:cy + th, cx:cx + tw]
        else:
            resize = _resolve_resize(src_w, src_h, target_long)
            if resize is not None:
                img = cv2.resize(img, resize, interpolation=cv2.INTER_LANCZOS4)
            if geom is not None and geom[0] == "crop":
                cur_w, cur_h = img.shape[1], img.shape[0]
                geom2 = _resolve_aspect_geometry(cur_w, cur_h, "crop")
                if geom2 is not None:
                    _, gw, gh = geom2
                    cx = max(0, (cur_w - gw) // 2)
                    cy = max(0, (cur_h - gh) // 2)
                    img = img[cy:cy + gh, cx:cx + gw]
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        cv2.imwrite(output_path, img)
        if progress_cb:
            progress_cb(1, 1)
        return

    # -------------------- filter graph --------------------
    # Order:
    #   * fill/auto: single scale-to-cover + crop (skip separate resize
    #     because the cover scale already lands at the right size).
    #   * crop:      resize to target long edge, then trim to target aspect.
    #   * none:      resize to target long edge only.
    # fps is last so drops/dupes happen at final dims.
    filters = []
    cur_w, cur_h = src_w, src_h
    geom = _resolve_aspect_geometry(src_w, src_h, aspect_mode, target_long)
    if geom is not None and geom[0] == "cover":
        _, tw, th, sw, sh = geom
        # Scale so both dims cover the target frame.
        filters.append(f"scale={sw}:{sh}:flags=lanczos")
        # Center-crop excess to reach exact target frame — no bars.
        cx = ((sw - tw) // 2 // 2) * 2
        cy = ((sh - th) // 2 // 2) * 2
        filters.append(f"crop={tw}:{th}:{cx}:{cy}")
        cur_w, cur_h = tw, th
    else:
        # crop / none path: optional resize first.
        resize = _resolve_resize(src_w, src_h, target_long)
        if resize is not None:
            cur_w, cur_h = resize
            filters.append(f"scale={cur_w}:{cur_h}:flags=lanczos")
        # Only crop mode continues to trim after resize.
        if geom is not None and geom[0] == "crop":
            _, gw, gh = geom
            # Re-resolve against post-resize dims so the trim math is
            # correct even when a resize happened above.
            geom2 = _resolve_aspect_geometry(cur_w, cur_h, "crop")
            if geom2 is not None:
                _, gw, gh = geom2
                filters.append(f"crop={gw}:{gh}")
                cur_w, cur_h = gw, gh

    # `fps=` inserted regardless — even Auto snaps to a clean rate. When
    # source fps ~ out_fps this is a cheap identity pass.
    filters.append(f"fps={out_fps:.6f}")

    # -------------------- codec args --------------------
    codec_args = []
    if final_codec == "h264":
        codec_args = ["-c:v", "libx264", "-preset", "medium", "-crf", "14",
                      "-pix_fmt", "yuv420p"]
    elif final_codec == "prores":
        profile_idx, pix_fmt = _PRORES_PROFILE_MAP.get(
            str(prores_profile), _PRORES_PROFILE_MAP["4444"])
        codec_args = ["-c:v", "prores_ks", "-profile:v", profile_idx,
                      "-pix_fmt", pix_fmt, "-vendor", "apl0"]
    elif final_codec == "image_seq":
        # Transcode-only into PNG sequences isn't part of the current
        # scope — deliverable clips only. Reject cleanly rather than
        # producing a surprise .png dump.
        raise RuntimeError(
            "PNG Sequence 는 최종 출력 정규화 모드에서 지원하지 않습니다.")
    else:
        raise RuntimeError(f"알 수 없는 코덱: {final_codec}")

    # Audio: -c:a copy is safe when the container accepts the source
    # audio codec (mp4 ↔ mov both do AAC). Fall back to re-encoding via
    # aac if the copy fails on exotic sources. Simpler: always re-encode
    # to aac 192k — Topaz outputs vary. Trade-off: negligible re-encode
    # noise vs. bulletproof compatibility.
    audio_args = (["-c:a", "aac", "-b:a", "192k"] if info.has_audio
                  else ["-an"])

    # -------------------- assemble command --------------------
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    ffmpeg = _get_ffmpeg()
    cmd = [
        ffmpeg, "-hide_banner", "-y",
        "-i", input_path,
        "-vf", ",".join(filters),
        "-r", f"{out_fps:.6f}",
        *codec_args,
        *audio_args,
        # Progress on pipe:1 — ffmpeg writes `frame=N\nout_time_ms=…\n`
        # lines we can parse for the UI without touching stderr's
        # unstructured log noise.
        "-progress", "pipe:1",
        "-nostats",
        output_path,
    ]
    log("[transcode] " + " ".join(cmd))

    # -------------------- progress + cancel loop --------------------
    total_frames = int(round((info.duration or 0.0) * out_fps))
    if total_frames <= 0:
        # Fall back to source frame_count when duration is missing —
        # ImageSequenceInfo hits this path.
        total_frames = int(info.frame_count or 0)
    frame_re = re.compile(rb"^frame=(\d+)")

    # stderr must not be an unread PIPE: long FFmpeg runs can fill the OS pipe
    # buffer and block at the very end, leaving the batch UI stuck on
    # "포맷 정규화" even though the encoded file is already complete.
    stderr_store = tempfile.TemporaryFile()
    proc = None
    rc = None
    last_report = 0.0
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=stderr_store,
            stdin=subprocess.DEVNULL,
        )
        assert proc.stdout is not None
        for raw in proc.stdout:
            if cancel_cb is not None and cancel_cb():
                proc.kill()
                raise RuntimeError("사용자가 취소했습니다.")
            m = frame_re.match(raw)
            if m and progress_cb is not None:
                # Throttle: at most ~20 Hz. UI can't render faster than
                # that on the QueuedConnection path anyway.
                now = time.monotonic()
                if now - last_report >= 0.05:
                    done = int(m.group(1))
                    progress_cb(done, max(total_frames, done))
                    last_report = now
        rc = proc.wait()
    finally:
        # Reap the child on cancellation, read failure, or any callback
        # exception. Closing stderr while FFmpeg is still alive can race with
        # its final log write and leak the process handle on Windows.
        if proc is not None:
            try:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait(timeout=10)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
                try:
                    proc.wait(timeout=5)
                except Exception:
                    pass
        try:
            stderr_store.seek(0)
            err_bytes = stderr_store.read()
        except Exception:
            err_bytes = b""
        finally:
            stderr_store.close()

    if rc != 0:
        err_text = (err_bytes.decode("utf-8", errors="replace")
                    if err_bytes else "")
        # Trim to the last ~30 lines so the error dialog stays readable.
        tail = "\n".join(err_text.splitlines()[-30:])
        raise RuntimeError(f"ffmpeg 실패 (rc={rc}):\n{tail}")

    # Final tick so the bar hits 100% cleanly even if the last progress
    # line was throttled out.
    if progress_cb is not None and total_frames > 0:
        progress_cb(total_frames, total_frames)

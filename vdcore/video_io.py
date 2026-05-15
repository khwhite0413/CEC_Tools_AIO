"""Frame reader for the inpaint / upscale / batch workers.

Used to host a parallel `FFmpegWriter`, `auto_format_from_codec`, and
a cv2-based `read_frames_batch` — leftovers from the VDInpaint
origin. All of those have been retired in favour of fdcore's writer
(better color tag handling + bgr48le precision + diagnostic stderr
capture). The only thing left here is the ffmpeg-backed frame reader,
which the workers still need.
"""

import os
import subprocess

import cv2
import numpy as np


class VideoInfo:
    """Thin cv2 probe — width / height / fps / frame_count / duration.

    Mirrors a small subset of fdcore.video_io.VideoInfo so the reader
    below stays self-contained. Workers that need richer info (color
    tags, codec name, audio presence) import the fdcore version
    directly.
    """

    def __init__(self, path):
        self.path = path
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0
        cap.release()


def _get_ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


def read_frames_batch(path, start, count, fps=None):
    """Read [start, start+count) frames as BGR uint8 numpy arrays.

    Backed by an ffmpeg subprocess that pipes rawvideo bgr24 to stdout.
    Always applies `-vf fps={target}` so the source is read as CFR,
    matching what fdcore.FFmpegReader does for the denoise path. Same
    rationale: VFR sources (Seedance 24.04 etc.) report a clean frame
    count in their container but the per-frame stts deltas drift —
    without the fps filter, pre-input `-ss` + `-frames:v` can miss
    frames at batch boundaries. Resampling to CFR makes seek_time
    arithmetic exact and frame counts predictable.

    Pass an explicit `fps` (e.g. the snapped effective_fps or the
    user's override) so reader and writer agree on the timeline.
    Defaults to the source's native fps when `fps` is None — pure
    container-rate reads, no retiming.
    """
    if count <= 0:
        return []
    try:
        info = VideoInfo(path)
    except Exception:
        return []
    width, height = info.width, info.height
    if width <= 0 or height <= 0:
        return []
    target_fps = float(fps) if (fps and fps > 0) else float(info.fps or 0.0)
    if target_fps <= 0:
        target_fps = 30.0  # last-resort fallback so seek_time is finite

    # Pre-input -ss with -accurate_seek (default on modern ffmpeg) gives
    # us frame-accurate seek + the speed of a keyframe jump on long
    # clips. seek_time uses the SAME target fps that -vf fps= resamples
    # to, so batch_start of N in the resampled timeline lands on the
    # right physical moment.
    seek_time = start / target_fps

    ffmpeg = _get_ffmpeg()
    cmd = [
        ffmpeg, "-loglevel", "error",
        "-accurate_seek",
        "-ss", f"{seek_time:.6f}",
        "-i", path,
        "-vf", f"fps={target_fps}",
        "-frames:v", str(int(count)),
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-",
    ]

    frame_size = width * height * 3
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            bufsize=max(frame_size * 4, 4 * 1024 * 1024))
    except Exception:
        return []

    frames = []
    try:
        while len(frames) < count:
            buf = proc.stdout.read(frame_size)
            if not buf or len(buf) < frame_size:
                break
            frames.append(
                np.frombuffer(buf, dtype=np.uint8)
                  .reshape((height, width, 3))
                  .copy())
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
        try:
            proc.wait(timeout=10)
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass
    return frames

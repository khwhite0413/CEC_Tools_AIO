import cv2
import numpy as np
import subprocess
import os


class VideoInfo:
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


def extract_frame(path, index):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    return frame


def read_frames_batch(path, start, count):
    """Read [start, start+count) frames as BGR uint8 numpy arrays.

    Backed by an ffmpeg subprocess that pipes rawvideo bgr24 to stdout.
    Previously cv2.VideoCapture, which used its own YUV->RGB swscale
    defaults independent of ffmpeg's. That asymmetry caused colour drift
    on chained workflows with untagged sources: Denoise wrote via ffmpeg
    (one default), Upscale then read via cv2 (another default) and wrote
    via ffmpeg again — net mismatch baked into the pixels of the final
    file. Routing reads through ffmpeg here keeps the matrix-decision
    chain end-to-end uniform across the four tabs.

    Function signature, return shape, and dtype match the old cv2 path
    exactly so worker code (inpaint / upscale / batch orchestrator) is
    unchanged.
    """
    if count <= 0:
        return []
    try:
        info = VideoInfo(path)
    except Exception:
        return []
    width, height, fps = info.width, info.height, info.fps
    if width <= 0 or height <= 0:
        return []

    # Pre-input -ss with -accurate_seek (default on modern ffmpeg) gives
    # us frame-accurate seek + the speed of a keyframe jump on long
    # clips. Time is computed at the source's native fps so the index
    # maps to the same physical frame the old cv2 path landed on.
    seek_time = (start / fps) if fps and fps > 0 else 0.0

    ffmpeg = _get_ffmpeg()
    cmd = [
        ffmpeg, "-loglevel", "error",
        "-accurate_seek",
        "-ss", f"{seek_time:.6f}",
        "-i", path,
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


def write_video(frames, output_path, fps, codec="mp4v"):
    if not frames:
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()


def _get_ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


def get_original_bitrate(path):
    ffmpeg = _get_ffmpeg()
    ffprobe = ffmpeg.replace("ffmpeg", "ffprobe")
    try:
        cmd = [
            ffprobe, "-v", "quiet",
            "-select_streams", "v:0",
            "-show_entries", "stream=bit_rate",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout.strip().isdigit():
            return int(result.stdout.strip())
    except Exception:
        pass
    try:
        info = VideoInfo(path)
        file_size = os.path.getsize(path)
        if info.duration > 0:
            return int(file_size * 8 / info.duration)
    except Exception:
        pass
    return 8_000_000


def auto_format_from_codec(codec_name):
    """Pick an output codec that matches the source.

    Used when the user leaves "Auto" selected in the output-format
    dropdown. ProRes-tagged sources stay on ProRes (preserves quality
    across multi-step Inpaint → Denoise → Upscale chains); everything
    else gets h264 (ubiquitous, smaller files).
    """
    s = (codec_name or "").lower()
    if "prores" in s:
        return "prores"
    return "h264"


# ffmpeg profile ids for prores_ks — kept narrow; the Inpaint/Upscale
# tabs don't expose profile selection (only Denoise does). HQ + yuv422p10le
# is the sane default for "preserve quality" intermediates.
_PRORES_PROFILE_ID = {
    "proxy": "0", "lt": "1", "standard": "2",
    "hq": "3", "4444": "4", "4444xq": "5",
}
_PRORES_PROFILE_4444 = {"4444", "4444xq"}


class FFmpegWriter:
    def __init__(self, output_path, fps, width, height, source_path=None,
                 color_tags=None,
                 output_format="h264", prores_profile="hq"):
        """Pipe BGR24 frames into ffmpeg.

        output_format:
            "h264"   — libx264 + yuv420p + AAC audio (default).
            "prores" — prores_ks intra codec (yuv422p10le for 422 profiles,
                       yuv444p10le for 4444). PCM_s16le audio. Larger files
                       but no inter-frame quality loss in chained workflows.

        color_tags: optional dict {primaries, trc, matrix, range}. When all
        three of primaries/trc/matrix are present, output is tagged to
        match the source so Nuke applies the same inverse transform on
        both files. When any tag is missing (source is untagged) the
        output is also left untagged — emitting our own bt709 default in
        that case would make Nuke read source vs output differently and
        shift midtones.
        """
        self.output_path = output_path
        self.ffmpeg = _get_ffmpeg()
        bitrate = get_original_bitrate(source_path) if source_path else 8_000_000

        tags_in = color_tags or {}
        if (tags_in.get("matrix") and tags_in.get("primaries")
                and tags_in.get("trc")):
            full_tags = dict(tags_in)
        else:
            full_tags = None

        cmd = [
            self.ffmpeg, "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", str(fps),
            "-i", "-",
        ]
        if source_path:
            cmd += ["-i", source_path, "-map", "0:v", "-map", "1:a?"]

        if output_format == "prores":
            profile_id = _PRORES_PROFILE_ID.get(prores_profile,
                                                 _PRORES_PROFILE_ID["hq"])
            pix_fmt = ("yuv444p10le" if prores_profile in _PRORES_PROFILE_4444
                       else "yuv422p10le")
            base_codec = [
                "-c:v", "prores_ks",
                "-profile:v", profile_id,
                "-vendor", "apl0",
                "-pix_fmt", pix_fmt,
            ]
            audio_codec = ["-c:a", "pcm_s16le"]
        else:  # h264
            base_codec = [
                "-c:v", "libx264",
                "-b:v", str(bitrate),
                "-maxrate", str(int(bitrate * 1.5)),
                "-bufsize", str(int(bitrate * 2)),
                "-preset", "medium", "-pix_fmt", "yuv420p",
            ]
            audio_codec = ["-c:a", "aac"]

        if full_tags is not None:
            matrix = full_tags["matrix"]
            primaries = full_tags["primaries"]
            trc = full_tags["trc"]
            src_range = (full_tags.get("range") or "").lower()
            # ProRes spec mandates tv range; for h264 we follow the source.
            if output_format == "prores":
                range_scale = "tv"
                range_tag = "tv"
            else:
                range_scale = {"tv": "tv", "pc": "full",
                               "limited": "tv", "full": "full"
                               }.get(src_range, "tv")
                range_tag = "tv" if range_scale == "tv" else "pc"
            cmd += [
                "-vf",
                f"scale=out_color_matrix={matrix}:out_range={range_scale}"
                f":flags=accurate_rnd+full_chroma_int",
                *base_codec,
                "-color_range", range_tag,
                "-colorspace", matrix,
                "-color_primaries", primaries,
                "-color_trc", trc,
                "-movflags", "+write_colr",
                *audio_codec, "-shortest",
                output_path,
            ]
        else:
            cmd += [
                *base_codec,
                *audio_codec, "-shortest",
                output_path,
            ]

        # stderr captured (rather than DEVNULL) so BrokenPipeError on
        # write() can surface the actual ffmpeg diagnostic — without
        # this, every encoder-side failure (bad output path, audio mux
        # error, codec issue, frame-size mismatch) just looked like
        # "[Errno 32] Broken pipe" with no context.
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        self._cmd = cmd

    def _drain_stderr(self) -> str:
        """Read whatever ffmpeg has written to stderr by now. Best-effort
        and non-blocking-ish: we try wait() with a short timeout so the
        pipe gets fully drained, then read."""
        try:
            self.proc.wait(timeout=2)
        except Exception:
            try:
                self.proc.terminate()
            except Exception:
                pass
            try:
                self.proc.wait(timeout=2)
            except Exception:
                pass
        try:
            err_bytes = self.proc.stderr.read() if self.proc.stderr else b""
        except Exception:
            err_bytes = b""
        try:
            return err_bytes.decode("utf-8", errors="replace").strip()
        except Exception:
            return ""

    def write(self, frame):
        try:
            self.proc.stdin.write(frame.tobytes())
        except BrokenPipeError:
            # ffmpeg already exited. Reword the error with whatever it
            # printed on stderr so the user sees a real diagnostic
            # instead of a bare "[Errno 32] Broken pipe".
            err = self._drain_stderr()
            tail = err[-1200:] if err else "(no stderr captured)"
            raise RuntimeError(
                "ffmpeg writer exited before all frames were sent. "
                "Last stderr output:\n" + tail
            ) from None

    def release(self):
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        try:
            rc = self.proc.wait(timeout=120)
        except Exception:
            rc = None
        if rc not in (None, 0):
            err = ""
            try:
                err_bytes = self.proc.stderr.read() if self.proc.stderr else b""
                err = err_bytes.decode("utf-8", errors="replace").strip()
            except Exception:
                pass
            tail = err[-1200:] if err else "(no stderr captured)"
            raise RuntimeError(
                f"ffmpeg writer exited with code {rc}. Last stderr "
                f"output:\n{tail}"
            )


def copy_audio(source_path, video_no_audio_path, output_path):
    """Copy audio from source video to output video using ffmpeg."""
    try:
        ffmpeg = _get_ffmpeg()
        cmd = [
            ffmpeg, "-y",
            "-i", video_no_audio_path,
            "-i", source_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-shortest",
            output_path
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300
        )
        if result.returncode == 0:
            if os.path.exists(video_no_audio_path) and video_no_audio_path != output_path:
                os.remove(video_no_audio_path)
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return False

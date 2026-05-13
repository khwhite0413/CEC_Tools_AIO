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
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for _ in range(count):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
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

        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

    def write(self, frame):
        self.proc.stdin.write(frame.tobytes())

    def release(self):
        self.proc.stdin.close()
        self.proc.wait(timeout=120)


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

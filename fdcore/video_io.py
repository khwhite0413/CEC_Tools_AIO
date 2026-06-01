"""Video read / write utilities. ffmpeg pipe + libx264, audio preservation.

Also handles image sequences (folder of numbered PNGs + meta.json) as a
first-class output format alongside MP4 / MOV.

Color management strategy (match-source, mirrors Premiere "match source"):
  - FFmpegReader decodes input via ffmpeg subprocess (matches Nuke's own
    ffmpeg) so the bgr24 we hand to the encoder represents the same RGB
    Nuke would derive on its read of the original.
  - FFmpegWriter writes the standard nclc / VUI color atoms with the same
    primaries / transfer / matrix / range as the source. Without them
    ffprobe shows "yuv420p(progressive)" with no range marker and Nuke
    falls back to its default colorspace, shifting tones vs the original.
  - Nuke-specific atoms (uk.co.thefoundry.* on Nuke-rendered files) are
    not reproducible from outside Nuke; the standard nclc atoms alone are
    sufficient for visual match — the user verified this against
    Premiere's match-source output.
"""

import os
import json
import re
import subprocess
import shutil
import numpy as np
import cv2


_DEFAULT_SEQ_PATTERN = "frame_%06d.png"


def is_image_sequence(path: str) -> bool:
    return bool(path) and os.path.isdir(path) and os.path.exists(
        os.path.join(path, "meta.json"))


class ImageSequenceInfo:
    """Mirrors VideoInfo's interface so callers can be format-agnostic."""

    def __init__(self, path: str):
        with open(os.path.join(path, "meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.path = path
        self.fps = float(meta.get("fps", 30.0))
        self.frame_count = int(meta.get("frames", 0))
        self.width = int(meta.get("width", 0))
        self.height = int(meta.get("height", 0))
        self.duration = (self.frame_count / self.fps) if self.fps > 0 else 0.0
        self.has_audio = False
        self.pattern = meta.get("pattern", _DEFAULT_SEQ_PATTERN)


def probe(path: str):
    """Polymorphic dispatch — image sequence dir or video file."""
    if is_image_sequence(path):
        return ImageSequenceInfo(path)
    return VideoInfo(path)


def _get_ffmpeg() -> str:
    """Find ffmpeg.exe — bundled folder, then imageio_ffmpeg, then PATH."""
    here = os.path.dirname(os.path.abspath(__file__))
    bundled = os.path.abspath(os.path.join(here, "..", "ffmpeg", "ffmpeg.exe"))
    if os.path.isfile(bundled):
        return bundled
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    found = shutil.which("ffmpeg")
    return found or "ffmpeg"


def _get_ffprobe() -> str:
    ffmpeg = _get_ffmpeg()
    candidate = ffmpeg.replace("ffmpeg.exe", "ffprobe.exe").replace("ffmpeg", "ffprobe")
    if os.path.isfile(candidate):
        return candidate
    found = shutil.which("ffprobe")
    return found or "ffprobe"


# Standard delivery frame rates that real-world content actually targets.
# AI-generated VFR sources (Seedance, Dreamina, etc.) report fractional
# averages like 24.04 in their metadata but the intended target is one of
# these. NTSC fractional rates (23.976, 29.97, 59.94) are included so the
# snap logic doesn't accidentally promote them up to the nearest integer.
_STANDARD_FPS = (23.976, 24.0, 25.0, 29.97, 30.0, 50.0, 59.94, 60.0)
_FPS_SNAP_TOLERANCE = 0.1  # snap only when within this many fps of a standard


def _snap_fps(fps):
    """Round a noisy detected fps to the nearest standard rate within
    `_FPS_SNAP_TOLERANCE`; otherwise return the input unchanged.

    Examples:
        24.04 → 24.0       (Seedance VFR — closest is 24.0 at 0.04)
        23.98 → 23.976     (closest is the NTSC fraction at 0.004, NOT 24.0)
        29.97 → 29.97      (already standard)
        27.534 → 27.534    (no standard within tolerance, leave alone)
        None / 0 → unchanged
    """
    if not fps or fps <= 0:
        return fps
    nearest = min(_STANDARD_FPS, key=lambda s: abs(s - fps))
    if abs(nearest - fps) <= _FPS_SNAP_TOLERANCE:
        return nearest
    return fps


def _probe_video_fields(path: str) -> dict:
    """ffprobe v:0 stream metadata: codec + color tags. Codec name is what
    decides whether stream-copy passthrough is viable. Color tags are
    forwarded to the writer so the ProRes / h264 output preserves the
    source's transfer / matrix / primaries.

    Falls back to parsing `ffmpeg -i` stderr when ffprobe is unavailable
    (imageio_ffmpeg ships ffmpeg only; ffprobe must come from system
    PATH). The fallback extracts codec_name reliably; color tags from
    stderr are best-effort.
    """
    fields = ("codec_name", "color_primaries", "color_trc",
              "color_space", "color_range",
              # avg_frame_rate is frame_count / duration — accurate for
              # both CFR and VFR. r_frame_rate is the stts-pattern
              # nominal rate which goes haywire for VFR (returns 60 for a
              # Seedance 24-fps-avg source because stts durations are all
              # multiples of a 60fps tick). Always use avg_frame_rate.
              "avg_frame_rate")
    try:
        cmd = [_get_ffprobe(), "-v", "quiet",
               "-select_streams", "v:0",
               "-show_entries", "stream=" + ",".join(fields),
               "-of", "default=noprint_wrappers=1",
               path]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            out = {}
            for line in r.stdout.splitlines():
                if "=" in line:
                    k, v = line.split("=", 1)
                    v = v.strip()
                    if v and v not in ("unknown", "N/A", "0/0"):
                        out[k.strip()] = v
            # ffprobe returns rationals like "24/1" or "30000/1001" —
            # promote to a single 'frame_rate' float key.
            fps = _parse_rational(out.pop("avg_frame_rate", None))
            if fps:
                out["frame_rate"] = fps
            if out:
                return out
    except Exception:
        pass
    # ffprobe unavailable / failed → combine the two text-free fallbacks:
    # ffmpeg stderr gives codec name + range marker + tbr/fps, mp4 atom
    # parser gives primaries / transfer / matrix from the colr box.
    out = _probe_via_ffmpeg(path)
    atom = _parse_mp4_colr_atom(path)
    # atom values only fill missing keys — never override what ffmpeg
    # already gave us (e.g. range from `yuv420p(tv, ...)`).
    for k, v in atom.items():
        out.setdefault(k, v)
    return out


def _parse_rational(s):
    """Parse 'num/den' or a plain decimal string into a float. None on
    miss. Used to normalize ffprobe's fractional frame_rate output."""
    if not s:
        return None
    try:
        if "/" in s:
            num, den = s.split("/", 1)
            n, d = float(num), float(den)
            return n / d if d > 0 else None
        return float(s)
    except (ValueError, ZeroDivisionError):
        return None


# Stream-line patterns inside `ffmpeg -i` stderr.
#  e.g. "Stream #0:0[0x1](und): Video: h264 (High) (avc1 / 0x31637661), yuv420p(tv, progressive), ..."
_FFMPEG_STREAM_RE = re.compile(
    r"Stream\s+#\d+:\d+.*?Video:\s+(?P<codec>\w+).*?,\s*(?P<pix>\w+)"
    r"(?:\((?P<pix_meta>[^)]*)\))?")

# Frame-rate hint on the stream line — e.g. "..., 24 fps, ...".
# We use the `fps` token (avg_frame_rate equivalent), NOT `tbr`: tbr is
# the stts-pattern nominal rate and goes wrong on VFR sources (a 24-fps
# avg Seedance mp4 with alternating 33/50ms deltas reports `60 tbr`
# because both durations are multiples of a 60fps tick).
_FFMPEG_FPS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*fps\b")


def _probe_via_ffmpeg(path: str) -> dict:
    """Extract codec / range / frame_rate from `ffmpeg -i` stderr.
    Best-effort; missing fields stay missing so callers fall back to
    bt709 defaults / cv2 fps like before."""
    try:
        cmd = [_get_ffmpeg(), "-hide_banner", "-i", path]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    except Exception:
        return {}
    out = {}
    for line in (r.stderr or "").splitlines():
        m = _FFMPEG_STREAM_RE.search(line)
        if not m:
            continue
        out["codec_name"] = m.group("codec").lower()
        meta = (m.group("pix_meta") or "").lower()
        # `yuv420p(tv, progressive)` → range = tv
        if "tv" in meta or "limited" in meta:
            out["color_range"] = "tv"
        elif "pc" in meta or "full" in meta:
            out["color_range"] = "pc"
        m_fps = _FFMPEG_FPS_RE.search(line)
        if m_fps:
            out["frame_rate"] = float(m_fps.group(1))
        break
    return out


# ISO/IEC 23001-8 (CICP) code points ↔ ffmpeg / ffprobe canonical names.
# Used to translate the integer codes inside an mp4 colr (nclc/nclx) box
# into the strings FFmpegWriter passes to -colorspace / -color_primaries
# / -color_trc, and back the other direction when patching the atom
# after ffmpeg writes it (some ffmpeg builds drop primaries/transfer
# down to "unspecified" inside the colr box even when explicitly given).
_ISO_PRIMARIES = {
    1: "bt709", 4: "bt470m", 5: "bt470bg", 6: "smpte170m",
    7: "smpte240m", 9: "bt2020",
}
_ISO_TRC = {
    1: "bt709", 4: "bt470m", 5: "bt470bg", 6: "smpte170m",
    7: "smpte240m", 8: "linear", 13: "iec61966-2-1",
    14: "bt2020-10", 15: "bt2020-12", 16: "smpte2084", 18: "arib-std-b67",
}
_ISO_MATRIX = {
    0: "rgb", 1: "bt709", 5: "bt470bg", 6: "smpte170m",
    7: "smpte240m", 9: "bt2020nc", 10: "bt2020c",
}
_NAME_TO_PRIMARIES = {v: k for k, v in _ISO_PRIMARIES.items()}
_NAME_TO_TRC = {v: k for k, v in _ISO_TRC.items()}
_NAME_TO_MATRIX = {v: k for k, v in _ISO_MATRIX.items()}


def _parse_mp4_colr_atom(path: str) -> dict:
    """Pull primaries / transfer / matrix straight out of the mp4 colr box.

    Why we need this: imageio_ffmpeg (the bundled binary) doesn't ship
    ffprobe. When ffprobe isn't on PATH our `_probe_video_fields` falls
    back to ffmpeg -i stderr parsing, which only surfaces codec name +
    range and silently drops the matrix / primaries / transfer tags. The
    writer then forwards a default 'bt709' matrix even when the source
    was tagged Rec601 — the encoder commits the BGR→YUV with the wrong
    matrix, while the output's nclc tag keeps saying 'bt709', so the
    output's actual encoding doesn't match its tag (or, equivalently,
    doesn't match the source's encoding even if both share a tag).
    Symptom in Nuke: switching the YCbCr matrix on either side toggles
    them into agreement.

    mp4 boxes are TLV: 4 bytes big-endian size, 4 bytes ASCII type,
    payload. The colr box is found inside moov/trak/mdia/minf/stbl/
    stsd/<sample-entry>. Box hierarchy doesn't matter for our purposes
    — a flat byte search for the four-character 'colr' tag is enough,
    because non-color boxes never embed that exact bytestring near a
    valid nclc/nclx flavor marker.

    Returns a dict in `_probe_video_fields`'s shape: keys
    'color_primaries' / 'color_trc' / 'color_space' / 'color_range'
    (range only when the box flavor is nclx). Empty dict on miss.
    """
    try:
        size = os.path.getsize(path)
        # First 10MB is more than enough — moov is at the front of a
        # standard (faststart) mp4 and nclc lives shallow inside it.
        with open(path, "rb") as f:
            data = f.read(min(size, 10_000_000))
    except Exception:
        return {}

    idx = data.find(b"colr")
    if idx < 4 or len(data) < idx + 14:
        return {}
    flavor = data[idx + 4:idx + 8]
    if flavor not in (b"nclc", b"nclx"):
        return {}

    primaries_code = int.from_bytes(data[idx + 8:idx + 10], "big")
    transfer_code = int.from_bytes(data[idx + 10:idx + 12], "big")
    matrix_code = int.from_bytes(data[idx + 12:idx + 14], "big")

    out = {}
    if primaries_code in _ISO_PRIMARIES:
        out["color_primaries"] = _ISO_PRIMARIES[primaries_code]
    if transfer_code in _ISO_TRC:
        out["color_trc"] = _ISO_TRC[transfer_code]
    if matrix_code in _ISO_MATRIX:
        out["color_space"] = _ISO_MATRIX[matrix_code]
    if flavor == b"nclx" and len(data) >= idx + 15:
        # nclx adds a full_range_flag byte (top bit set = pc/full range).
        out["color_range"] = "pc" if (data[idx + 14] & 0x80) else "tv"
    return out


def _patch_mp4_colr_atom(path: str, color_tags: dict) -> bool:
    """Rewrite the nclc/nclx box's primaries / transfer / matrix codes.

    ffmpeg's mov muxer (at least the 7.x line shipped in imageio_ffmpeg)
    only forwards `-colorspace` into the colr box; `-color_primaries`
    and `-color_trc` are dropped to ISO 'unspecified' (2) inside the
    box even when explicitly provided on the command line, even with
    `-movflags +write_colr`. The H264 elementary stream's VUI carries
    them correctly, but Nuke's mov64 reader prefers the container box
    and falls back to its colorspace heuristic when it sees
    'unspecified' there — exactly the symptom the user observed
    (Nuke showing both source and our output as Rec601, with colors
    only matching when one side's matrix was manually swapped).

    We patch the codes directly: locate the colr box near the end of
    the file (ffmpeg's default mov muxer puts moov after mdat, so
    scanning from the tail is fast) and overwrite the three 16-bit BE
    integers with codes derived from `color_tags`.
    """
    if not color_tags:
        return False
    p = _NAME_TO_PRIMARIES.get(color_tags.get("primaries"))
    t = _NAME_TO_TRC.get(color_tags.get("trc"))
    m = _NAME_TO_MATRIX.get(color_tags.get("matrix"))
    if not (p and t and m):
        return False
    try:
        with open(path, "r+b") as f:
            f.seek(0)
            data = f.read()
            idx = data.rfind(b"colr")
            if idx < 0 or idx + 14 > len(data):
                return False
            flavor = data[idx + 4:idx + 8]
            if flavor not in (b"nclc", b"nclx"):
                return False
            f.seek(idx + 8)
            f.write(p.to_bytes(2, "big"))
            f.write(t.to_bytes(2, "big"))
            f.write(m.to_bytes(2, "big"))
        return True
    except Exception:
        return False


class VideoInfo:
    def __init__(self, path: str):
        self.path = path
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        cv2_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        probe = _probe_video_fields(path)
        # Raw detected fps (kept as-is for the source-info display line).
        # Priority: ffprobe avg_frame_rate → ffmpeg-stderr fps token →
        # cv2.CAP_PROP_FPS. We deliberately avoid r_frame_rate / tbr —
        # those report the stts-pattern nominal rate which is 60 for a
        # Seedance 24-avg VFR source.
        self.fps = float(probe.get("frame_rate") or cv2_fps)
        # Standard-rate-snapped value the pipeline actually uses for
        # encoding / -vf fps=. A Seedance mp4 reports 24.04 here ←
        # _snap_fps rounds that to a clean 24.0 (within ±0.1 of the
        # 24fps standard). NTSC 23.976 is in the standard set so a
        # 23.98 reading snaps to 23.976 rather than 24.0.
        self.effective_fps = _snap_fps(self.fps)
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0

        self.codec_name = probe.get("codec_name")  # h264 | hevc | prores | ...
        self.color_primaries = probe.get("color_primaries")
        self.color_trc = probe.get("color_trc")
        self.color_space = probe.get("color_space")  # YUV->RGB matrix
        self.color_range = probe.get("color_range")  # tv | pc

        self.has_audio = self._probe_audio()

    def _probe_audio(self) -> bool:
        # Stage 1 (ffprobe) is the cleanest check; stage 2 (ffmpeg -i
        # stderr) is the fallback for environments where ffprobe is
        # missing. imageio_ffmpeg ships only ffmpeg, so on machines
        # without a system ffprobe the fallback used to be the only
        # thing keeping the audio label honest — and it was silently
        # losing in two real cases:
        #
        #   - non-ASCII metadata / path under a non-UTF-8 default
        #     locale (Windows CP949 was the trigger on another user's
        #     PC) → subprocess.run(..., text=True) raised
        #     UnicodeDecodeError before any "Audio:" line could be
        #     scanned. Force encoding="utf-8" with errors="replace" so
        #     decode never blows up.
        #   - slow / network / external drives where ffmpeg's stderr
        #     arrived later than the previous 10 s timeout. Bumped to
        #     30 s to give it real headroom without hanging the UI on
        #     genuinely broken files.
        #
        # Also: stdin=DEVNULL + ffmpeg's own -nostdin so ffmpeg never
        # tries to read from a closed pipe, which can deadlock on some
        # builds. setup.bat now bundles ffprobe.exe next to ffmpeg.exe
        # so stage 1 should be the path of least resistance on a fresh
        # install — this fallback hardening is the belt-and-braces.
        common = dict(capture_output=True, text=True,
                      encoding="utf-8", errors="replace",
                      timeout=30, stdin=subprocess.DEVNULL)
        try:
            cmd = [_get_ffprobe(), "-v", "quiet",
                   "-select_streams", "a:0",
                   "-show_entries", "stream=codec_type",
                   "-of", "default=noprint_wrappers=1:nokey=1",
                   self.path]
            r = subprocess.run(cmd, **common)
            if r.returncode == 0 and r.stdout.strip():
                return "audio" in r.stdout.lower()
        except Exception:
            pass
        try:
            cmd = [_get_ffmpeg(), "-hide_banner", "-nostdin", "-i", self.path]
            r = subprocess.run(cmd, **common)
            for line in (r.stderr or "").splitlines():
                if "Stream " in line and "Audio:" in line:
                    return True
        except Exception:
            pass
        return False

    def color_tags(self) -> dict:
        """Non-empty source color tags. Forwarded to FFmpegWriter so the
        ProRes path keeps the source's transfer/primaries/matrix — what
        Nuke uses to pick the inverse on read. Missing values stay missing
        so the writer falls back to bt709 defaults rather than guessing.
        """
        return {k: v for k, v in {
            "primaries": self.color_primaries,
            "trc": self.color_trc,
            "matrix": self.color_space,
            "range": self.color_range,
        }.items() if v}


# Codecs that we consider safe to remux into an mp4 container via -c copy.
# Restricted to what FFmpegWriter's h264 path produces or accepts as input;
# any other codec falls back to the bgr24 transcode path.
_STREAM_COPY_CODECS = frozenset({"h264", "avc1"})


def auto_format_from_codec(codec_name):
    """Pick an output codec that matches the source.

    Used when the user leaves "Auto" selected in the output-format
    dropdown. ProRes-tagged sources stay on ProRes (preserves quality
    across multi-step Inpaint -> Denoise -> Upscale chains); everything
    else gets h264 (ubiquitous, smaller files).

    Lives in fdcore so all three tabs (inpaint / denoise / upscale)
    can call into the same helper now that the formerly-separate
    vdcore.FFmpegWriter has been retired.
    """
    s = (codec_name or "").lower()
    if "prores" in s:
        return "prores"
    return "h264"


def bgr24_to_bgr48le(frame_bgr24):
    """Expand an 8-bit BGR frame to the 16-bit bgr48le layout that
    FFmpegWriter feeds ffmpeg.

    Maps each 8-bit value X to ((X << 8) | X), filling the 16-bit
    range [0, 65535] without skewing — equivalently `X * 257`. Lossless
    on the way in, so feeding upscale / inpaint outputs (uint8 BGR) into
    the denoise tab's writer doesn't introduce any rounding bias.
    """
    if frame_bgr24.dtype == np.uint16:
        return frame_bgr24
    u16 = frame_bgr24.astype(np.uint16)
    return (u16 << 8) | u16


def can_stream_copy(input_path: str, output_format: str) -> bool:
    """Return True iff we can passthrough the source's video stream into
    output_format with `ffmpeg -c copy` (zero decode/encode = bit-exact).

    Conditions:
      - Source is a regular video file (not a PNG sequence).
      - Output format is h264 mp4 (we only target the .mp4 container).
      - Source's video codec is itself h264 — otherwise the elementary
        stream wouldn't be a valid h264 bitstream inside the mp4.
    """
    if is_image_sequence(input_path):
        return False
    if output_format != "h264":
        return False
    try:
        info = VideoInfo(input_path)
    except Exception:
        return False
    return (info.codec_name or "").lower() in _STREAM_COPY_CODECS


def copy_stream(input_path: str, output_path: str, frame_cap: int = -1) -> None:
    """Bit-exact remux of input's video stream into output_path.

    Decodes nothing on the video side: `-c:v copy` lifts the encoded
    bitstream straight into the new container, so the output's pixel
    bytes match the input exactly (no bgr24 round-trip, no libx264
    re-encode, no chroma resample). Audio is also stream-copied (no
    re-encode) — the AAC track in a typical h264 mp4 is already
    container-compatible, and copying preserves it bit-exact instead
    of re-running it through the AAC encoder.

    frame_cap > 0 caps the output by *time* (frame_cap / fps), not by
    frame count: with `-c copy` the cut lands on the nearest preceding
    keyframe, so a strict frame count would leave dangling B-frames.
    A time-based limit yields a clean GOP boundary playable everywhere.
    """
    ffmpeg = _get_ffmpeg()
    cmd = [ffmpeg, "-y", "-i", input_path]
    if frame_cap > 0:
        info = VideoInfo(input_path)
        fps = float(info.fps) or 30.0
        # Slight overshoot (+1 frame) so the very last frame the caller
        # asked for is included even after keyframe-aligned trimming.
        duration = (frame_cap + 1) / fps
        cmd += ["-t", f"{duration:.6f}"]
    cmd += [
        "-map", "0:v", "-map", "0:a?",
        "-c:v", "copy",
        "-c:a", "copy",
        "-avoid_negative_ts", "make_zero",
        "-shortest",
        output_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg -c copy failed (rc={proc.returncode}): "
            f"{proc.stderr.strip()[:500]}")


def get_original_bitrate(path: str) -> int:
    try:
        cmd = [_get_ffprobe(), "-v", "quiet",
               "-select_streams", "v:0",
               "-show_entries", "stream=bit_rate",
               "-of", "default=noprint_wrappers=1:nokey=1",
               path]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if r.returncode == 0 and r.stdout.strip().isdigit():
            return int(r.stdout.strip())
    except Exception:
        pass
    try:
        info = VideoInfo(path)
        size = os.path.getsize(path)
        if info.duration > 0:
            return int(size * 8 / info.duration)
    except Exception:
        pass
    return 8_000_000


def extract_frame(path: str, index: int):
    """Read a single frame at index. Returns HxWx3 uint8 BGR or None."""
    cap = cv2.VideoCapture(path)
    if index > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


class FFmpegReader:
    """Decode a video file to raw BGR frames via ffmpeg subprocess.

    Pixel format is bgr48le — 16-bit per channel, little-endian. The
    8-bit bgr24 path was losing precision on the YUV→RGB matrix step:
    each channel got rounded independently with bt709 coefficients
    (0.2126 / 0.7152 / 0.0722), and that asymmetric ±1 LSB error baked
    into the bgr buffer survived all the way to the encode side, drifting
    saturation by a fraction of a code per pixel. 16-bit pulls that error
    8 bits below the LSB of the final 8-bit output — invisible.

    No -vf scale: ffmpeg auto-detects matrix/range from the source
    stream tags (tv→full expansion, bt709 vs bt601 matrix, etc.), same
    way Nuke's internal decoder does. -vf fps={self.fps} pins to cv2's
    nominal frame count so VFR sources don't desync against the
    cv2-based in-app preview player."""

    def __init__(self, path: str, fps: float = None):
        """fps overrides the auto-detected VideoInfo.fps for the -vf fps=
        resample target. Pass an explicit value when the user has supplied
        an FPS override in settings; leave as None for auto."""
        self.path = path
        info = VideoInfo(path)
        self.width = int(info.width)
        self.height = int(info.height)
        self.fps = float(fps) if fps and fps > 0 else float(info.fps)
        self.frame_count = int(info.frame_count)
        self.duration = float(info.duration)
        self.has_audio = bool(info.has_audio)
        # bgr48le = 6 bytes / pixel (3 channels * 16-bit each)
        self.frame_size = self.width * self.height * 6

        ffmpeg = _get_ffmpeg()
        cmd = [ffmpeg,
               "-loglevel", "error",
               "-i", path,
               "-f", "rawvideo",
               "-pix_fmt", "bgr48le",
               "-vf", f"fps={self.fps}",
               "-"]
        self.proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            bufsize=max(self.frame_size * 4, 4 * 1024 * 1024))

    def read(self):
        buf = self.proc.stdout.read(self.frame_size)
        if not buf or len(buf) < self.frame_size:
            return None
        # bgr48le is little-endian 16-bit; numpy uint16 on x86 matches.
        return np.frombuffer(buf, dtype=np.uint16).reshape(
            (self.height, self.width, 3)).copy()

    def release(self):
        try:
            self.proc.stdout.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=5)
        except Exception:
            try:
                self.proc.terminate()
            except Exception:
                pass


def read_frames_bgr(path: str, start: int = 0, count: int = -1,
                    fps: float = None):
    """Read raw BGR frames as a list of HxWx3 uint8/uint16 arrays.

    Image sequences go through cv2 (PNG decode is straightforward); video
    files go through FFmpegReader so our decode matches Nuke's ffmpeg.

    fps overrides the FFmpegReader's resample target (its -vf fps=X).
    Used by the pipeline when the user supplies an fps_override setting.
    Ignored for image sequences.
    """
    if is_image_sequence(path):
        info = ImageSequenceInfo(path)
        cap_path = os.path.join(path, info.pattern)
        cap = cv2.VideoCapture(cap_path, cv2.CAP_IMAGES)
        if start > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames = []
        n = 0
        while True:
            if count >= 0 and n >= count:
                break
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            n += 1
        cap.release()
        return frames

    reader = FFmpegReader(path, fps=fps)
    try:
        for _ in range(max(0, start)):
            if reader.read() is None:
                return []
        frames = []
        while count < 0 or len(frames) < count:
            f = reader.read()
            if f is None:
                break
            frames.append(f)
        return frames
    finally:
        reader.release()


def frames_bgr_to_tensor(frames):
    """List of HxWx3 BGR (uint8 OR uint16) -> BHWC float32 RGB [0,1].

    Auto-detects bit depth: FFmpegReader produces uint16 (bgr48le pipe),
    cv2.VideoCapture image-sequence path produces uint8. Both normalize
    to the same float32 [0,1] domain so downstream tensor code is
    bit-depth-agnostic.
    """
    import torch
    arr = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames], axis=0)
    scale = 65535.0 if arr.dtype == np.uint16 else 255.0
    return torch.from_numpy(arr).float() / scale


def tensor_to_frames_bgr(tensor):
    """BHWC float [0,1] RGB -> list of HxWx3 uint16 BGR.

    Returns 16-bit BGR so the encoder's RGB→YUV matrix step has 8 bits
    of headroom over the 8-bit (libx264 yuv420p) or 10-bit (prores
    yuv*p10le) output, eliminating the asymmetric per-channel rounding
    bias that the bgr24 path produced. cv2.imwrite handles uint16 BGR
    natively (16-bit PNG); FFmpegWriter uses bgr48le for the rawvideo
    pipe.

    Apply tpdf_dither(tensor) before calling this on the 8-bit h264
    path — the high-bit-depth pipe still benefits from triangular
    dither at the 8-bit-LSB scale to break dark-area banding when
    ffmpeg's swscale quantizes 16→8.
    """
    arr = np.rint(tensor.clamp(0, 1).cpu().numpy() * 65535.0).astype(np.uint16)
    return [cv2.cvtColor(arr[i], cv2.COLOR_RGB2BGR) for i in range(arr.shape[0])]


def tpdf_dither(tensor):
    """Triangular-PDF dither at 2 LSB peak-to-peak.

    The difference of two uniforms is triangular on [-1, +1] LSB. Adding it
    before 8-bit quantization decorrelates noise from signal so dark
    gradients don't contour into visible bands. Mean is preserved (zero
    bias) so a passthrough re-encode looks identical on average.

    Caller is responsible for the float -> uint8 step (tensor_to_frames_bgr).
    """
    import torch
    u1 = torch.rand_like(tensor)
    u2 = torch.rand_like(tensor)
    return tensor + (u1 - u2) / 255.0


_PRORES_PROFILE_MAP = {
    "proxy": "0",
    "lt": "1",
    "standard": "2",
    "hq": "3",
    "4444": "4",
    "4444xq": "5",
}
_PRORES_4444 = {"4444", "4444xq"}


class FFmpegWriter:
    """Pipe BGR frames into ffmpeg. Supports H.264 MP4 and ProRes MOV.

    Both paths preserve the source's color tags (primaries / transfer /
    matrix / range) on output so Nuke and other readers apply the same
    inverse transform on our output as on the original — see module
    docstring for why this matters.

    output_format:
        "h264"   — libx264 + yuv420p, original-bitrate target, AAC audio.
                   nclc/VUI tags inherited from source via color_tags.
        "prores" — prores_ks intra codec, 10-bit pixfmt. Profile selected
                   via `prores_profile` (proxy / lt / standard / hq /
                   4444 / 4444xq). Range stays tv per ProRes spec; matrix
                   / primaries / trc inherit from source.
    """

    def __init__(self, output_path: str, fps: float, width: int, height: int,
                 source_path: str = None,
                 output_format: str = "h264",
                 prores_profile: str = "hq",
                 color_tags: dict = None):
        self.output_path = output_path
        # Treat the source as "tagged" whenever it has a matrix entry —
        # AI generators (LTX, Runway, Sora) routinely emit only matrix +
        # primaries and drop the transfer characteristic, which under
        # the old all-of-three check fell into the untagged branch and
        # produced an output with no nclc atom. Players then applied
        # their BT.601 heuristic to the output while still reading the
        # source's matrix tag → visible color shift between source and
        # render. Fill in any missing fields with values derived from
        # the matrix (BT.709 spec ties matrix/primaries/trc together,
        # so this matches what MediaInfo / Nuke infer anyway). Truly
        # untagged sources (Seedance, raw screen grabs) have no matrix
        # entry and still fall into the untagged branch, preserving
        # the existing match-source-untagged behaviour.
        tags_in = color_tags or {}
        matrix = tags_in.get("matrix")
        if matrix:
            self._color_tags = {
                "matrix":    matrix,
                "primaries": tags_in.get("primaries") or matrix,
                "trc":       tags_in.get("trc") or matrix,
                "range":     tags_in.get("range") or "tv",
            }
        else:
            self._color_tags = None
        ffmpeg = _get_ffmpeg()

        # bgr48le rawvideo input — 16-bit per channel BGR. Gives ffmpeg's
        # RGB→YUV conversion 8 bits of headroom over the 8-bit (yuv420p)
        # or 10-bit (yuv*p10le) output, eliminating the per-channel
        # asymmetric ±1 LSB drift that a bgr24 pipe produced.
        # -loglevel error keeps stderr tiny (only actual problems) so
        # we can capture it via PIPE without risking the buffer-full
        # deadlock that bit the old vdcore writer. Captured stderr is
        # surfaced from write() / release() on failure so future
        # BrokenPipe errors come with a real diagnostic instead of
        # silently dropping ffmpeg's complaint.
        cmd = [ffmpeg, "-y", "-loglevel", "error",
               "-f", "rawvideo", "-vcodec", "rawvideo",
               "-pix_fmt", "bgr48le",
               "-s", f"{width}x{height}", "-r", str(fps),
               "-i", "-"]
        if source_path:
            cmd += ["-i", source_path, "-map", "0:v", "-map", "1:a?"]

        if output_format == "prores":
            profile_id = _PRORES_PROFILE_MAP.get(prores_profile,
                                                 _PRORES_PROFILE_MAP["hq"])
            pix_fmt = "yuv444p10le" if prores_profile in _PRORES_4444 else "yuv422p10le"
            # ProRes is the Nuke deliverable path. Preserve the source's
            # transfer / primaries / matrix on the output tags so Nuke
            # applies the same inverse on read for both source mp4 and
            # our ProRes — otherwise (e.g. forcing color_trc=bt709 on a
            # source tagged iec61966-2-1) Nuke uses BT.1886 inverse on
            # output but sRGB inverse on source, shifting midtones.
            # Range stays tv per ProRes spec (NLEs require it).
            tags = color_tags or {}
            matrix = tags.get("matrix") or "bt709"
            primaries = tags.get("primaries") or "bt709"
            trc = tags.get("trc") or "bt709"
            cmd += [
                "-vf", f"scale=out_color_matrix={matrix}:out_range=tv:flags=accurate_rnd+full_chroma_int",
                "-c:v", "prores_ks",
                "-profile:v", profile_id,
                "-vendor", "apl0",
                "-pix_fmt", pix_fmt,
                "-color_range", "tv",
                "-colorspace", matrix,
                "-color_primaries", primaries,
                "-color_trc", trc,
                # write_colr forces ffmpeg to add the nclc/nclx box to the
                # mov container in addition to the H264 / ProRes elementary
                # stream VUI. Nuke's mov64 reader prefers container atoms
                # over stream-level VUI; without this flag Nuke saw the
                # output as untagged and applied its default (Rec601 on
                # this user's setup), shifting colors vs the source.
                "-movflags", "+write_colr",
                "-c:a", "pcm_s16le", "-shortest",
                output_path]
        else:  # h264
            bitrate = get_original_bitrate(source_path) if source_path else 8_000_000
            base_h264 = [
                "-c:v", "libx264",
                "-b:v", str(bitrate),
                "-maxrate", str(int(bitrate * 1.5)),
                "-bufsize", str(int(bitrate * 2)),
                "-preset", "medium",
                "-pix_fmt", "yuv420p",
            ]

            if self._color_tags is not None:
                # Source has full nclc — match-source strategy: forward the
                # exact same primaries/transfer/matrix/range so Nuke's
                # mov64 reader uses an identical inverse transform on both
                # files. Patched again in release() because ffmpeg's mov
                # muxer drops primaries/transfer to 'unspecified' inside
                # the container box even when we set -color_*.
                tags = self._color_tags
                matrix = tags["matrix"]
                primaries = tags["primaries"]
                trc = tags["trc"]
                src_range = (tags.get("range") or "").lower()
                range_map = {"tv": "tv", "pc": "full",
                             "limited": "tv", "full": "full"}
                range_scale = range_map.get(src_range, "tv")
                range_tag = "tv" if range_scale == "tv" else "pc"
                cmd += [
                    "-vf", f"scale=out_color_matrix={matrix}:out_range={range_scale}:flags=accurate_rnd+full_chroma_int",
                    *base_h264,
                    "-color_range", range_tag,
                    "-colorspace", matrix,
                    "-color_primaries", primaries,
                    "-color_trc", trc,
                    "-movflags", "+write_colr",
                    "-c:a", "copy", "-shortest",
                    output_path,
                ]
            else:
                # Source is untagged (no nclc atom in the container).
                # Emit no color flags so the output is also untagged, and
                # downstream readers (Nuke, players) apply the same default
                # heuristic to source and output → they line up visually.
                # If we instead defaulted bt709, Nuke would interpret our
                # output as bt709 but still default-fallback the source to
                # Rec.601, breaking the match the user expects.
                cmd += [
                    *base_h264,
                    "-c:a", "copy", "-shortest",
                    output_path,
                ]
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    def _drain_stderr(self) -> str:
        """Pull whatever ffmpeg has put on stderr by now."""
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
            err_bytes = (self.proc.stderr.read()
                         if self.proc.stderr else b"")
        except Exception:
            err_bytes = b""
        try:
            return err_bytes.decode("utf-8", errors="replace").strip()
        except Exception:
            return ""

    def write(self, frame_bgr):
        try:
            self.proc.stdin.write(frame_bgr.tobytes())
        except BrokenPipeError:
            # ffmpeg already exited. Surface its stderr tail so we know
            # WHY — silent BrokenPipeError tells the user nothing and
            # hides bad codecs / paths / audio mux failures.
            err = self._drain_stderr()
            tail = err[-1500:] if err else "(no stderr captured)"
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
            rc = self.proc.wait(timeout=300)
        except Exception:
            rc = None
        # Non-zero exit code with non-empty stderr means ffmpeg failed
        # the encode even though our stdin pipe was happy. Surface that
        # before any colr-atom patching attempt would silently no-op.
        if rc not in (None, 0):
            try:
                err_bytes = (self.proc.stderr.read()
                             if self.proc.stderr else b"")
                err = err_bytes.decode("utf-8", errors="replace").strip()
            except Exception:
                err = ""
            tail = err[-1500:] if err else "(no stderr captured)"
            raise RuntimeError(
                f"ffmpeg writer exited with code {rc}. Last stderr "
                f"output:\n{tail}"
            )
        # Patch the colr box after ffmpeg releases the file. ffmpeg leaves
        # primaries / transfer as 'unspecified' inside the container box;
        # we rewrite them from the source tags so Nuke (and any other
        # reader that prefers container atoms over stream-level VUI) sees
        # exactly what the source had.
        if self._color_tags:
            _patch_mp4_colr_atom(self.output_path, self._color_tags)


class ImageSequenceWriter:
    """Save frames as numbered PNGs in `output_dir` + a meta.json sidecar.

    Same write/release contract as FFmpegWriter so the pipeline can swap
    them transparently. cv2.imwrite is used for the PNG encode (8-bit BGR
    direct, no colorspace conversion — bit-exact for our pipeline)."""

    def __init__(self, output_dir: str, fps: float, width: int, height: int,
                 source_path: str = None, digits: int = 6,
                 image_format: str = "png"):
        self.output_dir = output_dir
        self.fps = float(fps)
        self.width = int(width)
        self.height = int(height)
        self.digits = int(digits)
        self.image_format = image_format
        os.makedirs(output_dir, exist_ok=True)
        # Wipe any previous content so the sequence is clean.
        for old in os.listdir(output_dir):
            p = os.path.join(output_dir, old)
            try:
                if os.path.isfile(p):
                    os.remove(p)
            except Exception:
                pass
        self._index = 1
        self._pattern = f"frame_%0{self.digits}d.{self.image_format}"

    def write(self, frame_bgr):
        path = os.path.join(self.output_dir, self._pattern % self._index)
        cv2.imwrite(path, frame_bgr)
        self._index += 1

    def release(self):
        meta = {
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "frames": self._index - 1,
            "image_format": self.image_format,
            "digits": self.digits,
            "pattern": self._pattern,
        }
        with open(os.path.join(self.output_dir, "meta.json"),
                  "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

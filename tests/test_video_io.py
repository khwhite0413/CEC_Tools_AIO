"""Verify the encoded-passthrough pieces in fdcore.video_io.

Run from the project root:
    python -m unittest tests.test_video_io
"""

import unittest

import numpy as np
import torch

from fdcore import video_io


class RoundTripQuantizationTests(unittest.TestCase):
    """High-bit-depth pipe: tensor outputs are uint16. uint8 inputs (image
    sequences) get promoted to float [0,1] then quantized back to uint16.
    A back-conversion through /257 must recover the original uint8 codes
    exactly — verifying the 16-bit pipeline is a strict superset of the
    old 8-bit one."""

    def test_u8_input_round_trips_through_u16(self):
        """u8 in -> tensor -> u16 -> back to u8 must equal the input."""
        codes = np.arange(256, dtype=np.uint8)
        frame_bgr = np.broadcast_to(codes.reshape(1, -1, 1),
                                    (1, 256, 3)).astype(np.uint8)
        tensor = video_io.frames_bgr_to_tensor([frame_bgr])
        u16_out = video_io.tensor_to_frames_bgr(tensor)[0]
        self.assertEqual(u16_out.dtype, np.uint16)
        # u16 -> u8: 65535 / 255 = 257 exactly, so /257 round-trips losslessly
        u8_back = (u16_out // 257).astype(np.uint8)
        self.assertTrue(np.array_equal(frame_bgr, u8_back),
                        "u8 → tensor → u16 → u8 lost codes")

    def test_u16_input_round_trips_byte_exact(self):
        """The bgr48le decode path: every u16 code must survive /65535
        -> *65535 -> u16 unchanged."""
        rng = np.random.default_rng(0)
        frame = rng.integers(0, 65536, size=(32, 48, 3), dtype=np.uint16)
        tensor = video_io.frames_bgr_to_tensor([frame])
        out = video_io.tensor_to_frames_bgr(tensor)[0]
        self.assertEqual(out.dtype, np.uint16)
        self.assertTrue(np.array_equal(frame, out),
                        "u16 → tensor → u16 round-trip not bit-exact")

    def test_multi_frame_u16_zero_diff(self):
        rng = np.random.default_rng(0)
        frames = [rng.integers(0, 65536, size=(32, 48, 3), dtype=np.uint16)
                  for _ in range(4)]
        tensor = video_io.frames_bgr_to_tensor(frames)
        out = video_io.tensor_to_frames_bgr(tensor)
        for i, (a, b) in enumerate(zip(frames, out)):
            self.assertTrue(np.array_equal(a, b), f"frame {i} diverged")


class TpdfDitherTests(unittest.TestCase):
    def test_mean_preserved(self):
        torch.manual_seed(0)
        flat = torch.full((1, 3, 256, 256), 0.4)
        bias = (video_io.tpdf_dither(flat).mean() - flat.mean()).abs().item()
        self.assertLess(bias, 1e-3)

    def test_amplitude_within_2_lsb_pp(self):
        torch.manual_seed(0)
        flat = torch.full((1, 3, 64, 64), 0.5)
        d = video_io.tpdf_dither(flat) - flat
        self.assertLess(d.abs().max().item(), 1.0 / 255.0 + 1e-6)

    def test_dark_gradient_quantization_distributes(self):
        """In an 8-bit ramp where many floats round to one code, dither
        must spread them: dithered+quantize must produce at least as many
        unique codes as plain rounding."""
        torch.manual_seed(0)
        ramp = torch.linspace(0.0, 3.0 / 255.0, 1024)
        plain = torch.round(ramp * 255).to(torch.int32)
        dithered = torch.round(
            video_io.tpdf_dither(ramp).clamp(0, 1) * 255).to(torch.int32)
        self.assertGreaterEqual(int(dithered.unique().numel()),
                                int(plain.unique().numel()))


class ImageSequenceInfoColorTagsTests(unittest.TestCase):
    def test_color_tags_method_does_not_exist_on_image_seq(self):
        """ImageSequenceInfo intentionally has no color_tags() — image
        sequences don't carry stream-level color metadata. Pipeline guards
        with hasattr(...) so it falls back to {}."""
        info = video_io.ImageSequenceInfo.__new__(video_io.ImageSequenceInfo)
        self.assertFalse(hasattr(info, "color_tags"))


class ProResTagPreservationTests(unittest.TestCase):
    """The Nuke-matching fix: ProRes 4444 output must inherit the source's
    transfer / primaries / matrix tags so Nuke applies the same inverse
    transform on read."""

    def _capture_writer_cmd(self, **kwargs):
        import subprocess as sp
        captured = []

        def fake_popen(cmd, *a, **kw):
            captured.append(list(cmd))
            class P:
                stdin = type("S", (), {"write": lambda *a: None,
                                       "close": lambda *a: None})()
                def wait(self, timeout=None): return 0
            return P()

        original_get_ffmpeg = video_io._get_ffmpeg
        original_popen = sp.Popen
        video_io._get_ffmpeg = lambda: "ffmpeg"
        sp.Popen = fake_popen
        try:
            video_io.FFmpegWriter("out.mov", 30, 1920, 1080,
                                  output_format="prores", prores_profile="4444",
                                  **kwargs)
        finally:
            sp.Popen = original_popen
            video_io._get_ffmpeg = original_get_ffmpeg
        return captured[0]

    def _flag_after(self, cmd, flag):
        i = cmd.index(flag)
        return cmd[i + 1]

    def test_prores_uses_source_trc_when_provided(self):
        """When source is sRGB-tagged (iec61966-2-1), ProRes output must
        carry that same transfer tag — the previous behavior forced
        bt709, causing Nuke to use BT.1886 inverse instead of sRGB."""
        cmd = self._capture_writer_cmd(
            color_tags={"trc": "iec61966-2-1", "primaries": "bt709",
                        "matrix": "bt709"})
        self.assertEqual(self._flag_after(cmd, "-color_trc"), "iec61966-2-1")

    def test_prores_falls_back_to_bt709_when_no_tags(self):
        cmd = self._capture_writer_cmd(color_tags=None)
        self.assertEqual(self._flag_after(cmd, "-color_trc"), "bt709")
        self.assertEqual(self._flag_after(cmd, "-color_primaries"), "bt709")
        self.assertEqual(self._flag_after(cmd, "-colorspace"), "bt709")

    def test_prores_preserves_bt601_matrix_for_sd_content(self):
        cmd = self._capture_writer_cmd(
            color_tags={"matrix": "smpte170m", "primaries": "smpte170m",
                        "trc": "smpte170m"})
        self.assertEqual(self._flag_after(cmd, "-colorspace"), "smpte170m")
        self.assertEqual(self._flag_after(cmd, "-color_primaries"), "smpte170m")
        self.assertEqual(self._flag_after(cmd, "-color_trc"), "smpte170m")

    def test_prores_range_stays_tv_regardless(self):
        """ProRes spec convention is video range; never override per
        source even if source was pc-tagged."""
        cmd = self._capture_writer_cmd(color_tags={"range": "pc"})
        self.assertEqual(self._flag_after(cmd, "-color_range"), "tv")


class H264TagPreservationTests(unittest.TestCase):
    """The Premiere-match fix: h264 output must carry the source's nclc /
    VUI color tags so Nuke applies the same inverse on read. Without them
    ffprobe shows 'yuv420p(progressive)' (no range marker, no nclc atom)
    and Nuke uses its default colorspace, shifting tones."""

    def _capture_h264_cmd(self, **kwargs):
        import subprocess as sp
        captured = []

        def fake_popen(cmd, *a, **kw):
            captured.append(list(cmd))
            class P:
                stdin = type("S", (), {"write": lambda *a: None,
                                       "close": lambda *a: None})()
                def wait(self, timeout=None): return 0
            return P()

        original_get_ffmpeg = video_io._get_ffmpeg
        original_popen = sp.Popen
        video_io._get_ffmpeg = lambda: "ffmpeg"
        sp.Popen = fake_popen
        try:
            video_io.FFmpegWriter("out.mp4", 30, 1920, 1080,
                                  output_format="h264", **kwargs)
        finally:
            sp.Popen = original_popen
            video_io._get_ffmpeg = original_get_ffmpeg
        return captured[0]

    def _flag_after(self, cmd, flag):
        i = cmd.index(flag)
        return cmd[i + 1]

    def test_h264_writes_full_color_tags_from_source(self):
        cmd = self._capture_h264_cmd(
            color_tags={"matrix": "bt709", "primaries": "bt709",
                        "trc": "bt709", "range": "tv"})
        self.assertEqual(self._flag_after(cmd, "-colorspace"), "bt709")
        self.assertEqual(self._flag_after(cmd, "-color_primaries"), "bt709")
        self.assertEqual(self._flag_after(cmd, "-color_trc"), "bt709")
        self.assertEqual(self._flag_after(cmd, "-color_range"), "tv")
        # scale filter should pin the same values so the byte content
        # matches the tag the encoder emits
        vf = self._flag_after(cmd, "-vf")
        self.assertIn("out_color_matrix=bt709", vf)
        self.assertIn("out_range=tv", vf)

    def test_h264_inherits_srgb_trc_when_source_is_srgb(self):
        cmd = self._capture_h264_cmd(
            color_tags={"trc": "iec61966-2-1", "primaries": "bt709",
                        "matrix": "bt709", "range": "tv"})
        self.assertEqual(self._flag_after(cmd, "-color_trc"), "iec61966-2-1")

    def test_h264_emits_no_color_flags_when_source_is_untagged(self):
        """Untagged source (no nclc atom) — output stays untagged too.
        Defaulting to bt709 here would make Nuke read source vs output
        with different inverses (source falls back to its Rec.601
        heuristic, output gets read at the bt709 we tagged), which is
        exactly the bug this branch was added to fix."""
        cmd = self._capture_h264_cmd(color_tags=None)
        for flag in ("-color_range", "-colorspace",
                     "-color_primaries", "-color_trc",
                     "-movflags"):
            self.assertNotIn(flag, cmd, f"{flag} should not appear when untagged")
        self.assertNotIn("-vf", cmd)

    def test_h264_partial_tags_treated_as_untagged(self):
        """Source with only `range` (no matrix/primaries/trc) is
        effectively untagged — same no-flags policy applies."""
        cmd = self._capture_h264_cmd(color_tags={"range": "pc"})
        self.assertNotIn("-colorspace", cmd)
        self.assertNotIn("-color_primaries", cmd)
        self.assertNotIn("-color_trc", cmd)
        self.assertNotIn("-color_range", cmd)

    def test_h264_pc_range_forwarded_when_full_tags_present(self):
        cmd = self._capture_h264_cmd(
            color_tags={"matrix": "bt709", "primaries": "bt709",
                        "trc": "bt709", "range": "pc"})
        self.assertEqual(self._flag_after(cmd, "-color_range"), "pc")
        vf = self._flag_after(cmd, "-vf")
        self.assertIn("out_range=full", vf)  # scale-filter spelling


class StreamCopyTests(unittest.TestCase):
    """The bit-exact passthrough fix: when source is h264 and output is
    h264, ffmpeg -c copy lifts the encoded bitstream into the new
    container without any decode/encode. Eliminates the bgr24 round-trip
    bias that even our 'passthrough toggle' was still incurring."""

    def test_image_sequence_input_cannot_stream_copy(self):
        from unittest.mock import patch
        with patch("fdcore.video_io.is_image_sequence", return_value=True):
            self.assertFalse(video_io.can_stream_copy("dir/", "h264"))

    def test_non_h264_output_cannot_stream_copy(self):
        # ProRes / PNG sequence outputs need real encode, not stream copy.
        from unittest.mock import patch
        with patch("fdcore.video_io.is_image_sequence", return_value=False):
            self.assertFalse(video_io.can_stream_copy("in.mp4", "prores"))
            self.assertFalse(video_io.can_stream_copy("in.mp4", "image_seq"))

    def test_h264_source_to_h264_output_can_stream_copy(self):
        from unittest.mock import patch, MagicMock
        info = MagicMock()
        info.codec_name = "h264"
        with patch("fdcore.video_io.is_image_sequence", return_value=False), \
             patch("fdcore.video_io.VideoInfo", return_value=info):
            self.assertTrue(video_io.can_stream_copy("in.mp4", "h264"))

    def test_hevc_source_cannot_stream_copy_to_h264(self):
        """A non-h264 codec inside the source mp4 wouldn't form a valid
        elementary stream when remuxed as -c:v copy into another mp4."""
        from unittest.mock import patch, MagicMock
        info = MagicMock()
        info.codec_name = "hevc"
        with patch("fdcore.video_io.is_image_sequence", return_value=False), \
             patch("fdcore.video_io.VideoInfo", return_value=info):
            self.assertFalse(video_io.can_stream_copy("in.mp4", "h264"))

    def test_copy_stream_command_uses_c_copy(self):
        """The actual ffmpeg invocation must use -c:v copy (no re-encode)
        and pin -avoid_negative_ts so the player slider starts at zero."""
        import subprocess as sp
        captured = []

        def fake_run(cmd, *a, **kw):
            captured.append(list(cmd))
            class R:
                returncode = 0
                stderr = ""
            return R()

        original_get_ffmpeg = video_io._get_ffmpeg
        original_run = sp.run
        video_io._get_ffmpeg = lambda: "ffmpeg"
        sp.run = fake_run
        try:
            video_io.copy_stream("in.mp4", "out.mp4")
        finally:
            sp.run = original_run
            video_io._get_ffmpeg = original_get_ffmpeg

        cmd = captured[0]
        # video stream must be copied, not re-encoded
        self.assertEqual(cmd[cmd.index("-c:v") + 1], "copy")
        # audio is mapped optionally and stream-copied bit-exact (no
        # AAC round-trip): typical h264 mp4 already carries AAC which
        # is container-compatible, so copying preserves it unchanged.
        self.assertIn("-c:a", cmd)
        self.assertEqual(cmd[cmd.index("-c:a") + 1], "copy")
        # timestamps normalized so player slider is sensible
        self.assertIn("-avoid_negative_ts", cmd)
        # no scale / color flags / fps remap — pure remux
        for forbidden in ("-vf", "-color_range", "-colorspace",
                          "-color_primaries", "-color_trc",
                          "-pix_fmt", "-r"):
            self.assertNotIn(forbidden, cmd, f"{forbidden} should not appear")

    def test_copy_stream_with_frame_cap_uses_time_based_t(self):
        """frame_cap > 0 must translate to a -t duration (keyframe-aligned
        cuts), not -frames:v (which would leave dangling B-frames under
        -c copy)."""
        import subprocess as sp
        from unittest.mock import patch, MagicMock
        captured = []

        def fake_run(cmd, *a, **kw):
            captured.append(list(cmd))
            class R:
                returncode = 0
                stderr = ""
            return R()

        info = MagicMock()
        info.fps = 24.0

        original_get_ffmpeg = video_io._get_ffmpeg
        original_run = sp.run
        video_io._get_ffmpeg = lambda: "ffmpeg"
        sp.run = fake_run
        try:
            with patch("fdcore.video_io.VideoInfo", return_value=info):
                video_io.copy_stream("in.mp4", "out.mp4", frame_cap=80)
        finally:
            sp.run = original_run
            video_io._get_ffmpeg = original_get_ffmpeg

        cmd = captured[0]
        self.assertIn("-t", cmd)
        self.assertNotIn("-frames:v", cmd)
        # Slight overshoot (+1 frame) to survive keyframe alignment
        t_value = float(cmd[cmd.index("-t") + 1])
        self.assertAlmostEqual(t_value, 81 / 24.0, places=4)


class FpsDetectionTests(unittest.TestCase):
    """Auto fps detection priority: ffprobe avg_frame_rate → ffmpeg-stderr
    fps token → cv2.CAP_PROP_FPS fallback. NEVER use ffprobe's
    r_frame_rate / ffmpeg's tbr — those return the stts-pattern nominal
    rate which is 60 for a Seedance 24-fps-avg VFR source."""

    def test_parse_rational_handles_fraction(self):
        self.assertAlmostEqual(video_io._parse_rational("30000/1001"),
                               29.970, places=3)

    def test_parse_rational_handles_decimal(self):
        self.assertEqual(video_io._parse_rational("24.04"), 24.04)

    def test_parse_rational_handles_integer_string(self):
        self.assertEqual(video_io._parse_rational("24"), 24.0)

    def test_parse_rational_rejects_zero_denominator(self):
        self.assertIsNone(video_io._parse_rational("24/0"))

    def test_parse_rational_rejects_empty(self):
        self.assertIsNone(video_io._parse_rational(""))
        self.assertIsNone(video_io._parse_rational(None))

    def test_ffmpeg_stream_regex_extracts_fps(self):
        line = ("Stream #0:0[0x1](und): Video: h264 (High) (avc1 / 0x31637661), "
                "yuv420p(progressive), 1470x630, 6327 kb/s, 24.04 fps, 60 tbr, "
                "15360 tbn (default)")
        m_fps = video_io._FFMPEG_FPS_RE.search(line)
        self.assertIsNotNone(m_fps)
        self.assertEqual(float(m_fps.group(1)), 24.04)
        # Sanity: tbr regex no longer exists — we explicitly avoid it
        # because tbr=60 for a 24-avg VFR source.
        self.assertFalse(hasattr(video_io, "_FFMPEG_TBR_RE"))


class FpsSnapTests(unittest.TestCase):
    """Snap noisy detected fps to a nearby standard delivery rate (24,
    23.976, 25, 29.97, 30, 50, 59.94, 60). Within ±0.1 of a standard →
    snap to it. Otherwise leave the value alone (real fractional fps
    like 27.534 must survive untouched)."""

    def test_seedance_vfr_24_04_snaps_to_24(self):
        self.assertEqual(video_io._snap_fps(24.04), 24.0)

    def test_already_standard_24_passes_through(self):
        self.assertEqual(video_io._snap_fps(24.0), 24.0)

    def test_ntsc_23_976_preserved(self):
        self.assertAlmostEqual(video_io._snap_fps(23.976), 23.976, places=3)

    def test_almost_ntsc_23_98_snaps_to_ntsc_not_integer(self):
        """0.004 from 23.976 vs 0.020 from 24.0 — must pick 23.976."""
        self.assertAlmostEqual(video_io._snap_fps(23.98), 23.976, places=3)

    def test_ntsc_29_97_preserved(self):
        self.assertAlmostEqual(video_io._snap_fps(29.97), 29.97, places=3)

    def test_30_04_snaps_to_30(self):
        self.assertEqual(video_io._snap_fps(30.04), 30.0)

    def test_exotic_27_534_left_alone(self):
        """Outside tolerance of any standard → must NOT snap."""
        self.assertEqual(video_io._snap_fps(27.534), 27.534)

    def test_60_04_snaps_to_60_not_59_94(self):
        """|60.04 - 60| = 0.04 vs |60.04 - 59.94| = 0.10 — pick 60."""
        self.assertEqual(video_io._snap_fps(60.04), 60.0)

    def test_zero_and_none_pass_through(self):
        self.assertEqual(video_io._snap_fps(0), 0)
        self.assertIsNone(video_io._snap_fps(None))

    def test_negative_passes_through(self):
        self.assertEqual(video_io._snap_fps(-1), -1)


class FpsOverrideTests(unittest.TestCase):
    """User-supplied fps_override flows from settings → pipeline →
    FFmpegReader's -vf fps= (and FFmpegWriter's -r via effective_fps)."""

    def test_ffmpeg_reader_accepts_explicit_fps(self):
        """FFmpegReader(path, fps=24.0) must pin its -vf fps= to that
        value, not the auto-detected one."""
        import subprocess as sp
        from unittest.mock import patch, MagicMock
        captured = []

        def fake_popen(cmd, *a, **kw):
            captured.append(list(cmd))
            class P:
                stdout = type("S", (), {"close": lambda *a: None,
                                        "read": lambda *a, **kw: b""})()
                def wait(self, timeout=None): return 0
                def terminate(self): pass
            return P()

        info = MagicMock()
        info.width = 1280; info.height = 720; info.fps = 24.04
        info.frame_count = 360; info.duration = 15.0; info.has_audio = False

        original_ffmpeg = video_io._get_ffmpeg
        original_popen = sp.Popen
        video_io._get_ffmpeg = lambda: "ffmpeg"
        sp.Popen = fake_popen
        try:
            with patch("fdcore.video_io.VideoInfo", return_value=info):
                # Auto path: uses info.fps (24.04)
                video_io.FFmpegReader("in.mp4")
                # Override path: pin to 24.0
                video_io.FFmpegReader("in.mp4", fps=24.0)
        finally:
            sp.Popen = original_popen
            video_io._get_ffmpeg = original_ffmpeg

        # First call (auto): -vf fps=24.04
        self.assertIn("fps=24.04", " ".join(captured[0]))
        # Second call (override): -vf fps=24.0
        self.assertIn("fps=24.0", " ".join(captured[1]))


if __name__ == "__main__":
    unittest.main()

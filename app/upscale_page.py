"""Upscale tab — spandrel-based SR (OpenProteus etc.).

Layout mirrors inpaint_page: VideoPlayer on the left, a parameter panel
on the right with a dev-mode toggle, hover tooltips on every widget,
and the same progress / cancel pattern.
"""

import os
import shutil
import time
from math import gcd

import cv2
import torch
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                              QPushButton, QFileDialog, QProgressBar,
                              QPlainTextEdit, QMessageBox, QScrollArea,
                              QSplitter, QLineEdit, QGroupBox, QSpinBox,
                              QCheckBox, QComboBox, QFormLayout, QFrame,
                              QSizePolicy)

from app.i18n import tr
from app.video_player import VideoPlayer
from app.upscale_worker import UpscaleWorker, MODELS, model_available
from vdcore.video_io import auto_format_from_codec


_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_MODULE_DIR, ".."))
_OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "output")
_CACHE_DIR = os.path.join(_PROJECT_ROOT, "cache")
# Viewer-only copy of the final deliverable. cv2's VideoPlayer would
# lock the real output in output/ on Windows; MainWindow's sweep
# (start + close) deletes this on next launch / shutdown.
_VIEWER_CACHE_NAME = "upscale_view.mp4"

# ProRes profile choices — mirrors fdcore parameter_panel so the
# Inpaint/Upscale tabs offer the same options the Denoise tab does.
_PRORES_KEYS = ("proxy", "lt", "standard", "hq", "4444", "4444xq")
_PRORES_LABELS = {
    "proxy": "Proxy", "lt": "LT", "standard": "422",
    "hq": "422 HQ", "4444": "4444", "4444xq": "4444 XQ",
}


class UpscalePage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.video_path = None
        self.video_info = None  # (h, w, fps, fc)
        self.video_codec = ""    # populated by _set_video via fdcore probe
        self.worker = None
        self._t_start = 0.0
        self._last_done = 0
        self._last_total = 0

        os.makedirs(_OUTPUT_DIR, exist_ok=True)

        root = QHBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 16)
        root.setSpacing(12)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter, 1)

        splitter.addWidget(self._build_left())
        splitter.addWidget(self._build_right())
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([900, 420])

    # ------------------------------------------------------------------
    # Layout — left column
    # ------------------------------------------------------------------
    def _build_left(self) -> QWidget:
        left = QWidget()
        lay = QVBoxLayout(left)
        lay.setContentsMargins(0, 0, 0, 0)

        self.player = VideoPlayer(click_to_load=True)
        self.player.empty_clicked.connect(self._pick_file)
        lay.addWidget(self.player, 1)

        meta_row = QHBoxLayout()
        self.info_label = QLabel(tr("info_no_video"))
        self.info_label.setWordWrap(True)
        pick = QPushButton(tr("btn_pick_file"))
        pick.clicked.connect(self._pick_file)
        meta_row.addWidget(self.info_label, 1)
        meta_row.addWidget(pick, 0)
        lay.addLayout(meta_row)

        out_row = QHBoxLayout()
        out_row.addWidget(QLabel(tr("label_output")))
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText(tr("upscale_placeholder_output"))
        self.output_edit.setToolTip(tr("tooltip_output_path"))
        out_row.addWidget(self.output_edit, 1)
        browse = QPushButton(tr("btn_browse"))
        browse.clicked.connect(self._browse_output)
        out_row.addWidget(browse, 0)
        lay.addLayout(out_row)

        self.res_label = QLabel(tr("upscale_no_preview"))
        self.res_label.setStyleSheet(
            "color: rgba(180,184,200,200); padding: 2px 0;")
        lay.addWidget(self.res_label)

        btn_row = QHBoxLayout()
        self.btn_run = QPushButton(tr("btn_run"))
        self.btn_run.setStyleSheet("font-weight: bold;")
        self.btn_run.clicked.connect(self._run)
        self.btn_cancel = QPushButton(tr("btn_cancel"))
        self.btn_cancel.clicked.connect(self._cancel)
        self.btn_cancel.setEnabled(False)
        btn_row.addWidget(self.btn_run)
        btn_row.addWidget(self.btn_cancel)
        lay.addLayout(btn_row)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        lay.addWidget(self.progress)
        self._progress_normal_palette = QPalette(self.progress.palette())
        self._progress_done_palette = QPalette(self.progress.palette())
        self._progress_done_palette.setColor(
            QPalette.Highlight, QColor("#a5d6a7"))

        self.eta_label = QLabel("")
        self.eta_label.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.eta_label)

        lay.addWidget(QLabel(tr("label_log")))
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        self.log.setMaximumHeight(140)
        lay.addWidget(self.log, 0)

        return left

    # ------------------------------------------------------------------
    # Layout — right column (params)
    # ------------------------------------------------------------------
    def _build_right(self) -> QWidget:
        right = QWidget()
        right.setMinimumWidth(360)
        right.setMaximumWidth(520)
        lay = QVBoxLayout(right)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)

        # QGroupBox (plain — no setStyleSheet) so QComboBox children
        # render in Fusion-native style instead of falling back to the
        # solid-black QSS chrome. Same rationale as the inpaint tab.
        box = QGroupBox(tr("upscale_section_params"))
        box.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        box_lay = QVBoxLayout(box)
        box_lay.setSpacing(8)

        head = QHBoxLayout()
        head.addStretch(1)
        self.dev_mode = QCheckBox(tr("label_dev_mode"))
        self.dev_mode.setToolTip(tr("tooltip_dev_mode"))
        head.addWidget(self.dev_mode)
        box_lay.addLayout(head)

        basic = QFormLayout()
        basic.setLabelAlignment(Qt.AlignRight)
        # SetMinAndMaxSize keeps the form's reported height tight so the
        # parent Card frame hugs only what's visible (FD pattern).
        basic.setSizeConstraint(QFormLayout.SetMinAndMaxSize)
        basic.setContentsMargins(0, 0, 0, 0)
        basic.setHorizontalSpacing(10)
        basic.setVerticalSpacing(6)

        self.model_combo = QComboBox()
        for key, info in MODELS.items():
            self.model_combo.addItem(tr(info["label_key"]), key)
        self.model_combo.setToolTip(tr("upscale_tooltip_model"))
        self.model_combo.currentIndexChanged.connect(
            lambda _: self._refresh_res_preview())
        lbl_model = QLabel(tr("upscale_label_model"))
        lbl_model.setToolTip(tr("upscale_tooltip_model"))
        basic.addRow(lbl_model, self.model_combo)

        # Target resolution: model is fixed 2x; cv2 then resizes the
        # model output to the picked long-edge. 0 = native 2x (no resize).
        self.target_combo = QComboBox()
        self.target_combo.addItem(tr("upscale_target_native"), 0)
        self.target_combo.addItem("HD (1280)",  1280)
        self.target_combo.addItem("FHD (1920)", 1920)
        self.target_combo.addItem("QHD (2560)", 2560)
        self.target_combo.addItem("4K UHD (3840)", 3840)
        self.target_combo.addItem("8K UHD (7680)", 7680)
        self.target_combo.setToolTip(tr("upscale_tooltip_target"))
        self.target_combo.currentIndexChanged.connect(
            lambda _: self._refresh_res_preview())
        lbl_target = QLabel(tr("upscale_label_target"))
        lbl_target.setToolTip(tr("upscale_tooltip_target"))
        basic.addRow(lbl_target, self.target_combo)

        # Predicted final dimensions + aspect ratio. Always visible —
        # populates from _refresh_res_preview whenever the source video
        # or the target dropdown changes.
        self.target_info_label = QLabel(tr("upscale_target_info_empty"))
        self.target_info_label.setStyleSheet(
            "color: rgba(160, 164, 180, 200); "
            "font-size: 11px; padding-top: 2px;")
        basic.addRow("", self.target_info_label)

        self.sb_batch_frames = QSpinBox()
        self.sb_batch_frames.setRange(0, 5000)
        self.sb_batch_frames.setValue(0)
        self.sb_batch_frames.setToolTip(tr("upscale_tooltip_batch_frames"))
        lbl_batch = QLabel(tr("upscale_label_batch_frames"))
        lbl_batch.setToolTip(tr("upscale_tooltip_batch_frames"))
        basic.addRow(lbl_batch, self.sb_batch_frames)
        box_lay.addLayout(basic)

        # Wrap dev-mode rows in a QWidget so the form's geometry
        # collapses entirely when dev_mode is off — hiding individual
        # children would leave reserved row slots and the card would
        # keep its taller height.
        self._adv_widget = QWidget()
        adv = QFormLayout(self._adv_widget)
        adv.setLabelAlignment(Qt.AlignRight)
        adv.setContentsMargins(0, 0, 0, 0)
        adv.setSizeConstraint(QFormLayout.SetMinAndMaxSize)
        adv.setHorizontalSpacing(10)
        adv.setVerticalSpacing(6)

        def _add_adv_check(tip_key: str, checkbox: QCheckBox):
            checkbox.setToolTip(tr(tip_key))
            adv.addRow(checkbox)

        self.chk_fp16 = QCheckBox(tr("inpaint_fp16"))
        self.chk_fp16.setChecked(True)
        _add_adv_check("upscale_tooltip_fp16", self.chk_fp16)

        box_lay.addWidget(self._adv_widget)

        self.dev_mode.toggled.connect(self._apply_dev_mode)
        self._apply_dev_mode(False)

        # Plain QGroupBox — same reason as params box above.
        fmt_box = QGroupBox(tr("io_section_format"))
        fmt_box.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        fmt_outer = QVBoxLayout(fmt_box)
        fmt_outer.setSpacing(6)

        codec_form = QFormLayout()
        codec_form.setLabelAlignment(Qt.AlignRight)
        codec_form.setSizeConstraint(QFormLayout.SetMinAndMaxSize)
        codec_form.setContentsMargins(0, 0, 0, 0)
        self.format_combo = QComboBox()
        self.format_combo.addItem(tr("io_format_auto"), "auto")
        self.format_combo.addItem(tr("io_format_h264"), "h264")
        self.format_combo.addItem(tr("io_format_prores"), "prores")
        self.format_combo.setToolTip(tr("io_tooltip_format"))
        self.format_combo.currentIndexChanged.connect(
            lambda _: self._on_format_changed())
        lbl_codec = QLabel(tr("io_label_codec"))
        lbl_codec.setToolTip(tr("io_tooltip_format"))
        codec_form.addRow(lbl_codec, self.format_combo)
        fmt_outer.addLayout(codec_form)

        # ProRes profile drawer — visible only when the resolved format
        # is "prores" (whether via Auto on a prores source or explicit).
        self.prores_drawer = QWidget()
        prof_form = QFormLayout(self.prores_drawer)
        prof_form.setContentsMargins(0, 0, 0, 0)
        prof_form.setLabelAlignment(Qt.AlignRight)
        prof_form.setSizeConstraint(QFormLayout.SetMinAndMaxSize)
        self.prores_profile = QComboBox()
        for k in _PRORES_KEYS:
            self.prores_profile.addItem(_PRORES_LABELS[k], k)
        # Default 4444 — same as the Denoise tab; alpha/color-faithful
        # for compositing chains.
        idx = _PRORES_KEYS.index("4444")
        self.prores_profile.setCurrentIndex(idx)
        self.prores_profile.setToolTip(tr("io_tooltip_prores_profile"))
        lbl_profile = QLabel(tr("io_label_prores_profile"))
        lbl_profile.setToolTip(tr("io_tooltip_prores_profile"))
        prof_form.addRow(lbl_profile, self.prores_profile)
        fmt_outer.addWidget(self.prores_drawer)
        self.prores_drawer.setVisible(False)

        # --- Pack params card + format card into a single scrollable
        # column anchored to the top. Without this, the params card was
        # stretching to fill all vertical space and pushing the format
        # card to the bottom edge. ---
        # No setStyleSheet on inner / scroll — any QSS on a parent
        # forces descendant QComboBox / QSpinBox into QSS rendering
        # mode, which makes the dropdown chrome look solid black.
        inner = QWidget()
        inner_lay = QVBoxLayout(inner)
        inner_lay.setContentsMargins(0, 0, 0, 0)
        inner_lay.setSpacing(12)
        inner_lay.addWidget(box)
        inner_lay.addWidget(fmt_box)
        inner_lay.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidget(inner)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        lay.addWidget(scroll, 1)

        return right

    def _resolved_format(self) -> str:
        fmt = (self.format_combo.currentData()
               if hasattr(self, "format_combo") else "auto")
        if fmt == "auto":
            return auto_format_from_codec(self.video_codec)
        return fmt

    def _on_format_changed(self):
        fmt = self._resolved_format()
        # Reveal the ProRes profile sub-row only when the resolved
        # codec is prores (either explicit or auto-detected from source).
        if hasattr(self, "prores_drawer"):
            self.prores_drawer.setVisible(fmt == "prores")
        cur = self.output_edit.text().strip()
        if not cur:
            return
        new_ext = ".mov" if fmt == "prores" else ".mp4"
        base, _ = os.path.splitext(cur)
        new = base + new_ext
        if new != cur:
            self.output_edit.setText(new)

    def _apply_dev_mode(self, on: bool):
        """Toggle the wrapping QWidget so dev-mode rows collapse entirely
        and the card hugs only the visible content."""
        self._adv_widget.setVisible(on)

    # ------------------------------------------------------------------
    # File picking
    # ------------------------------------------------------------------
    def _pick_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "비디오 선택", "",
            "비디오 (*.mp4 *.mov *.mkv *.avi *.webm *.m4v);;모든 파일 (*.*)")
        if path:
            self._set_video(path)

    def _set_video(self, path: str):
        if not os.path.isfile(path):
            QMessageBox.warning(self, "Upscale", f"파일이 존재하지 않습니다:\n{path}")
            return
        try:
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                raise ValueError("cv2가 영상을 열지 못했습니다.")
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        except Exception as e:
            QMessageBox.critical(self, "Upscale", f"비디오 열기 실패:\n{e}")
            return

        self.video_info = (h, w, fps, fc)
        # Light codec probe so "Auto" format picks the right codec to
        # avoid h264 re-compression in chained workflows.
        self.video_codec = ""
        try:
            from fdcore.video_io import VideoInfo as _FdInfo
            self.video_codec = (_FdInfo(path).codec_name or "").lower()
        except Exception:
            pass
        try:
            self.player.disable_compare()
        except Exception:
            pass
        self.player.load(path)
        self.video_path = path
        self.info_label.setText(
            f"<b>{os.path.basename(path)}</b> &nbsp; "
            f"{w}×{h} @ {fps:.2f}fps · {fc} frames")
        self.output_edit.setText(self._default_output_path(path))
        if hasattr(self, "format_combo"):
            self._on_format_changed()
        self._refresh_res_preview()
        self._log(f"입력 로드: {path}")

    def _browse_output(self):
        start = self.output_edit.text().strip() or (self.video_path or "")
        path, _ = QFileDialog.getSaveFileName(
            self, "출력 파일", start, "MP4 (*.mp4);;모든 파일 (*.*)")
        if path:
            self.output_edit.setText(path)

    @staticmethod
    def _default_output_path(input_path: str) -> str:
        os.makedirs(_OUTPUT_DIR, exist_ok=True)
        base = os.path.splitext(os.path.basename(input_path))[0]
        candidate = os.path.join(_OUTPUT_DIR, f"{base}_upscaled.mp4")
        if not os.path.exists(candidate):
            return candidate
        n = 1
        while True:
            candidate = os.path.join(
                _OUTPUT_DIR, f"{base}_upscaled_{n:03d}.mp4")
            if not os.path.exists(candidate):
                return candidate
            n += 1

    def _refresh_res_preview(self):
        if self.video_info is None:
            self.res_label.setText(tr("upscale_no_preview"))
            if hasattr(self, "target_info_label"):
                self.target_info_label.setText(tr("upscale_target_info_empty"))
            return
        h, w, _, _ = self.video_info
        info = MODELS.get(self.model_combo.currentData(), {})
        scale = int(info.get("scale_fallback", 2))
        model_w, model_h = w * scale, h * scale
        target_long = int(self.target_combo.currentData() or 0)
        if target_long > 0:
            model_long = max(model_w, model_h)
            if model_long != target_long:
                ratio = target_long / float(model_long)
                dst_w = max(2, int(round(model_w * ratio / 2.0)) * 2)
                dst_h = max(2, int(round(model_h * ratio / 2.0)) * 2)
            else:
                dst_w, dst_h = model_w, model_h
        else:
            dst_w, dst_h = model_w, model_h

        # Pick the wording dynamically. The end-to-end transform (source
        # vs final) is what the user cares about, NOT the cv2-internal
        # direction (model 2x output → final). Comparing dst to model_w
        # decides "줄여서" vs "더 키워서" because by that point the model
        # has already produced model_w × model_h.
        if (dst_w, dst_h) == (model_w, model_h):
            key = "upscale_preview_native"
        elif dst_w > model_w:
            key = "upscale_preview_upscale"
        else:
            key = "upscale_preview_downscale"
        self.res_label.setText(
            tr(key).format(
                src_w=w, src_h=h,
                mid_w=model_w, mid_h=model_h,
                dst_w=dst_w, dst_h=dst_h))

        # Aspect ratio follows the source (cv2 resize preserves it via
        # the proportional ratio math above). gcd-simplified — odd source
        # ratios like 1470×630 get reduced to 7:3 rather than 16:9.
        if dst_w > 0 and dst_h > 0:
            g = gcd(dst_w, dst_h)
            ar_w, ar_h = dst_w // g, dst_h // g
        else:
            ar_w, ar_h = 0, 0
        if hasattr(self, "target_info_label"):
            self.target_info_label.setText(
                tr("upscale_target_info_fmt").format(
                    w=dst_w, h=dst_h, ar_w=ar_w, ar_h=ar_h))

    # ------------------------------------------------------------------
    # Run / cancel
    # ------------------------------------------------------------------
    def _run(self):
        if not self.video_path:
            QMessageBox.information(self, "Upscale", "비디오를 먼저 선택하세요.")
            return
        model_key = self.model_combo.currentData()
        if not model_available(model_key):
            QMessageBox.warning(
                self, "Upscale",
                "업스케일 가중치가 없습니다.\n"
                "setup.bat을 다시 실행하거나 weights/openproteus/ 폴더를 확인하세요.")
            return

        out_path = self.output_edit.text().strip() or \
            self._default_output_path(self.video_path)
        fmt = self._resolved_format()
        expected_ext = ".mov" if fmt == "prores" else ".mp4"
        if not out_path.lower().endswith(expected_ext):
            out_path = os.path.splitext(out_path)[0] + expected_ext
        out_dir = os.path.dirname(out_path)
        if out_dir and not os.path.isdir(out_dir):
            try:
                os.makedirs(out_dir, exist_ok=True)
            except Exception as e:
                QMessageBox.critical(self, "Upscale",
                                     f"출력 폴더 생성 실패:\n{e}")
                return

        settings = {
            "model":             model_key,
            "fp16":              self.chk_fp16.isChecked(),
            "max_batch_frames":  self.sb_batch_frames.value(),
            "target_long_edge":  int(self.target_combo.currentData() or 0),
            "output_format":     fmt,
            "prores_profile":    (self.prores_profile.currentData()
                                  or "4444"),
        }
        device = (torch.device("cuda") if torch.cuda.is_available()
                  else torch.device("cpu"))
        self._log(f"처리 시작 → {out_path}")
        self._log(f"  device={device}, model={model_key}, "
                  f"fp16={settings['fp16']}, "
                  f"max_batch={settings['max_batch_frames']}")

        self.worker = UpscaleWorker(
            self.video_path, out_path, settings, device)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_signal.connect(self._on_finished)
        self.worker.error_signal.connect(self._on_error)
        self.worker.cancelled_signal.connect(self._on_cancelled)
        self.worker.phase_changed.connect(self._on_phase_changed)

        self.progress.setPalette(self._progress_normal_palette)
        self.progress.setValue(0)
        self.eta_label.setText("")
        self._last_done = 0
        self._last_total = 0
        self._t_start = time.time()
        self._set_running(True)
        self.worker.start()

    def _cancel(self):
        if self.worker is not None:
            self.worker.cancel()
            self._log("취소 요청됨...")

    def _set_running(self, running: bool):
        self.btn_run.setEnabled(not running)
        self.btn_cancel.setEnabled(running)
        self.model_combo.setEnabled(not running)

    # ------------------------------------------------------------------
    # Worker signal handlers
    # ------------------------------------------------------------------
    def _on_progress(self, done: int, total: int):
        if total <= 0:
            return
        self._last_done = done
        self._last_total = total
        pct = int(done * 100 / total)
        self.progress.setValue(pct)
        elapsed = time.time() - self._t_start
        if done > 0:
            eta = elapsed * (total - done) / done
            self.eta_label.setText(
                f"{done}/{total} | 경과 {elapsed:.0f}s | "
                f"남은 시간 ≈ {eta:.0f}s")

    def _on_phase_changed(self, phase: str):
        if phase == "encoding":
            self.progress.setValue(100)
            self.eta_label.setText("마무리 중 — 비디오 인코딩")
            self.btn_cancel.setEnabled(False)
        elif phase == "done":
            self.eta_label.setText("저장 완료, 마감 처리 중...")

    def _on_finished(self, output_path: str):
        self._set_running(False)
        self.progress.setValue(100)
        elapsed = time.time() - self._t_start
        self.progress.setPalette(self._progress_done_palette)
        self.eta_label.setText(tr("finished_summary").format(
            done=self._last_done, total=self._last_total, elapsed=elapsed))
        self._log(f"완료 ({elapsed:.1f}s) → {output_path}")
        self.worker = None
        if output_path and os.path.exists(output_path):
            try:
                # Stage a viewer-only copy in cache/ so the real
                # deliverable in output/ stays writable while the player
                # keeps its cv2 handle.
                viewer_path = self._stage_for_viewer(output_path)
                self.player.disable_compare()
                self.player.load(viewer_path)
                if self.video_path and os.path.exists(self.video_path):
                    self.player.enable_compare(self.video_path)
            except Exception as e:
                self._log(f"[WARN] 결과 미리보기 전환 실패: {e}")

    @staticmethod
    def _stage_for_viewer(output_path: str) -> str:
        """Copy the deliverable into cache/upscale_view.<ext> so the
        in-app player can hold a cv2 handle without locking output/.
        Extension follows the deliverable (.mp4 for h264, .mov for prores)."""
        try:
            os.makedirs(_CACHE_DIR, exist_ok=True)
            ext = os.path.splitext(output_path)[1].lower() or ".mp4"
            if ext not in (".mp4", ".mov"):
                ext = ".mp4"
            base = os.path.splitext(_VIEWER_CACHE_NAME)[0]
            dst = os.path.join(_CACHE_DIR, f"{base}{ext}")
            if os.path.exists(dst):
                os.remove(dst)
            shutil.copy2(output_path, dst)
            return dst
        except Exception:
            return output_path

    def _on_cancelled(self):
        self._set_running(False)
        elapsed = time.time() - self._t_start
        pal = QPalette(self.progress.palette())
        pal.setColor(QPalette.Highlight, QColor("#e57373"))
        self.progress.setPalette(pal)
        self.eta_label.setText(tr("cancelled_summary").format(
            done=self._last_done, total=self._last_total, elapsed=elapsed))
        self._log(f"취소됨 · {self._last_done}/{self._last_total} · "
                  f"{elapsed:.1f}s")
        self.worker = None

    def _on_error(self, msg: str):
        self._set_running(False)
        self._log(f"[ERROR] {msg}")
        self.worker = None
        QMessageBox.critical(self, "Upscale", msg)

    def _log(self, msg: str):
        self.log.appendPlainText(msg)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def shutdown(self):
        if self.worker is not None and self.worker.isRunning():
            try:
                self.worker.cancel()
                self.worker.wait(3000)
                if self.worker.isRunning():
                    self.worker.terminate()
                    self.worker.wait(1000)
            except Exception:
                pass
            self.worker = None
        try:
            self.player.clear()
        except Exception:
            pass

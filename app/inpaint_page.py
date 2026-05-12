"""Inpaint tab — ProPainter-based video inpainting with PNG mask presets.

Brush / rectangle ROI deliberately omitted (future work). Mask input
comes from a fixed preset (currently: Dreamina AI watermark removal).
"""

import os
import time

import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap, QPalette, QColor
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                              QPushButton, QFileDialog, QProgressBar,
                              QPlainTextEdit, QMessageBox, QScrollArea,
                              QSplitter, QLineEdit, QGroupBox, QSpinBox,
                              QCheckBox, QToolButton, QMenu, QFrame,
                              QFormLayout, QComboBox)

from app.i18n import tr
from app.video_player import VideoPlayer
from app.inpaint_worker import InpaintWorker
from vdcore.config import get_device, weights_available


_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_MODULE_DIR, ".."))
_PRESETS_DIR = os.path.join(_PROJECT_ROOT, "presets")
_OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "output")


# Mirror of VDInpaint's PRESETS dict — extend here when adding new
# generator presets. Anchor positions the scaled mask when the preset
# aspect ratio doesn't match the input video.
_PRESETS = {
    "dreamina": {
        "landscape": "dreamina_ai.png",
        "portrait":  "dreamina_9_16_watermark.png",
        "anchor":    "top-left",
        "label_key": "inpaint_preset_dreamina",
    },
}


class InpaintPage(QWidget):
    """Single tab — file pick, preset selection, ProPainter run."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.video_path = None
        self.video_info = None  # (h, w, fps, frame_count)
        self.current_mask = None  # uint8 H x W, full-frame
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
    # Layout
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
        self.output_edit.setPlaceholderText(tr("inpaint_placeholder_output"))
        self.output_edit.setToolTip(tr("tooltip_output_path"))
        out_row.addWidget(self.output_edit, 1)
        browse = QPushButton(tr("btn_browse"))
        browse.clicked.connect(self._browse_output)
        out_row.addWidget(browse, 0)
        lay.addLayout(out_row)

        preset_row = QHBoxLayout()
        preset_menu = QMenu(self)
        for key, preset in _PRESETS.items():
            act = preset_menu.addAction(tr(preset["label_key"]))
            act.triggered.connect(lambda _, k=key: self._apply_preset(k))
        self.preset_btn = QToolButton(self)
        self.preset_btn.setText(tr("inpaint_preset_btn"))
        self.preset_btn.setMenu(preset_menu)
        self.preset_btn.setPopupMode(QToolButton.InstantPopup)
        self.preset_btn.setFixedHeight(28)
        preset_row.addWidget(self.preset_btn, 0)

        self.mask_status = QLabel(tr("inpaint_no_mask"))
        self.mask_status.setStyleSheet(
            "color: rgba(180,184,200,200); padding-left: 6px;")
        preset_row.addWidget(self.mask_status, 1)

        self.mask_thumb = QLabel()
        self.mask_thumb.setFixedSize(64, 36)
        self.mask_thumb.setFrameShape(QFrame.StyledPanel)
        self.mask_thumb.setAlignment(Qt.AlignCenter)
        self.mask_thumb.setStyleSheet(
            "background-color: rgba(20,20,28,200); "
            "border: 1px solid rgba(100,100,120,80);")
        preset_row.addWidget(self.mask_thumb, 0)
        lay.addLayout(preset_row)

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

    def _build_right(self) -> QWidget:
        right = QWidget()
        right.setMinimumWidth(360)
        right.setMaximumWidth(520)
        lay = QVBoxLayout(right)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)

        params_box = QGroupBox(tr("inpaint_section_params"))
        params_lay = QVBoxLayout(params_box)
        params_lay.setSpacing(8)

        # Header row: dev-mode toggle on the right, like the denoise tab.
        head = QHBoxLayout()
        head.addStretch(1)
        self.dev_mode = QCheckBox(tr("label_dev_mode"))
        self.dev_mode.setToolTip(tr("tooltip_dev_mode"))
        head.addWidget(self.dev_mode)
        params_lay.addLayout(head)

        # --- Always-visible row: model + max_batch_frames ---
        basic = QFormLayout()
        basic.setLabelAlignment(Qt.AlignRight)

        self.model_combo = QComboBox()
        self.model_combo.addItem(tr("inpaint_model_propainter"), "propainter")
        self.model_combo.setToolTip(tr("inpaint_tooltip_model"))
        lbl_model = QLabel(tr("inpaint_label_model"))
        lbl_model.setToolTip(tr("inpaint_tooltip_model"))
        basic.addRow(lbl_model, self.model_combo)

        self.sb_batch_frames = QSpinBox()
        self.sb_batch_frames.setRange(0, 5000)
        self.sb_batch_frames.setValue(0)
        self.sb_batch_frames.setToolTip(tr("inpaint_tooltip_batch_frames"))
        lbl_batch = QLabel(tr("inpaint_label_batch_frames"))
        lbl_batch.setToolTip(tr("inpaint_tooltip_batch_frames"))
        basic.addRow(lbl_batch, self.sb_batch_frames)
        params_lay.addLayout(basic)

        # --- Dev-mode-only rows (hidden by default) ---
        adv = QFormLayout()
        adv.setLabelAlignment(Qt.AlignRight)

        self._adv_widgets = []  # collected so _apply_dev_mode can toggle

        def _add_adv_row(label_key: str, tip_key: str, widget):
            widget.setToolTip(tr(tip_key))
            lbl = QLabel(tr(label_key))
            lbl.setToolTip(tr(tip_key))
            adv.addRow(lbl, widget)
            self._adv_widgets.extend([lbl, widget])

        def _add_adv_check(tip_key: str, checkbox: QCheckBox):
            checkbox.setToolTip(tr(tip_key))
            adv.addRow(checkbox)
            self._adv_widgets.append(checkbox)

        self.chk_fp16 = QCheckBox(tr("inpaint_fp16"))
        self.chk_fp16.setChecked(True)
        _add_adv_check("inpaint_tooltip_fp16", self.chk_fp16)

        self.chk_feather = QCheckBox(tr("inpaint_feather"))
        self.chk_feather.setChecked(True)
        _add_adv_check("inpaint_tooltip_feather", self.chk_feather)

        self.sb_neighbor = QSpinBox()
        self.sb_neighbor.setRange(2, 30); self.sb_neighbor.setValue(10)
        _add_adv_row("inpaint_label_neighbor",
                     "inpaint_tooltip_neighbor", self.sb_neighbor)

        self.sb_ref_stride = QSpinBox()
        self.sb_ref_stride.setRange(1, 30); self.sb_ref_stride.setValue(10)
        _add_adv_row("inpaint_label_ref_stride",
                     "inpaint_tooltip_ref_stride", self.sb_ref_stride)

        self.sb_subvideo = QSpinBox()
        self.sb_subvideo.setRange(10, 200); self.sb_subvideo.setValue(80)
        _add_adv_row("inpaint_label_subvideo",
                     "inpaint_tooltip_subvideo", self.sb_subvideo)

        self.sb_raft = QSpinBox()
        self.sb_raft.setRange(5, 60); self.sb_raft.setValue(20)
        _add_adv_row("inpaint_label_raft_iter",
                     "inpaint_tooltip_raft_iter", self.sb_raft)

        self.sb_dilation = QSpinBox()
        self.sb_dilation.setRange(0, 20); self.sb_dilation.setValue(4)
        _add_adv_row("inpaint_label_dilation",
                     "inpaint_tooltip_dilation", self.sb_dilation)

        params_lay.addLayout(adv)
        params_lay.addStretch(1)

        # Wire & hide-by-default. Same pattern as the denoise tab.
        self.dev_mode.toggled.connect(self._apply_dev_mode)
        self._apply_dev_mode(False)

        scroll = QScrollArea()
        scroll.setWidget(params_box)
        scroll.setWidgetResizable(True)
        lay.addWidget(scroll, 1)

        return right

    def _apply_dev_mode(self, on: bool):
        """Show/hide advanced rows. Model + max_batch stay visible regardless."""
        for w in self._adv_widgets:
            w.setVisible(on)

    # ------------------------------------------------------------------
    # File / input handling
    # ------------------------------------------------------------------
    def _pick_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "비디오 선택", "",
            "비디오 (*.mp4 *.mov *.mkv *.avi *.webm *.m4v);;모든 파일 (*.*)")
        if path:
            self._set_video(path)

    def _set_video(self, path: str):
        if not os.path.isfile(path):
            QMessageBox.warning(self, "Inpaint", f"파일이 존재하지 않습니다:\n{path}")
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
            QMessageBox.critical(self, "Inpaint", f"비디오 열기 실패:\n{e}")
            return

        self.video_info = (h, w, fps, fc)
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
        # Stale mask — new video may have different dimensions.
        self.current_mask = None
        self.mask_status.setText(tr("inpaint_no_mask"))
        self.mask_thumb.clear()
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
        candidate = os.path.join(_OUTPUT_DIR, f"{base}_inpainted.mp4")
        if not os.path.exists(candidate):
            return candidate
        n = 1
        while True:
            candidate = os.path.join(
                _OUTPUT_DIR, f"{base}_inpainted_{n:03d}.mp4")
            if not os.path.exists(candidate):
                return candidate
            n += 1

    # ------------------------------------------------------------------
    # Preset → full-frame mask
    # ------------------------------------------------------------------
    def _apply_preset(self, name: str):
        if not self.video_path or self.video_info is None:
            QMessageBox.warning(self, "Inpaint", "비디오를 먼저 선택하세요.")
            return
        preset = _PRESETS.get(name)
        if not preset:
            return

        vh, vw, _, _ = self.video_info
        is_portrait = vh > vw
        filename = preset["portrait"] if is_portrait else preset["landscape"]
        anchor = preset.get("anchor", "top-left")

        preset_path = os.path.join(_PRESETS_DIR, filename)
        if not os.path.exists(preset_path):
            QMessageBox.warning(self, "Inpaint",
                                f"프리셋 파일이 없습니다:\n{preset_path}")
            return

        preset_mask = cv2.imread(preset_path, cv2.IMREAD_GRAYSCALE)
        if preset_mask is None:
            QMessageBox.warning(self, "Inpaint",
                                f"프리셋 마스크를 읽지 못했습니다:\n{preset_path}")
            return
        ph, pw = preset_mask.shape[:2]

        preset_ar = pw / ph if ph else 1.0
        video_ar = vw / vh if vh else 1.0
        ar_match = abs(preset_ar - video_ar) / max(preset_ar, 1e-9) < 0.01

        if ar_match:
            mask = cv2.resize(preset_mask, (vw, vh),
                              interpolation=cv2.INTER_LINEAR)
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            scale_info = f"{preset_ar:.3f} (비율 일치, 스케일링)"
        else:
            scale = min(vw / pw, vh / ph)
            new_w, new_h = int(pw * scale), int(ph * scale)
            scaled = cv2.resize(preset_mask, (new_w, new_h),
                                interpolation=cv2.INTER_LINEAR)
            _, scaled = cv2.threshold(scaled, 127, 255, cv2.THRESH_BINARY)
            mask = np.zeros((vh, vw), dtype=np.uint8)
            if anchor == "top-left":
                ox, oy = 0, 0
            elif anchor == "top-right":
                ox, oy = vw - new_w, 0
            elif anchor == "bottom-left":
                ox, oy = 0, vh - new_h
            else:  # bottom-right
                ox, oy = vw - new_w, vh - new_h
            mask[oy:oy + new_h, ox:ox + new_w] = scaled
            scale_info = (f"비율 다름 ({preset_ar:.3f} → {video_ar:.3f}, "
                          f"{anchor} 정렬)")

        if self.chk_feather.isChecked():
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.dilate(mask, kernel, iterations=1)
            mask = cv2.GaussianBlur(mask, (3, 3), 0)
            _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)

        self.current_mask = mask
        coverage = float((mask > 127).mean()) * 100.0
        self.mask_status.setText(
            f"{tr('inpaint_mask_set')} — {scale_info} · {coverage:.1f}%")
        self._update_mask_thumb(mask)
        feather_str = " +feather" if self.chk_feather.isChecked() else ""
        self._log(f"프리셋 적용: {name} · {scale_info}{feather_str}")

    def _update_mask_thumb(self, mask: np.ndarray):
        tw, th = 64, 36
        thumb = cv2.resize(mask, (tw, th), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(thumb, cv2.COLOR_GRAY2RGB)
        qimg = QImage(rgb.data, tw, th, rgb.strides[0], QImage.Format_RGB888)
        self.mask_thumb.setPixmap(QPixmap.fromImage(qimg.copy()))

    # ------------------------------------------------------------------
    # Run / cancel
    # ------------------------------------------------------------------
    def _run(self):
        if not self.video_path:
            QMessageBox.information(self, "Inpaint", "비디오를 먼저 선택하세요.")
            return
        if self.current_mask is None:
            QMessageBox.information(
                self, "Inpaint",
                "마스크가 없습니다 — 먼저 프리셋을 적용하세요.")
            return
        if not weights_available():
            QMessageBox.warning(
                self, "Inpaint",
                "ProPainter 가중치가 없습니다.\n"
                "setup.bat을 다시 실행하거나, weights/ 폴더를 확인하세요.")
            return

        out_path = self.output_edit.text().strip() or \
            self._default_output_path(self.video_path)
        if not out_path.lower().endswith(".mp4"):
            out_path = os.path.splitext(out_path)[0] + ".mp4"
        out_dir = os.path.dirname(out_path)
        if out_dir and not os.path.isdir(out_dir):
            try:
                os.makedirs(out_dir, exist_ok=True)
            except Exception as e:
                QMessageBox.critical(self, "Inpaint",
                                     f"출력 폴더 생성 실패:\n{e}")
                return

        settings = {
            "fp16":              self.chk_fp16.isChecked(),
            "neighbor_length":   self.sb_neighbor.value(),
            "ref_stride":        self.sb_ref_stride.value(),
            "subvideo_length":   self.sb_subvideo.value(),
            "raft_iter":         self.sb_raft.value(),
            "mask_dilation":     self.sb_dilation.value(),
            "max_batch_frames":  self.sb_batch_frames.value(),
        }
        device = get_device()
        self._log(f"처리 시작 → {out_path}")
        self._log(f"  device={device}, fp16={settings['fp16']}, "
                  f"neighbor={settings['neighbor_length']}, "
                  f"ref_stride={settings['ref_stride']}, "
                  f"subvideo={settings['subvideo_length']}")

        self.worker = InpaintWorker(
            self.video_path, self.current_mask, settings, out_path, device)
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
        self.preset_btn.setEnabled(not running)

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
                self.player.disable_compare()
                self.player.load(output_path)
                if self.video_path and os.path.exists(self.video_path):
                    self.player.enable_compare(self.video_path)
            except Exception as e:
                self._log(f"[WARN] 결과 미리보기 전환 실패: {e}")

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
        QMessageBox.critical(self, "Inpaint", msg)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def _log(self, msg: str):
        self.log.appendPlainText(msg)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def shutdown(self):
        """Called by MainWindow.closeEvent — cancel worker and release player."""
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

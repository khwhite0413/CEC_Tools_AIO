"""Inpaint tab — ProPainter-based video inpainting with PNG mask presets.

Brush / rectangle ROI deliberately omitted (future work). Mask input
comes from a fixed preset (currently: Dreamina AI watermark removal).
"""

import os
import shutil
import time

import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap, QPalette, QColor, QKeySequence
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                              QPushButton, QFileDialog, QProgressBar,
                              QPlainTextEdit, QMessageBox, QScrollArea,
                              QSplitter, QLineEdit, QGroupBox, QSpinBox,
                              QDoubleSpinBox, QCheckBox, QToolButton, QMenu,
                              QFrame, QFormLayout, QComboBox, QSizePolicy,
                              QDialog, QShortcut)

from app.i18n import tr
from app.video_player import VideoPlayer
from app.inpaint_worker import InpaintWorker
from app.mask_editor_dialog import MaskEditorDialog
from app.preview_dialog import PreviewDialog
from vdcore.config import get_device, weights_available
from fdcore.video_io import auto_format_from_codec
from fdcore.scene_detection import DEFAULT_SCENE_MSE_THRESHOLD


# Mirrors fdcore.video_io._STANDARD_FPS so the inpaint tab's FPS
# override drawer offers the exact rates the auto-snap would have
# chosen. Same list the denoise tab uses.
_STANDARD_FPS_OPTIONS = (
    (23.976, "23.976"),
    (24.0, "24"),
    (25.0, "25"),
    (29.97, "29.97"),
    (30.0, "30"),
    (50.0, "50"),
    (59.94, "59.94"),
    (60.0, "60"),
)


_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_MODULE_DIR, ".."))
_PRESETS_DIR = os.path.join(_PROJECT_ROOT, "presets")
_OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "output")
_CACHE_DIR = os.path.join(_PROJECT_ROOT, "cache")

# Frame count for the [미리보기] button — matches the Denoise tab's
# 80-frame preview so iteration on params is fast across both tabs.
PREVIEW_FRAMES = 80
_PREVIEW_CACHE_NAME = "inpaint_preview.mp4"
# Viewer-only copy of the final deliverable. Same rationale as the
# Denoise tab: cv2's VideoPlayer keeps a file handle and would lock the
# real output in output/ on Windows. MainWindow's preview-cache sweep
# (start + close) deletes this on next launch / shutdown.
_VIEWER_CACHE_NAME = "inpaint_view.mp4"

# ProRes profile choices — mirrors fdcore parameter_panel so all three
# tabs offer the same profile options.
_PRORES_KEYS = ("proxy", "lt", "standard", "hq", "4444", "4444xq")
_PRORES_LABELS = {
    "proxy": "Proxy", "lt": "LT", "standard": "422",
    "hq": "422 HQ", "4444": "4444", "4444xq": "4444 XQ",
}


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
        self.video_codec = ""    # populated by _set_video via fdcore probe
        self.current_mask = None  # uint8 H x W, full-frame
        # Name of the preset that produced current_mask. Set in
        # _apply_preset, cleared when the user draws via the brush
        # editor or loads a new video. Used by the live mask-dilation
        # slider to re-derive the preset mask on the fly so the A-key
        # overlay reflects the latest value.
        self._active_preset_name = None
        # Auto-detected source fps (raw VideoInfo.fps, pre-snap).
        # Drives the effective_fps_label readout. Set on _set_video.
        self._detected_fps = None
        # Last preview run output + the input it came from. Used by
        # "이전 미리보기 다시 보기" to reopen the modal PreviewDialog
        # without re-running the worker.
        self._last_preview_path = None
        self._last_preview_source = None
        self.worker = None
        # Distinguishes preview vs full run in _on_finished — preview
        # output lives in cache/ and shouldn't be re-staged.
        self._is_preview = False
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

        # Manual brush editor — opens a modal dialog with the video's
        # first frame and brush/rect/ellipse/eraser tools so the user
        # can paint a custom mask when no preset fits.
        self.btn_edit_mask = QPushButton(tr("inpaint_btn_edit_mask"))
        self.btn_edit_mask.setFixedHeight(28)
        self.btn_edit_mask.clicked.connect(self._open_mask_editor)
        preset_row.addWidget(self.btn_edit_mask, 0)

        self.mask_status = QLabel(tr("inpaint_no_mask"))
        self.mask_status.setStyleSheet(
            "color: rgba(180,184,200,200); padding-left: 6px;")
        preset_row.addWidget(self.mask_status, 1)
        lay.addLayout(preset_row)

        # Page-scoped A shortcut: toggle the alpha-mask overlay on the
        # viewer above. WidgetWithChildrenShortcut limits scope so the
        # key doesn't fire while the user is on a different tab — they'd
        # have focus inside that other tab's widgets, not ours.
        sc_overlay = QShortcut(Qt.Key_A, self)
        sc_overlay.setContext(Qt.WidgetWithChildrenShortcut)
        sc_overlay.activated.connect(self._toggle_mask_overlay)

        btn_row = QHBoxLayout()
        self.btn_preview = QPushButton(tr("btn_preview"))
        self.btn_preview.clicked.connect(lambda: self._run(preview=True))
        # Re-open the last preview dialog without re-running the worker
        # (same UX as the denoise tab).
        self.btn_show_preview = QPushButton(tr("btn_show_preview"))
        self.btn_show_preview.clicked.connect(self._show_last_preview)
        self.btn_show_preview.setEnabled(False)
        self.btn_run = QPushButton(tr("btn_run"))
        self.btn_run.setStyleSheet("font-weight: bold;")
        self.btn_run.clicked.connect(self._run)
        self.btn_cancel = QPushButton(tr("btn_cancel"))
        self.btn_cancel.clicked.connect(self._cancel)
        self.btn_cancel.setEnabled(False)
        btn_row.addWidget(self.btn_preview)
        btn_row.addWidget(self.btn_show_preview)
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

        # QGroupBox (plain — no setStyleSheet) so Qt keeps the QComboBox
        # and QSpinBox children in Fusion-native rendering. Setting a
        # stylesheet on the parent QFrame would push descendants into
        # QSS rendering mode, which makes the dropdown chrome go solid
        # black. FD's Output-Settings card uses the same plain-QGroupBox
        # approach for exactly this reason; only its Temporal-Flow-
        # Average card (which has no always-visible dropdowns) uses the
        # stylesheet-driven QFrame#TfaCard pattern.
        params_box = QGroupBox(tr("inpaint_section_params"))
        params_box.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        params_lay = QVBoxLayout(params_box)
        params_lay.setSpacing(8)

        head = QHBoxLayout()
        head.addStretch(1)
        self.dev_mode = QCheckBox(tr("label_dev_mode"))
        self.dev_mode.setToolTip(tr("tooltip_dev_mode"))
        head.addWidget(self.dev_mode)
        params_lay.addLayout(head)

        # --- Always-visible row: model + max_batch_frames ---
        basic = QFormLayout()
        basic.setLabelAlignment(Qt.AlignRight)
        # SetMinAndMaxSize makes the form report a tight bounding height
        # so the parent Card frame hugs only what's visible — without this
        # FD-style constraint the Card grows past its content.
        basic.setSizeConstraint(QFormLayout.SetMinAndMaxSize)
        basic.setContentsMargins(0, 0, 0, 0)
        basic.setHorizontalSpacing(10)
        basic.setVerticalSpacing(6)

        self.model_combo = QComboBox()
        self.model_combo.addItem(tr("inpaint_model_propainter"), "propainter")
        self.model_combo.setToolTip(tr("inpaint_tooltip_model"))
        lbl_model = QLabel(tr("inpaint_label_model"))
        lbl_model.setToolTip(tr("inpaint_tooltip_model"))
        basic.addRow(lbl_model, self.model_combo)

        self.scene_detection_mode = QComboBox()
        self.scene_detection_mode.addItem(tr("scene_detection_mse"), "mse")
        self.scene_detection_mode.addItem(
            tr("scene_detection_adaptive"), "pyscenedetect")
        self.scene_detection_mode.setToolTip(tr("scene_detection_tooltip"))
        basic.addRow(tr("scene_detection_label"), self.scene_detection_mode)

        self.sb_batch_frames = QSpinBox()
        self.sb_batch_frames.setRange(0, 5000)
        self.sb_batch_frames.setValue(0)
        self.sb_batch_frames.setToolTip(tr("inpaint_tooltip_batch_frames"))
        lbl_batch = QLabel(tr("inpaint_label_batch_frames"))
        lbl_batch.setToolTip(tr("inpaint_tooltip_batch_frames"))
        basic.addRow(lbl_batch, self.sb_batch_frames)

        # Mask expansion (formerly dev-mode-only). Promoted to the
        # always-visible block because users routinely need to nudge
        # this to cover watermark edges. Value applies as a 2D cv2
        # dilation to the preset mask in _apply_preset, NOT as the
        # ProPainter-internal scipy dilation — that internal pass is
        # now zeroed out at worker-submit time to avoid double-dilation.
        # Range mirrors batch_page (0..50 px). Default 4 was the old
        # dev-mode default; keeping it preserves existing visual results.
        self.sb_dilation = QSpinBox()
        self.sb_dilation.setRange(0, 50)
        self.sb_dilation.setValue(4)
        self.sb_dilation.setSuffix(" px")
        self.sb_dilation.setToolTip(tr("inpaint_tooltip_dilation"))
        self.sb_dilation.valueChanged.connect(self._on_mask_settings_changed)
        lbl_dilation = QLabel(tr("inpaint_label_dilation"))
        lbl_dilation.setToolTip(tr("inpaint_tooltip_dilation"))
        basic.addRow(lbl_dilation, self.sb_dilation)

        # Effective FPS readout — mirrors the denoise tab's same-named
        # label. Always visible; reflects auto-snapped source fps when
        # Dev mode + Auto is on, or the override value otherwise.
        self.effective_fps_label = QLabel(tr("fps_unknown"))
        self.effective_fps_label.setStyleSheet(
            "color: rgba(200, 205, 220, 220); font-weight: 500;")
        basic.addRow(tr("label_input_fps"), self.effective_fps_label)

        params_lay.addLayout(basic)

        # --- Dev-mode-only rows (hidden by default) ---
        # Wrapped in a QWidget so QFormLayout's row geometry collapses
        # when dev_mode is off. Hiding individual children would still
        # leave each row's vertical slot reserved.
        self._adv_widget = QWidget()
        adv = QFormLayout(self._adv_widget)
        adv.setLabelAlignment(Qt.AlignRight)
        adv.setContentsMargins(0, 0, 0, 0)
        adv.setSizeConstraint(QFormLayout.SetMinAndMaxSize)
        adv.setHorizontalSpacing(10)
        adv.setVerticalSpacing(6)

        def _add_adv_row(label_key: str, tip_key: str, widget):
            widget.setToolTip(tr(tip_key))
            lbl = QLabel(tr(label_key))
            lbl.setToolTip(tr(tip_key))
            adv.addRow(lbl, widget)

        def _add_adv_check(tip_key: str, checkbox: QCheckBox):
            checkbox.setToolTip(tr(tip_key))
            adv.addRow(checkbox)

        self.chk_fp16 = QCheckBox(tr("inpaint_fp16"))
        self.chk_fp16.setChecked(True)
        _add_adv_check("inpaint_tooltip_fp16", self.chk_fp16)

        # sb_dilation was promoted to the basic block above. The
        # former "feather" checkbox was removed entirely — its
        # implementation (1px dilate + Gaussian + re-threshold) just
        # produced a slightly-rounded BINARY mask, not a real soft-
        # alpha edge, and downstream binarizes anyway, so the toggle
        # was misleading. Kept out of the dev section here so there's
        # exactly one widget instance feeding settings().

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

        # FPS override drawer — same Auto checkbox + dropdown pattern as
        # the denoise tab. Forwarded to the writer's -r so AI sources
        # tagged at weird rates (24.04 etc.) can be locked to a clean
        # standard rate without changing the actual frame timing.
        fps_row = QWidget()
        fps_h = QHBoxLayout(fps_row)
        fps_h.setContentsMargins(0, 0, 0, 0)
        fps_h.setSpacing(8)
        self.fps_auto_check = QCheckBox(tr("label_fps_auto"))
        self.fps_auto_check.setChecked(True)
        self.fps_auto_check.setToolTip(tr("tooltip_fps_override"))
        self.fps_override = QComboBox()
        for value, text in _STANDARD_FPS_OPTIONS:
            self.fps_override.addItem(text, value)
        self.fps_override.addItem(tr("fps_custom"), -1.0)
        self.fps_override.setCurrentIndex(
            _STANDARD_FPS_OPTIONS.index((24.0, "24")))
        self.fps_override.setEnabled(False)
        self.fps_override.setToolTip(tr("tooltip_fps_override"))

        self.fps_custom_spin = QDoubleSpinBox()
        self.fps_custom_spin.setRange(0.001, 1000.0)
        self.fps_custom_spin.setDecimals(3)
        self.fps_custom_spin.setSingleStep(1.0)
        self.fps_custom_spin.setValue(24.000)
        self.fps_custom_spin.setToolTip(tr("tooltip_fps_custom"))
        self.fps_custom_spin.setVisible(False)
        self.fps_custom_spin.setEnabled(False)

        fps_h.addWidget(self.fps_auto_check)
        fps_h.addWidget(self.fps_override, 1)
        fps_h.addWidget(self.fps_custom_spin, 1)
        adv.addRow(tr("label_fps_override"), fps_row)

        def _on_fps_auto_toggled(checked: bool):
            self.fps_override.setEnabled(not checked)
            is_custom = (not checked) and (self.fps_override.currentData() == -1.0)
            self.fps_custom_spin.setEnabled(is_custom)
            self.fps_custom_spin.setVisible(is_custom)

        def _on_fps_override_changed(_idx: int):
            is_custom = (not self.fps_auto_check.isChecked()) and (self.fps_override.currentData() == -1.0)
            self.fps_custom_spin.setEnabled(is_custom)
            self.fps_custom_spin.setVisible(is_custom)
            self._refresh_effective_fps_label()

        self.fps_auto_check.toggled.connect(_on_fps_auto_toggled)
        self.fps_auto_check.toggled.connect(self._refresh_effective_fps_label)
        self.fps_override.currentIndexChanged.connect(_on_fps_override_changed)
        self.fps_custom_spin.valueChanged.connect(self._refresh_effective_fps_label)

        # sb_dilation moved to the basic block (always visible).

        params_lay.addWidget(self._adv_widget)

        # Wire & hide-by-default. Same pattern as the denoise tab.
        self.dev_mode.toggled.connect(self._apply_dev_mode)
        self.dev_mode.toggled.connect(self._refresh_effective_fps_label)
        self._apply_dev_mode(False)

        # Plain QGroupBox — same reason as params_box above.
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
        idx = _PRORES_KEYS.index("4444")
        self.prores_profile.setCurrentIndex(idx)
        self.prores_profile.setToolTip(tr("io_tooltip_prores_profile"))
        lbl_profile = QLabel(tr("io_label_prores_profile"))
        lbl_profile.setToolTip(tr("io_tooltip_prores_profile"))
        prof_form.addRow(lbl_profile, self.prores_profile)
        fmt_outer.addWidget(self.prores_drawer)
        self.prores_drawer.setVisible(False)

        # Pack params + format into a single scrollable column, top-
        # anchored. Without the trailing stretch the params card was
        # absorbing all vertical space and pushing the format card to
        # the bottom edge.
        # Do NOT call setStyleSheet on inner / scroll. Any QSS on a
        # parent forces every descendant — including QComboBox and
        # QSpinBox — into QSS rendering mode, which makes the dropdown
        # chrome look solid black instead of Fusion-native.
        inner = QWidget()
        inner_lay = QVBoxLayout(inner)
        inner_lay.setContentsMargins(0, 0, 0, 0)
        inner_lay.setSpacing(12)
        inner_lay.addWidget(params_box)
        inner_lay.addWidget(fmt_box)
        inner_lay.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidget(inner)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        lay.addWidget(scroll, 1)

        return right

    # ------------------------------------------------------------------
    # Output format handling
    # ------------------------------------------------------------------
    def _resolved_format(self) -> str:
        """Return the concrete codec ("h264" or "prores"). Resolves "auto"
        via the source codec cached during _set_video."""
        fmt = (self.format_combo.currentData()
               if hasattr(self, "format_combo") else "auto")
        if fmt == "auto":
            return auto_format_from_codec(self.video_codec)
        return fmt

    def _on_format_changed(self):
        """Reveal the ProRes profile drawer when needed and keep the
        visible output-path extension in sync (mp4 ↔ mov)."""
        fmt = self._resolved_format()
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
        """Show/hide advanced rows. Model + max_batch stay visible regardless.
        Toggling the wrapping QWidget collapses the dev-mode rows entirely
        so the card hugs only its visible content (no reserved empty space)."""
        self._adv_widget.setVisible(on)

    # ------------------------------------------------------------------
    # FPS override / display — mirrors the denoise tab's parameter panel.
    # ------------------------------------------------------------------
    def set_input_fps(self, fps) -> None:
        """Cache the auto-detected snapped fps and redraw the readout."""
        self._detected_fps = (
            float(fps) if (fps is not None and fps > 0) else None)
        self._refresh_effective_fps_label()

    def _get_override_fps(self) -> float:
        val = float(self.fps_override.currentData() or 0.0)
        if val <= 0:
            return float(self.fps_custom_spin.value())
        return val

    def _refresh_effective_fps_label(self) -> None:
        """Update the effective-fps readout based on current state.
        Reflects the override value only when Dev mode is on AND Auto
        is unchecked — otherwise the auto-detected snapped value."""
        if self._detected_fps is None:
            self.effective_fps_label.setText(tr("fps_unknown"))
            return
        if (self.dev_mode.isChecked()
                and not self.fps_auto_check.isChecked()):
            val = self._get_override_fps()
            self.effective_fps_label.setText(f"{val:.3f}")
        else:
            self.effective_fps_label.setText(f"{self._detected_fps:.3f}")

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
        # Light codec probe so "Auto" format picks the right codec
        # without re-encoding more than necessary in chained workflows.
        # Also pull the SNAPPED effective fps (24.04 -> 24.0 etc.) so
        # the effective_fps_label shows the value the writer will
        # actually emit, not the raw container metadata.
        self.video_codec = ""
        effective_fps = fps
        try:
            from fdcore.video_io import VideoInfo as _FdInfo
            _fd = _FdInfo(path)
            self.video_codec = (_fd.codec_name or "").lower()
            effective_fps = getattr(_fd, "effective_fps", fps)
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
        # Refresh extension based on the freshly probed codec (Auto may
        # now resolve to ProRes if source is ProRes).
        if hasattr(self, "format_combo"):
            self._on_format_changed()
        # Stale mask — new video may have different dimensions.
        self.current_mask = None
        self._active_preset_name = None
        self.mask_status.setText(tr("inpaint_no_mask"))
        self.player.clear_overlay()
        # Refresh the effective-FPS readout. self._detected_fps stores
        # the snapped value so the label tracks what the writer will
        # actually use; the dev-mode override (when on) replaces it.
        self.set_input_fps(effective_fps)
        # Drop any cached preview reference — it belongs to the prior input.
        self._last_preview_path = None
        self._last_preview_source = None
        if hasattr(self, "btn_show_preview"):
            self.btn_show_preview.setEnabled(False)
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

        # Mask expansion — pre-applied here so the worker's
        # mask_dilation setting is zeroed at submit time (ProPainter's
        # internal scipy dilation would otherwise double-expand on
        # top of this).
        dilation_px = int(self.sb_dilation.value())
        if dilation_px > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.dilate(mask, kernel, iterations=dilation_px)

        self.current_mask = mask
        self._active_preset_name = name
        coverage = float((mask > 127).mean()) * 100.0
        self.mask_status.setText(
            f"{tr('inpaint_mask_set')} — {scale_info} · {coverage:.1f}% "
            f"({tr('inpaint_overlay_hint')})")
        # Push the mask up to the viewer overlay (off by default — user
        # presses A to toggle it on). Replaces the old 64x36 thumbnail
        # box that used to live next to the preset button.
        self.player.set_overlay_mask(mask)
        dil_str = f" +dil{dilation_px}" if dilation_px > 0 else ""
        self._log(f"프리셋 적용: {name} · {scale_info}{dil_str}")

    def _on_mask_settings_changed(self, *_):
        """Live re-apply when sb_dilation changes AND a preset is
        currently active. Brush-edited masks aren't touched
        (we shouldn't dilate user-drawn pixels — that's the same
        convention the batch tab follows)."""
        if self._active_preset_name is None:
            return
        # Calling _apply_preset rebuilds current_mask and pushes the new
        # overlay so the A-key preview reflects the change live. The
        # mask_status label and log line update too — that's intentional
        # because the user *did* change a mask-affecting setting.
        self._apply_preset(self._active_preset_name)

    def _toggle_mask_overlay(self):
        if self.current_mask is None:
            self._log("[A] 마스크가 없습니다 — 프리셋 적용 또는 마스크 편집 먼저 하세요.")
            return
        visible = self.player.toggle_overlay()
        self._log(f"[A] 마스크 오버레이 {'ON' if visible else 'OFF'}")

    # ------------------------------------------------------------------
    # Run / cancel
    # ------------------------------------------------------------------
    def _gather_settings(self, output_format=None):
        return {
            "fp16": self.chk_fp16.isChecked(),
            "neighbor_length": self.sb_neighbor.value(),
            "ref_stride": self.sb_ref_stride.value(),
            "subvideo_length": self.sb_subvideo.value(),
            "raft_iter": self.sb_raft.value(),
            # Dilation is already baked into current_mask.
            "mask_dilation": 0,
            "max_batch_frames": self.sb_batch_frames.value(),
            "scene_detection_mode": (
                self.scene_detection_mode.currentData() or "mse"),
            "scene_threshold": DEFAULT_SCENE_MSE_THRESHOLD,
            "output_format": output_format or self._resolved_format(),
            "prores_profile": self.prores_profile.currentData() or "4444",
            "fps_override": (self._get_override_fps()
                             if (self.dev_mode.isChecked()
                                 and not self.fps_auto_check.isChecked())
                             else 0.0),
        }

    def _run(self, preview: bool = False):
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

        # Preview goes to cache as h264 mp4 (fast, throwaway). Full
        # run goes to the user-chosen output path in the user-chosen
        # codec.
        if preview:
            os.makedirs(_CACHE_DIR, exist_ok=True)
            out_path = os.path.join(_CACHE_DIR, _PREVIEW_CACHE_NAME)
            fmt = "h264"
            frame_cap = PREVIEW_FRAMES
        else:
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
                    QMessageBox.critical(self, "Inpaint",
                                         f"출력 폴더 생성 실패:\n{e}")
                    return
            frame_cap = -1

        settings = self._gather_settings(fmt)
        device = get_device()
        kind = f"미리보기 (앞 {PREVIEW_FRAMES}프레임)" if preview else "처리 시작"
        self._log(f"{kind} → {out_path}")
        self._log(f"  device={device}, fp16={settings['fp16']}, "
                  f"neighbor={settings['neighbor_length']}, "
                  f"ref_stride={settings['ref_stride']}, "
                  f"subvideo={settings['subvideo_length']}")

        self._is_preview = preview
        self.worker = InpaintWorker(
            self.video_path, self.current_mask, settings, out_path, device,
            frame_cap=frame_cap)
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
        self.btn_preview.setEnabled(not running)
        # btn_show_preview stays disabled while running; flips back
        # when finished only if a preview output is available.
        self.btn_show_preview.setEnabled(
            (not running) and bool(self._last_preview_path)
            and os.path.exists(self._last_preview_path or ""))
        self.btn_cancel.setEnabled(running)
        self.preset_btn.setEnabled(not running)
        self.btn_edit_mask.setEnabled(not running)

    # ------------------------------------------------------------------
    # Manual brush mask editor
    # ------------------------------------------------------------------
    def _open_mask_editor(self):
        if not self.video_path:
            QMessageBox.information(self, "Inpaint",
                                     "비디오를 먼저 선택하세요.")
            return
        try:
            dlg = MaskEditorDialog(
                self.video_path, self.current_mask, parent=self)
        except Exception as e:
            QMessageBox.critical(self, "Inpaint",
                                  f"마스크 편집기 열기 실패:\n{e}")
            return
        if dlg.exec_() != QDialog.Accepted:
            return
        new_mask = dlg.get_mask()
        if new_mask is None or not np.any(new_mask > 0):
            QMessageBox.information(
                self, "Inpaint",
                "그려진 영역이 없습니다. 마스크가 변경되지 않았습니다.")
            return
        self.current_mask = new_mask
        # Brush-drawn mask leaves preset-tracking state cleared so the
        # dilation slider doesn't silently overwrite the user's
        # hand-drawn pixels on the next valueChanged signal. Same
        # convention as the batch tab.
        self._active_preset_name = None
        coverage = float((new_mask > 127).mean()) * 100.0
        self.mask_status.setText(
            f"{tr('inpaint_mask_set')} — 브러시 편집 · {coverage:.1f}% "
            f"({tr('inpaint_overlay_hint')})")
        self.player.set_overlay_mask(new_mask)
        self._log(f"브러시 마스크 적용 ({new_mask.shape[1]}×{new_mask.shape[0]}, "
                  f"커버리지 {coverage:.1f}%)")

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
        was_preview = self._is_preview
        self._is_preview = False
        self.worker = None
        if not output_path or not os.path.exists(output_path):
            return
        if was_preview:
            # Preview run — same UX as the denoise tab: open a modal
            # PreviewDialog with the source as the wipe "before" and
            # the inpaint output as "after". The main viewer is left
            # untouched so the user can keep scrubbing the source
            # while comparing in the dialog.
            self._last_preview_path = output_path
            self._last_preview_source = self.video_path
            self.btn_show_preview.setEnabled(True)
            dlg = PreviewDialog(self.video_path, output_path,
                                self, stage_label="인페인트")
            dlg.exec_()
            return
        # Full run — stage a viewer-only copy in cache/ so the real
        # deliverable in output/ stays writable while the player keeps
        # its cv2 handle.
        try:
            viewer_path = self._stage_for_viewer(output_path)
            self.player.disable_compare()
            self.player.load(viewer_path)
            if self.video_path and os.path.exists(self.video_path):
                self.player.enable_compare(self.video_path)
            self._log("메인 뷰어가 결과물로 전환됨 (와이프로 원본 비교 가능).")
        except Exception as e:
            self._log(f"[WARN] 결과 미리보기 전환 실패: {e}")

    def _show_last_preview(self):
        path = self._last_preview_path
        source = self._last_preview_source or self.video_path
        if not path or not os.path.exists(path):
            self.btn_show_preview.setEnabled(False)
            self._last_preview_path = None
            self._last_preview_source = None
            QMessageBox.information(
                self, "Inpaint",
                "미리보기 파일을 찾을 수 없습니다. 다시 생성해 주세요.")
            return
        dlg = PreviewDialog(source, path, self, stage_label="인페인트")
        dlg.exec_()

    @staticmethod
    def _stage_for_viewer(output_path: str) -> str:
        """Copy the deliverable into cache/inpaint_view.<ext> so the
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
            # Viewer-locking the real output is better than losing the
            # ability to show the result at all.
            return output_path

    def _on_cancelled(self):
        self._set_running(False)
        self._is_preview = False
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
        self._is_preview = False
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

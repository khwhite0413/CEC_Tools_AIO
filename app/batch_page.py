"""Batch tab — Adobe Media Encoder-style queue + 3-stage pipeline.

Left pane is the video viewer (currently-selected clip in the queue).
Right pane stacks the queue, per-stage settings cards, and the run
controls in a scroll area. The QSplitter defaults to roughly 45 / 55 so
the viewer reads as the smaller half — but the user can drag the handle.

Per-stage codec defaults to H.264 (universal). Selecting "ProRes (MOV)"
reveals an inline profile dropdown so the user can pin 422 / 4444 / etc.
"""

from __future__ import annotations

import os
from typing import List, Optional

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QDragEnterEvent, QDropEvent, QBrush, QColor, QPalette
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                              QLabel, QListWidget, QListWidgetItem, QSplitter,
                              QScrollArea, QSpinBox, QComboBox, QGroupBox,
                              QFileDialog, QProgressBar, QMessageBox,
                              QAbstractItemView, QFormLayout, QSizePolicy,
                              QLineEdit, QMenu, QShortcut, QCheckBox)

from app.mask_editor_dialog import MaskEditorDialog


# Spinner used in queue-row "running" state. Braille block animation
# matches the visual density of monospace text columns so the row width
# doesn't jitter as the frame cycles. Total cycle ≈ 1s at 100ms tick.
_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

# Mirrors upscale_page._STANDARD_FPS_OPTIONS so the batch tab's upscale
# FPS override drawer offers the exact rates the single-tab does.
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

# Queue-row foreground colours. Picked to match the bottom progress bar
# palette (green done / red cancel) for visual consistency. Orange is a
# distinct hue so "currently running" is unmistakable next to the others.
_COLOR_RUNNING = QColor("#f0a35a")
_COLOR_DONE    = QColor("#a5d6a7")
_COLOR_ERROR   = QColor("#e57373")

from app.i18n import tr
from app.video_player import VideoPlayer
from app.batch_orchestrator import (BatchOrchestrator, STAGE_INPAINT,
                                     STAGE_DENOISE, STAGE_UPSCALE,
                                     _compute_dreamina_mask)


def _probe_meta(path: str) -> str:
    """Probe a video for a short single-line metadata string.

    Uses fdcore.VideoInfo so the surfaced fps matches what the workers
    will *actually* use (effective_fps post-snap, not the raw container
    metadata). Falls back silently if probing fails — the user still
    gets the filename + status, just no resolution.
    """
    try:
        from fdcore.video_io import VideoInfo
        info = VideoInfo(path)
        eff = getattr(info, "effective_fps", info.fps)
        dur = info.duration if info.duration else 0.0
        audio = "audio" if info.has_audio else "no audio"
        return (f"{info.width}×{info.height} @ {eff:.2f}fps · "
                f"{info.frame_count}f · {dur:.1f}s · {audio}")
    except Exception:
        return ""


_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_MODULE_DIR, ".."))


# ProRes profile keys mirror inpaint/denoise pages so all four tabs read
# the same way to a user comparing profile lists across them.
_PRORES_KEYS = ("proxy", "lt", "standard", "hq", "4444", "4444xq")
_PRORES_LABELS = {
    "proxy": "Proxy", "lt": "LT", "standard": "422",
    "hq": "422 HQ", "4444": "4444", "4444xq": "4444 XQ",
}

_VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v")


def _is_video(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in _VIDEO_EXTS


class _CodecCard(QWidget):
    """Mini card: codec radio + ProRes profile drawer.

    Used three times on the batch page (one per stage). Default codec is
    H.264 — the user said "보편적인게 최고" so all three start there. The
    profile combo is hidden until ProRes is chosen, mirroring the
    Inpaint/Upscale "ProRes drawer" pattern.
    """

    def __init__(self, title: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        lay = QFormLayout(self)
        lay.setContentsMargins(0, 4, 0, 4)
        lay.setSpacing(6)

        self.codec_combo = QComboBox()
        self.codec_combo.addItem(tr("batch_codec_h264"), "h264")
        self.codec_combo.addItem(tr("batch_codec_prores"), "prores")
        self.codec_combo.currentIndexChanged.connect(self._on_codec_changed)
        lay.addRow(QLabel(tr("batch_label_codec") + ":"), self.codec_combo)

        self.profile_combo = QComboBox()
        for key in _PRORES_KEYS:
            self.profile_combo.addItem(_PRORES_LABELS[key], key)
        # Default to 4444 — the highest fidelity profile, used as the
        # quality-preserving choice in the other tabs' parameter panels.
        self.profile_combo.setCurrentIndex(_PRORES_KEYS.index("4444"))
        self.profile_label = QLabel(tr("batch_label_profile") + ":")
        lay.addRow(self.profile_label, self.profile_combo)
        self._set_profile_visible(False)

    def _on_codec_changed(self, _idx: int):
        self._set_profile_visible(self.codec() == "prores")

    def _set_profile_visible(self, visible: bool):
        self.profile_label.setVisible(visible)
        self.profile_combo.setVisible(visible)

    def codec(self) -> str:
        return self.codec_combo.currentData() or "h264"

    def prores_profile(self) -> str:
        return self.profile_combo.currentData() or "4444"


class _QueueList(QListWidget):
    """QListWidget that accepts file drops for the queue."""

    files_dropped = pyqtSignal(list)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setAlternatingRowColors(True)

    def dragEnterEvent(self, e: QDragEnterEvent):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
        else:
            super().dragEnterEvent(e)

    def dragMoveEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
        else:
            super().dragMoveEvent(e)

    def dropEvent(self, e: QDropEvent):
        if not e.mimeData().hasUrls():
            return super().dropEvent(e)
        paths: List[str] = []
        for url in e.mimeData().urls():
            p = url.toLocalFile()
            if not p:
                continue
            if os.path.isdir(p):
                for name in sorted(os.listdir(p)):
                    full = os.path.join(p, name)
                    if os.path.isfile(full) and _is_video(full):
                        paths.append(full)
            elif _is_video(p):
                paths.append(p)
        if paths:
            self.files_dropped.emit(paths)
        e.acceptProposedAction()


class BatchPage(QWidget):
    """4th tab — full automation across all queued clips."""

    log = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setAcceptDrops(True)

        self._files: List[str] = []     # canonical sorted list
        # path -> metadata display string (probed once on add, reused on
        # every queue refresh). Files that fail probing simply map to "".
        self._meta: dict = {}
        # path -> custom mask ndarray (uint8 H x W). Volatile by design
        # — lost on app restart. Overrides the auto-Dreamina preset for
        # that specific clip during the inpaint stage. Queue rows show
        # a ✱ marker when an entry exists here.
        self._masks: dict = {}
        self._orchestrator: Optional[BatchOrchestrator] = None
        # Stage names enabled for the current run, in canonical order.
        # Used by _on_progress() to derive stage rank for the overall
        # percentage bar. Filled in _run() right before orchestrator start.
        self._active_stages: List[str] = []

        # Per-row spinner state. _running_idx == -1 when nothing is in
        # flight; >=0 means that queue row is the one being processed.
        # The spinner timer ticks every 100ms regardless of how often
        # the worker emits frame progress — keeps the animation smooth
        # even on slow CPU stages.
        self._running_idx = -1
        self._running_stage = ""
        self._running_pct = 0.0
        self._spinner_idx = 0
        self._spinner_timer = QTimer(self)
        self._spinner_timer.setInterval(100)
        self._spinner_timer.timeout.connect(self._on_spinner_tick)

        root = QHBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 16)
        root.setSpacing(12)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter, 1)

        splitter.addWidget(self._build_left())
        splitter.addWidget(self._build_right())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        # Viewer takes slightly less than the queue/settings column —
        # roughly 45 / 55 at a 1320-wide window. User can drag to taste.
        splitter.setSizes([580, 700])
        self._splitter = splitter

        # A-key toggles the alpha-mask overlay on the batch viewer for
        # the currently-selected clip. Same UX as the single Inpaint
        # tab; WidgetWithChildrenShortcut so the key doesn't leak
        # across to other tabs (focus has to be inside this page).
        sc_overlay = QShortcut(Qt.Key_A, self)
        sc_overlay.setContext(Qt.WidgetWithChildrenShortcut)
        sc_overlay.activated.connect(self._toggle_mask_overlay)

    # ------------------------------------------------------------------
    # Layout — left (viewer)
    # ------------------------------------------------------------------
    def _build_left(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        self.player = VideoPlayer(click_to_load=False)
        lay.addWidget(self.player, 1)

        self.viewer_label = QLabel(tr("batch_status_idle"))
        self.viewer_label.setWordWrap(True)
        self.viewer_label.setStyleSheet(
            "color: rgba(200,204,220,180); font-size: 9pt;")
        lay.addWidget(self.viewer_label, 0)

        return w

    # ------------------------------------------------------------------
    # Layout — right (queue + settings + run)
    # ------------------------------------------------------------------
    def _build_right(self) -> QWidget:
        outer = QWidget()
        outer_lay = QVBoxLayout(outer)
        outer_lay.setContentsMargins(0, 0, 0, 0)
        outer_lay.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        inner = QWidget()
        scroll.setWidget(inner)
        outer_lay.addWidget(scroll, 1)

        lay = QVBoxLayout(inner)
        lay.setContentsMargins(2, 2, 6, 2)
        lay.setSpacing(10)

        # ----- Output folder (optional override) -----
        # Empty -> falls back to project/output/<yymmdd>_result/. When
        # set, the orchestrator still wraps the result in a
        # {yymmdd}_result subfolder so re-running into the same chosen
        # folder doesn't co-mingle batches.
        out_box = QGroupBox(tr("batch_output_folder"))
        out_lay = QHBoxLayout(out_box)
        out_lay.setSpacing(6)
        self.out_dir_edit = QLineEdit()
        self.out_dir_edit.setPlaceholderText(tr("batch_output_placeholder"))
        self.out_dir_edit.setToolTip(tr("batch_output_tooltip"))
        out_lay.addWidget(self.out_dir_edit, 1)
        self.btn_out_browse = QPushButton(tr("btn_browse"))
        self.btn_out_browse.clicked.connect(self._browse_output_dir)
        out_lay.addWidget(self.btn_out_browse, 0)
        lay.addWidget(out_box)

        # ----- Queue -----
        self.queue_box = QGroupBox(tr("batch_queue_title").format(n=0))
        qb = QVBoxLayout(self.queue_box)
        qb.setSpacing(6)
        self.queue_hint = QLabel(tr("batch_queue_empty"))
        self.queue_hint.setStyleSheet("color: rgba(170,174,190,180);")
        self.queue_hint.setWordWrap(True)
        qb.addWidget(self.queue_hint)

        self.queue_list = _QueueList()
        self.queue_list.setMinimumHeight(160)
        self.queue_list.files_dropped.connect(self._add_files)
        self.queue_list.itemSelectionChanged.connect(self._on_select_changed)
        # Right-click context menu for per-clip custom mask editing.
        # Volatile (memory-only) — the ✱ marker in the row text is the
        # sole visual cue and disappears as soon as the user removes
        # the override or quits the app.
        self.queue_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.queue_list.customContextMenuRequested.connect(
            self._show_queue_menu)
        qb.addWidget(self.queue_list, 1)

        qbtns = QHBoxLayout()
        self.btn_add = QPushButton(tr("batch_btn_add"))
        self.btn_add.clicked.connect(self._pick_files)
        self.btn_remove = QPushButton(tr("batch_btn_remove"))
        self.btn_remove.clicked.connect(self._remove_selected)
        self.btn_clear = QPushButton(tr("batch_btn_clear"))
        self.btn_clear.clicked.connect(self._clear)
        qbtns.addWidget(self.btn_add)
        qbtns.addWidget(self.btn_remove)
        qbtns.addWidget(self.btn_clear)
        qbtns.addStretch(1)
        qb.addLayout(qbtns)

        # Always-visible affordance hint — without this, the right-click
        # menu and the A-key overlay toggle are completely invisible
        # affordances. One line of muted text saves a support ticket.
        tip_label = QLabel(tr("batch_queue_tip"))
        tip_label.setWordWrap(True)
        tip_label.setStyleSheet(
            "color: rgba(170,174,190,160); font-size: 9pt; "
            "padding: 4px 2px;")
        qb.addWidget(tip_label)
        lay.addWidget(self.queue_box)

        # ----- Stage 1: Inpaint -----
        # QGroupBox.setCheckable(True) puts a real checkbox in the title
        # bar; Qt then auto-disables every child when it's unchecked.
        # That's the entire mechanism for "skip this stage" — no need
        # for a parallel hand-rolled checkbox + manual setEnabled chain.
        self.inp_box = QGroupBox(tr("batch_section_inpaint"))
        self.inp_box.setCheckable(True)
        self.inp_box.setChecked(True)
        inp_lay = QFormLayout(self.inp_box)
        inp_lay.setSpacing(6)
        self.inp_preset = QComboBox()
        self.inp_preset.addItem(tr("batch_preset_dreamina"), "dreamina")
        inp_lay.addRow(QLabel(tr("batch_label_preset") + ":"), self.inp_preset)
        # Dreamina preset expansion knob. Only affects clips that use
        # the AUTO preset — clips with a manual brush mask (✱) keep
        # exactly what the user drew. Updates the live overlay preview
        # via _refresh_preview_mask so the user sees the effect.
        self.inp_mask_dilation = QSpinBox()
        self.inp_mask_dilation.setRange(0, 50)
        # Default 4 matches the single Inpaint tab's sb_dilation
        # default. Keeping them in sync means switching between the
        # two tabs doesn't silently change the mask coverage.
        self.inp_mask_dilation.setValue(4)
        self.inp_mask_dilation.setSuffix(" px")
        self.inp_mask_dilation.setToolTip(tr("batch_tooltip_mask_dilation"))
        self.inp_mask_dilation.valueChanged.connect(self._refresh_preview_mask)
        inp_lay.addRow(QLabel(tr("batch_label_mask_dilation") + ":"),
                       self.inp_mask_dilation)
        self.inp_batch_frames = QSpinBox()
        self.inp_batch_frames.setRange(0, 5000)
        self.inp_batch_frames.setValue(0)
        self.inp_batch_frames.setToolTip(tr("batch_tooltip_batch_frames"))
        inp_lay.addRow(QLabel(tr("batch_label_batch_frames") + ":"),
                       self.inp_batch_frames)
        self.inp_codec = _CodecCard("inpaint")
        inp_lay.addRow(self.inp_codec)
        lay.addWidget(self.inp_box)

        # ----- Stage 2: Denoise -----
        self.den_box = QGroupBox(tr("batch_section_denoise"))
        self.den_box.setCheckable(True)
        self.den_box.setChecked(True)
        den_lay = QFormLayout(self.den_box)
        den_lay.setSpacing(6)
        self.den_window = QSpinBox()
        self.den_window.setRange(2, 9)
        self.den_window.setValue(3)
        den_lay.addRow(QLabel(tr("batch_label_window") + ":"), self.den_window)
        self.den_batch = QSpinBox()
        self.den_batch.setRange(1, 32)
        self.den_batch.setValue(24)
        self.den_batch.setToolTip(tr("batch_tooltip_batch_size"))
        den_lay.addRow(QLabel(tr("batch_label_batch_size") + ":"),
                       self.den_batch)
        self.den_codec = _CodecCard("denoise")
        den_lay.addRow(self.den_codec)
        lay.addWidget(self.den_box)

        # ----- Stage 3: Upscale -----
        self.ups_box = QGroupBox(tr("batch_section_upscale"))
        self.ups_box.setCheckable(True)
        self.ups_box.setChecked(True)
        ups_lay = QFormLayout(self.ups_box)
        ups_lay.setSpacing(6)
        self.ups_model = QComboBox()
        self.ups_model.addItem(tr("upscale_model_openproteus"), "openproteus_2x")
        ups_lay.addRow(QLabel(tr("batch_label_model") + ":"), self.ups_model)
        self.ups_target = QComboBox()
        self.ups_target.addItem(tr("batch_target_native"), 0)
        self.ups_target.addItem("HD (1280)",  1280)
        self.ups_target.addItem("FHD (1920)", 1920)
        self.ups_target.addItem("QHD (2560)", 2560)
        self.ups_target.addItem("4K UHD (3840)", 3840)
        self.ups_target.addItem("8K UHD (7680)", 7680)
        ups_lay.addRow(QLabel(tr("batch_label_target") + ":"), self.ups_target)
        self.ups_batch_frames = QSpinBox()
        self.ups_batch_frames.setRange(0, 5000)
        self.ups_batch_frames.setValue(0)
        self.ups_batch_frames.setToolTip(tr("batch_tooltip_batch_frames"))
        ups_lay.addRow(QLabel(tr("batch_label_batch_frames") + ":"),
                       self.ups_batch_frames)
        # FPS override row — Auto checkbox + standard-rate dropdown.
        # Mirrors the single-tab pattern in upscale_page.py. The
        # batch_orchestrator forwards `fps_override` straight into
        # UpscaleWorker.settings, where 0.0 means "use detected".
        ups_fps_row = QWidget()
        ups_fps_h = QHBoxLayout(ups_fps_row)
        ups_fps_h.setContentsMargins(0, 0, 0, 0)
        ups_fps_h.setSpacing(8)
        self.ups_fps_auto = QCheckBox(tr("label_fps_auto"))
        self.ups_fps_auto.setChecked(True)
        self.ups_fps_auto.setToolTip(tr("tooltip_fps_override"))
        self.ups_fps_override = QComboBox()
        for value, text in _STANDARD_FPS_OPTIONS:
            self.ups_fps_override.addItem(text, value)
        self.ups_fps_override.setCurrentIndex(
            _STANDARD_FPS_OPTIONS.index((24.0, "24")))
        self.ups_fps_override.setEnabled(False)
        self.ups_fps_override.setToolTip(tr("tooltip_fps_override"))
        self.ups_fps_auto.toggled.connect(
            lambda checked: self.ups_fps_override.setEnabled(not checked))
        ups_fps_h.addWidget(self.ups_fps_auto)
        ups_fps_h.addWidget(self.ups_fps_override, 1)
        ups_lay.addRow(QLabel(tr("label_fps_override") + ":"), ups_fps_row)
        self.ups_codec = _CodecCard("upscale")
        ups_lay.addRow(self.ups_codec)
        lay.addWidget(self.ups_box)

        # ----- Run / status -----
        self.run_box = QGroupBox("")
        run_lay = QVBoxLayout(self.run_box)
        run_lay.setSpacing(6)
        self.status_label = QLabel(tr("batch_status_idle"))
        self.status_label.setWordWrap(True)
        run_lay.addWidget(self.status_label)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        run_lay.addWidget(self.progress)
        # Snapshot the default palette so we can restore it on a new
        # run after a previous cancel/error painted the bar red/green.
        # Same pattern as the Denoise tab's progress bar — stylesheet
        # would put the bar in non-native rendering and shift the
        # percentage text.
        self._progress_normal_palette = QPalette(self.progress.palette())
        self._progress_done_palette = QPalette(self.progress.palette())
        self._progress_done_palette.setColor(
            QPalette.Highlight, QColor("#a5d6a7"))
        self._progress_cancel_palette = QPalette(self.progress.palette())
        self._progress_cancel_palette.setColor(
            QPalette.Highlight, QColor("#e57373"))
        run_btns = QHBoxLayout()
        self.btn_run = QPushButton(tr("batch_btn_run"))
        self.btn_run.setStyleSheet("font-weight: bold;")
        self.btn_run.clicked.connect(self._run)
        self.btn_stop = QPushButton(tr("batch_btn_stop"))
        self.btn_stop.clicked.connect(self._stop)
        self.btn_stop.setEnabled(False)
        run_btns.addWidget(self.btn_run)
        run_btns.addWidget(self.btn_stop)
        run_btns.addStretch(1)
        run_lay.addLayout(run_btns)
        lay.addWidget(self.run_box)

        lay.addStretch(1)
        return outer

    # ------------------------------------------------------------------
    # Queue management
    # ------------------------------------------------------------------
    def _add_files(self, paths: List[str]):
        added = 0
        for p in paths:
            p = os.path.abspath(p)
            if p in self._files:
                continue
            if not os.path.isfile(p):
                continue
            self._files.append(p)
            # Probe once on add — synchronous read of header only, takes
            # a few ms even on a slow disk. Cached in self._meta so the
            # repeated queue-refresh paths never re-probe.
            self._meta[p] = _probe_meta(p)
            added += 1
        if added:
            # User said: "A-Z 1-9 가-하 순으로 정렬" — Python's default
            # unicode sort already orders numbers < uppercase < lowercase
            # < hangul, which matches that intent. Case-insensitive so a
            # mix of `Clip_A.mp4` and `clip_b.mp4` stays predictable.
            self._files.sort(key=lambda p: os.path.basename(p).lower())
            self._refresh_queue_widget()

    def _browse_output_dir(self):
        start = self.out_dir_edit.text().strip() or _PROJECT_ROOT
        path = QFileDialog.getExistingDirectory(
            self, tr("batch_output_pick"), start)
        if path:
            self.out_dir_edit.setText(path)

    def _pick_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, tr("batch_pick_files"), "",
            "비디오 (*.mp4 *.mov *.mkv *.avi *.webm *.m4v);;모든 파일 (*.*)")
        if paths:
            self._add_files(paths)

    def _remove_selected(self):
        rows = sorted({self.queue_list.row(it)
                       for it in self.queue_list.selectedItems()},
                      reverse=True)
        for r in rows:
            if 0 <= r < len(self._files):
                p = self._files.pop(r)
                self._meta.pop(p, None)
                self._masks.pop(p, None)
        self._refresh_queue_widget()

    def _clear(self):
        self._files.clear()
        self._meta.clear()
        self._masks.clear()
        self._refresh_queue_widget()

    # ------------------------------------------------------------------
    # Custom mask editing (per-clip override of the auto-Dreamina preset)
    # ------------------------------------------------------------------
    def _show_queue_menu(self, pos):
        item = self.queue_list.itemAt(pos)
        if item is None:
            return
        idx = self.queue_list.row(item)
        if not (0 <= idx < len(self._files)):
            return
        path = self._files[idx]
        has_custom = path in self._masks

        menu = QMenu(self)
        edit_act = menu.addAction(
            tr("batch_menu_edit_mask") if not has_custom
            else tr("batch_menu_edit_mask_existing"))
        remove_act = menu.addAction(tr("batch_menu_remove_mask"))
        remove_act.setEnabled(has_custom)

        chosen = menu.exec_(self.queue_list.mapToGlobal(pos))
        if chosen is None:
            return
        if chosen == edit_act:
            self._edit_mask_for(path)
        elif chosen == remove_act:
            self._masks.pop(path, None)
            self._refresh_queue_widget()
            self._refresh_preview_mask()

    def _edit_mask_for(self, path: str):
        try:
            dlg = MaskEditorDialog(
                path, current_mask=self._masks.get(path), parent=self)
        except Exception as e:
            QMessageBox.warning(
                self, "Batch",
                f"마스크 편집기 열기 실패:\n{e}")
            return
        if dlg.exec_() != dlg.Accepted:
            return
        mask = dlg.get_mask()
        # Empty mask = user implicitly cleared the override. Drop the
        # entry so the clip reverts to auto-Dreamina at run time.
        if mask is None or not (mask > 0).any():
            self._masks.pop(path, None)
        else:
            self._masks[path] = mask
        self._refresh_queue_widget()
        self._refresh_preview_mask()

    def _format_queue_line(self, path: str, status_text: str) -> str:
        """One canonical place to format
        `<basename> [✱ if custom mask] · <status> · <meta>` so every
        status transition picks up the same metadata tail and the
        ✱ marker stays in sync with the custom-masks dict."""
        base = os.path.basename(path)
        mask_marker = "  ✱" if path in self._masks else ""
        meta = self._meta.get(path) or ""
        tail = f"  ·  {meta}" if meta else ""
        return f"{base}{mask_marker}  ·  {status_text}{tail}"

    def _refresh_queue_widget(self):
        # Preserve selection across clear+rebuild. Without this, editing
        # a custom mask (which calls _refresh_queue_widget so the row
        # picks up its ✱ marker) silently wipes the selection — then
        # the immediately-following _refresh_preview_mask() finds no
        # selected item and never stages the new mask onto the viewer,
        # so the A key shows nothing until the user clicks the row
        # again. Restoring selection by path keeps the chain intact.
        prev_selected = {
            self._files[self.queue_list.row(it)]
            for it in self.queue_list.selectedItems()
            if 0 <= self.queue_list.row(it) < len(self._files)
        }
        self.queue_list.clear()
        for p in self._files:
            item = QListWidgetItem(
                self._format_queue_line(p, tr("batch_item_queued")))
            item.setToolTip(p)
            self.queue_list.addItem(item)
            if p in prev_selected:
                item.setSelected(True)
        self.queue_box.setTitle(
            tr("batch_queue_title").format(n=len(self._files)))
        self.queue_hint.setVisible(not self._files)

    # Maps stage key -> i18n key for the short "Inpaint / Denoise /
    # Upscale" label used inside running and stage-done text.
    _STAGE_LABEL_KEY = {
        STAGE_INPAINT: "batch_stage_inpaint",
        STAGE_DENOISE: "batch_stage_denoise",
        STAGE_UPSCALE: "batch_stage_upscale",
    }

    def _set_item_status(self, idx: int, status_key: str, payload: str):
        if not (0 <= idx < self.queue_list.count()):
            return
        item = self.queue_list.item(idx)
        if status_key.startswith("running:"):
            # Spinner tick will paint the actual row text + colour. We
            # just record state here and make sure the timer is alive.
            stage = status_key.split(":", 1)[1]
            self._running_idx = idx
            self._running_stage = stage
            self._running_pct = 0.0
            if not self._spinner_timer.isActive():
                self._spinner_timer.start()
            self._paint_running_row()
            return

        # Any non-running transition clears the spinner for THIS row.
        if idx == self._running_idx:
            self._running_idx = -1
            if not self._any_running():
                self._spinner_timer.stop()

        path = self._files[idx]
        color = None
        if status_key == "queued":
            status_text = tr("batch_item_queued")
        elif status_key.startswith("stage_done:"):
            stage = status_key.split(":", 1)[1]
            stage_label = tr(self._STAGE_LABEL_KEY.get(stage, ""))
            status_text = tr("batch_item_done_stage").format(stage=stage_label)
            color = _COLOR_DONE
        elif status_key == "all_done":
            status_text = tr("batch_item_done_all")
            color = _COLOR_DONE
        elif status_key == "error":
            msg = payload or ""
            status_text = f"{tr('batch_item_error')} — {msg}"
            color = _COLOR_ERROR
        else:
            status_text = status_key
        item.setText(self._format_queue_line(path, status_text))
        if color is not None:
            item.setForeground(QBrush(color))
        else:
            item.setForeground(QBrush())  # default

    def _any_running(self) -> bool:
        return self._running_idx >= 0

    def _paint_running_row(self) -> None:
        """Render the currently-processing queue row: stage label +
        spinner glyph + per-clip percentage. Called from both the
        spinner tick and the worker's progress signal so the row
        repaints whichever event fires first."""
        idx = self._running_idx
        if not (0 <= idx < self.queue_list.count()):
            return
        spin = _SPINNER_FRAMES[self._spinner_idx % len(_SPINNER_FRAMES)]
        stage_label = tr(self._STAGE_LABEL_KEY.get(self._running_stage, ""))
        status_text = (f"{stage_label} {spin} "
                       f"({self._running_pct:.0f}%)")
        item = self.queue_list.item(idx)
        item.setText(self._format_queue_line(self._files[idx], status_text))
        item.setForeground(QBrush(_COLOR_RUNNING))

    def _on_spinner_tick(self):
        self._spinner_idx = (self._spinner_idx + 1) % len(_SPINNER_FRAMES)
        if self._running_idx < 0:
            self._spinner_timer.stop()
            return
        self._paint_running_row()

    # ------------------------------------------------------------------
    # Viewer wiring
    # ------------------------------------------------------------------
    def _on_select_changed(self):
        sel = self.queue_list.selectedItems()
        if not sel:
            # Nothing selected — no clip to preview, drop any overlay
            # left over from the previous selection.
            self.player.clear_overlay()
            return
        idx = self.queue_list.row(sel[0])
        if 0 <= idx < len(self._files):
            try:
                self.player.load(self._files[idx])
            except Exception:
                pass
            # Stage the appropriate mask onto the viewer's overlay
            # layer for the A-key toggle. Custom mask wins; otherwise
            # compute Dreamina auto-preset with the current dilation.
            self._refresh_preview_mask()

    def _refresh_preview_mask(self):
        """Recompute the overlay mask for whichever queue row is
        currently selected. Called on selection change, after the
        dilation spinner moves, and after the user edits a custom
        mask via the right-click menu."""
        sel = self.queue_list.selectedItems()
        if not sel:
            self.player.clear_overlay()
            return
        idx = self.queue_list.row(sel[0])
        if not (0 <= idx < len(self._files)):
            self.player.clear_overlay()
            return
        path = self._files[idx]
        mask = self._masks.get(path)
        if mask is None:
            try:
                mask = _compute_dreamina_mask(
                    path,
                    dilation=int(self.inp_mask_dilation.value()))
            except Exception:
                mask = None
        if mask is not None:
            self.player.set_overlay_mask(mask)
        else:
            self.player.clear_overlay()

    def _toggle_mask_overlay(self):
        # Make sure the latest mask is staged before we toggle — the
        # user might have changed dilation or edited the custom mask
        # since the last selection event.
        self._refresh_preview_mask()
        self.player.toggle_overlay()

    # ------------------------------------------------------------------
    # Run / stop
    # ------------------------------------------------------------------
    def _gather_settings(self):
        inpaint = {
            "enabled": self.inp_box.isChecked(),
            "output_format": self.inp_codec.codec(),
            "prores_profile": self.inp_codec.prores_profile(),
            "fp16": True,
            # ProPainter defaults (matches InpaintPage). The user didn't
            # ask for per-stage advanced controls in v1 of the batch tab.
            "subvideo_length": 80,
            "ref_stride": 10,
            "neighbor_length": 10,
            "raft_iter": 20,
            "mask_dilation": 0,
            # ProPainter's internal `mask_dilation` (above) is a model
            # knob; this one expands the Dreamina PRESET MASK in 2D
            # before the worker ever sees it. Different layer.
            "dreamina_dilation_px": int(self.inp_mask_dilation.value()),
            "max_batch_frames": int(self.inp_batch_frames.value()),
        }
        denoise = {
            "enabled": self.den_box.isChecked(),
            "flow_model": "memfof",
            "window_size": int(self.den_window.value()),
            "batch_size": int(self.den_batch.value()),
            "occlusion_threshold": 0.02,
            "cut_threshold": 0.05,
            # flow_scale matches fdcore DEFAULT_SETTINGS (0.5) so the
            # batch tab denoise runs at the same MEMFOF resolution the
            # single denoise tab uses. Forcing 1.0 here would push
            # MEMFOF to full source resolution — 4x the tensor size,
            # which appeared as a "hang at 0%" on bs=24 because each
            # chunk's forward took minutes instead of seconds. The
            # single tab worked fine with bs=24 because it stayed at
            # 0.5; only the batch tab carried this inflated value.
            "flow_scale": 0.5,
            "fp16": True,
            "compute_noise_preview": False,
            "output_format": self.den_codec.codec(),
            "prores_profile": self.den_codec.prores_profile(),
        }
        upscale = {
            "enabled": self.ups_box.isChecked(),
            "model": self.ups_model.currentData() or "openproteus_2x",
            "target_long_edge": int(self.ups_target.currentData() or 0),
            "max_batch_frames": int(self.ups_batch_frames.value()),
            # 0.0 = follow auto-detected (snapped) fps in UpscaleWorker;
            # >0 forces the rate the user picked in the dropdown.
            "fps_override": (0.0 if self.ups_fps_auto.isChecked()
                             else float(self.ups_fps_override.currentData())),
            "fp16": True,
            "output_format": self.ups_codec.codec(),
            "prores_profile": self.ups_codec.prores_profile(),
        }
        return inpaint, denoise, upscale

    def _run(self):
        if not self._files:
            QMessageBox.information(self, "Batch", tr("batch_no_files"))
            return
        inpaint, denoise, upscale = self._gather_settings()
        if not (inpaint["enabled"] or denoise["enabled"]
                or upscale["enabled"]):
            QMessageBox.information(self, "Batch", tr("batch_no_stages"))
            return

        custom_dir = self.out_dir_edit.text().strip() or None
        if custom_dir:
            # Best-effort sanity check — if the user typed a path that
            # can't be created, fail fast with a clear message instead
            # of letting the orchestrator crash mid-run.
            try:
                os.makedirs(custom_dir, exist_ok=True)
            except Exception as e:
                QMessageBox.warning(
                    self, "Batch",
                    f"{tr('batch_output_bad')}\n{custom_dir}\n\n{e}")
                return

        self._active_stages = []
        if inpaint["enabled"]:
            self._active_stages.append(STAGE_INPAINT)
        if denoise["enabled"]:
            self._active_stages.append(STAGE_DENOISE)
        if upscale["enabled"]:
            self._active_stages.append(STAGE_UPSCALE)

        self._orchestrator = BatchOrchestrator(
            self._files, inpaint, denoise, upscale,
            output_dir_base=custom_dir,
            custom_masks=self._masks,
            parent=self)
        self._orchestrator.overall_progress.connect(self._on_progress)
        self._orchestrator.item_status.connect(self._set_item_status)
        self._orchestrator.log.connect(self.log.emit)
        self._orchestrator.error.connect(self._on_error)
        self._orchestrator.cancelled.connect(self._on_cancelled)
        self._orchestrator.all_done.connect(self._on_all_done)

        self._set_running(True)
        self.progress.setValue(0)
        # Reset visual state from any prior run — green/red palette from
        # the last completion shouldn't bleed into this run, and queue
        # row colours from the previous batch shouldn't either.
        self.progress.setPalette(self._progress_normal_palette)
        self._reset_queue_colors()
        # Pick the first ENABLED stage's label so the status line doesn't
        # mislead the user when they've disabled inpaint.
        first_label = tr({
            True: "batch_stage_inpaint",
        }.get(inpaint["enabled"], "")) if inpaint["enabled"] else (
            tr("batch_stage_denoise") if denoise["enabled"]
            else tr("batch_stage_upscale"))
        self.status_label.setText(
            tr("batch_status_stage").format(
                stage=first_label,
                fname="...", idx=1, total=len(self._files)))
        self._orchestrator.start()

    def _stop(self):
        if self._orchestrator is None:
            return
        ret = QMessageBox.question(
            self, "Batch", tr("batch_confirm_stop"),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if ret == QMessageBox.Yes:
            try:
                self._orchestrator.cancel()
            except Exception:
                pass

    def _set_running(self, running: bool):
        self.btn_run.setEnabled(not running)
        self.btn_stop.setEnabled(running)
        self.btn_add.setEnabled(not running)
        self.btn_remove.setEnabled(not running)
        self.btn_clear.setEnabled(not running)
        self.queue_list.setEnabled(not running)
        self.inp_box.setEnabled(not running)
        self.den_box.setEnabled(not running)
        self.ups_box.setEnabled(not running)
        self.out_dir_edit.setEnabled(not running)
        self.btn_out_browse.setEnabled(not running)

    # ------------------------------------------------------------------
    # Orchestrator signal handlers
    # ------------------------------------------------------------------
    def _on_progress(self, stage: str, file_idx: int, file_total: int,
                     done: int, total: int):
        stage_label = tr(self._STAGE_LABEL_KEY.get(stage, ""))
        fname = ""
        if 0 <= file_idx - 1 < len(self._files):
            fname = os.path.basename(self._files[file_idx - 1])
        self.status_label.setText(tr("batch_status_stage").format(
            stage=stage_label, fname=fname,
            idx=file_idx, total=file_total))

        # Per-clip percentage feeds the queue row's "(NN%)" tail. Pushing
        # it through _paint_running_row() also forces a repaint on every
        # frame-progress signal — between spinner ticks the percentage
        # stays current without waiting for the 100ms timer slot.
        if total > 0:
            self._running_pct = (done / float(total)) * 100.0
        if self._running_idx >= 0:
            self._paint_running_row()
        # Overall progress: combine stage rank (0..N-1 within the active
        # set), file index, and within-file frame ratio into a single
        # 0..100 bar. Uses the dynamic active-stages count so a 1-stage
        # batch (e.g. upscale-only) still fills the bar end to end.
        stage_count = max(1, len(self._active_stages))
        try:
            stage_rank = self._active_stages.index(stage)
        except ValueError:
            stage_rank = 0
        if total > 0 and file_total > 0:
            per_clip = 1.0 / (file_total * stage_count)
            stage_offset = stage_rank / float(stage_count)
            file_offset = ((file_idx - 1) / float(file_total)
                           / float(stage_count))
            frame_progress = (done / float(total)) * per_clip
            pct = (stage_offset + file_offset + frame_progress) * 100.0
            self.progress.setValue(int(max(0, min(100, pct))))

    def _on_error(self, msg: str):
        self._set_running(False)
        self.status_label.setText(tr("batch_status_error").format(msg=msg))
        self.progress.setPalette(self._progress_cancel_palette)
        self._stop_spinner_and_mark_row(_COLOR_ERROR, tr("batch_status_error")
                                        .format(msg=msg))
        QMessageBox.critical(self, "Batch", msg)
        self._orchestrator = None

    def _on_cancelled(self):
        self._set_running(False)
        self.status_label.setText(tr("batch_status_cancelled"))
        # Soft red — matches the Denoise tab's cancel state and our
        # error palette, so the user reads "interrupted" at a glance.
        self.progress.setPalette(self._progress_cancel_palette)
        self._stop_spinner_and_mark_row(None, tr("batch_status_cancelled"))
        self._orchestrator = None

    def _on_all_done(self, output_dir: str):
        self._set_running(False)
        self.progress.setValue(100)
        self.progress.setPalette(self._progress_done_palette)
        self.status_label.setText(tr("batch_status_done").format(
            n=len(self._files), out=output_dir))
        self._orchestrator = None
        # Reveal the output folder so the user can verify the deliverables.
        try:
            os.startfile(output_dir)  # type: ignore[attr-defined]
        except Exception:
            pass

    def _stop_spinner_and_mark_row(self, color, status_text: str):
        """Cleanup helper for cancel/error paths: stops the spinner and
        repaints whichever row was running with a final status + colour.
        `color=None` resets to default text colour (used on cancel where
        we don't want the row to scream red since cancel was intentional)."""
        if self._spinner_timer.isActive():
            self._spinner_timer.stop()
        idx = self._running_idx
        self._running_idx = -1
        if 0 <= idx < self.queue_list.count():
            item = self.queue_list.item(idx)
            item.setText(self._format_queue_line(
                self._files[idx], status_text))
            item.setForeground(QBrush(color) if color is not None
                               else QBrush())

    def _reset_queue_colors(self):
        for i in range(self.queue_list.count()):
            self.queue_list.item(i).setForeground(QBrush())

    # ------------------------------------------------------------------
    # External (drag/drop from window)
    # ------------------------------------------------------------------
    def dragEnterEvent(self, e: QDragEnterEvent):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e: QDropEvent):
        paths: List[str] = []
        for url in e.mimeData().urls():
            p = url.toLocalFile()
            if not p:
                continue
            if os.path.isdir(p):
                for name in sorted(os.listdir(p)):
                    full = os.path.join(p, name)
                    if os.path.isfile(full) and _is_video(full):
                        paths.append(full)
            elif _is_video(p):
                paths.append(p)
        if paths:
            self._add_files(paths)

    def shutdown(self):
        if self._spinner_timer.isActive():
            self._spinner_timer.stop()
        if self._orchestrator is not None:
            try:
                self._orchestrator.cancel()
                self._orchestrator.wait(3000)
            except Exception:
                pass
        try:
            self.player.clear()
        except Exception:
            pass

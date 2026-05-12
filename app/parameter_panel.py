"""Parameter panel (pure PyQt5).

Default view exposes only batch_size in the Temporal Flow Average card —
plus the noise preview toggle and output format selector. Toggling
"Dev mode" reveals every parameter that's otherwise locked to its
fast-path default (window_size, weight_decay, flow_model,
flow_iterations, color_threshold, scene_threshold, precision, flow_scale).
"""

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
                              QFrame, QFormLayout, QSpinBox, QDoubleSpinBox,
                              QSlider, QComboBox, QCheckBox, QLabel,
                              QSizePolicy)

from app.i18n import tr


_FORMAT_KEYS = ("h264", "prores", "image_seq")
_FORMAT_LABELS = {"h264": "H.264 MP4", "prores": "ProRes MOV",
                  "image_seq": "PNG Sequence"}
_PRORES_KEYS = ("proxy", "lt", "standard", "hq", "4444", "4444xq")

# Standard delivery frame rates exposed in the Dev-mode FPS override
# dropdown. Values mirror fdcore.video_io._STANDARD_FPS so the override
# lands on the exact same number the auto-snap would have chosen for a
# matching source. Order is roughly low → high.
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
_PRORES_LABELS = {
    "proxy": "Proxy",
    "lt": "LT",
    "standard": "422",
    "hq": "422 HQ",
    "4444": "4444",
    "4444xq": "4444 XQ",
}


def _slider_with_spin(value, lo, hi, tooltip=""):
    container = QWidget()
    h = QHBoxLayout(container)
    h.setContentsMargins(0, 0, 0, 0)
    h.setSpacing(6)
    slider = QSlider(Qt.Horizontal)
    slider.setRange(lo, hi)
    slider.setValue(value)
    spin = QSpinBox()
    spin.setRange(lo, hi)
    spin.setValue(value)
    spin.setFixedWidth(70)
    slider.valueChanged.connect(spin.setValue)
    spin.valueChanged.connect(slider.setValue)
    if tooltip:
        slider.setToolTip(tooltip)
        spin.setToolTip(tooltip)
    h.addWidget(slider, 1)
    h.addWidget(spin, 0)
    return container, spin


class ParameterPanel(QWidget):
    format_changed = pyqtSignal(str)  # "h264" or "prores" — main_window uses this to fix the output extension

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # --- Temporal Flow Average ---
        # The title + dev-mode toggle live OUTSIDE the bordered box so they
        # read like a section heading rather than card chrome. Only the
        # parameter form sits inside the box.
        section = QWidget()
        section_lay = QVBoxLayout(section)
        section_lay.setContentsMargins(0, 0, 0, 0)
        section_lay.setSpacing(4)

        # Section heading row (outside the box)
        title_row = QHBoxLayout()
        title_row.setContentsMargins(2, 0, 2, 0)
        tfa_title = QLabel(tr("section_temporal"))
        tfa_title.setStyleSheet("font-weight: 600;")
        title_row.addWidget(tfa_title)
        title_row.addStretch(1)
        self.dev_mode = QCheckBox(tr("label_dev_mode"))
        self.dev_mode.setToolTip(tr("tooltip_dev_mode"))
        title_row.addWidget(self.dev_mode)
        section_lay.addLayout(title_row)

        # Bordered box that holds only the form. Vertical SizePolicy=Maximum
        # so the box never grows past its content height — important when
        # only batch_size is visible (default mode).
        tfa_box = QFrame()
        tfa_box.setObjectName("TfaCard")
        tfa_box.setStyleSheet(
            "QFrame#TfaCard { "
            "  border: 1px solid rgba(150, 150, 170, 80); "
            "  border-radius: 5px; "
            "} "
            "QFrame#TfaCard QLabel { border: none; background: transparent; }")
        tfa_box.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        tfa_box_lay = QVBoxLayout(tfa_box)
        tfa_box_lay.setContentsMargins(10, 6, 10, 6)
        # 0 spacing between sub-widgets so a hidden advanced group leaves
        # zero residual gap. The forms inside still keep verticalSpacing(6)
        # between their own visible rows.
        tfa_box_lay.setSpacing(0)

        # Helper: a fresh QWidget+QFormLayout pair with the same styling.
        def _new_form():
            w = QWidget()
            f = QFormLayout(w)
            f.setLabelAlignment(Qt.AlignRight)
            f.setContentsMargins(0, 0, 0, 0)
            f.setHorizontalSpacing(10)
            f.setVerticalSpacing(6)
            f.setSizeConstraint(QFormLayout.SetMinAndMaxSize)
            return w, f

        # Three groups so hidden advanced rows actually take 0 height
        # (single QFormLayout would still budget verticalSpacing between
        # consecutive hidden rows, which is what was eating space above
        # batch_size in default mode).
        adv_pre, pre_form = _new_form()
        basic, basic_form = _new_form()
        adv_post, post_form = _new_form()

        # --- pre-batch_size advanced rows ---
        ws_widget, self.window_size = _slider_with_spin(
            3, 1, 15, tooltip=tr("tooltip_window_size"))
        pre_form.addRow("window_size", ws_widget)

        self.weight_decay = QDoubleSpinBox()
        self.weight_decay.setRange(0.0, 1.0)
        self.weight_decay.setSingleStep(0.05)
        self.weight_decay.setDecimals(2)
        self.weight_decay.setValue(0.80)
        self.weight_decay.setToolTip(tr("tooltip_weight_decay"))
        pre_form.addRow("weight_decay", self.weight_decay)

        self.flow_model = QComboBox()
        self.flow_model.addItems(["memfof", "raft_small", "raft_large"])
        self.flow_model.setCurrentText("memfof")
        self.flow_model.setToolTip(tr("tooltip_flow_model"))
        pre_form.addRow("flow_model", self.flow_model)

        self.flow_iterations = QSpinBox()
        self.flow_iterations.setRange(1, 32)
        self.flow_iterations.setValue(4)
        self.flow_iterations.setToolTip(tr("tooltip_flow_iterations"))
        pre_form.addRow("flow_iterations", self.flow_iterations)

        self.color_threshold = QDoubleSpinBox()
        self.color_threshold.setRange(0.005, 0.5)
        self.color_threshold.setSingleStep(0.005)
        self.color_threshold.setDecimals(3)
        self.color_threshold.setValue(0.020)
        self.color_threshold.setToolTip(tr("tooltip_color_threshold"))
        pre_form.addRow("color_threshold", self.color_threshold)

        self.scene_threshold = QDoubleSpinBox()
        self.scene_threshold.setRange(0.001, 0.5)
        self.scene_threshold.setSingleStep(0.001)
        self.scene_threshold.setDecimals(3)
        self.scene_threshold.setValue(0.030)
        self.scene_threshold.setToolTip(tr("tooltip_scene_threshold"))
        pre_form.addRow("scene_threshold", self.scene_threshold)

        # --- always-visible row ---
        bs_widget, self.batch_size = _slider_with_spin(
            24, 1, 128, tooltip=tr("tooltip_batch_size"))
        basic_form.addRow("batch_size", bs_widget)

        # --- post-batch_size advanced rows ---
        # Passthrough sits first under Dev mode because it's the most
        # invasive override: when on, the entire denoise pipeline is
        # skipped — frames are decoded and re-encoded with nothing in
        # between. Used to verify the encode/decode chain in isolation.
        self.passthrough = QCheckBox()
        self.passthrough.setChecked(False)
        self.passthrough.setToolTip(tr("tooltip_passthrough"))
        post_form.addRow(tr("label_passthrough"), self.passthrough)

        self.precision = QComboBox()
        self.precision.addItems(["bf16", "fp32"])
        self.precision.setCurrentText("bf16")
        self.precision.setToolTip(tr("tooltip_precision"))
        post_form.addRow("precision", self.precision)

        self.flow_scale = QDoubleSpinBox()
        self.flow_scale.setRange(0.3, 1.0)
        self.flow_scale.setSingleStep(0.1)
        self.flow_scale.setDecimals(1)
        self.flow_scale.setValue(0.5)
        self.flow_scale.setToolTip(tr("tooltip_flow_scale"))
        post_form.addRow("flow_scale", self.flow_scale)

        # clamp_output — was part of the deleted Selective Denoise stage,
        # kept here as a dev toggle for HDR / float-domain debugging.
        self.clamp_output = QCheckBox()
        self.clamp_output.setChecked(True)
        self.clamp_output.setToolTip(tr("tooltip_clamp_output"))
        post_form.addRow("clamp_output", self.clamp_output)

        tfa_box_lay.addWidget(adv_pre)
        tfa_box_lay.addWidget(basic)
        tfa_box_lay.addWidget(adv_post)
        self._adv_widgets = [adv_pre, adv_post]

        section_lay.addWidget(tfa_box)
        tfa = section  # used by the addWidget call below

        layout.addWidget(tfa)

        # Wire after rows exist so the initial _apply_dev_mode finds them.
        self.dev_mode.toggled.connect(self._apply_dev_mode)
        self._apply_dev_mode(False)  # default: hide advanced params

        # --- Noise preview (drawer) ---
        npv = QGroupBox(tr("section_noise_preview"))
        npv_lay = QVBoxLayout(npv)

        top = QHBoxLayout()
        self.compute_noise_preview = QCheckBox(tr("checkbox_enable"))
        self.compute_noise_preview.setChecked(False)
        self.compute_noise_preview.setToolTip(tr("tooltip_noise_enable"))
        top.addWidget(self.compute_noise_preview)
        top.addStretch(1)
        npv_lay.addLayout(top)

        self._drawer = QWidget()
        d_form = QFormLayout(self._drawer)
        d_form.setContentsMargins(0, 4, 0, 0)
        self.noise_preview = QComboBox()
        self.noise_preview.addItems(["heatmap", "signed", "gray"])
        self.noise_preview.setCurrentText("heatmap")
        self.noise_preview.setToolTip(tr("tooltip_noise_preview"))
        d_form.addRow("noise_preview", self.noise_preview)
        npv_lay.addWidget(self._drawer)
        self._drawer.setVisible(False)
        self.compute_noise_preview.toggled.connect(self._drawer.setVisible)

        layout.addWidget(npv)

        # --- Output settings ---
        out = QGroupBox(tr("section_output"))
        out_lay = QVBoxLayout(out)

        # Always-visible info row: the EFFECTIVE fps the pipeline will use
        # end-to-end. Distinct from the always-on source-info line below
        # the player ("…@ 24.04fps · 361 frames · …") which is the raw
        # source metadata — this row instead reflects whatever value the
        # pipeline is actually going to feed to the reader's -vf fps= and
        # the writer's -r. With Auto on it mirrors the auto-detected fps;
        # with Auto off it shows the user's override. Updated via
        # set_input_fps() (new source loaded) and whenever dev_mode /
        # fps_auto_check / fps_override change.
        self._detected_fps = None
        info_form = QFormLayout()
        info_form.setLabelAlignment(Qt.AlignRight)
        info_form.setContentsMargins(0, 0, 0, 0)
        self.effective_fps_label = QLabel(tr("fps_unknown"))
        self.effective_fps_label.setStyleSheet(
            "color: rgba(200, 205, 220, 220); font-weight: 500;")
        info_form.addRow(tr("label_input_fps"), self.effective_fps_label)
        out_lay.addLayout(info_form)

        fmt_form = QFormLayout()
        fmt_form.setLabelAlignment(Qt.AlignRight)
        self.output_format = QComboBox()
        for k in _FORMAT_KEYS:
            self.output_format.addItem(_FORMAT_LABELS[k])
        # Default to H.264 — small, broadly compatible. ProRes 4444 is one
        # click away for NLE/Nuke deliverables.
        self.output_format.setCurrentIndex(_FORMAT_KEYS.index("h264"))
        self.output_format.setToolTip(tr("tooltip_output_format"))
        fmt_form.addRow(tr("section_output"), self.output_format)
        out_lay.addLayout(fmt_form)

        # ProRes profile drawer (visible whenever ProRes is selected)
        self._prores_drawer = QWidget()
        prof_form = QFormLayout(self._prores_drawer)
        prof_form.setLabelAlignment(Qt.AlignRight)
        prof_form.setContentsMargins(0, 4, 0, 0)
        self.prores_profile = QComboBox()
        for k in _PRORES_KEYS:
            self.prores_profile.addItem(_PRORES_LABELS[k])
        self.prores_profile.setCurrentText(_PRORES_LABELS["4444"])
        self.prores_profile.setToolTip(tr("tooltip_prores_profile"))
        prof_form.addRow(tr("label_prores_profile"), self.prores_profile)
        out_lay.addWidget(self._prores_drawer)
        # Drawer hidden initially because the default format is H.264;
        # _on_format_changed reveals it when the user picks ProRes.
        self._prores_drawer.setVisible(False)

        # FPS override drawer — Dev-mode only. Auto checkbox + dropdown
        # pattern: when Auto is on (default), the dropdown is disabled
        # and the pipeline uses the auto-snapped fps. Unchecking Auto
        # enables the dropdown; the user picks one of the standard
        # delivery rates (23.976 / 24 / 25 / 29.97 / 30 / 50 / 59.94 /
        # 60). The selected value is forwarded as fps_override to both
        # the reader (-vf fps=) and writer (-r). Free-form numeric
        # entry was removed by user request — only canonical standard
        # rates are exposed.
        self._fps_override_drawer = QWidget()
        fps_form = QFormLayout(self._fps_override_drawer)
        fps_form.setLabelAlignment(Qt.AlignRight)
        fps_form.setContentsMargins(0, 4, 0, 0)

        fps_row = QWidget()
        fps_h = QHBoxLayout(fps_row)
        fps_h.setContentsMargins(0, 0, 0, 0)
        fps_h.setSpacing(8)
        self.fps_auto_check = QCheckBox(tr("label_fps_auto"))
        self.fps_auto_check.setChecked(True)
        self.fps_auto_check.setToolTip(tr("tooltip_fps_override"))
        self.fps_override = QComboBox()
        # (display, value) pairs. Values match video_io._STANDARD_FPS
        # exactly so the override snaps to the same value the auto path
        # would land on for the corresponding source.
        for value, text in _STANDARD_FPS_OPTIONS:
            self.fps_override.addItem(text, value)
        # 24 is the most common target for AI VFR sources, so default
        # selection lands there.
        self.fps_override.setCurrentIndex(_STANDARD_FPS_OPTIONS.index((24.0, "24")))
        self.fps_override.setEnabled(False)
        self.fps_override.setToolTip(tr("tooltip_fps_override"))
        fps_h.addWidget(self.fps_auto_check)
        fps_h.addWidget(self.fps_override, 1)
        fps_form.addRow(tr("label_fps_override"), fps_row)
        out_lay.addWidget(self._fps_override_drawer)
        self._fps_override_drawer.setVisible(False)
        # Tie this drawer to the existing dev_mode toggle alongside the
        # advanced-temporal rows.
        self._adv_widgets.append(self._fps_override_drawer)

        # Wiring: Auto toggle drives the dropdown's enabled state, and
        # any change to dev_mode / Auto / override selection should
        # refresh the effective fps display.
        self.fps_auto_check.toggled.connect(
            lambda checked: self.fps_override.setEnabled(not checked))
        self.fps_auto_check.toggled.connect(self._refresh_effective_fps_label)
        self.fps_override.currentIndexChanged.connect(
            self._refresh_effective_fps_label)
        self.dev_mode.toggled.connect(self._refresh_effective_fps_label)

        self.output_format.currentIndexChanged.connect(self._on_format_changed)

        layout.addWidget(out)

        # Small grey hint sitting just outside the card, right-aligned —
        # tells the user the preview will use whatever they pick here.
        hint = QLabel(tr("hint_preview_format_match"))
        hint.setAlignment(Qt.AlignRight)
        hint.setStyleSheet(
            "color: rgba(180, 184, 200, 140); font-size: 8pt; "
            "padding-right: 4px;")
        layout.addWidget(hint)

        layout.addStretch(1)

    def _apply_dev_mode(self, on: bool):
        """Show/hide both advanced row groups. batch_size stays visible regardless."""
        for w in self._adv_widgets:
            w.setVisible(on)

    def _on_format_changed(self, idx: int):
        if idx < 0 or idx >= len(_FORMAT_KEYS):
            return
        key = _FORMAT_KEYS[idx]
        self._prores_drawer.setVisible(key == "prores")
        self.format_changed.emit(key)

    def settings(self) -> dict:
        fmt_idx = self.output_format.currentIndex()
        prof_idx = self.prores_profile.currentIndex()
        out = {
            "window_size": self.window_size.value(),
            "weight_decay": self.weight_decay.value(),
            "flow_model": self.flow_model.currentText(),
            "flow_iterations": self.flow_iterations.value(),
            "color_threshold": self.color_threshold.value(),
            "scene_threshold": self.scene_threshold.value(),
            "batch_size": self.batch_size.value(),
            "precision": self.precision.currentText(),
            "flow_scale": self.flow_scale.value(),
            "clamp_output": self.clamp_output.isChecked(),
            "compute_noise_preview": self.compute_noise_preview.isChecked(),
            "noise_preview": self.noise_preview.currentText(),
            "output_format": _FORMAT_KEYS[fmt_idx] if 0 <= fmt_idx < len(_FORMAT_KEYS) else "h264",
            "prores_profile": _PRORES_KEYS[prof_idx] if 0 <= prof_idx < len(_PRORES_KEYS) else "hq",
        }
        # Passthrough is a dev-mode-only override. If dev_mode is off the
        # checkbox is hidden, so we drop the key so its default (False)
        # applies regardless of the checkbox's residual state.
        if self.dev_mode.isChecked() and self.passthrough.isChecked():
            out["passthrough"] = True
        # fps_override forwarded only when Dev mode is on AND Auto is
        # unchecked. The dropdown's currentData() is always a positive
        # float from _STANDARD_FPS_OPTIONS so no extra guard needed.
        if (self.dev_mode.isChecked()
                and not self.fps_auto_check.isChecked()):
            out["fps_override"] = float(self.fps_override.currentData())
        return out

    def current_format(self) -> str:
        idx = self.output_format.currentIndex()
        return _FORMAT_KEYS[idx] if 0 <= idx < len(_FORMAT_KEYS) else "h264"

    def set_input_fps(self, fps) -> None:
        """Remember the auto-detected source fps and refresh the
        effective-fps display.

        Called by MainWindow whenever a new video / image sequence is
        loaded. The label itself reflects what the pipeline will actually
        use — auto-detected when Auto is checked, override value when
        Auto is unchecked — so this just stashes the detected number and
        triggers a redraw.
        """
        self._detected_fps = (
            float(fps) if (fps is not None and fps > 0) else None)
        self._refresh_effective_fps_label()

    def _refresh_effective_fps_label(self) -> None:
        """Update the effective-fps readout based on current state."""
        if self._detected_fps is None:
            self.effective_fps_label.setText(tr("fps_unknown"))
            return
        # Override applies only when Dev mode is on AND Auto is unchecked.
        if (self.dev_mode.isChecked()
                and not self.fps_auto_check.isChecked()):
            val = float(self.fps_override.currentData())
            self.effective_fps_label.setText(f"{val:.3f}")
        else:
            self.effective_fps_label.setText(
                f"{self._detected_fps:.3f}")

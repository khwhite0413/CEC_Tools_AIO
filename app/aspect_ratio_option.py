"""16:9 / 9:16 forced-aspect option, shared by the upscale tab and the
batch tab's upscale section.

Behaviour when enabled:
- After the model output is resized to the user's target long-edge, the
  worker center-crops the frame to a 16:9 (landscape input) or 9:16
  (portrait input) box. Orientation is auto-detected from the post-
  resize dimensions — no separate landscape/portrait toggle.
- 'Fill / cover' style: the long edge stays at the user's target, the
  short edge gets trimmed equally on both ends. No letterbox bars.
- Default: OFF. The checkbox lives next to the target long-edge field
  on both tabs so the dependency is obvious.

The actual cropping happens in `UpscaleWorker.run` — this module just
owns the toggle widget so the two tabs stay in lockstep without copy-
pasted UI code.
"""

from PyQt5.QtWidgets import QWidget, QHBoxLayout, QCheckBox, QComboBox


_TOOLTIP = (
    "켜면 업스케일 결과가 자동으로 16:9 (가로 영상) 또는 9:16 (세로 영상)\n"
    "비율에 맞도록 가운데 기준 crop 됩니다. 예: 3840×2400 → 3840×2160.\n"
    "긴 변은 유지되고 짧은 변이 양쪽 가장자리에서 같은 양만큼 잘려요.\n"
    "Letterbox 검은 띠는 추가하지 않습니다."
)


# Canonical aspect-mode keys shared by AspectRatioMode, UpscaleWorker,
# and fdcore.transcode. UI currently exposes only ("none", "fill") —
# "crop" and "auto" are retained in the internal geometry solver for
# backward compat with the single Upscale tab's legacy `force_16_9`
# checkbox (which still maps to a mode string via the compat shim) but
# are not offered to Batch-tab users because Fill Frame is the only
# semantic that reliably reaches the exact target resolution.
ASPECT_MODES = ("none", "fill")


class AspectRatioOption(QWidget):
    """Single-checkbox widget — used by the single Upscale tab.

    Kept as a plain crop-only checkbox for backward compatibility with the
    Upscale page. The Batch tab uses `AspectRatioMode` (below) instead so
    it can offer fill / auto for pass-through transcode workflows.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self.check = QCheckBox("16:9 / 9:16 자동 비율 강제")
        self.check.setChecked(False)
        self.check.setToolTip(_TOOLTIP)
        layout.addWidget(self.check)
        layout.addStretch(1)

    def enabled(self) -> bool:
        return self.check.isChecked()

    def mode(self) -> str:
        """Adapter so callers that only see this class still get the
        canonical mode string. Checkbox maps to crop-or-none."""
        return "crop" if self.check.isChecked() else "none"


class AspectRatioMode(QWidget):
    """Four-mode aspect selector used by the Batch tab's Final Output group.

    Modes:
        none — passthrough, keep source aspect
        crop — center-crop to 16:9 (landscape) / 9:16 (portrait); source
               must be WIDER than target — otherwise no-op
        fill — letterbox (black bars) to 16:9 / 9:16; source must be
               NARROWER than target — otherwise no-op
        auto — pick crop or fill per clip based on source dimensions

    Implementation lives in `UpscaleWorker` (when upscale is the last
    stage) and `fdcore.transcode.run_transcode_only` (for the standalone
    pass-through path).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self.combo = QComboBox()
        # The user-facing labels are set in batch_page.py via i18n.tr()
        # (this widget stays i18n-free so it can be reused). Batch page
        # populates the combo after construction.
        layout.addWidget(self.combo, 1)

    def populate(self, labels: dict) -> None:
        """Fill the combo with `{mode_key: display_label}` — called by
        the batch page so translated strings live at the callsite."""
        self.combo.clear()
        for key in ASPECT_MODES:
            self.combo.addItem(labels.get(key, key), key)

    def mode(self) -> str:
        return self.combo.currentData() or "none"

    def enabled(self) -> bool:
        """Compatibility shim mirroring AspectRatioOption — True unless
        mode is 'none'. Any non-none mode reads as 'aspect enforcement on'."""
        return self.mode() != "none"

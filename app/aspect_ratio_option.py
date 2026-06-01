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

from PyQt5.QtWidgets import QWidget, QHBoxLayout, QCheckBox


_TOOLTIP = (
    "켜면 업스케일 결과가 자동으로 16:9 (가로 영상) 또는 9:16 (세로 영상)\n"
    "비율에 맞도록 가운데 기준 crop 됩니다. 예: 3840×2400 → 3840×2160.\n"
    "긴 변은 유지되고 짧은 변이 양쪽 가장자리에서 같은 양만큼 잘려요.\n"
    "Letterbox 검은 띠는 추가하지 않습니다."
)


class AspectRatioOption(QWidget):
    """Single-checkbox widget — shared instance pattern between tabs."""

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

"""App-wide guard that stops the mouse wheel from changing parameter
values.

The parameter panels are tall enough to scroll, and users naturally
scroll with the wheel to reach a control further down. Qt's default is
to consume the wheel event the moment the cursor enters a spinbox /
combobox / slider mid-scroll, silently bumping the value while the
panel stops scrolling. Almost never intended.

A single QObject installed on QApplication intercepts QEvent.Wheel for
the three offending widget types and `ignore()`s the event so it
propagates up to the enclosing QScrollArea (which actually wants the
wheel). Everything else — QScrollBar, QTextEdit, QListView, our
ViewportController on the video viewers — passes through untouched.
"""

from PyQt5.QtCore import QEvent, QObject
from PyQt5.QtWidgets import QAbstractSpinBox, QComboBox, QSlider


_GUARDED_TYPES = (QAbstractSpinBox, QComboBox, QSlider)


class _WheelGuard(QObject):
    def eventFilter(self, obj, event):
        if event.type() == QEvent.Wheel and isinstance(obj, _GUARDED_TYPES):
            event.ignore()
            return True
        return False


_singleton = None


def install(app) -> None:
    """Idempotent — calling more than once on the same QApplication is
    safe but does nothing extra."""
    global _singleton
    if _singleton is None:
        _singleton = _WheelGuard()
        app.installEventFilter(_singleton)

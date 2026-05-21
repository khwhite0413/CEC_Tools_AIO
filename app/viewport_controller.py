"""Nuke-style viewport zoom + pan, reusable across any QWidget.

Drop-in behaviour:
  - wheel up   = zoom in, anchored at the cursor
  - wheel down = zoom out, anchored at the cursor
  - middle-button drag = pan
  - F key      = reset zoom + pan to identity (1.0, (0, 0))

The controller installs an event filter on a target widget and never
touches the widget's painting path directly. The host widget calls
`controller.transform()` inside `paintEvent` and lets QPainter do the
work. For widgets that interpret mouse coords (mask drawing, hit
testing, etc.), `controller.map_from_view(QPoint)` undoes the zoom/pan
so the host's existing widget→source mapping keeps working unchanged.

Notes for hosts
---------------
* Wheel events are consumed unconditionally — they never propagate.
* Middle-button events are consumed only while a pan is active, so the
  host's own left/right button handlers are untouched.
* F is consumed only when the target has keyboard focus. Widgets that
  delegate focus to a parent (e.g. VideoPlayer._FrameLabel forwards
  focus to VideoPlayer) should call `controller.reset()` from the
  parent's keyPressEvent instead of relying on the filter.
"""

from PyQt5.QtCore import QEvent, QObject, QPointF, Qt, pyqtSignal
from PyQt5.QtGui import QTransform


_MIN_ZOOM = 0.1
_MAX_ZOOM = 32.0
# Per-notch multiplier. 1.25 matches Nuke's default — slow enough that
# precision zoom is doable without modifier keys, fast enough that
# getting to 4x or 8x doesn't take a dozen wheel ticks.
_ZOOM_STEP = 1.25


class ViewportController(QObject):
    """Pan/zoom state + event filter for a single target widget."""

    transform_changed = pyqtSignal()

    def __init__(self, target):
        super().__init__(target)
        self._target = target
        self._zoom = 1.0
        self._offset = QPointF(0.0, 0.0)
        self._panning = False
        self._last_pan_pos = None
        target.setMouseTracking(True)
        # StrongFocus lets the F-key reset land on widgets that take
        # focus directly. Hosts that route focus to a parent can ignore
        # this — the extra policy is harmless.
        target.setFocusPolicy(Qt.StrongFocus)
        target.installEventFilter(self)

    # --- state accessors ------------------------------------------------

    def zoom(self) -> float:
        return self._zoom

    def offset(self) -> QPointF:
        return QPointF(self._offset)

    def is_identity(self) -> bool:
        return (self._zoom == 1.0
                and self._offset.x() == 0.0 and self._offset.y() == 0.0)

    def reset(self):
        if self.is_identity():
            return
        self._zoom = 1.0
        self._offset = QPointF(0.0, 0.0)
        self._emit()

    def transform(self) -> QTransform:
        """Affine that maps base widget coords -> view (zoomed/panned) coords."""
        t = QTransform()
        t.translate(self._offset.x(), self._offset.y())
        t.scale(self._zoom, self._zoom)
        return t

    def map_from_view(self, view_point) -> QPointF:
        """Inverse of `transform()`. Use on incoming mouse positions
        before passing them to your host widget's coord-to-source code."""
        inv, ok = self.transform().inverted()
        if not ok:
            return QPointF(view_point)
        return inv.map(QPointF(view_point))

    # --- event filter ---------------------------------------------------

    def eventFilter(self, obj, event):
        if obj is not self._target:
            return False
        et = event.type()
        if et == QEvent.Wheel:
            return self._handle_wheel(event)
        if et == QEvent.MouseButtonPress and event.button() == Qt.MiddleButton:
            self._panning = True
            self._last_pan_pos = event.pos()
            try:
                self._target.setCursor(Qt.ClosedHandCursor)
            except Exception:
                pass
            return True
        if (et == QEvent.MouseMove and self._panning
                and self._last_pan_pos is not None):
            delta = event.pos() - self._last_pan_pos
            self._last_pan_pos = event.pos()
            self._offset = QPointF(self._offset.x() + delta.x(),
                                   self._offset.y() + delta.y())
            self._emit()
            return True
        if (et == QEvent.MouseButtonRelease
                and event.button() == Qt.MiddleButton):
            if self._panning:
                self._panning = False
                self._last_pan_pos = None
                try:
                    self._target.unsetCursor()
                except Exception:
                    pass
                return True
        if et == QEvent.KeyPress and event.key() == Qt.Key_F:
            self.reset()
            return True
        return False

    # --- internals ------------------------------------------------------

    def _handle_wheel(self, event) -> bool:
        delta = event.angleDelta().y()
        if delta == 0:
            return False
        cursor = QPointF(event.pos())
        # Anchor the zoom at the cursor: figure out which base-widget
        # point currently sits under the cursor, then after scaling,
        # re-solve the offset so that same base point lands back under
        # the cursor. Standard DCC viewer behaviour.
        inv, ok = self.transform().inverted()
        if not ok:
            return False
        world = inv.map(cursor)
        factor = _ZOOM_STEP if delta > 0 else (1.0 / _ZOOM_STEP)
        new_zoom = max(_MIN_ZOOM, min(_MAX_ZOOM, self._zoom * factor))
        if new_zoom == self._zoom:
            # Already at limit. Still consume the event so wheel scroll
            # doesn't escape and trigger a parent scrollbar.
            return True
        self._zoom = new_zoom
        # cursor = offset + zoom * world  =>  offset = cursor - zoom * world
        self._offset = QPointF(
            cursor.x() - self._zoom * world.x(),
            cursor.y() - self._zoom * world.y(),
        )
        self._emit()
        return True

    def _emit(self):
        self.transform_changed.emit()
        try:
            self._target.update()
        except Exception:
            pass

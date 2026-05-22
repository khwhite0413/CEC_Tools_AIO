"""Modal mask editor — brush / rect / ellipse / eraser on top of the
video's first frame. Returns a uint8 H×W mask numpy array.

Loosely based on VDInpaint's MaskCanvas but rewritten as a standalone
dialog (the original was a transparent overlay sitting on top of the
running video player). Adds the ellipse tool the user requested.
"""

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QPoint, QRect, pyqtSignal
from PyQt5.QtGui import QPainter, QImage, QColor, QPen, QBrush, QPixmap
from PyQt5.QtWidgets import (QDialog, QWidget, QVBoxLayout, QHBoxLayout,
                              QPushButton, QSlider, QLabel, QButtonGroup,
                              QDialogButtonBox, QSizePolicy)

from app.i18n import tr
from app.viewport_controller import ViewportController


class MaskCanvas(QWidget):
    """Drawable canvas: background frame + mask overlay.

    Mask lives at full video resolution internally; the widget paints
    a scaled, aspect-fitted view. Mouse events convert widget→video
    coords so drawing is independent of widget size.
    """

    TOOL_BRUSH = "brush"
    TOOL_RECT = "rect"
    TOOL_ELLIPSE = "ellipse"
    TOOL_ERASER = "eraser"

    mask_updated = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(640, 360)
        self.setFocusPolicy(Qt.StrongFocus)

        self.tool = self.TOOL_BRUSH
        self.brush_size = 20  # in video-pixel radius
        self.mask_opacity = 0.45
        self.mask_color = QColor(255, 64, 64)

        self._bg_pixmap = None
        self._video_w = 0
        self._video_h = 0
        self._mask_image = None      # QImage Format_Grayscale8 at video res
        self._mask_arr_ref = None    # keep numpy buffer alive

        self._display_rect = QRect()
        self._scale = 1.0

        self._drawing = False
        self._rect_start = None
        self._rect_current = None
        self._last_point = None
        self._cursor_pos = None

        self._undo_stack = []
        self._max_undo = 20

        # Nuke-style wheel zoom / middle-drag pan / F reset. All
        # internal mouse-derived state (cursor_pos, rect_start,
        # last_point, etc.) is stored in *base* widget coords — the
        # view-space pos that arrives in mouse events is fed through
        # `_viewport.map_from_view` first so the rest of the canvas
        # logic stays oblivious to the zoom layer.
        self._viewport = ViewportController(self)
        self._viewport.transform_changed.connect(self.update)

    # ------------------------------------------------------------------
    # External API
    # ------------------------------------------------------------------
    def set_frame(self, frame_bgr: np.ndarray):
        """Install the background frame (BGR). Initializes a blank mask
        at the same resolution if no mask exists yet."""
        h, w = frame_bgr.shape[:2]
        self._video_w, self._video_h = w, h
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        qimg = QImage(rgb.data, w, h, rgb.strides[0],
                      QImage.Format_RGB888).copy()
        self._bg_pixmap = QPixmap.fromImage(qimg)
        if (self._mask_image is None
                or self._mask_image.width() != w
                or self._mask_image.height() != h):
            self._mask_image = QImage(w, h, QImage.Format_Grayscale8)
            self._mask_image.fill(0)
            self._undo_stack.clear()
        self._compute_display_mapping()
        self.update()

    def set_mask(self, mask_np):
        if mask_np is None or self._video_w == 0:
            return
        h, w = mask_np.shape[:2]
        if (h, w) != (self._video_h, self._video_w):
            mask_np = cv2.resize(mask_np, (self._video_w, self._video_h),
                                  interpolation=cv2.INTER_NEAREST)
        arr = np.ascontiguousarray(mask_np.astype(np.uint8))
        self._mask_arr_ref = arr
        self._mask_image = QImage(
            arr.data, self._video_w, self._video_h, self._video_w,
            QImage.Format_Grayscale8).copy()
        self.mask_updated.emit()
        self.update()

    def get_mask_numpy(self):
        if self._mask_image is None:
            return None
        w = self._mask_image.width()
        h = self._mask_image.height()
        # Qt pads Format_Grayscale8 rows up to a 4-byte alignment for
        # SIMD reasons, so bytesPerLine may exceed `w` (e.g. width 1470
        # gets bpl 1472 → 2 bytes of padding per row). Reading h*w bytes
        # flat would drift each row by `bpl - w` bytes and project a
        # painted dot into a diagonal streak — the 21:9 mask bug.
        # Respect the real stride and slice the padding off.
        bpl = self._mask_image.bytesPerLine()
        ptr = self._mask_image.bits()
        ptr.setsize(h * bpl)
        raw = np.frombuffer(ptr, dtype=np.uint8).reshape((h, bpl))
        return raw[:, :w].copy()

    def set_tool(self, tool):
        self.tool = tool
        self.update()

    def set_brush_size(self, size):
        self.brush_size = max(1, int(size))
        self.update()

    def clear_mask(self):
        if self._mask_image:
            self._save_undo()
            self._mask_image.fill(0)
            self.mask_updated.emit()
            self.update()

    def undo(self):
        if self._undo_stack and self._mask_image is not None:
            self._mask_image = self._undo_stack.pop()
            self.mask_updated.emit()
            self.update()

    # ------------------------------------------------------------------
    # Display mapping
    # ------------------------------------------------------------------
    def _compute_display_mapping(self):
        if not self._bg_pixmap or self._bg_pixmap.isNull():
            return
        ww, wh = self.width(), self.height()
        if ww <= 0 or wh <= 0 or self._video_w == 0:
            return
        scale = min(ww / self._video_w, wh / self._video_h)
        dw = int(self._video_w * scale)
        dh = int(self._video_h * scale)
        ox = (ww - dw) // 2
        oy = (wh - dh) // 2
        self._display_rect = QRect(ox, oy, dw, dh)
        self._scale = scale

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._compute_display_mapping()

    def _widget_to_video(self, wx, wy):
        if self._scale <= 0:
            return 0, 0
        vx = (wx - self._display_rect.x()) / self._scale
        vy = (wy - self._display_rect.y()) / self._scale
        return (max(0, min(int(vx), self._video_w - 1)),
                max(0, min(int(vy), self._video_h - 1)))

    # ------------------------------------------------------------------
    # Mouse handling
    # ------------------------------------------------------------------
    def mousePressEvent(self, e):
        if e.button() != Qt.LeftButton or self._mask_image is None:
            return
        self._save_undo()
        self._drawing = True
        # Undo the viewport zoom/pan so the rest of this method
        # operates in the same base-widget space `_widget_to_video`
        # expects. Without this, painting at high zoom would land far
        # from where the user's brush is visible.
        base_pt = self._viewport.map_from_view(e.pos()).toPoint()
        if self.tool in (self.TOOL_RECT, self.TOOL_ELLIPSE):
            self._rect_start = base_pt
            self._rect_current = base_pt
        else:
            vx, vy = self._widget_to_video(base_pt.x(), base_pt.y())
            self._last_point = (vx, vy)
            self._paint_circle(vx, vy)
            self.mask_updated.emit()
        self.update()

    def mouseMoveEvent(self, e):
        base_pt = self._viewport.map_from_view(e.pos()).toPoint()
        self._cursor_pos = base_pt
        if not self._drawing:
            self.update()
            return
        if self.tool in (self.TOOL_RECT, self.TOOL_ELLIPSE):
            self._rect_current = base_pt
            self.update()
        else:
            vx, vy = self._widget_to_video(base_pt.x(), base_pt.y())
            if self._last_point:
                self._paint_line(self._last_point[0], self._last_point[1],
                                  vx, vy)
            self._last_point = (vx, vy)
            self.mask_updated.emit()
            self.update()

    def mouseReleaseEvent(self, e):
        if e.button() != Qt.LeftButton or not self._drawing:
            return
        self._drawing = False
        if (self.tool in (self.TOOL_RECT, self.TOOL_ELLIPSE)
                and self._rect_start and self._rect_current):
            self._fill_shape(is_rect=(self.tool == self.TOOL_RECT))
        self._rect_start = None
        self._rect_current = None
        self._last_point = None
        self.mask_updated.emit()
        self.update()

    def leaveEvent(self, e):
        self._cursor_pos = None
        self.update()

    def keyPressEvent(self, e):
        # Ctrl+Z = undo. [ / ] = brush size −/+ like Photoshop.
        if e.key() == Qt.Key_Z and (e.modifiers() & Qt.ControlModifier):
            self.undo()
            return
        if e.key() == Qt.Key_BracketLeft:
            self.set_brush_size(max(1, self.brush_size - 4))
            return
        if e.key() == Qt.Key_BracketRight:
            self.set_brush_size(min(500, self.brush_size + 4))
            return
        super().keyPressEvent(e)

    # ------------------------------------------------------------------
    # Mask painting (operate on _mask_image)
    # ------------------------------------------------------------------
    def _save_undo(self):
        if self._mask_image is None:
            return
        if len(self._undo_stack) >= self._max_undo:
            self._undo_stack.pop(0)
        self._undo_stack.append(self._mask_image.copy())

    def _erase(self):
        return self.tool == self.TOOL_ERASER

    def _paint_circle(self, vx, vy):
        painter = QPainter(self._mask_image)
        color = QColor(0, 0, 0) if self._erase() else QColor(255, 255, 255)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(color))
        painter.drawEllipse(QPoint(vx, vy), self.brush_size, self.brush_size)
        painter.end()

    def _paint_line(self, x1, y1, x2, y2):
        painter = QPainter(self._mask_image)
        color = QColor(0, 0, 0) if self._erase() else QColor(255, 255, 255)
        pen = QPen(color, self.brush_size * 2, Qt.SolidLine,
                    Qt.RoundCap, Qt.RoundJoin)
        painter.setPen(pen)
        painter.drawLine(QPoint(x1, y1), QPoint(x2, y2))
        painter.end()

    def _fill_shape(self, is_rect: bool):
        if not self._rect_start or not self._rect_current:
            return
        x1, y1 = self._widget_to_video(self._rect_start.x(),
                                        self._rect_start.y())
        x2, y2 = self._widget_to_video(self._rect_current.x(),
                                        self._rect_current.y())
        left, top = min(x1, x2), min(y1, y2)
        right, bottom = max(x1, x2), max(y1, y2)
        if right - left < 1 or bottom - top < 1:
            return
        painter = QPainter(self._mask_image)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(255, 255, 255)))
        r = QRect(left, top, right - left, bottom - top)
        if is_rect:
            painter.drawRect(r)
        else:
            painter.drawEllipse(r)
        painter.end()

    # ------------------------------------------------------------------
    # paintEvent
    # ------------------------------------------------------------------
    def paintEvent(self, e):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(16, 16, 24))

        # All content below lives in *base* widget coords; the viewport
        # transform zooms/pans the whole stack — bg, mask overlay,
        # rect/ellipse preview, and the brush hover circle — together.
        painter.save()
        painter.setTransform(self._viewport.transform(), True)

        if self._bg_pixmap and not self._bg_pixmap.isNull():
            painter.drawPixmap(self._display_rect, self._bg_pixmap)

        # Mask overlay — convert grayscale mask to RGBA with the brand
        # red color. Keep the numpy buffer alive via _last_overlay_buf
        # so QImage's view doesn't dangle.
        if self._mask_image is not None and self._video_w > 0:
            mask_np = self.get_mask_numpy()
            if mask_np is not None and np.any(mask_np > 0):
                h, w = mask_np.shape
                rgba = np.zeros((h, w, 4), dtype=np.uint8)
                m = mask_np > 0
                rgba[m, 0] = self.mask_color.red()
                rgba[m, 1] = self.mask_color.green()
                rgba[m, 2] = self.mask_color.blue()
                rgba[m, 3] = int(255 * self.mask_opacity)
                self._last_overlay_buf = rgba  # keep alive
                overlay = QImage(rgba.data, w, h, w * 4,
                                  QImage.Format_RGBA8888)
                painter.drawImage(self._display_rect, overlay)

        # Shape preview while dragging rect / ellipse
        if (self._drawing
                and self.tool in (self.TOOL_RECT, self.TOOL_ELLIPSE)
                and self._rect_start and self._rect_current):
            # Cosmetic pen keeps the dashed outline at a sensible
            # ~2px regardless of zoom.
            pen = QPen(QColor(255, 220, 80), 2, Qt.DashLine)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.setBrush(QBrush(QColor(255, 220, 80, 50)))
            r = QRect(self._rect_start, self._rect_current).normalized()
            if self.tool == self.TOOL_RECT:
                painter.drawRect(r)
            else:
                painter.drawEllipse(r)

        # Brush hover cursor — radius uses the base display scale only;
        # the viewport zoom scales the circle automatically. Pen is
        # cosmetic so the outline stays 1px on screen at any zoom.
        if (self.tool in (self.TOOL_BRUSH, self.TOOL_ERASER)
                and self._cursor_pos is not None):
            r_widget = int(self.brush_size * self._scale)
            color = (QColor(80, 200, 255) if self.tool == self.TOOL_BRUSH
                     else QColor(255, 180, 80))
            pen = QPen(color, 1, Qt.SolidLine)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(self._cursor_pos, r_widget, r_widget)

        painter.restore()
        painter.end()


class MaskEditorDialog(QDialog):
    """Modal mask editor dialog."""

    def __init__(self, video_path, current_mask=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("mask_editor_title"))
        self.resize(1100, 720)
        self.setModal(True)

        # Keep the cv2 capture open so the frame seeker can scrub
        # without re-opening the file on every slider move. The mask
        # itself is video-resolution-static, so changing the visible
        # frame only swaps the canvas background — it's a verification
        # aid for watermarks that drift across the clip.
        self._cap = cv2.VideoCapture(video_path)
        if not self._cap.isOpened():
            raise ValueError("비디오를 열지 못했습니다.")
        self._total_frames = max(1, int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        ret, frame = self._cap.read()
        if not ret or frame is None:
            self._cap.release()
            self._cap = None
            raise ValueError("첫 프레임을 읽지 못했습니다.")

        root = QVBoxLayout(self)
        root.setSpacing(8)

        # --- Toolbar ---
        toolbar = QHBoxLayout()
        self._tool_group = QButtonGroup(self)
        self._tool_group.setExclusive(True)
        for tool, key in (
            (MaskCanvas.TOOL_BRUSH,   "mask_tool_brush"),
            (MaskCanvas.TOOL_RECT,    "mask_tool_rect"),
            (MaskCanvas.TOOL_ELLIPSE, "mask_tool_ellipse"),
            (MaskCanvas.TOOL_ERASER,  "mask_tool_eraser"),
        ):
            btn = QPushButton(tr(key))
            btn.setCheckable(True)
            btn.clicked.connect(lambda _, t=tool: self.canvas.set_tool(t))
            self._tool_group.addButton(btn)
            toolbar.addWidget(btn)
            if tool == MaskCanvas.TOOL_BRUSH:
                btn.setChecked(True)

        toolbar.addSpacing(12)
        toolbar.addWidget(QLabel(tr("mask_brush_size")))
        self.size_slider = QSlider(Qt.Horizontal)
        self.size_slider.setRange(2, 200)
        self.size_slider.setValue(20)
        self.size_slider.setFixedWidth(220)
        self.size_slider.valueChanged.connect(self._on_size_changed)
        toolbar.addWidget(self.size_slider)
        self.size_value_label = QLabel("20 px")
        self.size_value_label.setMinimumWidth(54)
        toolbar.addWidget(self.size_value_label)

        toolbar.addStretch(1)

        self.btn_undo = QPushButton(tr("mask_undo"))
        self.btn_undo.setToolTip("Ctrl+Z")
        self.btn_undo.clicked.connect(lambda: self.canvas.undo())
        toolbar.addWidget(self.btn_undo)
        self.btn_clear = QPushButton(tr("mask_clear"))
        self.btn_clear.clicked.connect(lambda: self.canvas.clear_mask())
        toolbar.addWidget(self.btn_clear)
        root.addLayout(toolbar)

        # --- Canvas ---
        self.canvas = MaskCanvas()
        self.canvas.set_frame(frame)
        if current_mask is not None and np.any(current_mask > 0):
            self.canvas.set_mask(current_mask)
        root.addWidget(self.canvas, 1)

        # --- Frame seeker --------------------------------------------------
        # The mask is video-static, but the user often wants to verify
        # it tracks the watermark across the whole clip — not just frame
        # 0. Slider scrubs the canvas background only; mask layer stays
        # put because set_frame() detects same-resolution input and
        # keeps the existing mask image intact.
        seek_row = QHBoxLayout()
        seek_row.addWidget(QLabel(tr("mask_frame_label")))
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setRange(0, max(0, self._total_frames - 1))
        self.frame_slider.setValue(0)
        self.frame_slider.valueChanged.connect(self._on_frame_changed)
        seek_row.addWidget(self.frame_slider, 1)
        self.frame_count_label = QLabel(f"1 / {self._total_frames}")
        self.frame_count_label.setMinimumWidth(80)
        self.frame_count_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        seek_row.addWidget(self.frame_count_label)
        root.addLayout(seek_row)

        # --- OK / Cancel ---
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.button(QDialogButtonBox.Ok).setText(tr("mask_apply"))
        btns.button(QDialogButtonBox.Cancel).setText(tr("btn_cancel"))
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)

    def _on_size_changed(self, v):
        self.canvas.set_brush_size(v)
        self.size_value_label.setText(f"{v} px")

    def _on_frame_changed(self, idx: int):
        if self._cap is None:
            return
        cur = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))
        # cap.read() advances the position by one, so plain re-reads
        # would drift. Seek when our idea of "where to be" doesn't match
        # the cap's next-frame position.
        if cur != idx:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self._cap.read()
        if not ret or frame is None:
            return
        self.canvas.set_frame(frame)
        self.frame_count_label.setText(f"{idx + 1} / {self._total_frames}")

    def _release_cap(self):
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def accept(self):
        self._release_cap()
        super().accept()

    def reject(self):
        self._release_cap()
        super().reject()

    def closeEvent(self, e):
        # Covers the X-button path and any other reason the dialog
        # gets torn down without accept/reject going through.
        self._release_cap()
        super().closeEvent(e)

    def get_mask(self):
        return self.canvas.get_mask_numpy()

"""OpenCV-based video frame viewer with playback controls (pure PyQt5)."""

import os
import shutil

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer, QRect, QUrl, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen
from PyQt5.QtMultimedia import (QMediaPlayer, QMediaContent,
                                 QAbstractVideoSurface, QVideoFrame)
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                              QSlider, QPushButton, QSizePolicy, QStyle,
                              QMenu, QFileDialog, QMessageBox)

from fdcore.video_io import (VideoInfo, ImageSequenceInfo, is_image_sequence,
                              probe as _probe_video)
from app.i18n import tr


def _fmt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    m = int(seconds // 60)
    s = seconds - m * 60
    return f"{m:02d}:{s:05.2f}"


class _NullVideoSurface(QAbstractVideoSurface):
    """Discards every video frame.

    QMediaPlayer on the WMF backend refuses to build its playback graph
    ("Failed to set topology") for a media that contains video unless
    a video sink is wired up — even when the caller only cares about
    audio. cv2 already owns the frame viewer in this app, so we plug
    this no-op surface into QMediaPlayer just to satisfy the topology
    builder and let audio through.
    """

    def supportedPixelFormats(self, handle_type=None):
        return [QVideoFrame.Format_RGB32, QVideoFrame.Format_YUV420P,
                QVideoFrame.Format_NV12, QVideoFrame.Format_YV12,
                QVideoFrame.Format_BGRA32, QVideoFrame.Format_BGR32,
                QVideoFrame.Format_BGR24, QVideoFrame.Format_RGB24,
                QVideoFrame.Format_RGB565]

    def present(self, _frame):
        return True


class _FrameLabel(QLabel):
    """Pixmap host for a single frame.

    Also owns the optional A/B wipe overlay: when wipe is enabled, paints a
    thin vertical line at `_wipe_pos` (0..1) and lets the user drag it.
    The actual frame compositing happens in VideoPlayer; this label only
    handles the line/handle UI and emits wipe_changed when the user drags.
    """

    clicked = pyqtSignal()
    wipe_changed = pyqtSignal(float)  # 0.0 = all after, 1.0 = all before

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(480, 270)
        self.setStyleSheet("background-color: #18181f; border: 1px solid #2a2a3a;")
        self.setText(tr("video_placeholder"))
        self.setMouseTracking(True)  # for hover state on the wipe line
        self._current = None
        self._wipe_enabled = False
        self._wipe_pos = 0.0
        self._dragging_wipe = False
        self._hover = False

    def show_bgr(self, frame):
        self._current = frame
        self._refresh()

    def clear_image(self):
        self._current = None
        self.setPixmap(QPixmap())
        self.setText(tr("video_placeholder"))

    # --- wipe overlay -------------------------------------------------
    def set_wipe_enabled(self, on: bool):
        self._wipe_enabled = bool(on)
        if not self._wipe_enabled:
            self._dragging_wipe = False
            self._wipe_pos = 0.0
        self.update()

    def set_wipe_pos(self, pos: float):
        self._wipe_pos = max(0.0, min(1.0, float(pos)))
        self.update()

    def wipe_pos(self) -> float:
        return self._wipe_pos

    def enterEvent(self, e):
        self._hover = True
        if self._wipe_enabled:
            self.update()
        super().enterEvent(e)

    def leaveEvent(self, e):
        self._hover = False
        self._dragging_wipe = False
        if self._wipe_enabled:
            self.update()
        super().leaveEvent(e)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._focus_player()
            if self._wipe_enabled:
                self._dragging_wipe = True
                self._set_wipe_from_x(e.x())
                # don't fire `clicked` while dragging the wipe — clicked is
                # the click-to-load entry path, irrelevant in compare mode
                return
            self.clicked.emit()
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._dragging_wipe:
            self._set_wipe_from_x(e.x())
            return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if self._dragging_wipe and e.button() == Qt.LeftButton:
            self._dragging_wipe = False
            self.update()
            return
        super().mouseReleaseEvent(e)

    def _focus_player(self):
        # Walk up to give the parent VideoPlayer keyboard focus for arrow keys.
        p = self.parent()
        while p is not None:
            if isinstance(p, VideoPlayer):
                p.setFocus()
                return
            p = p.parent()

    def _pixmap_geometry(self) -> QRect:
        pix = self.pixmap()
        if pix is None or pix.isNull():
            return self.rect()
        lw, lh = self.width(), self.height()
        pw, ph = pix.width(), pix.height()
        return QRect((lw - pw) // 2, (lh - ph) // 2, pw, ph)

    def _set_wipe_from_x(self, mx: int):
        rect = self._pixmap_geometry()
        if rect.width() <= 0:
            return
        local_x = mx - rect.left()
        pos = max(0.0, min(1.0, local_x / rect.width()))
        if pos != self._wipe_pos:
            self._wipe_pos = pos
            self.wipe_changed.emit(pos)
            self.update()

    def paintEvent(self, e):
        super().paintEvent(e)
        if not self._wipe_enabled:
            return
        # Show the line subtly when hovering, fade out otherwise unless the
        # user has actually moved the wipe away from "all after".
        active = self._hover or self._dragging_wipe or self._wipe_pos > 0.001
        if not active:
            return
        rect = self._pixmap_geometry()
        if rect.width() <= 0:
            return
        line_x = rect.left() + int(self._wipe_pos * rect.width())
        line_alpha = 200 if self._dragging_wipe else (140 if self._hover else 70)
        handle_alpha = 230 if self._dragging_wipe else (180 if self._hover else 110)
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.Antialiasing, True)
            painter.setPen(QPen(QColor(255, 255, 255, line_alpha), 1.5))
            painter.drawLine(line_x, rect.top(), line_x, rect.bottom())
            cy = (rect.top() + rect.bottom()) // 2
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(255, 255, 255, handle_alpha))
            painter.drawRoundedRect(line_x - 3, cy - 18, 6, 36, 3, 3)
        finally:
            painter.end()

    def _refresh(self):
        if self._current is None:
            return
        rgb = cv2.cvtColor(self._current, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(pix)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._refresh()


class VideoPlayer(QWidget):
    """Frame viewer + timeline + transport controls."""

    frame_changed = pyqtSignal(int)
    empty_clicked = pyqtSignal()  # frame area clicked while no video is loaded

    def __init__(self, parent=None, show_controls: bool = True,
                 click_to_load: bool = False,
                 audio_enabled: bool = True):
        super().__init__(parent)
        # When audio_enabled is False (e.g. the noise-preview viewer),
        # we skip mounting any media on QMediaPlayer and hide the
        # volume widgets — extracted noise is a residual stream, the
        # "audio" it would carry is meaningless or absent.
        self._audio_enabled = bool(audio_enabled)
        self.video_path = None
        self.video_info = None
        self._cap = None
        self._idx = 0
        self._playing = False
        self._loop = True
        self._click_to_load = click_to_load
        # Optional A/B compare state: a second capture (the "before"
        # version), composited on top of the primary frame using the
        # frame_view's wipe position.
        self._before_cap = None
        self._before_path = None
        # Optional alpha-mask overlay (Inpaint tab uses this — pressing
        # A toggles it on/off so the user can verify which pixels the
        # preset / brush will inpaint). _raw_frame caches the most recent
        # composited frame so toggling is instant — no cap re-read.
        self._overlay_mask = None        # uint8 H x W binary mask
        self._overlay_visible = False
        self._raw_frame = None           # cached BGR uint8 frame, no overlay

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_tick)

        # Audio is played by a separate QMediaPlayer (audio-only role):
        # the frame viewer above is cv2-based for frame-accurate stepping
        # and image-sequence support, which means we can't piggyback on
        # QMediaPlayer's video. Instead we run audio in parallel, sync
        # its position whenever the user seeks/plays/pauses, and let it
        # free-run between seeks. Small drift over a clip is acceptable
        # for a denoise preview; perfect lip-sync would need a single
        # master clock (a much bigger rewrite).
        self._audio = QMediaPlayer(self)
        # Required by the WMF backend to build a playback topology when
        # the source has a video stream; the surface drops every frame
        # so picture stays with the cv2 viewer.
        self._null_video_sink = _NullVideoSurface(self)
        self._audio.setVideoOutput(self._null_video_sink)
        self._audio.setVolume(50)
        self._volume = 50
        self._has_audio = False
        self._was_audio_playing = False  # restore audio after scrub
        self._audio_pending_play = False  # play once media finishes loading
        self._audio.error.connect(self._on_audio_error)
        self._audio.mediaStatusChanged.connect(self._on_audio_status)

        # Keyboard focus: arrow keys step ±1 frame
        self.setFocusPolicy(Qt.StrongFocus)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.frame_view = _FrameLabel()
        self.frame_view.clicked.connect(self._on_frame_clicked)
        self.frame_view.wipe_changed.connect(self._on_wipe_changed)
        layout.addWidget(self.frame_view, 1)

        bar = QHBoxLayout()
        self.time_label = QLabel("00:00.00 / 00:00.00")
        self.time_label.setMinimumWidth(140)
        self.frame_label = QLabel("0 / 0")
        self.frame_label.setMinimumWidth(110)
        self.frame_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.setEnabled(False)
        self.slider.sliderPressed.connect(self._on_slider_pressed)
        self.slider.sliderReleased.connect(self._on_slider_released)
        self.slider.valueChanged.connect(self._on_slider_value)
        bar.addWidget(self.time_label)
        bar.addWidget(self.slider, 1)
        bar.addWidget(self.frame_label)
        layout.addLayout(bar)

        self._controls = QWidget()
        ctrl = QHBoxLayout(self._controls)
        ctrl.setContentsMargins(0, 0, 0, 0)
        style = self.style()
        self.btn_first = QPushButton()
        self.btn_first.setIcon(style.standardIcon(QStyle.SP_MediaSkipBackward))
        self.btn_first.setToolTip(tr("tooltip_first"))
        self.btn_prev = QPushButton()
        self.btn_prev.setIcon(style.standardIcon(QStyle.SP_MediaSeekBackward))
        self.btn_prev.setToolTip(tr("tooltip_prev"))
        self.btn_play = QPushButton()
        self.btn_play.setIcon(style.standardIcon(QStyle.SP_MediaPlay))
        self.btn_play.setToolTip(tr("tooltip_play"))
        self.btn_play.setShortcut(Qt.Key_Space)
        self.btn_next = QPushButton()
        self.btn_next.setIcon(style.standardIcon(QStyle.SP_MediaSeekForward))
        self.btn_next.setToolTip(tr("tooltip_next"))
        self.btn_last = QPushButton()
        self.btn_last.setIcon(style.standardIcon(QStyle.SP_MediaSkipForward))
        self.btn_last.setToolTip(tr("tooltip_last"))
        self.btn_loop = QPushButton()
        self.btn_loop.setIcon(style.standardIcon(QStyle.SP_BrowserReload))
        self.btn_loop.setCheckable(True)
        self.btn_loop.setChecked(True)
        self.btn_loop.setToolTip(tr("tooltip_loop"))
        for b in (self.btn_first, self.btn_prev, self.btn_play,
                  self.btn_next, self.btn_last, self.btn_loop):
            b.setEnabled(False)
            ctrl.addWidget(b)
        ctrl.addStretch(1)
        # Volume controls: mute toggle + 0..100 slider. Disabled until a
        # file with an audio track loads. Default 50% per user request.
        self.btn_mute = QPushButton()
        self.btn_mute.setIcon(style.standardIcon(QStyle.SP_MediaVolume))
        self.btn_mute.setCheckable(True)
        self.btn_mute.setToolTip(tr("tooltip_mute"))
        self.btn_mute.setEnabled(False)
        self.vol_slider = QSlider(Qt.Horizontal)
        self.vol_slider.setRange(0, 100)
        self.vol_slider.setValue(50)
        self.vol_slider.setFixedWidth(90)
        self.vol_slider.setEnabled(False)
        self.vol_slider.setToolTip(tr("tooltip_volume"))
        ctrl.addWidget(self.btn_mute)
        ctrl.addWidget(self.vol_slider)
        self.btn_first.clicked.connect(self.seek_first)
        self.btn_prev.clicked.connect(lambda: self.step(-1))
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_next.clicked.connect(lambda: self.step(+1))
        self.btn_last.clicked.connect(self.seek_last)
        self.btn_loop.toggled.connect(self.set_loop)
        self.btn_mute.toggled.connect(self._on_mute_toggled)
        self.vol_slider.valueChanged.connect(self._on_volume_changed)
        if not self._audio_enabled:
            # Stay invisible in any container (controls row or
            # re-parented into a dialog) when the owner asked for a
            # silent viewer.
            self.btn_mute.setVisible(False)
            self.vol_slider.setVisible(False)
        layout.addWidget(self._controls)
        self._controls.setVisible(show_controls)

        self._scrubbing = False
        self._internal_seek = False
        self._save_as_label = None
        self._save_as_default_name = None
        self._refresh_cursor()

    def load(self, path: str):
        self._release_cap()
        self.video_path = path
        self.video_info = _probe_video(path)
        if is_image_sequence(path):
            cap_path = os.path.join(path, self.video_info.pattern)
            self._cap = cv2.VideoCapture(cap_path, cv2.CAP_IMAGES)
        else:
            self._cap = cv2.VideoCapture(path)
        total = self.video_info.frame_count

        self._internal_seek = True
        self.slider.setRange(0, max(0, total - 1))
        self.slider.setValue(0)
        self.slider.setEnabled(total > 0)
        self.slider.update()
        self._internal_seek = False

        for b in (self.btn_first, self.btn_prev, self.btn_play,
                  self.btn_next, self.btn_last, self.btn_loop):
            b.setEnabled(total > 0)

        fps = self.video_info.fps or 30.0
        self._timer.setInterval(max(10, int(1000.0 / fps)))
        self._idx = 0
        self._show_frame_at(0)
        self._setup_audio_for(path)
        self._refresh_cursor()

    def clear(self):
        self.pause()
        self._release_cap()
        self._release_before_cap()
        self._audio.stop()
        self._audio.setMedia(QMediaContent())
        self._has_audio = False
        self.btn_mute.setEnabled(False)
        self.vol_slider.setEnabled(False)
        self.frame_view.set_wipe_enabled(False)
        self.video_path = None
        self.video_info = None
        self._idx = 0
        self._internal_seek = True
        self.slider.setRange(0, 0)
        self.slider.setEnabled(False)
        self._internal_seek = False
        for b in (self.btn_first, self.btn_prev, self.btn_play,
                  self.btn_next, self.btn_last, self.btn_loop):
            b.setEnabled(False)
        self.frame_view.clear_image()
        self.time_label.setText("00:00.00 / 00:00.00")
        self.frame_label.setText("0 / 0")
        self._raw_frame = None
        self._refresh_cursor()

    def play(self):
        if not self._cap or not self.video_info:
            return
        if self._idx >= self.video_info.frame_count - 1:
            self.seek_to(0)
        self._timer.start()
        self._playing = True
        if self._has_audio:
            # setMedia() is asynchronous, so on the very first play after a
            # load the media may still be loading — calling play() now is a
            # no-op on some Qt backends. Mark a pending-play and let the
            # mediaStatusChanged → LoadedMedia handler start it then.
            status = self._audio.mediaStatus()
            ready = status in (QMediaPlayer.LoadedMedia,
                               QMediaPlayer.BufferedMedia,
                               QMediaPlayer.BufferingMedia,
                               QMediaPlayer.EndOfMedia)
            if ready:
                self._sync_audio_position()
                self._audio.play()
            else:
                self._audio_pending_play = True
        self.btn_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))

    def pause(self):
        self._timer.stop()
        self._playing = False
        if self._has_audio:
            self._audio_pending_play = False
            self._audio.pause()
        if hasattr(self, "btn_play"):
            self.btn_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))

    def toggle_play(self):
        if self._playing:
            self.pause()
        else:
            self.play()

    def seek_to(self, idx: int):
        if not self.video_info:
            return
        idx = max(0, min(int(idx), self.video_info.frame_count - 1))
        self._show_frame_at(idx)
        self._internal_seek = True
        self.slider.setValue(idx)
        self.slider.update()
        self._internal_seek = False
        if self._has_audio:
            self._sync_audio_position()

    def seek_first(self):
        self.seek_to(0)

    def seek_last(self):
        if self.video_info:
            self.seek_to(self.video_info.frame_count - 1)

    def step(self, delta: int):
        self.seek_to(self._idx + delta)

    # --- A/B compare ---------------------------------------------------
    def enable_compare(self, before_path: str) -> bool:
        """Mount `before_path` as the secondary stream for wipe comparison.
        Primary capture (after) keeps showing as default; wipe at 0.0 hides
        the before. Accepts a video file path or an image-sequence dir."""
        if not before_path or not os.path.exists(before_path):
            return False
        self._release_before_cap()
        if is_image_sequence(before_path):
            info = ImageSequenceInfo(before_path)
            cap_path = os.path.join(before_path, info.pattern)
            cap = cv2.VideoCapture(cap_path, cv2.CAP_IMAGES)
        else:
            cap = cv2.VideoCapture(before_path)
        if not cap.isOpened():
            return False
        self._before_cap = cap
        self._before_path = before_path
        self.frame_view.set_wipe_enabled(True)
        self.frame_view.set_wipe_pos(0.0)
        # Re-render current frame with the new compositing path.
        self._show_frame_at(self._idx)
        return True

    def disable_compare(self):
        self._release_before_cap()
        self.frame_view.set_wipe_enabled(False)
        if self._cap is not None:
            self._show_frame_at(self._idx)

    def _release_before_cap(self):
        if self._before_cap is not None:
            try:
                self._before_cap.release()
            except Exception:
                pass
        self._before_cap = None
        self._before_path = None

    def _on_wipe_changed(self, pos: float):
        # Re-render the current frame with the new wipe position.
        if self._cap is not None and self.video_info is not None:
            self._show_frame_at(self._idx)

    # ------------------------------------------------------------------
    # Alpha-mask overlay
    # ------------------------------------------------------------------
    # The inpaint tab uses these to let the user verify the mask against
    # the live frame. Default state is off — other tabs never call into
    # this API so their viewers keep rendering frames untouched.
    def set_overlay_mask(self, mask):
        """Bind a binary mask (uint8 H x W). Stale masks at different
        resolutions get resized to the current frame on render."""
        self._overlay_mask = mask
        if self._raw_frame is not None:
            self._render_current()

    def clear_overlay(self):
        self._overlay_mask = None
        self._overlay_visible = False
        if self._raw_frame is not None:
            self._render_current()

    def set_overlay_visible(self, visible: bool):
        new_state = bool(visible) and self._overlay_mask is not None
        if new_state == self._overlay_visible:
            return
        self._overlay_visible = new_state
        if self._raw_frame is not None:
            self._render_current()

    def toggle_overlay(self) -> bool:
        """Flip the overlay. Returns the new visible state so callers
        can update status labels. Falls back to False if no mask is
        bound — toggling without a mask is a no-op."""
        if self._overlay_mask is None:
            return False
        self._overlay_visible = not self._overlay_visible
        if self._raw_frame is not None:
            self._render_current()
        return self._overlay_visible

    def overlay_visible(self) -> bool:
        return self._overlay_visible

    def _apply_overlay(self, frame):
        """Tint the frame red where the mask is white. Returns a new
        ndarray so the caller's cached _raw_frame stays untouched."""
        if (self._overlay_mask is None
                or not self._overlay_visible or frame is None):
            return frame
        h, w = frame.shape[:2]
        m = self._overlay_mask
        if m.shape[:2] != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        # ~50% red tint inside the mask. Blend against the original so
        # the underlying image stays readable rather than going solid.
        tint = np.zeros_like(frame)
        tint[:, :, 2] = 255  # BGR red channel
        out = frame.copy()
        mask3 = (m > 127)
        if mask3.any():
            out[mask3] = cv2.addWeighted(
                frame, 0.45, tint, 0.55, 0)[mask3]
        return out

    def _render_current(self):
        """Re-blit the cached raw frame through the overlay step.
        Used when overlay toggles without a frame seek."""
        if self._raw_frame is None:
            return
        self.frame_view.show_bgr(self._apply_overlay(self._raw_frame))

    def _composite_with_before(self, after_frame, idx: int):
        """If compare mode is on, blend the before frame in on the left side
        of the wipe line. Otherwise return the after frame untouched."""
        if self._before_cap is None or after_frame is None:
            return after_frame
        cur_b = int(self._before_cap.get(cv2.CAP_PROP_POS_FRAMES))
        if idx != cur_b:
            self._before_cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret_b, before_frame = self._before_cap.read()
        if not ret_b or before_frame is None:
            return after_frame
        h_a, w_a = after_frame.shape[:2]
        h_b, w_b = before_frame.shape[:2]
        if (h_b, w_b) != (h_a, w_a):
            before_frame = cv2.resize(before_frame, (w_a, h_a))
        wipe_pos = self.frame_view.wipe_pos()
        boundary_x = int(wipe_pos * w_a)
        if boundary_x <= 0:
            return after_frame
        composite = after_frame.copy()
        composite[:, :boundary_x] = before_frame[:, :boundary_x]
        return composite

    # --- keyboard ------------------------------------------------------
    def keyPressEvent(self, e):
        if self.video_path and self.video_info:
            if e.key() == Qt.Key_Left:
                self.step(-1)
                e.accept()
                return
            if e.key() == Qt.Key_Right:
                self.step(+1)
                e.accept()
                return
        super().keyPressEvent(e)

    def set_loop(self, on: bool):
        self._loop = bool(on)
        if hasattr(self, "btn_loop") and self.btn_loop.isChecked() != self._loop:
            self.btn_loop.blockSignals(True)
            self.btn_loop.setChecked(self._loop)
            self.btn_loop.blockSignals(False)

    def enable_save_as_menu(self, label: str = "비디오",
                            default_name: str = None):
        self._save_as_label = label
        self._save_as_default_name = default_name
        self.frame_view.setContextMenuPolicy(Qt.CustomContextMenu)
        try:
            self.frame_view.customContextMenuRequested.disconnect(self._on_context_menu)
        except Exception:
            pass
        self.frame_view.customContextMenuRequested.connect(self._on_context_menu)

    def set_save_as_default_name(self, name: str):
        self._save_as_default_name = name

    def _on_context_menu(self, pos):
        if not self.video_path:
            return
        menu = QMenu(self)
        label = self._save_as_label or "Video"
        act_save = menu.addAction(f"{label}: {tr('menu_save_as')}")
        chosen = menu.exec_(self.frame_view.mapToGlobal(pos))
        if chosen is act_save:
            self._save_as()

    def _on_frame_clicked(self):
        # Click-to-open: only fire when this player is configured for it
        # AND there's no video currently loaded.
        if self._click_to_load and self.video_path is None:
            self.empty_clicked.emit()

    def _refresh_cursor(self):
        if self._click_to_load and self.video_path is None:
            self.frame_view.setCursor(Qt.PointingHandCursor)
        else:
            self.frame_view.setCursor(Qt.ArrowCursor)

    def _save_as(self):
        if not self.video_path or not os.path.exists(self.video_path):
            return
        suggested = (self._save_as_default_name
                     or os.path.basename(self.video_path))
        path, _ = QFileDialog.getSaveFileName(
            self, f"{self._save_as_label or '비디오'} 저장",
            suggested,
            "MP4 (*.mp4);;MOV (*.mov);;MKV (*.mkv);;모든 파일 (*.*)")
        if not path:
            return
        try:
            shutil.copy2(self.video_path, path)
        except Exception as e:
            QMessageBox.warning(self, "저장 실패", str(e))

    def _release_cap(self):
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def _show_frame_at(self, idx: int):
        if not self._cap or not self.video_info:
            return
        cur = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))
        if idx != cur:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self._cap.read()
        if not ret:
            self.pause()
            return
        frame = self._composite_with_before(frame, idx)
        self._idx = idx
        self._raw_frame = frame
        self.frame_view.show_bgr(self._apply_overlay(frame))
        fps = self.video_info.fps or 30.0
        total = self.video_info.frame_count
        self.time_label.setText(
            f"{_fmt_time(idx / fps)} / {_fmt_time(total / fps)}")
        self.frame_label.setText(f"{idx + 1} / {total}")
        self.frame_changed.emit(idx)

    def _on_tick(self):
        if not self.video_info:
            return
        nxt = self._idx + 1
        wrapped = False
        if nxt >= self.video_info.frame_count:
            if self._loop:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self._idx = -1
                nxt = 0
                wrapped = True
            else:
                self.pause()
                return
        ret, frame = self._cap.read()
        if not ret:
            if self._loop:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self._cap.read()
                if not ret:
                    self.pause()
                    return
                nxt = 0
                self._idx = -1
                wrapped = True
            else:
                self.pause()
                return
        if wrapped and self._has_audio and not self.btn_mute.isChecked():
            # Video duration is often a few ms shorter than audio (encoder
            # rounding on the last partial frame). Without this, the cv2
            # tick wraps to frame 0 while audio is still playing the tail
            # of the previous lap — they desync by ~50 ms per loop. Snap
            # audio to 0 on every wrap to keep them locked.
            self._audio.setPosition(0)
            if self._audio.state() != QMediaPlayer.PlayingState:
                self._audio.play()
        frame = self._composite_with_before(frame, nxt)
        self._idx = nxt
        self._raw_frame = frame
        self.frame_view.show_bgr(self._apply_overlay(frame))
        fps = self.video_info.fps or 30.0
        total = self.video_info.frame_count
        self.time_label.setText(
            f"{_fmt_time(nxt / fps)} / {_fmt_time(total / fps)}")
        self.frame_label.setText(f"{nxt + 1} / {total}")
        self._internal_seek = True
        self.slider.setValue(nxt)
        self.slider.update()
        self._internal_seek = False
        self.frame_changed.emit(nxt)

    def _on_slider_pressed(self):
        self._scrubbing = True
        if self._playing:
            self._timer.stop()
        # Mute audio while scrubbing: setPosition spam during a drag
        # produces audible chirps, and the user only cares about the
        # landing frame anyway.
        if self._has_audio:
            self._was_audio_playing = (
                self._audio.state() == QMediaPlayer.PlayingState)
            self._audio.pause()

    def _on_slider_released(self):
        self._scrubbing = False
        if self._playing:
            self._timer.start()
        if self._has_audio:
            self._sync_audio_position()
            if self._was_audio_playing:
                self._audio.play()
            self._was_audio_playing = False

    def _on_slider_value(self, val: int):
        if self._internal_seek:
            return
        if val != self._idx:
            self._show_frame_at(val)

    def closeEvent(self, e):
        self.pause()
        self._release_cap()
        self._release_before_cap()
        self._audio.stop()
        self._audio.setMedia(QMediaContent())
        super().closeEvent(e)

    # --- audio --------------------------------------------------------
    def _setup_audio_for(self, path: str):
        """Mount the file's audio track on the parallel QMediaPlayer.

        Image sequences and audio-less videos take the silent path —
        we just disable the volume widgets and leave the player blank.
        """
        self._audio.stop()
        self._audio.setMedia(QMediaContent())
        self._audio_pending_play = False
        info = self.video_info
        has_audio = (self._audio_enabled
                     and info is not None
                     and not is_image_sequence(path)
                     and bool(getattr(info, "has_audio", False)))
        self._has_audio = has_audio
        self.btn_mute.setEnabled(has_audio)
        self.vol_slider.setEnabled(has_audio)
        if not has_audio:
            return
        # Normalize the path: QUrl.fromLocalFile chokes on a mix of forward
        # and backward slashes on Windows, leaving the URL invalid and the
        # backend silently failing to load. Pass the absolute, slash-
        # normalized form.
        abs_path = os.path.abspath(path).replace("\\", "/")
        self._audio.setMedia(QMediaContent(QUrl.fromLocalFile(abs_path)))
        self._audio.setVolume(0 if self.btn_mute.isChecked() else self._volume)

    def _sync_audio_position(self):
        if not self._has_audio or not self.video_info:
            return
        fps = self.video_info.fps or 30.0
        if fps <= 0:
            return
        self._audio.setPosition(int(self._idx * 1000.0 / fps))

    def _on_mute_toggled(self, muted: bool):
        if muted:
            self._audio.setVolume(0)
            self.btn_mute.setIcon(
                self.style().standardIcon(QStyle.SP_MediaVolumeMuted))
        else:
            self._audio.setVolume(self._volume)
            self.btn_mute.setIcon(
                self.style().standardIcon(QStyle.SP_MediaVolume))

    def _on_audio_status(self, status):
        # When setMedia finishes loading, flush any pending play() that
        # was issued while the media was still buffering.
        if status in (QMediaPlayer.LoadedMedia, QMediaPlayer.BufferedMedia):
            if self._audio_pending_play and self._has_audio:
                self._audio_pending_play = False
                self._sync_audio_position()
                self._audio.play()
        elif status == QMediaPlayer.EndOfMedia:
            # QMediaPlayer stops at EOF by default — but the cv2 timer
            # keeps ticking and wraps the video back to frame 0 in loop
            # mode, leaving the audio mute for every subsequent lap.
            # Re-seek to the start and resume to keep them in step.
            # Audio is usually a few ms longer than video (encoder
            # rounding), so without this the second loop onward is silent.
            if (self._loop and self._playing and self._has_audio
                    and not self.btn_mute.isChecked()):
                self._audio.setPosition(0)
                self._audio.play()

    def _on_audio_error(self, err):
        # QMediaPlayer swallows backend failures silently — log them so a
        # missing codec on the user's system doesn't just produce mute
        # output with no clue why. Include the enum code: errorString()
        # is often empty on the Windows backend even when err != NoError.
        names = {
            QMediaPlayer.NoError: "NoError",
            QMediaPlayer.ResourceError: "ResourceError",
            QMediaPlayer.FormatError: "FormatError",
            QMediaPlayer.NetworkError: "NetworkError",
            QMediaPlayer.AccessDeniedError: "AccessDeniedError",
            QMediaPlayer.ServiceMissingError: "ServiceMissingError",
        }
        name = names.get(err, f"error#{int(err)}")
        msg = self._audio.errorString() or "(no message)"
        media = self._audio.currentMedia()
        try:
            url = media.canonicalUrl().toString() if not media.isNull() else "(no media)"
        except Exception:
            url = "(url unavailable)"
        import sys
        print(f"[VideoPlayer] audio {name}: {msg} | url={url}",
              file=sys.stderr)

    def _on_volume_changed(self, val: int):
        self._volume = int(val)
        if self.btn_mute.isChecked() and val > 0:
            # Sliding off zero implicitly un-mutes — feels less surprising
            # than having the slider do nothing while muted.
            self.btn_mute.blockSignals(True)
            self.btn_mute.setChecked(False)
            self.btn_mute.setIcon(
                self.style().standardIcon(QStyle.SP_MediaVolume))
            self.btn_mute.blockSignals(False)
        if not self.btn_mute.isChecked():
            self._audio.setVolume(self._volume)

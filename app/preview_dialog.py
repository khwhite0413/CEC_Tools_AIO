"""Preview viewer: single VideoPlayer in A/B wipe mode (after = denoised,
before = original input). Arrow keys step ±1 frame, Space toggles play/pause."""

import os

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                              QStyle, QLabel, QWidget)

from app.video_player import VideoPlayer
from app.i18n import tr


class PreviewDialog(QDialog):
    def __init__(self, original_path: str, preview_path: str, parent=None,
                 stage_label: str = "디노이즈"):
        super().__init__(parent)
        # Each dialog instance owns its own QMediaPlayer. Without
        # WA_DeleteOnClose Qt keeps the dialog alive as a child of the
        # parent MainWindow even after exec_() returns, and re-opening
        # via "이전 미리보기 다시 보기" spawns a second player that
        # plays in parallel with the lingering old one.
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        # stage_label lets each tab brand its own preview: "디노이즈",
        # "인페인트", "업스케일". Used to be hardcoded "디노이즈".
        self.setWindowTitle(
            f"미리보기 — 원본 / {stage_label} (앞 80프레임)")
        self.resize(1280, 720)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        hint = QLabel(
            "마우스를 화면 위에 올린 뒤 가운데 핸들을 좌우로 끌어 "
            f"원본/{stage_label}를 비교하세요.")
        hint.setAlignment(Qt.AlignCenter)
        hint.setStyleSheet("color: rgba(180, 184, 200, 200);")
        layout.addWidget(hint)

        self.player = VideoPlayer(show_controls=False)
        self.player.set_loop(True)
        if os.path.exists(preview_path):
            self.player.load(preview_path)
            if os.path.exists(original_path):
                self.player.enable_compare(original_path)
        layout.addWidget(self.player, 1)

        # Master transport — laid out so the icons stay centered relative to
        # the dialog width regardless of the close button width.
        close_btn = QPushButton("닫기")
        close_btn.clicked.connect(self.accept)

        controls = QHBoxLayout()
        style = self.style()

        # Invisible spacer on the left, the same width as the close button,
        # so the centered transport row sits in the visual middle of the
        # dialog rather than being pushed left by the close button.
        left_spacer = QWidget()
        left_spacer.setFixedWidth(close_btn.sizeHint().width())
        controls.addWidget(left_spacer)
        controls.addStretch(1)

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
            controls.addWidget(b)

        # Re-parent the player's volume widgets into this dialog's
        # transport row. The player is constructed with show_controls
        # =False so its own control row is hidden; the widgets stay
        # wired to the player's audio slots, so simply adding them
        # into this layout makes them visible and functional here.
        controls.addWidget(self.player.btn_mute)
        controls.addWidget(self.player.vol_slider)

        controls.addStretch(1)
        controls.addWidget(close_btn)
        layout.addLayout(controls)

        self.btn_first.clicked.connect(self.player.seek_first)
        self.btn_prev.clicked.connect(lambda: self.player.step(-1))
        self.btn_play.clicked.connect(self._toggle_play)
        self.btn_next.clicked.connect(lambda: self.player.step(+1))
        self.btn_last.clicked.connect(self.player.seek_last)
        self.btn_loop.toggled.connect(self.player.set_loop)

        # Auto-play and refresh play icon based on the player state.
        self.player.play()
        self._refresh_play_icon()

    def _toggle_play(self):
        self.player.toggle_play()
        self._refresh_play_icon()

    def _refresh_play_icon(self):
        playing = self.player._playing
        self.btn_play.setIcon(self.style().standardIcon(
            QStyle.SP_MediaPause if playing else QStyle.SP_MediaPlay))

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Left:
            self.player.step(-1)
            e.accept()
            return
        if e.key() == Qt.Key_Right:
            self.player.step(+1)
            e.accept()
            return
        super().keyPressEvent(e)

    def closeEvent(self, e):
        # Full teardown — clear() stops the cv2 capture AND drops the
        # audio media. pause() alone leaves QMediaPlayer in Paused
        # state with the file still loaded, which on the WMF backend
        # can keep flushing buffered audio for a moment after the
        # dialog vanishes — the user heard that as lingering sound.
        self.player.clear()
        super().closeEvent(e)

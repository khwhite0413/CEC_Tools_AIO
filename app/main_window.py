"""CEC All-In-One main window (pure PyQt5).

Hosts three tabs: Inpaint (ProPainter) | Denoise (FlowDenoise) | About.
Denoise logic is identical to FlowDenoise v1.0.0; the Inpaint tab is
new and lives in app/inpaint_page.py.
"""

import os
import shutil
import time

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDragEnterEvent, QDropEvent, QPixmap, QPalette, QColor
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                              QLabel, QPushButton, QFileDialog, QProgressBar,
                              QPlainTextEdit, QMessageBox, QScrollArea,
                              QSplitter, QLineEdit, QGroupBox, QStackedWidget,
                              QFrame, QButtonGroup, QComboBox)

from app.parameter_panel import ParameterPanel
from app.preview_dialog import PreviewDialog
from app.video_player import VideoPlayer
from app.workers import DenoiseWorker, PreviewWorker, derive_output_path, probe_video
from app.about_page import AboutPage
from app.inpaint_page import InpaintPage
from app.i18n import tr, get_lang, set_lang, LANGS, LANG_LABELS


_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_MODULE_DIR, ".."))
_CACHE_DIR = os.path.join(_PROJECT_ROOT, "cache")
_OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "output")
_LOGO_DIR = os.path.join(_PROJECT_ROOT, "assets", "logo")


# Files / folders the pipeline writes into cache/ for preview playback.
# Listed explicitly so the cleanup never touches anything the user might
# have parked in cache/ themselves.
_PREVIEW_CACHE_FILES = (
    "preview.mp4", "preview.mov",
    "noise_preview.mp4", "noise_preview_short.mp4",
    # Viewer copy of a full-run deliverable: the in-app wipe player
    # holds a cv2 handle on whatever it's loaded, which locks the file
    # on Windows. We stage a copy here so the user can freely rename /
    # move / delete the real output in output/ during the session.
    "result_view.mp4", "result_view.mov",
)
_PREVIEW_CACHE_DIRS = ("preview_seq", "result_view_seq")


def _cleanup_preview_cache(cache_dir: str) -> None:
    """Best-effort delete of preview / noise cache artefacts.

    Called both on app start and on close: the close-side delete can fail
    silently when cv2 or ffmpeg still holds a Windows file handle on the
    output (PreviewDialog had a player open, the OS hasn't released the
    lock yet, etc.). Calling the same routine on the next start guarantees
    leftovers get cleaned up eventually instead of accumulating.
    """
    if not os.path.isdir(cache_dir):
        return
    for name in _PREVIEW_CACHE_FILES:
        p = os.path.join(cache_dir, name)
        try:
            if os.path.isfile(p):
                os.remove(p)
        except Exception:
            pass
    for name in _PREVIEW_CACHE_DIRS:
        p = os.path.join(cache_dir, name)
        try:
            if os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)
        except Exception:
            pass


_TAB_BTN_QSS = """
QPushButton {
    background: transparent;
    border: none;
    padding: 4px 12px;
    font-size: 9pt;
    color: rgba(200, 204, 220, 180);
}
QPushButton:hover { color: rgba(220, 224, 240, 240); }
QPushButton:checked {
    color: rgba(220, 224, 240, 255);
    border-bottom: 2px solid #7aa2f7;
    font-weight: 500;
}
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CEC All-In-One v0.1.0")
        self.resize(1320, 860)
        self.setAcceptDrops(True)

        self.video_path = None
        self.worker = None
        self._t_start = 0.0
        self._last_done = 0
        self._last_total = 0
        # Last completed preview's output + the input it came from, so the
        # user can reopen the side-by-side dialog after closing it.
        self._last_preview_path = None
        self._last_preview_source = None
        self.log_path = None
        os.makedirs(_CACHE_DIR, exist_ok=True)
        os.makedirs(_OUTPUT_DIR, exist_ok=True)
        # Sweep any preview leftovers from a prior session — closeEvent's
        # cleanup can race the OS file lock when cv2 hasn't released the
        # cap yet, leaving artefacts behind.
        _cleanup_preview_cache(_CACHE_DIR)

        central = QWidget()
        self.setCentralWidget(central)
        shell = QVBoxLayout(central)
        shell.setContentsMargins(0, 0, 0, 0)
        shell.setSpacing(0)

        shell.addWidget(self._build_header())
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(
            "background-color: rgba(120,120,140,40); border: none; max-height: 1px;")
        sep.setMaximumHeight(1)
        shell.addWidget(sep)

        self.stack = QStackedWidget()
        shell.addWidget(self.stack, 1)

        self.inpaint_page = InpaintPage()
        self.denoise_page = self._build_denoise_page()
        self.about_page = AboutPage()
        # Tab order: 0 = Inpaint, 1 = Denoise, 2 = About.
        self.stack.addWidget(self.inpaint_page)
        self.stack.addWidget(self.denoise_page)
        self.stack.addWidget(self.about_page)

        self._log("CEC All-In-One 시작.")
        self._log_torch_status()

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    def _build_header(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(40)
        h = QHBoxLayout(bar)
        h.setContentsMargins(14, 5, 12, 5)
        h.setSpacing(10)

        self.header_logo = QLabel()
        self.header_logo.setFixedSize(22, 22)
        self.header_logo.setScaledContents(True)
        logo_path = os.path.join(_LOGO_DIR, "aimz_symbol.png")
        if os.path.exists(logo_path):
            self.header_logo.setPixmap(QPixmap(logo_path))
        h.addWidget(self.header_logo)

        brand = QLabel("CEC All-In-One")
        brand.setStyleSheet("font-size: 9pt; font-weight: 500;")
        h.addWidget(brand)

        h.addStretch(1)

        # Tab buttons (checkable, mutually exclusive). Inpaint is the
        # first tab so it's the one users land on at startup.
        self.tab_group = QButtonGroup(self)
        self.tab_group.setExclusive(True)
        self.btn_tab_inpaint = QPushButton(tr("tab_inpaint"))
        self.btn_tab_inpaint.setCheckable(True)
        self.btn_tab_inpaint.setChecked(True)
        self.btn_tab_inpaint.setStyleSheet(_TAB_BTN_QSS)
        self.btn_tab_inpaint.clicked.connect(lambda: self.stack.setCurrentIndex(0))
        self.btn_tab_denoise = QPushButton(tr("tab_denoise"))
        self.btn_tab_denoise.setCheckable(True)
        self.btn_tab_denoise.setStyleSheet(_TAB_BTN_QSS)
        self.btn_tab_denoise.clicked.connect(lambda: self.stack.setCurrentIndex(1))
        self.btn_tab_about = QPushButton(tr("tab_about"))
        self.btn_tab_about.setCheckable(True)
        self.btn_tab_about.setStyleSheet(_TAB_BTN_QSS)
        self.btn_tab_about.clicked.connect(lambda: self.stack.setCurrentIndex(2))
        self.tab_group.addButton(self.btn_tab_inpaint)
        self.tab_group.addButton(self.btn_tab_denoise)
        self.tab_group.addButton(self.btn_tab_about)
        h.addWidget(self.btn_tab_inpaint)
        h.addWidget(self.btn_tab_denoise)
        h.addWidget(self.btn_tab_about)

        h.addStretch(1)

        # Language combo
        self.lang_combo = QComboBox()
        for code in LANGS:
            self.lang_combo.addItem(LANG_LABELS[code])
        cur = get_lang()
        try:
            self.lang_combo.setCurrentIndex(LANGS.index(cur))
        except ValueError:
            self.lang_combo.setCurrentIndex(0)
        self.lang_combo.setFixedWidth(110)
        self.lang_combo.setToolTip(tr("tooltip_lang"))
        self.lang_combo.currentIndexChanged.connect(self._on_lang_changed)
        h.addWidget(self.lang_combo)

        return bar

    def _on_lang_changed(self, idx: int):
        if idx < 0 or idx >= len(LANGS):
            return
        new_lang = LANGS[idx]
        if new_lang == get_lang():
            return
        set_lang(new_lang)
        QMessageBox.information(self, "FlowDenoise", tr("restart_to_apply"))

    # ------------------------------------------------------------------
    # Denoise page (extracted from old __init__)
    # ------------------------------------------------------------------
    def _build_denoise_page(self) -> QWidget:
        page = QWidget()
        root = QHBoxLayout(page)
        root.setContentsMargins(16, 12, 16, 16)
        root.setSpacing(12)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter, 1)

        # ----- Left: video player + meta + actions + log -----
        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)

        self.player = VideoPlayer(click_to_load=True)
        self.player.empty_clicked.connect(self._pick_file)
        left_lay.addWidget(self.player, 1)

        meta_row = QHBoxLayout()
        self.info_label = QLabel(tr("info_no_video"))
        self.info_label.setWordWrap(True)
        pick = QPushButton(tr("btn_pick_file"))
        pick.clicked.connect(self._pick_file)
        meta_row.addWidget(self.info_label, 1)
        meta_row.addWidget(pick, 0)
        left_lay.addLayout(meta_row)

        out_row = QHBoxLayout()
        out_row.addWidget(QLabel(tr("label_output")))
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText(tr("placeholder_output"))
        self.output_edit.setToolTip(tr("tooltip_output_path"))
        out_row.addWidget(self.output_edit, 1)
        browse = QPushButton(tr("btn_browse"))
        browse.clicked.connect(self._browse_output)
        out_row.addWidget(browse, 0)
        left_lay.addLayout(out_row)

        btn_row = QHBoxLayout()
        self.btn_preview = QPushButton(tr("btn_preview"))
        self.btn_preview.clicked.connect(self._run_preview)
        self.btn_show_preview = QPushButton(tr("btn_show_preview"))
        self.btn_show_preview.clicked.connect(self._show_last_preview)
        self.btn_show_preview.setEnabled(False)
        self.btn_run = QPushButton(tr("btn_run"))
        self.btn_run.setStyleSheet("font-weight: bold;")
        self.btn_run.clicked.connect(self._run_full)
        self.btn_cancel = QPushButton(tr("btn_cancel"))
        self.btn_cancel.clicked.connect(self._cancel)
        self.btn_cancel.setEnabled(False)
        btn_row.addWidget(self.btn_preview)
        btn_row.addWidget(self.btn_show_preview)
        btn_row.addWidget(self.btn_run)
        btn_row.addWidget(self.btn_cancel)
        left_lay.addLayout(btn_row)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        left_lay.addWidget(self.progress)
        # Snapshot the default palette so we can restore it after a cancel.
        self._progress_normal_palette = QPalette(self.progress.palette())
        # Soft green for the completion state — visually distinct from
        # both the default in-progress blue and the cancellation red.
        self._progress_done_palette = QPalette(self.progress.palette())
        self._progress_done_palette.setColor(
            QPalette.Highlight, QColor("#a5d6a7"))

        self.eta_label = QLabel("")
        self.eta_label.setAlignment(Qt.AlignCenter)
        left_lay.addWidget(self.eta_label)

        log_header = QHBoxLayout()
        log_header.addWidget(QLabel(tr("label_log")))
        log_header.addStretch(1)
        self.btn_open_log = QPushButton(tr("btn_open_log"))
        self.btn_open_log.clicked.connect(self._open_log_file)
        log_header.addWidget(self.btn_open_log)
        left_lay.addLayout(log_header)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        self.log.setMaximumHeight(140)
        left_lay.addWidget(self.log, 0)

        splitter.addWidget(left)

        # ----- Right: parameter panel + noise mini viewer -----
        right_pane = QWidget()
        right_lay = QVBoxLayout(right_pane)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.setSpacing(12)

        self.params = ParameterPanel()
        self.params.format_changed.connect(self._on_format_changed)
        scroll = QScrollArea()
        scroll.setWidget(self.params)
        scroll.setWidgetResizable(True)
        right_lay.addWidget(scroll, 3)

        self.noise_box = QGroupBox(tr("noise_box_title"))
        nb_lay = QVBoxLayout(self.noise_box)
        # Noise preview is a residual stream — there's no meaningful
        # audio to play, and the source video's track piped through
        # would just be confusing. Silence the player explicitly.
        self.noise_player = VideoPlayer(show_controls=True,
                                        audio_enabled=False)
        self.noise_player.set_loop(True)
        self.noise_player.enable_save_as_menu(
            label=tr("noise_save_label"),
            default_name="noise_preview.mp4")
        nb_lay.addWidget(self.noise_player)
        self.noise_box.setVisible(False)
        right_lay.addWidget(self.noise_box, 4)

        right_pane.setMinimumWidth(360)
        right_pane.setMaximumWidth(520)

        splitter.addWidget(right_pane)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([900, 420])

        return page

    def _log(self, msg: str):
        self.log.appendPlainText(msg)

    def _log_torch_status(self):
        try:
            import torch
            cuda = torch.cuda.is_available()
            self._log(f"PyTorch {torch.__version__}, CUDA={'사용 가능' if cuda else '없음 (CPU 모드)'}")
            if cuda:
                self._log(f"GPU: {torch.cuda.get_device_name(0)}")
        except Exception as e:
            self._log(f"PyTorch 미설치 또는 import 실패: {e}")

    def dragEnterEvent(self, e: QDragEnterEvent):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e: QDropEvent):
        for url in e.mimeData().urls():
            p = url.toLocalFile()
            if p:
                self._set_video(p)
                break

    def _set_video(self, path: str):
        if not os.path.isfile(path):
            QMessageBox.warning(self, "FlowDenoise", f"파일이 존재하지 않습니다:\n{path}")
            return
        try:
            info = probe_video(path)
        except Exception as e:
            QMessageBox.critical(self, "FlowDenoise", f"비디오 열기 실패:\n{e}")
            return
        # Drop any previous wipe-compare overlay, then load the new input.
        try:
            self.player.disable_compare()
        except Exception:
            pass
        self.player.load(path)
        self.video_path = path
        audio = "있음" if info.has_audio else "없음"
        self.info_label.setText(
            f"<b>{os.path.basename(path)}</b> &nbsp; "
            f"{info.width}×{info.height} @ {info.fps:.2f}fps · "
            f"{info.frame_count} frames · {info.duration:.1f}s · audio: {audio}")
        # Surface the snapped EFFECTIVE fps in the Output Settings card —
        # this is what the pipeline will actually use end-to-end (24.04
        # source metadata → 24.0 after auto-snap). The raw 24.04 stays
        # visible on the info line above the player.
        self.params.set_input_fps(getattr(info, "effective_fps", info.fps))
        self.output_edit.setText(
            self._default_output_path(path, self.params.current_format()))
        # Drop any cached preview reference — it belongs to the previous input.
        self._last_preview_path = None
        self._last_preview_source = None
        self.btn_show_preview.setEnabled(False)
        self._log(f"입력 로드: {path}")

    def _pick_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "비디오 선택", "",
            "비디오 (*.mp4 *.mov *.mkv *.avi *.webm *.m4v);;모든 파일 (*.*)")
        if path:
            self._set_video(path)

    def _browse_output(self):
        start = self.output_edit.text().strip() or (self.video_path or "")
        path, _ = QFileDialog.getSaveFileName(
            self, "출력 파일 저장 위치", start,
            "MP4 (*.mp4);;MOV (*.mov);;MKV (*.mkv);;모든 파일 (*.*)")
        if path:
            self.output_edit.setText(path)

    def _ensure_video(self) -> bool:
        if not self.video_path:
            QMessageBox.information(self, "FlowDenoise", "먼저 비디오 파일을 선택하세요.")
            return False
        return True

    def _start(self, worker_cls, suffix: str):
        if not self._ensure_video():
            return
        settings = self.params.settings()
        if suffix == "_preview":
            # Preview is ephemeral and shown by the cv2-based in-app
            # VideoPlayer. Force h264 here regardless of the user's chosen
            # full-run format: cv2 ↔ ffmpeg ↔ cv2 with h264 (no color flags)
            # is the cycle that keeps mp4 → mp4 passthrough color-stable.
            # ProRes/PNG are deliverable formats, irrelevant to the wipe.
            settings["output_format"] = "h264"
            out_path = os.path.join(_CACHE_DIR, "preview.mp4")
            # Disable "show last preview" while a new one is being baked —
            # the cache file is being overwritten and would play partially.
            self.btn_show_preview.setEnabled(False)
        else:
            user_path = self.output_edit.text().strip()
            fmt = settings.get("output_format", "h264")
            if user_path:
                # Explicit path — only adjust the extension to match the codec.
                out_path = self._force_format_extension(user_path, fmt)
            else:
                # Empty field falls back to project/output/<basename>_denoised.ext
                out_path = self._default_output_path(self.video_path, fmt)
            out_dir = os.path.dirname(out_path)
            if out_dir and not os.path.isdir(out_dir):
                try:
                    os.makedirs(out_dir, exist_ok=True)
                except Exception as e:
                    QMessageBox.critical(self, "FlowDenoise",
                                         f"출력 폴더 생성 실패:\n{e}")
                    return
        self._log(f"처리 시작 → {out_path}")
        self._log(f"  flow_model={settings['flow_model']}, "
                  f"window_size={settings['window_size']}, "
                  f"batch_size={settings['batch_size']}, "
                  f"noise_preview={settings['compute_noise_preview']}")

        noise_path = None
        if settings.get("compute_noise_preview"):
            self.noise_player.clear()
            self.noise_box.setVisible(False)
            noise_name = ("noise_preview_short.mp4" if suffix == "_preview"
                          else "noise_preview.mp4")
            noise_path = os.path.join(_CACHE_DIR, noise_name)
            try:
                if os.path.exists(noise_path):
                    os.remove(noise_path)
            except Exception:
                pass

        self.worker = worker_cls(self.video_path, settings, out_path,
                                 noise_path=noise_path)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_signal.connect(self._on_finished)
        self.worker.error_signal.connect(self._on_error)
        self.worker.noise_ready.connect(self._on_noise_ready)
        self.worker.phase_changed.connect(self._on_phase_changed)
        self.worker.cancelled_signal.connect(self._on_cancelled)
        # Reset visual state for the new run (clears any prior cancel state).
        self.progress.setPalette(self._progress_normal_palette)
        self.progress.setValue(0)
        self.eta_label.setText("")
        self._last_done = 0
        self._last_total = 0
        self._t_start = time.time()
        self._set_running(True)
        self.worker.start()

    def _run_preview(self):
        self._start(PreviewWorker, suffix="_preview")

    def _run_full(self):
        self._start(DenoiseWorker, suffix="_denoised")

    def _cancel(self):
        if self.worker is not None:
            self.worker.cancel()
            self._log("취소 요청됨...")

    def _set_running(self, running: bool):
        self.btn_run.setEnabled(not running)
        self.btn_preview.setEnabled(not running)
        self.btn_cancel.setEnabled(running)

    @staticmethod
    def _force_format_extension(path: str, fmt: str) -> str:
        if fmt == "image_seq":
            # Image sequence is a folder, not a file — drop any extension.
            base, _ = os.path.splitext(path)
            return base
        target_ext = ".mov" if fmt == "prores" else ".mp4"
        base, _ = os.path.splitext(path)
        return base + target_ext

    @classmethod
    def _default_output_path(cls, input_path: str, fmt: str) -> str:
        """Default destination when the user hasn't set an explicit path.
        Path lives under <project>/output/<basename>_denoised<ext>. If
        that exact path already exists (same extension), append _001,
        _002, ... until a free name is found. Different extensions
        (e.g. .mp4 vs .mov vs image_seq folder) don't collide with each
        other — only same-extension duplicates bump the counter, so
        re-running with a different output_format keeps the clean name.
        """
        os.makedirs(_OUTPUT_DIR, exist_ok=True)
        base = os.path.splitext(os.path.basename(input_path))[0]
        ext = "" if fmt == "image_seq" else (".mov" if fmt == "prores" else ".mp4")

        candidate = os.path.join(_OUTPUT_DIR, f"{base}_denoised{ext}")
        if not os.path.exists(candidate):
            return candidate

        n = 1
        while True:
            candidate = os.path.join(
                _OUTPUT_DIR, f"{base}_denoised_{n:03d}{ext}")
            if not os.path.exists(candidate):
                return candidate
            n += 1

    def _on_format_changed(self, fmt: str):
        # Keep the visible output path in sync so the user always sees the
        # extension they'll actually get.
        cur = self.output_edit.text().strip()
        if not cur:
            return
        new = self._force_format_extension(cur, fmt)
        if new != cur:
            self.output_edit.setText(new)

    def _on_phase_changed(self, phase: str):
        if phase == "processing":
            return
        if phase == "encoding":
            self.progress.setValue(100)
            self.eta_label.setText("마무리 중 — 비디오 인코딩")
            self.btn_cancel.setEnabled(False)
        elif phase == "encoding_noise":
            self.eta_label.setText("마무리 중 — 노이즈 미리보기 인코딩")
        elif phase == "done":
            self.eta_label.setText("저장 완료, 마감 처리 중...")

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
                f"{done}/{total} | 경과 {elapsed:.0f}s | 남은 시간 ≈ {eta:.0f}s")

    def _on_cancelled(self):
        self._set_running(False)
        elapsed = time.time() - self._t_start
        # Color the chunk red via palette — QSS on QProgressBar would flip
        # the bar to a non-native renderer and shift the percentage text.
        pal = QPalette(self.progress.palette())
        pal.setColor(QPalette.Highlight, QColor("#e57373"))
        self.progress.setPalette(pal)
        # Same eta_label slot, alignment unchanged — only the text changes.
        self.eta_label.setText(tr("cancelled_summary").format(
            done=self._last_done, total=self._last_total, elapsed=elapsed))
        self._log(f"취소됨 · {self._last_done}/{self._last_total} · {elapsed:.1f}s")
        self.worker = None

    def _on_finished(self, output_path: str):
        self._set_running(False)
        self.progress.setValue(100)
        elapsed = time.time() - self._t_start
        # Mark the bar visually as "done" — soft green palette + a real
        # completion summary in the eta slot. Without this the bar stays
        # blue and eta_label stays stuck on "저장 완료, 마감 처리 중…"
        # from _on_phase_changed("done"), making it ambiguous whether
        # the run actually finished or hung at the very end.
        self.progress.setPalette(self._progress_done_palette)
        self.eta_label.setText(tr("finished_summary").format(
            done=self._last_done, total=self._last_total, elapsed=elapsed))
        self._log(f"완료 ({elapsed:.1f}s) → {output_path}")
        was_preview = isinstance(self.worker, PreviewWorker)
        self.worker = None
        if not output_path or not os.path.exists(output_path):
            return
        if was_preview:
            # Remember this preview so the user can reopen it later.
            self._last_preview_path = output_path
            self._last_preview_source = self.video_path
            self.btn_show_preview.setEnabled(True)
            dlg = PreviewDialog(self.video_path, output_path, self)
            dlg.exec_()
        else:
            # Full processing finished — swap the main viewer to a CACHED
            # COPY of the result with the input as the wipe "before". The
            # cv2 handle the player keeps would lock the real deliverable
            # in output/, so the user couldn't rename / move it without
            # closing the app first. Staging a copy in cache/ keeps the
            # output writable and the cleanup at exit (and at next
            # startup) sweeps the dummy.
            input_path = self.video_path
            try:
                self.player.disable_compare()
                viewer_path = self._stage_for_viewer(output_path)
                self.player.load(viewer_path)
                if input_path and os.path.exists(input_path):
                    self.player.enable_compare(input_path)
                self._log("메인 뷰어가 결과물로 전환됨 (와이프로 원본 비교 가능).")
            except Exception as e:
                self._log(f"[WARN] 결과 미리보기 전환 실패: {e}")

    @staticmethod
    def _stage_for_viewer(output_path: str) -> str:
        """Copy the deliverable into cache/result_view.<ext> so the
        in-app player can hold a cv2 handle on it without locking the
        user's real output. Falls back to the original path on any
        copy failure — viewer-locking is preferable to losing the
        ability to show the result at all."""
        if os.path.isdir(output_path):
            dst = os.path.join(_CACHE_DIR, "result_view_seq")
            try:
                if os.path.isdir(dst):
                    shutil.rmtree(dst, ignore_errors=True)
                shutil.copytree(output_path, dst)
                return dst
            except Exception:
                return output_path
        ext = os.path.splitext(output_path)[1].lower() or ".mp4"
        if ext not in (".mp4", ".mov"):
            ext = ".mp4"
        dst = os.path.join(_CACHE_DIR, f"result_view{ext}")
        try:
            if os.path.exists(dst):
                os.remove(dst)
            shutil.copy2(output_path, dst)
            return dst
        except Exception:
            return output_path

    def _show_last_preview(self):
        path = self._last_preview_path
        source = self._last_preview_source or self.video_path
        if not path or not os.path.exists(path):
            self.btn_show_preview.setEnabled(False)
            self._last_preview_path = None
            self._last_preview_source = None
            QMessageBox.information(
                self, "FlowDenoise",
                "미리보기 파일을 찾을 수 없습니다. 다시 생성해 주세요.")
            return
        dlg = PreviewDialog(source, path, self)
        dlg.exec_()

    def _on_error(self, msg: str):
        self._set_running(False)
        self._log(f"[ERROR] {msg}")
        self.worker = None
        QMessageBox.critical(self, "FlowDenoise", msg)

    def _on_noise_ready(self, path: str):
        if not path or not os.path.exists(path):
            return
        try:
            self.noise_player.load(path)
            base = os.path.splitext(os.path.basename(self.video_path or "noise"))[0]
            self.noise_player.set_save_as_default_name(f"{base}_noise.mp4")
            self.noise_player.play()
            self.noise_box.setVisible(True)
            self._log(f"노이즈 미리보기 준비됨 (캐시): {path}")
        except Exception as e:
            self._log(f"[ERROR] 노이즈 미리보기 로드 실패: {e}")

    def _open_log_file(self):
        if not self.log_path or not os.path.exists(self.log_path):
            QMessageBox.information(self, "FlowDenoise",
                                    "로그 파일이 아직 없습니다.")
            return
        try:
            os.startfile(self.log_path)  # type: ignore[attr-defined]
        except Exception as e:
            QMessageBox.warning(self, "FlowDenoise",
                                f"로그 파일을 열지 못했습니다:\n{e}")

    def closeEvent(self, e):
        if self.worker is not None and self.worker.isRunning():
            self._log("종료 중 — 작업을 취소합니다...")
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
        try:
            self.noise_player.clear()
        except Exception:
            pass
        try:
            self.inpaint_page.shutdown()
        except Exception:
            pass
        # Force the cv2 caps to drop their Windows file handles before we
        # try to delete the underlying files — without this gc step the
        # remove() can silently fail and leave the cache file behind.
        import gc
        gc.collect()
        _cleanup_preview_cache(_CACHE_DIR)
        super().closeEvent(e)

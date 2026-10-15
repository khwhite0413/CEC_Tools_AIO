"""Standalone Frame Interpolation tab."""

import os
import shutil
import time

import torch
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QSplitter,
    QLabel, QPushButton, QLineEdit, QFileDialog, QProgressBar, QPlainTextEdit,
    QGroupBox, QFormLayout, QComboBox, QDoubleSpinBox, QCheckBox,
    QMessageBox, QScrollArea, QFrame, QSizePolicy)

from app.interpolation_fps import build_fps_plan, output_frame_count
from app.interpolation_models import INTERPOLATION_MODELS, model_available
from app.interpolation_worker import (
    DEFAULT_SCENE_CUT_THRESHOLD, InterpolationWorker)
from app.video_player import VideoPlayer
from fdcore.video_io import VideoInfo, auto_format_from_codec


_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_OUTPUT = os.path.join(_ROOT, "output")
_CACHE = os.path.join(_ROOT, "cache")
_FPS_PRESETS = (23.976, 24.0, 25.0, 29.97, 30.0, 48.0, 50.0,
                59.94, 60.0, 100.0, 119.88, 120.0)


class InterpolationPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.video_path = None
        self.video_info = None
        self.video_codec = ""
        self.worker = None
        self._pending_video_path = None
        self._started = 0.0
        self._last_done = self._last_total = 0
        os.makedirs(_OUTPUT, exist_ok=True)

        root = QHBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 16)
        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter, 1)
        splitter.addWidget(self._build_left())
        splitter.addWidget(self._build_right())
        splitter.setStretchFactor(0, 1)
        splitter.setSizes([900, 420])

    def _build_left(self):
        panel = QWidget()
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)
        self.player = VideoPlayer(click_to_load=True)
        self.player.empty_clicked.connect(self._pick_file)
        lay.addWidget(self.player, 1)

        row = QHBoxLayout()
        self.info_label = QLabel("영상이 선택되지 않았습니다.")
        self.info_label.setWordWrap(True)
        pick = QPushButton("파일 선택")
        pick.clicked.connect(self._pick_file)
        row.addWidget(self.info_label, 1)
        row.addWidget(pick)
        lay.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("출력"))
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("출력 파일 경로")
        browse = QPushButton("찾아보기")
        browse.clicked.connect(self._browse_output)
        row.addWidget(self.output_edit, 1)
        row.addWidget(browse)
        lay.addLayout(row)

        self.fps_summary = QLabel("입력 영상을 선택하면 최종 FPS를 계산합니다.")
        self.fps_summary.setStyleSheet("color: rgba(180,184,200,210); padding: 3px 0;")
        lay.addWidget(self.fps_summary)

        row = QHBoxLayout()
        self.btn_run = QPushButton("전체 처리")
        self.btn_run.setStyleSheet("font-weight: bold;")
        self.btn_run.clicked.connect(self._run)
        self.btn_cancel = QPushButton("취소")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._cancel)
        row.addWidget(self.btn_run)
        row.addWidget(self.btn_cancel)
        lay.addLayout(row)

        self.progress = QProgressBar()
        lay.addWidget(self.progress)
        self._normal_palette = QPalette(self.progress.palette())
        self._done_palette = QPalette(self.progress.palette())
        self._done_palette.setColor(QPalette.Highlight, QColor("#a5d6a7"))
        self.eta_label = QLabel("")
        self.eta_label.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.eta_label)
        lay.addWidget(QLabel("로그"))
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        self.log.setMaximumHeight(140)
        lay.addWidget(self.log)
        return panel

    def _build_right(self):
        right = QWidget()
        right.setMinimumWidth(360)
        right.setMaximumWidth(520)
        outer = QVBoxLayout(right)
        outer.setContentsMargins(0, 0, 0, 0)

        inner = QWidget()
        inner_lay = QVBoxLayout(inner)
        inner_lay.setContentsMargins(0, 0, 0, 0)

        params = QGroupBox("프레임 보간")
        params.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        form = QFormLayout(params)
        form.setLabelAlignment(Qt.AlignRight)

        self.model_combo = QComboBox()
        for key, item in INTERPOLATION_MODELS.items():
            self.model_combo.addItem(f"{item['rank']}위 · {item['name']}", key)
            self.model_combo.setItemData(
                self.model_combo.count() - 1,
                f"{item['description']} · {item['license']}", Qt.ToolTipRole)
        form.addRow("모델", self.model_combo)

        self.multiplier_combo = QComboBox()
        for value in (2, 4, 8):
            self.multiplier_combo.addItem(f"{value}x", value)
        self.multiplier_combo.currentIndexChanged.connect(self._refresh_fps_summary)
        form.addRow("보간 배율", self.multiplier_combo)

        self.output_fps_combo = QComboBox()
        self.output_fps_combo.addItem("끄기 (처리된 FPS 그대로)", None)
        for fps in _FPS_PRESETS:
            label = f"{fps:g}" if fps == int(fps) else f"{fps:.3f}"
            self.output_fps_combo.addItem(label, fps)
        self.output_fps_combo.addItem("직접 입력", -1.0)
        self.output_fps_combo.currentIndexChanged.connect(self._output_fps_changed)
        form.addRow("최종 출력 FPS", self.output_fps_combo)

        self.custom_fps = QDoubleSpinBox()
        self.custom_fps.setRange(1.0, 240.0)
        self.custom_fps.setDecimals(3)
        self.custom_fps.setValue(60.0)
        self.custom_fps.setVisible(False)
        self.custom_fps.valueChanged.connect(self._refresh_fps_summary)
        form.addRow("직접 입력", self.custom_fps)

        self.fp16 = QCheckBox("FP16 사용 (NVIDIA GPU 권장)")
        self.fp16.setChecked(True)
        form.addRow(self.fp16)
        self.scene_cut = QCheckBox("장면 전환 자동 보호")
        self.scene_cut.setChecked(True)
        self.scene_cut.setToolTip("장면 전환 사이에 가짜 중간 프레임을 만들지 않습니다.")
        form.addRow(self.scene_cut)
        self.scene_detection_mode = QComboBox()
        self.scene_detection_mode.addItem("기존 MSE (기본)", "mse")
        self.scene_detection_mode.addItem(
            "PySceneDetect Adaptive", "pyscenedetect")
        self.scene_detection_mode.setToolTip(
            "기존 임계값 방식 또는 주변 프레임을 참고하는 Adaptive 방식을 선택합니다.")
        self.scene_cut.toggled.connect(self.scene_detection_mode.setEnabled)
        form.addRow("장면 전환 감지", self.scene_detection_mode)
        inner_lay.addWidget(params)

        fmt = QGroupBox("최종 출력")
        fmt_form = QFormLayout(fmt)
        fmt_form.setLabelAlignment(Qt.AlignRight)
        self.format_combo = QComboBox()
        self.format_combo.addItem("자동 (입력 코덱 기준)", "auto")
        self.format_combo.addItem("H.264 / MP4", "h264")
        self.format_combo.addItem("ProRes / MOV", "prores")
        self.format_combo.currentIndexChanged.connect(self._format_changed)
        fmt_form.addRow("포맷", self.format_combo)
        self.prores_profile = QComboBox()
        for label, key in (("Proxy", "proxy"), ("LT", "lt"), ("422", "standard"),
                           ("422 HQ", "hq"), ("4444", "4444"), ("4444 XQ", "4444xq")):
            self.prores_profile.addItem(label, key)
        self.prores_profile.setCurrentIndex(4)
        self.prores_profile.setVisible(False)
        self.profile_label = QLabel("ProRes 프로파일")
        self.profile_label.setVisible(False)
        fmt_form.addRow(self.profile_label, self.prores_profile)
        inner_lay.addWidget(fmt)
        inner_lay.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(inner)
        outer.addWidget(scroll)
        return right

    def _pick_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "비디오 선택", "",
            "비디오 (*.mp4 *.mov *.mkv *.avi *.webm *.m4v);;모든 파일 (*.*)")
        if path:
            self._set_video(path)

    def _set_video(self, path):
        if not os.path.isfile(path):
            QMessageBox.warning(self, "Frame Interpolation", "파일이 존재하지 않습니다.")
            return
        if self.worker is not None:
            self._pending_video_path = path
            self.worker.cancel()
            self._log(
                "새 입력 대기 중 — 현재 작업이 완전히 종료되면 불러옵니다.")
            return
        self._apply_video(path)

    def _apply_video(self, path):
        try:
            info = VideoInfo(path)
        except Exception as exc:
            QMessageBox.critical(self, "Frame Interpolation", f"비디오 열기 실패:\n{exc}")
            return
        self.video_path, self.video_info = path, info
        self.video_codec = (info.codec_name or "").lower()
        self.player.load(path)
        self.info_label.setText(
            f"<b>{os.path.basename(path)}</b> &nbsp; {info.width}×{info.height} @ "
            f"{info.fps:.3f}fps · {info.frame_count} frames · {info.duration:.1f}s")
        self.output_edit.setText(self._default_output_path(path))
        self._format_changed()
        self._reset_run_progress(info.frame_count)
        self._refresh_fps_summary()
        self._log(f"입력 로드: {path}")

    def _target_fps(self):
        value = self.output_fps_combo.currentData()
        if value is None:
            return None
        return float(self.custom_fps.value() if float(value) < 0 else value)

    def _output_fps_changed(self):
        value = self.output_fps_combo.currentData()
        self.custom_fps.setVisible(value is not None and float(value) < 0)
        self._refresh_fps_summary()

    def _refresh_fps_summary(self):
        if self.video_info is None:
            return
        source = float(getattr(self.video_info, "effective_fps", None)
                       or self.video_info.fps or 30.0)
        try:
            plan = build_fps_plan(source, int(self.multiplier_combo.currentData()),
                                  self._target_fps())
            count = output_frame_count(self.video_info.frame_count, source, plan.output_fps)
            extra = (f" · 내부 {plan.interpolation_multiplier}x 생성"
                     if plan.interpolation_multiplier != plan.requested_multiplier else "")
            self.fps_summary.setText(
                f"{source:.3f} fps → <b>{plan.output_fps:.3f} fps</b> "
                f"· 예상 {count} frames{extra}")
            self.btn_run.setEnabled(self.worker is None)
        except ValueError as exc:
            self.fps_summary.setText(f"설정 오류: {exc}")
            self.btn_run.setEnabled(False)

    def _resolved_format(self):
        value = self.format_combo.currentData()
        return auto_format_from_codec(self.video_codec) if value == "auto" else value

    def _format_changed(self):
        prores = self._resolved_format() == "prores"
        self.prores_profile.setVisible(prores)
        self.profile_label.setVisible(prores)
        current = self.output_edit.text().strip()
        if current:
            self.output_edit.setText(os.path.splitext(current)[0] + (".mov" if prores else ".mp4"))

    def _browse_output(self):
        start = self.output_edit.text().strip() or (self.video_path or "")
        path, _ = QFileDialog.getSaveFileName(
            self, "출력 파일", start, "MP4 (*.mp4);;MOV (*.mov);;모든 파일 (*.*)")
        if path:
            self.output_edit.setText(path)

    @staticmethod
    def _default_output_path(path):
        base = os.path.splitext(os.path.basename(path))[0]
        candidate = os.path.join(_OUTPUT, base + "_interpolated.mp4")
        index = 1
        while os.path.exists(candidate):
            candidate = os.path.join(_OUTPUT, f"{base}_interpolated_{index:03d}.mp4")
            index += 1
        return candidate

    def _gather_settings(self):
        return {
            "model": self.model_combo.currentData(),
            "multiplier": int(self.multiplier_combo.currentData()),
            "output_fps": self._target_fps(),
            "fp16": self.fp16.isChecked(),
            "scene_detection_mode": (
                self.scene_detection_mode.currentData() or "mse"),
            "scene_cut_threshold": (
                DEFAULT_SCENE_CUT_THRESHOLD if self.scene_cut.isChecked()
                else 0.0),
            "output_format": self._resolved_format(),
            "prores_profile": self.prores_profile.currentData() or "4444",
        }

    def _run(self):
        if self.worker is not None:
            return
        if not self.video_path:
            QMessageBox.information(self, "Frame Interpolation", "비디오를 먼저 선택하세요.")
            return
        model = self.model_combo.currentData()
        if not model_available(model):
            QMessageBox.warning(self, "Frame Interpolation",
                                "선택한 모델 또는 런타임이 없습니다.\nsetup.bat을 다시 실행하세요.")
            return
        output = self.output_edit.text().strip() or self._default_output_path(self.video_path)
        fmt = self._resolved_format()
        output = os.path.splitext(output)[0] + (".mov" if fmt == "prores" else ".mp4")
        os.makedirs(os.path.dirname(output) or _OUTPUT, exist_ok=True)
        settings = self._gather_settings()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.worker = InterpolationWorker(self.video_path, output, settings, device)
        self.worker.progress.connect(self._on_progress)
        self.worker.phase_changed.connect(self._on_phase)
        self.worker.finished_signal.connect(self._on_finished)
        self.worker.error_signal.connect(self._on_error)
        self.worker.cancelled_signal.connect(self._on_cancelled)
        self.worker.finished.connect(self._on_worker_thread_finished)
        self._reset_run_progress(
            int(getattr(self.video_info, "frame_count", 0) or 0))
        self._started = time.time()
        self._set_running(True)
        self._log(f"처리 시작 · {model} · {settings['multiplier']}x → {output}")
        self.worker.start()

    def _cancel(self):
        if self.worker:
            self.worker.cancel()
            self._log("취소 요청됨...")

    def _set_running(self, running):
        self.btn_run.setEnabled(not running)
        self.btn_cancel.setEnabled(running)
        self.model_combo.setEnabled(not running)

    def _reset_run_progress(self, total_frames):
        total = max(0, int(total_frames or 0))
        self.progress.setPalette(self._normal_palette)
        self.progress.setValue(0)
        self._last_done, self._last_total = 0, total
        self.eta_label.setText(
            f"0/{total} | 준비 중..." if total else "준비 중...")

    def _is_current_worker_signal(self):
        sender = self.sender()
        return sender is None or sender is self.worker

    def _on_progress(self, done, total):
        if not self._is_current_worker_signal():
            return
        if total <= 0:
            return
        self._last_done, self._last_total = done, total
        self.progress.setValue(int(done * 100 / total))
        elapsed = time.time() - self._started
        eta = elapsed * (total - done) / done if done else 0
        self.eta_label.setText(f"{done}/{total} | 경과 {elapsed:.0f}s | 남은 시간 ≈ {eta:.0f}s")

    def _on_phase(self, phase):
        if not self._is_current_worker_signal():
            return
        if phase == "encoding":
            self.progress.setValue(100)
            self.eta_label.setText("마무리 중 — 비디오 인코딩")

    def _on_finished(self, path):
        if not self._is_current_worker_signal():
            return
        elapsed = time.time() - self._started
        self.progress.setValue(100)
        self.progress.setPalette(self._done_palette)
        self.eta_label.setText(f"완료 · {elapsed:.1f}s")
        self._log(f"완료 ({elapsed:.1f}s) → {path}")
        signal_worker = self.sender()
        source_path = getattr(signal_worker, "video_path", self.video_path)
        try:
            ext = os.path.splitext(path)[1].lower()
            cached = os.path.join(_CACHE, "interpolation_view" + ext)
            os.makedirs(_CACHE, exist_ok=True)
            if os.path.exists(cached): os.remove(cached)
            shutil.copy2(path, cached)
            self.player.disable_compare()
            self.player.load(cached)
            if source_path and os.path.exists(source_path):
                self.player.enable_compare(
                    source_path, sync_by_time=True)
            self._log("메인 뷰어가 결과물로 전환됨 (와이프로 원본 비교 가능).")
        except Exception as exc:
            self._log(f"[WARN] 결과 미리보기 실패: {exc}")

    def _on_cancelled(self):
        if not self._is_current_worker_signal():
            return
        self._log("취소됨")

    def _on_error(self, message):
        if not self._is_current_worker_signal():
            return
        self._log("[ERROR] " + message)
        QMessageBox.critical(self, "Frame Interpolation", message)

    def _on_worker_thread_finished(self):
        finished_worker = self.sender()
        if finished_worker is not None and finished_worker is not self.worker:
            return
        worker = self.worker
        if worker is None:
            return
        self.worker = None
        self._set_running(False)
        pending_path = self._pending_video_path
        self._pending_video_path = None
        if pending_path:
            self._apply_video(pending_path)
        else:
            self._refresh_fps_summary()
        try:
            worker.deleteLater()
        except Exception:
            pass

    def _log(self, message):
        self.log.appendPlainText(message)

    def shutdown(self):
        self._pending_video_path = None
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(3000)
            if self.worker.isRunning():
                self.worker.terminate()
                self.worker.wait(1000)
        self.worker = None
        try:
            self.player.clear()
        except Exception:
            pass

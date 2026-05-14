"""Floating chatbot bubble + popup chat panel.

Talks to a Cloudflare Workers proxy that forwards to Nvidia NIM-hosted
chat-completion models. Wire diagram:
    PyQt5 ChatPanel → ChatWorker QThread → HTTPS POST → proxy → NIM

The same proxy is shared with khwhite0413/idol-prompt-builder, so no new
secret/account is needed — the worker handles auth on its side.
"""

import json
import urllib.request
import urllib.error

from PyQt5.QtCore import Qt, QTimer, QThread, QPoint, QSize, pyqtSignal
from PyQt5.QtGui import QKeyEvent, QPainter, QPen
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                              QLabel, QPlainTextEdit, QScrollArea, QFrame)

from app.i18n import tr


PROXY_URL = "https://prompt-builder-proxy.khwhite0413.workers.dev"
DEFAULT_MODEL = "meta/llama-4-maverick-17b-128e-instruct"
MAX_HISTORY = 20  # last N turns kept in the request body


# Manual + persona for the chatbot. Korean primary; the model is asked
# to mirror the user's language for general questions.
SYSTEM_PROMPT = """\
당신은 "CEC All-In-One" 데스크톱 애플리케이션 안에 내장된 도우미 챗봇입니다.

# 응답 규칙
- 사용자 언어에 맞춰 답하세요 (한국어 질문 → 한국어, English question → English).
- CEC All-In-One 외 일반적인 질문에도 자유롭게 답해도 됩니다.
- 모르는 건 솔직히 모른다고 답하세요.
- 답변은 간결하게. 필요한 경우 마크다운(굵게/리스트)을 사용해도 됩니다.

# CEC All-In-One v0.1.0 개요
AI 영상 후처리 통합 데스크톱 도구 (PyQt5 기반). 세 가지 탭이 있습니다:
1. 인페인트 (Inpaint) — ProPainter로 워터마크/객체 제거
2. 디노이즈 (Denoise) — FlowDenoise로 시간적 광학흐름 기반 노이즈 감소
3. 업스케일 (Upscale) — OpenProteus 2x로 해상도 확대

모든 탭이 동일한 영상 IO 정책을 공유합니다: untagged 영상은 untagged 그대로
출력하고, 24.04처럼 비정수 fps는 24.0 같은 표준값으로 자동 스냅합니다.

# 인페인트 탭 사용법
1. 좌측 영상 영역 클릭 또는 드래그&드롭으로 입력 영상 로드
2. [프리셋 ▼] 버튼 클릭 → "Dreamina (자동)" 선택
   → 영상 가로/세로 비율 자동 감지해서 적절한 마스크 PNG 적용
   → 미리보기 썸네일로 마스크 영역 확인 가능
3. (선택) "마스크 가장자리 부드럽게" 체크 — 경계 잔상 줄임
4. [전체 처리] 클릭 → output 폴더에 mp4 저장
처리 시 마스크 영역 + 50px 패딩만 잘라서 ProPainter에 넣고 결과를 원본에
다시 덮어쓰는 방식이라, 워터마크 바깥 픽셀은 손실 0입니다.

# 디노이즈 탭 사용법
1. 영상 로드
2. 우측 파라미터 패널:
   - 기본값(window_size 3, flow_model memfof) 권장
   - 노이즈가 여전히 보이면 window_size를 5-7로 늘리세요
3. [미리보기 (앞 80프레임)] 클릭 → 빠르게 확인
4. [전체 처리] 클릭 → output 폴더에 mp4/mov/png seq 저장
출력 포맷: H.264 MP4(기본), ProRes MOV(후작업용), PNG sequence(무손실)

# 업스케일 탭 사용법
1. 영상 로드 (1080p 이하 라이브액션 권장)
2. 우측 파라미터 패널:
   - 모델: OpenProteus 2x (Compact) — 현재 유일 모델
   - 출력 해상도: Native(원본×2) / HD(1280) / FHD(1920) / QHD(2560) /
     4K(3840) / 8K(7680)
   - "예상: WxH · NxM" 라벨로 최종 결과 미리 확인 가능
   - Max batch frames: 0(전체) 권장. VRAM 부족할 때만 분할
3. [전체 처리] 클릭
중요: 모델은 항상 입력의 정확히 2배만 처리합니다. 4K나 8K를 골라도 모델은
그 중간(2x) 결과만 만들고 cv2가 마지막에 Lanczos4(키울 때) / Area(줄일 때)로
리사이즈하는 구조입니다.

# 자주 묻는 질문
Q: 720p 영상에서 1080p 골랐는데 "줄여서 출력"이라고 떠요?
A: 모델이 720p → 1440p로 키운 다음 1080p로 줄이는 거라 마지막 cv2 단계는
다운샘플이 맞습니다. 사용자 시작점(720p) 기준 1080p는 업스케일이 맞고,
이건 의도된 동작입니다.

Q: GPU 메모리 부족 오류가 나요
A: 인페인트는 개발자 모드에서 subvideo_length를 60-40으로 줄이고,
업스케일은 max batch frames를 50 등으로 분할하세요. 디노이즈는 batch_size를
절반으로 줄이는 게 가장 효과적입니다.

Q: 노이즈가 많은 4K 영상을 업스케일하고 싶어요
A: 권장 순서 — 디노이즈 탭으로 먼저 깔끔하게 만든 다음, 업스케일 탭에서
원하는 해상도로. 그래야 노이즈가 증폭되지 않습니다.

Q: 색이 원본과 어긋나 보여요 (Nuke 같은 도구에서)
A: 도구는 untagged 영상은 untagged 그대로 출력합니다. About 탭의 디노이즈
FAQ에 색상 태그에 대한 자세한 설명이 있습니다.

# 추가 정보
- 언어 변경: 우측 상단 한국어/English 드롭다운 (변경 후 재시작)
- 로그 파일: <앱폴더>/logs/cec_aio.log
- 모델 가중치: <앱폴더>/weights/ (setup.bat이 자동 다운로드)
"""


class ChatWorker(QThread):
    """One-shot HTTPS POST to the proxy. Emits the reply text on done."""

    response_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(self, messages, model=DEFAULT_MODEL, parent=None):
        super().__init__(parent)
        self.messages = messages
        self.model = model

    def run(self):
        try:
            body = json.dumps({
                "model": self.model,
                "messages": self.messages,
                "max_tokens": 1024,
                "temperature": 0.7,
                "stream": False,
            }).encode("utf-8")
            # The Cloudflare Worker fronting NIM is shared with the
            # idol-prompt-builder GitHub-Pages site and (very likely)
            # gates requests by the browser-supplied Origin header.
            # urllib doesn't send one by default → 403. Spoofing the
            # GitHub-Pages origin is the cheapest workaround; the
            # alternative is editing the Worker to expand its allowlist.
            req = urllib.request.Request(
                PROXY_URL,
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "Origin": "https://khwhite0413.github.io",
                    "Referer":
                        "https://khwhite0413.github.io/idol-prompt-builder/",
                    "User-Agent":
                        "CEC-AllInOne/0.1.0 (+khwhite0413)",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.loads(r.read().decode("utf-8"))
            content = (data.get("choices") or [{}])[0].get(
                "message", {}).get("content", "")
            self.response_signal.emit(content.strip() or "(응답 없음)")
        except urllib.error.HTTPError as e:
            self.error_signal.emit(f"HTTP {e.code}: {e.reason}")
        except Exception as e:
            self.error_signal.emit(str(e))


class _ChatInput(QPlainTextEdit):
    """QPlainTextEdit that sends on Enter, newlines on Shift+Enter."""

    submit_requested = pyqtSignal()

    def keyPressEvent(self, e: QKeyEvent):
        if e.key() in (Qt.Key_Return, Qt.Key_Enter):
            if e.modifiers() & Qt.ShiftModifier:
                super().keyPressEvent(e)
                return
            self.submit_requested.emit()
            return
        super().keyPressEvent(e)


class _DragHeader(QWidget):
    """Title-bar widget that drags the parent panel when held with the
    left mouse button. Children (close button etc.) still receive their
    own clicks normally — Qt routes the event to the deepest child first.
    """

    def __init__(self, panel, overlay, parent=None):
        super().__init__(parent)
        self.panel = panel
        self.overlay = overlay
        self.setCursor(Qt.SizeAllCursor)
        self._panel_origin = None  # panel.pos() at drag start
        self._mouse_origin = None  # global mouse pos at drag start

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._panel_origin = self.panel.pos()
            self._mouse_origin = e.globalPos()
            e.accept()

    def mouseMoveEvent(self, e):
        if (e.buttons() & Qt.LeftButton
                and self._panel_origin is not None
                and self._mouse_origin is not None):
            delta = e.globalPos() - self._mouse_origin
            new_x = self._panel_origin.x() + delta.x()
            new_y = self._panel_origin.y() + delta.y()
            parent = self.panel.parent()
            if parent is not None:
                new_x = max(0, min(new_x, parent.width() - self.panel.width()))
                new_y = max(0, min(new_y, parent.height() - self.panel.height()))
            self.panel.move(new_x, new_y)
            self.overlay._user_positioned = True
            e.accept()

    def mouseReleaseEvent(self, e):
        self._panel_origin = None
        self._mouse_origin = None
        e.accept()


class _PanelResizeGrip(QWidget):
    """Bottom-right resize handle that resizes ONLY the chatbot panel.

    Qt's stock QSizeGrip walks up to `parentWidget().window()` and resizes
    that — for a child overlay panel inside a QMainWindow, that means the
    main window grows instead of the panel. We track mouse deltas
    ourselves and call panel.resize() directly, clamped to the panel's
    minimumSize() and to the host widget's bounds so the panel can't
    escape the visible area.
    """

    def __init__(self, panel, parent=None):
        super().__init__(parent)
        self.panel = panel
        self.setFixedSize(16, 16)
        self.setCursor(Qt.SizeFDiagCursor)
        self._start_size = None
        self._mouse_origin = None

    def paintEvent(self, _e):
        p = QPainter(self)
        pen = QPen(Qt.gray)
        pen.setWidth(1)
        p.setPen(pen)
        w, h = self.width(), self.height()
        for off in (3, 7, 11):
            p.drawLine(w - off, h - 2, w - 2, h - off)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._start_size = self.panel.size()
            self._mouse_origin = e.globalPos()
            e.accept()

    def mouseMoveEvent(self, e):
        if (e.buttons() & Qt.LeftButton
                and self._start_size is not None
                and self._mouse_origin is not None):
            delta = e.globalPos() - self._mouse_origin
            new_w = self._start_size.width() + delta.x()
            new_h = self._start_size.height() + delta.y()
            min_w = self.panel.minimumWidth()
            min_h = self.panel.minimumHeight()
            new_w = max(min_w, new_w)
            new_h = max(min_h, new_h)
            host = self.panel.parent()
            if host is not None:
                new_w = min(new_w, host.width() - self.panel.x())
                new_h = min(new_h, host.height() - self.panel.y())
            self.panel.resize(new_w, new_h)
            e.accept()

    def mouseReleaseEvent(self, e):
        self._start_size = None
        self._mouse_origin = None
        e.accept()


class ChatbotOverlay:
    """Bottom-right floating bubble + popup chat panel.

    Not a QWidget itself — just a controller that owns two top-level
    children of the host widget so each can be raised independently
    above the QStackedWidget / tab content. Host should call .reposition()
    from its resizeEvent.
    """

    def __init__(self, parent_widget):
        self.parent = parent_widget
        self._open = False
        self._messages = []
        self._worker = None
        self._typing_wrap = None
        # Set to True the moment the user drags the panel by its header.
        # reposition() then leaves the panel where they put it instead of
        # re-snapping to the bottom-right default on window resize.
        self._user_positioned = False

        self._build_bubble()
        self._build_panel()
        self.reposition()

    # ------------------------------------------------------------------
    # Widget construction
    # ------------------------------------------------------------------
    def _build_bubble(self):
        self.bubble = QPushButton("💬", self.parent)
        self.bubble.setFixedSize(52, 52)
        self.bubble.setCursor(Qt.PointingHandCursor)
        self.bubble.setToolTip(tr("chatbot_tooltip"))
        self.bubble.setStyleSheet(
            "QPushButton {"
            "  background: #7aa2f7;"
            "  color: white;"
            "  font-size: 22px;"
            "  border: none;"
            "  border-radius: 26px;"
            "}"
            "QPushButton:hover { background: #8db4ff; }"
            "QPushButton:pressed { background: #6a92e7; }"
        )
        self.bubble.clicked.connect(self.toggle)

    def _build_panel(self):
        self.panel = QFrame(self.parent)
        # Initial size; user can resize via the QSizeGrip below. Min
        # size keeps the panel usable even when shrunk hard.
        self.panel.resize(380, 520)
        self.panel.setMinimumSize(300, 360)
        self.panel.setStyleSheet(
            "QFrame {"
            "  background-color: #1a1a26;"
            "  border: 1px solid #2e2e3e;"
            "  border-radius: 12px;"
            "}"
        )
        lay = QVBoxLayout(self.panel)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(8)

        # Title bar acts as the drag handle for moving the whole panel.
        # Wrap the row in a _DragHeader widget so its mouse events
        # translate into panel.move().
        head_widget = _DragHeader(self.panel, self)
        head = QHBoxLayout(head_widget)
        head.setContentsMargins(0, 0, 0, 0)
        title = QLabel(tr("chatbot_title"))
        title.setStyleSheet(
            "font-weight: 600; font-size: 11pt; "
            "border: none; background: transparent;")
        head.addWidget(title)
        head.addStretch(1)
        close_btn = QPushButton("✕")
        close_btn.setFixedSize(24, 24)
        close_btn.setCursor(Qt.PointingHandCursor)
        close_btn.setStyleSheet(
            "QPushButton {"
            "  background: transparent; border: none;"
            "  color: rgba(200,204,220,160); font-size: 14px;"
            "}"
            "QPushButton:hover { color: rgba(220,224,240,255); }"
        )
        close_btn.clicked.connect(self.close_panel)
        head.addWidget(close_btn)
        lay.addWidget(head_widget)

        # Message area
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setStyleSheet(
            "QScrollArea { background: transparent; border: none; }")
        self.msg_host = QWidget()
        self.msg_host.setStyleSheet("background: transparent;")
        self.msg_lay = QVBoxLayout(self.msg_host)
        self.msg_lay.setContentsMargins(0, 0, 0, 0)
        self.msg_lay.setSpacing(8)
        self.msg_lay.addStretch(1)
        self.scroll.setWidget(self.msg_host)
        lay.addWidget(self.scroll, 1)

        # Input
        in_row = QHBoxLayout()
        self.input = _ChatInput()
        self.input.setPlaceholderText(tr("chatbot_placeholder"))
        self.input.setFixedHeight(64)
        self.input.setStyleSheet(
            "QPlainTextEdit { background: #20202c; border: 1px solid #2e2e3e;"
            " border-radius: 6px; padding: 6px; color: #d0d4e0; }"
        )
        self.input.submit_requested.connect(self._on_send)
        in_row.addWidget(self.input, 1)
        self.send_btn = QPushButton(tr("chatbot_send"))
        self.send_btn.setFixedWidth(60)
        self.send_btn.setCursor(Qt.PointingHandCursor)
        self.send_btn.clicked.connect(self._on_send)
        in_row.addWidget(self.send_btn, 0)
        lay.addLayout(in_row)

        # Resize handle at the bottom-right corner. Custom grip — Qt's
        # QSizeGrip would resize the top-level window (the QMainWindow)
        # because self.panel is a child widget, not its own top-level.
        grip_row = QHBoxLayout()
        grip_row.setContentsMargins(0, 0, 0, 0)
        grip_row.setSpacing(0)
        grip_row.addStretch(1)
        grip = _PanelResizeGrip(self.panel, parent=self.panel)
        grip_row.addWidget(grip, 0, Qt.AlignBottom | Qt.AlignRight)
        lay.addLayout(grip_row)

        # Auto-scroll when the scroll content's range grows — covers the
        # case where a long assistant reply pushes scrollbar.max up
        # asynchronously after the bubble's word-wrap height is
        # measured. A one-shot QTimer fires too early in that case.
        # Toggled off when the user scrolls up manually so we don't
        # yank them back mid-read.
        self._follow_bottom = True
        sb = self.scroll.verticalScrollBar()
        sb.rangeChanged.connect(self._on_scroll_range_changed)
        sb.valueChanged.connect(self._on_user_scroll)

        # Greeting bubble
        self._append_message("assistant", tr("chatbot_greeting"))
        self.panel.setVisible(False)

    # ------------------------------------------------------------------
    # Message rendering
    # ------------------------------------------------------------------
    def _append_message(self, role: str, content: str) -> QWidget:
        bubble = QLabel(content)
        bubble.setWordWrap(True)
        bubble.setTextInteractionFlags(Qt.TextSelectableByMouse)
        bubble.setMaximumWidth(280)
        if role == "user":
            bubble.setStyleSheet(
                "QLabel {"
                "  background: #2a3550; color: #e8edf6;"
                "  border-radius: 10px; padding: 8px 11px;"
                "  font-size: 10pt;"
                "}")
        else:
            bubble.setStyleSheet(
                "QLabel {"
                "  background: #20202c; color: #d0d4e0;"
                "  border-radius: 10px; padding: 8px 11px;"
                "  font-size: 10pt;"
                "}")
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        if role == "user":
            row.addStretch(1)
            row.addWidget(bubble, 0)
        else:
            row.addWidget(bubble, 0)
            row.addStretch(1)
        wrap = QWidget()
        wrap.setStyleSheet("background: transparent;")
        wrap.setLayout(row)
        self.msg_lay.insertWidget(self.msg_lay.count() - 1, wrap)
        QTimer.singleShot(0, self._scroll_to_bottom)
        return wrap

    def _scroll_to_bottom(self):
        sb = self.scroll.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_scroll_range_changed(self, _min, _max):
        """Fires after the scrollarea recalculates its content height,
        which is when word-wrapped QLabels finally know how tall they
        are. Following _max here is what guarantees a new reply lands
        in view."""
        if self._follow_bottom:
            self.scroll.verticalScrollBar().setValue(_max)

    def _on_user_scroll(self, value):
        """If the user drags the scrollbar above the bottom, stop
        auto-following so we don't yank them back mid-read. Resumes
        when they scroll back down."""
        sb = self.scroll.verticalScrollBar()
        self._follow_bottom = value >= sb.maximum() - 4

    # ------------------------------------------------------------------
    # Send / receive
    # ------------------------------------------------------------------
    def _on_send(self):
        text = self.input.toPlainText().strip()
        if not text or self._worker is not None:
            return
        # User just typed and hit send — they're engaged with the new
        # turn, so resume auto-follow even if they had scrolled up.
        self._follow_bottom = True
        self.input.clear()
        self._messages.append({"role": "user", "content": text})
        self._append_message("user", text)
        self._typing_wrap = self._append_message(
            "assistant", tr("chatbot_thinking"))

        history = self._messages[-MAX_HISTORY:]
        api_messages = ([{"role": "system", "content": SYSTEM_PROMPT}]
                        + history)
        self.send_btn.setEnabled(False)
        self._worker = ChatWorker(api_messages)
        self._worker.response_signal.connect(self._on_response)
        self._worker.error_signal.connect(self._on_error)
        self._worker.finished.connect(self._on_worker_done)
        self._worker.start()

    def _on_response(self, reply: str):
        self._clear_typing()
        self._messages.append({"role": "assistant", "content": reply})
        self._append_message("assistant", reply)

    def _on_error(self, msg: str):
        self._clear_typing()
        self._append_message(
            "assistant", tr("chatbot_error").format(err=msg))

    def _on_worker_done(self):
        self.send_btn.setEnabled(True)
        self._worker = None

    def _clear_typing(self):
        if self._typing_wrap is not None:
            try:
                self._typing_wrap.deleteLater()
            except Exception:
                pass
            self._typing_wrap = None

    # ------------------------------------------------------------------
    # Open / close / position
    # ------------------------------------------------------------------
    def toggle(self):
        if self._open:
            self.close_panel()
        else:
            self.open_panel()

    def open_panel(self):
        self._open = True
        self.panel.setVisible(True)
        self.panel.raise_()
        self.bubble.raise_()
        self.input.setFocus()
        self.reposition()

    def close_panel(self):
        self._open = False
        self.panel.setVisible(False)

    def reposition(self):
        if not self.parent:
            return
        m = 18
        pw, ph = self.parent.width(), self.parent.height()
        # Bubble always pins to bottom-right (it's the anchor; user
        # never moves it, only the panel).
        self.bubble.move(pw - self.bubble.width() - m,
                          ph - self.bubble.height() - m)

        if self._user_positioned:
            # Respect the user's drag — only clamp back into bounds
            # so the panel can't escape after a window shrink.
            x = max(0, min(self.panel.x(),
                            max(0, pw - self.panel.width())))
            y = max(0, min(self.panel.y(),
                            max(0, ph - self.panel.height())))
            self.panel.move(x, y)
        else:
            # First-time / default placement: above the bubble, anchored
            # bottom-right with the same margin.
            panel_x = pw - self.panel.width() - m
            panel_y = ph - self.bubble.height() - m - self.panel.height() - 10
            if panel_y < 10:
                panel_y = 10
            self.panel.move(panel_x, panel_y)

        self.bubble.raise_()
        if self._open:
            self.panel.raise_()

"""About page — AIMZ GFX logo + product info + contact + FAQ (pure PyQt5)."""

import os

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                              QFrame, QPushButton, QScrollArea, QSizePolicy)

from app.i18n import tr, get_lang


_HERE = os.path.dirname(os.path.abspath(__file__))
_LOGO_DIR = os.path.abspath(os.path.join(_HERE, "..", "assets", "logo"))


# FAQ content is kept inline (not in i18n.py) because each answer is a
# short paragraph — fanning that out into 20+ string keys clutters i18n
# without making translation easier. Stays as (question, answer) tuples
# per language.
_FAQ_KO = [
    ("FlowDenoise가 뭔가요?",
     "AI로 뽑은 영상이나 촬영본에 끼는 지글거리는 노이즈를 떨어뜨려 주는 "
     "도구입니다. 한 프레임만 보고 처리하는 일반 디노이저와 달리, 앞뒤 프레임을 "
     "같이 보면서 같은 물체를 추적해서 평균을 내기 때문에 가만히 있는 부분의 "
     "디테일은 깎이지 않고 노이즈만 줄어듭니다."),

    ("Seedance 2.0이 24.04fps 같은 어중간한 프레임레이트로 영상을 뽑던데, "
     "그대로 써도 되나요?",
     "Seedance 같은 AI 영상 생성기는 정확한 24fps가 아니라 24.04, 24.06처럼 "
     "조금씩 어긋난 프레임레이트로 영상을 뽑는 버릇이 있습니다. 이걸 편집 "
     "프로그램 타임라인에 그냥 올리면 시간이 갈수록 영상이 한두 프레임씩 흘러 "
     "엇박이 납니다. FlowDenoise는 이런 어중간한 fps를 자동으로 가장 가까운 "
     "표준값(23.976 / 24 / 25 / 29.97 / 30 / 50 / 59.94 / 60)으로 맞춰서 "
     "출력합니다. 자동 결과가 마음에 안 들면 Dev mode → 출력 세팅에서 직접 "
     "선택할 수도 있습니다."),

    ("Nuke 같은 도구에서 색감이 원본이랑 어긋나 보일 때가 있는데 왜 그래요?",
     "영상 파일에는 보통 「이 영상은 이런 색공간으로 만들었어요」라고 알려주는 "
     "색상 태그가 같이 박혀 있고, Nuke 같은 도구는 그 태그를 보고 색을 풀어서 "
     "화면에 보여줍니다. 그런데 태그가 빠진 영상이 들어오면 도구마다 「아마 "
     "이거겠지」라고 추측을 합니다 — Nuke는 SD 시절 표준(Rec.601)이라고 가정하고, "
     "일부 플레이어는 HD 표준(Rec.709)이라고 가정합니다. 픽셀 값은 똑같아도 어떤 "
     "표준이라고 가정하느냐에 따라 보이는 색이 달라지는 거예요."),

    ("그럼 Seedance가 만든 영상에는 왜 색상 태그가 빠져 있는 건가요?",
     "AI 영상 생성기는 영상 자체를 만드는 데 집중하지, 출력 파일에 색상 태그를 "
     "박는 단계를 신경 쓰지 않는 경우가 많습니다. Seedance 2.0이 그런 경우라서 "
     "출력 mp4에 태그가 비어 있어요. FlowDenoise는 이렇게 태그가 없는 영상이 "
     "들어오면 출력에도 일부러 태그를 안 박습니다 — 양쪽 다 태그가 없으면 Nuke가 "
     "두 파일에 같은 추측을 적용해서 색이 일치하기 때문이죠. 우리 출력에만 "
     "「Rec.709에요」 태그를 박아 버리면, Nuke가 원본은 Rec.601로 풀고 우리 "
     "출력은 Rec.709로 풀어서 미드톤이 어긋나 보이는 사고가 납니다."),

    ("미리보기와 본 처리는 뭐가 다른가요?",
     "미리보기는 입력 영상의 앞 80프레임만 짧게 처리해 보는 거예요. 노이즈 강도 "
     "같은 파라미터를 바꿔가며 결과를 빠르게 확인하라고 만들어둔 기능입니다. "
     "최종 품질 확인은 꼭 본 처리 결과로 해주세요."),

    ("노이즈 미리보기는 뭘 보여주는 건가요?",
     "원본에서 디노이즈된 영상을 빼고 남은 부분, 즉 「뭐가 떨어져 나갔는지」를 "
     "보여줍니다. 노이즈가 잘 떨어졌는지 vs 진짜 디테일까지 같이 깎이고 있는지 "
     "눈으로 확인할 때 씁니다. 정상이면 지글거리는 노이즈만 보이고, 사람 얼굴 "
     "윤곽이나 글자 같은 게 비치면 안 됩니다 — 그러면 너무 세게 깎고 있다는 "
     "신호예요."),
]

_FAQ_EN = [
    ("What is FlowDenoise?",
     "A tool that scrubs the flickering noise out of AI-generated "
     "and live-shot footage. Unlike single-frame denoisers, it looks "
     "at the frames before and after, tracks the same object across "
     "them, and averages — so still details stay sharp and only the "
     "noise drops out."),

    ("Seedance 2.0 outputs odd frame rates like 24.04 fps — is that "
     "a problem?",
     "AI video generators like Seedance tend to emit slightly off "
     "rates (24.04, 24.06 …) instead of clean 24 fps. Drop that "
     "straight onto an NLE timeline and the picture slips by a frame "
     "or two as the clip plays out. FlowDenoise automatically snaps "
     "the input rate to the nearest standard (23.976 / 24 / 25 / "
     "29.97 / 30 / 50 / 59.94 / 60) before writing. If the auto "
     "guess is wrong, override it from Dev mode → Output Settings."),

    ("Why does the colour sometimes look off in Nuke compared to the "
     "original?",
     "Video files normally carry a small tag that says \"I was made "
     "in this colour space.\" Tools like Nuke read that tag and "
     "decode accordingly. When the tag is missing, each tool falls "
     "back to a guess — Nuke assumes the old SD standard (Rec.601), "
     "some players assume the HD standard (Rec.709). Same pixels, "
     "different assumption, different colour."),

    ("So why does Seedance's output have that tag missing?",
     "AI generators focus on producing the picture and tend to skip "
     "the step of attaching colour-space tags when they mux the "
     "file. Seedance 2.0 does exactly that — its mp4 comes out "
     "untagged. FlowDenoise deliberately leaves the output untagged "
     "too in that case, so the *same* guess applies to both files "
     "and they match in a comparison. If we instead stamped \"this "
     "is Rec.709\" on our output, Nuke would inverse-transform the "
     "source as Rec.601 and our output as Rec.709, and the midtones "
     "would visibly drift."),

    ("What's the difference between Preview and the full run?",
     "Preview only processes the first 80 frames, so you can iterate "
     "quickly on settings like noise threshold. Always validate the "
     "final quality on the full-run output."),

    ("What does the noise preview actually show?",
     "It's `source − denoised` — the stuff that got subtracted. Use "
     "it to check whether real detail is leaking into the noise "
     "channel. Healthy noise looks like flickering grain only; if "
     "you can make out face outlines or text, the threshold is too "
     "aggressive."),
]


class _FaqItem(QFrame):
    """One question/answer row. Click the question to toggle the answer."""

    def __init__(self, question: str, answer: str, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.NoFrame)
        self.setStyleSheet("QFrame { background: transparent; }")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        self._question = question
        self._q_btn = QPushButton(self._fmt(False))
        self._q_btn.setFlat(True)
        self._q_btn.setCursor(Qt.PointingHandCursor)
        self._q_btn.setStyleSheet(
            "QPushButton {"
            "  text-align: left;"
            "  padding: 8px 10px;"
            "  border: none;"
            "  background: rgba(255,255,255,8);"
            "  border-radius: 6px;"
            "  font-size: 13px;"
            "  font-weight: 600;"
            "}"
            "QPushButton:hover { background: rgba(255,255,255,18); }")
        self._q_btn.clicked.connect(self._toggle)
        lay.addWidget(self._q_btn)

        self._a_lbl = QLabel(answer)
        self._a_lbl.setWordWrap(True)
        self._a_lbl.setVisible(False)
        self._a_lbl.setStyleSheet(
            "color: rgba(190, 194, 210, 220);"
            "padding: 4px 12px 10px 22px;"
            "font-size: 12px;"
            "line-height: 150%;")
        lay.addWidget(self._a_lbl)

    def _fmt(self, expanded: bool) -> str:
        arrow = "▾" if expanded else "▸"
        return f"{arrow}  {self._question}"

    def _toggle(self):
        v = not self._a_lbl.isVisible()
        self._a_lbl.setVisible(v)
        self._q_btn.setText(self._fmt(v))


class AboutPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        outer = QHBoxLayout(self)
        outer.setContentsMargins(40, 40, 40, 40)
        outer.setSpacing(28)

        outer.addWidget(self._build_card(), 0, Qt.AlignTop)
        outer.addWidget(self._build_faq(), 1)

        self.refresh_logo()

    # ------------------------------------------------------------------
    # left column — product card
    # ------------------------------------------------------------------
    def _build_card(self) -> QFrame:
        card = QFrame()
        card.setFrameShape(QFrame.StyledPanel)
        card.setFixedWidth(440)
        card.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        card.setStyleSheet(
            "QFrame {"
            "  background-color: #20202c;"
            "  border: 1px solid #2e2e3e;"
            "  border-radius: 10px;"
            "}")
        card_lay = QVBoxLayout(card)
        card_lay.setContentsMargins(40, 36, 40, 36)
        card_lay.setSpacing(12)
        card_lay.setAlignment(Qt.AlignHCenter)

        self._logo_label = QLabel()
        self._logo_label.setAlignment(Qt.AlignCenter)
        self._logo_label.setMinimumHeight(110)
        self._logo_label.setStyleSheet("border: none; background: transparent;")
        card_lay.addWidget(self._logo_label, 0, Qt.AlignHCenter)

        title = QLabel("FlowDenoise")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(
            "font-size: 20px; font-weight: 700; "
            "border: none; background: transparent;")
        card_lay.addWidget(title)

        version = QLabel("v1.0.0")
        version.setAlignment(Qt.AlignCenter)
        version.setStyleSheet(
            "color: rgba(150, 154, 170, 200); "
            "font-size: 12px; "
            "border: none; background: transparent;")
        card_lay.addWidget(version)

        tagline = QLabel(tr("about_tagline"))
        tagline.setAlignment(Qt.AlignCenter)
        tagline.setStyleSheet(
            "color: rgba(180, 184, 200, 220); "
            "border: none; background: transparent;")
        card_lay.addWidget(tagline)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(
            "background-color: rgba(120,120,140,80); border: none; max-height: 1px;")
        sep.setMaximumHeight(1)
        card_lay.addSpacing(8)
        card_lay.addWidget(sep)
        card_lay.addSpacing(4)

        contact_title = QLabel(tr("about_contact"))
        contact_title.setAlignment(Qt.AlignCenter)
        contact_title.setStyleSheet(
            "font-size: 13px; font-weight: 600; "
            "border: none; background: transparent;")
        card_lay.addWidget(contact_title)

        name = QLabel("Hayeong Kim")
        name.setAlignment(Qt.AlignCenter)
        name.setStyleSheet("border: none; background: transparent;")
        card_lay.addWidget(name)

        email_text = "khwhite0413@aimz.media"
        email = QLabel(
            f'<a href="mailto:{email_text}" '
            f'style="color: #7aa2f7; text-decoration: none;">{email_text}</a>')
        email.setOpenExternalLinks(True)
        email.setAlignment(Qt.AlignCenter)
        email.setStyleSheet("border: none; background: transparent;")
        email.setCursor(Qt.PointingHandCursor)
        card_lay.addWidget(email)
        return card

    # ------------------------------------------------------------------
    # right column — FAQ
    # ------------------------------------------------------------------
    def _build_faq(self) -> QWidget:
        entries = _FAQ_KO if get_lang() == "ko" else _FAQ_EN

        wrap = QFrame()
        wrap.setStyleSheet(
            "QFrame {"
            "  background-color: #1a1a26;"
            "  border: 1px solid #2a2a3a;"
            "  border-radius: 10px;"
            "}")
        wrap_lay = QVBoxLayout(wrap)
        wrap_lay.setContentsMargins(20, 18, 20, 18)
        wrap_lay.setSpacing(8)

        header = QLabel("FAQ" if get_lang() == "en" else "자주 묻는 질문")
        header.setStyleSheet(
            "font-size: 15px; font-weight: 700; "
            "border: none; background: transparent;"
            "padding-bottom: 4px;")
        wrap_lay.addWidget(header)

        # Scrollable list — once a few entries get expanded the FAQ
        # can grow taller than the visible area.
        scroll = QScrollArea()
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")

        list_host = QWidget()
        list_host.setStyleSheet("background: transparent;")
        list_lay = QVBoxLayout(list_host)
        list_lay.setContentsMargins(0, 0, 0, 0)
        list_lay.setSpacing(6)
        for q, a in entries:
            list_lay.addWidget(_FaqItem(q, a))
        list_lay.addStretch(1)

        scroll.setWidget(list_host)
        wrap_lay.addWidget(scroll, 1)
        return wrap

    def refresh_logo(self):
        path = os.path.join(_LOGO_DIR, "aimz_white.png")
        if not os.path.exists(path):
            self._logo_label.setText("AIMZ")
            return
        pix = QPixmap(path)
        scaled = pix.scaledToWidth(280, Qt.SmoothTransformation)
        self._logo_label.setPixmap(scaled)

"""About page — AIMZ GFX logo + product info + contact + FAQ (pure PyQt5)."""

import os

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                              QFrame, QPushButton, QScrollArea, QSizePolicy,
                              QButtonGroup, QStackedWidget)

from app.i18n import tr, get_lang


_HERE = os.path.dirname(os.path.abspath(__file__))
_LOGO_DIR = os.path.abspath(os.path.join(_HERE, "..", "assets", "logo"))


# FAQ content is kept inline (not in i18n.py) because each answer is a
# short paragraph — fanning that out into 20+ string keys clutters i18n
# without making translation easier. Stays as (question, answer) tuples
# per language.
_FAQ_DENOISE_KO = [
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

_FAQ_DENOISE_EN = [
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


# Inpaint FAQ — explains the crop-and-paste pipeline, preset matching,
# and the dev-mode knobs. Detailed enough to answer "wait, why...".
_FAQ_INPAINT_KO = [
    ("인페인트가 영상을 통째로 처리하나요?",
     "아니에요. 마스크가 있는 영역만 잘라서 (가장자리 50px 패딩 포함) 작은 "
     "박스로 모델에 넣고, 결과를 원본 위에 다시 덮어쓰는 방식입니다. "
     "워터마크 바깥 픽셀은 손실 0이고 처리 시간도 짧아져요. 4K 영상이라도 "
     "워터마크 영역이 320×160 정도면 모델은 그 작은 박스만 봅니다."),

    ("뷰어에서 마스크가 어디에 적용되는지 미리 보려면?",
     "인페인트 탭의 뷰어에 포커스가 있을 때 **A** 키를 누르면 알파 마스크 "
     "오버레이가 토글됩니다. 빨간색 반투명으로 마스크 영역이 영상 위에 "
     "표시되어, 재생하면서 모든 프레임에서 워터마크 위치가 잘 덮이는지 "
     "확인할 수 있어요. 다시 A를 누르면 꺼지고, 이 단축키는 인페인트 탭 "
     "전용입니다."),

    ("Dreamina 프리셋이 어떻게 자동으로 맞춰지나요?",
     "입력 영상의 가로/세로 비율을 보고 가로 영상이면 가로용 PNG, 세로 "
     "영상이면 9:16 PNG를 자동 선택합니다. 비율이 완전히 일치하면 그대로 "
     "스케일링하고, 비율이 다르면 anchor 위치(Dreamina는 좌상단)에 맞춰 "
     "배치합니다. 「마스크 확장(px)」 슬라이더로 외곽을 N픽셀 키워 워터마크 "
     "경계를 안정적으로 덮을 수 있습니다."),

    ("ProPainter는 어떤 영상에서 잘 안 되나요?",
     "(1) 카메라가 완전히 정지하고 피사체도 안 움직이는 정지 화면 — 인접 "
     "프레임에서 정보 빌릴 게 없어서 결과가 흐릿함. (2) 마스크 영역이 화면의 "
     "30% 이상처럼 너무 큰 경우. (3) 워터마크가 사람 얼굴이나 글자 같은 핵심 "
     "정보를 가린 경우. 이런 케이스에선 결과가 어색해질 수 있습니다."),

    ("개발자 모드의 neighbor / ref_stride / subvideo 옵션은 뭔가요?",
     "ProPainter가 시간 일관성을 유지하려고 참조하는 이웃 프레임 수와 "
     "간격입니다. 기본값(10 / 10 / 80)에서 결과가 잘 나오면 손대지 마세요. "
     "VRAM 부족하면 subvideo를 60이나 40으로 줄이는 게 가장 효과적이에요."),

    ("인페인트 탭에서도 fps를 강제할 수 있나요?",
     "네. 우측 파라미터 패널 상단의 「FPS」 라벨에 현재 사용 중인 effective fps "
     "(자동 스냅된 값)가 항상 표시됩니다. 직접 지정하려면 **개발자 모드** 체크 → "
     "「FPS Override」 행에서 Auto 체크 해제 → 표준 rate 드롭다운에서 선택. "
     "리더(-vf fps=)와 라이터(-r) 둘 다에 동일하게 적용되어 출력이 정확히 그 "
     "fps로 떨어집니다. 디노이즈 / 업스케일 탭도 동일한 UX."),
]

_FAQ_INPAINT_EN = [
    ("Does inpainting process the whole frame?",
     "No. Only the masked region (plus 50px padding) is cropped, fed "
     "to the model as a small box, and the result is pasted back into "
     "the original frame. Pixels outside the mask are untouched (zero "
     "loss) and processing is much faster. Even on a 4K source, if the "
     "watermark sits in a ~320×160 corner, the model only ever sees "
     "that small box."),

    ("How do I preview where the mask will be applied?",
     "With focus inside the Inpaint tab viewer, press **A** to toggle "
     "a semi-transparent red overlay showing the mask on top of the "
     "video. Play the clip with the overlay on to verify the mask "
     "tracks the watermark across every frame. Press A again to hide "
     "it. The shortcut is Inpaint-tab-only."),

    ("How does the Dreamina preset auto-fit?",
     "It checks the input video's aspect ratio. Landscape videos get "
     "the landscape PNG, portrait get the 9:16 PNG. If the aspect "
     "matches the preset, it scales 1:1. If not, it scales by the "
     "smaller dimension and anchors at the preset's corner (top-left "
     "for Dreamina). The «Mask expansion (px)» spinner dilates the "
     "preset outward by N pixels to comfortably cover the watermark "
     "edges."),

    ("When does ProPainter struggle?",
     "(1) Truly static shots — no temporal context to borrow from, "
     "so the inpaint comes out blurry. (2) Masks covering more than "
     "~30% of the frame. (3) The watermark sits on important content "
     "like faces or text. Results can look off in these cases."),

    ("What are the neighbor / ref_stride / subvideo dev-mode options?",
     "Number and spacing of frames ProPainter uses for temporal "
     "consistency. Default 10 / 10 / 80 works for most cases. If "
     "you hit VRAM limits, lowering subvideo to 60 or 40 is the most "
     "effective single knob."),

    ("Can I force a specific fps on the inpaint output too?",
     "Yes. The «FPS» readout on the right panel always shows the "
     "effective fps (auto-snapped from the source). To pick a specific "
     "rate, toggle **Dev mode** → uncheck Auto on the «FPS Override» "
     "row → choose one of 23.976 / 24 / 25 / 29.97 / 30 / 50 / 59.94 / "
     "60. The value drives both the reader's -vf fps= (resampling VFR "
     "to CFR) and the writer's -r so the output lands exactly on that "
     "rate. Same UX on the denoise + upscale tabs."),
]


# Upscale FAQ — answers the "why does it say downscale" question
# explicitly, plus the resize-algorithm detail moved out of the tab UI.
_FAQ_UPSCALE_KO = [
    ("출력 해상도를 8K로 골라도 모델이 8K 처리를 하나요?",
     "아니에요. OpenProteus는 2x 모델이라 항상 입력의 2배까지만 처리합니다. "
     "사용자가 더 큰 해상도(예: 1080p 입력 + 8K 출력)를 골랐다면 모델이 "
     "먼저 1080p → 2160p (2x) 결과를 만들고, cv2가 그 결과를 한 번 더 "
     "키워서 8K로 출력합니다. 반대로 더 작은 해상도(예: 1080p 입력 + "
     "1080p 출력)를 골랐다면 모델이 2160p를 만든 다음 cv2가 1080p로 "
     "줄여서 출력해요."),

    ("마지막 리사이즈는 어떤 알고리즘을 쓰나요?",
     "모델 결과보다 더 키울 때는 OpenCV의 Lanczos4 (8×8 커널, 4 lobes), "
     "줄일 때는 Area 보간을 씁니다. Lanczos는 신경망 결과의 미세 디테일을 "
     "살리는 데 강하고, Area는 모아레/링잉 없이 깨끗하게 다운샘플합니다. "
     "DaVinci Resolve나 Premiere 같은 NLE 도구들이 흔히 쓰는 조합이에요."),

    ("OpenProteus는 어떤 영상에 잘 맞나요?",
     "노이즈가 적고 깨끗한 HD/FHD 라이브액션 영상을 4K로 늘릴 때 가장 잘 "
     "작동합니다. 오버샤프닝이나 아티팩트가 거의 없어요. 노이즈가 많은 "
     "영상이라면 디노이즈 탭에서 먼저 처리한 다음 업스케일을 적용하는 "
     "걸 권장합니다."),

    ("4K나 8K 출력을 골랐는데 처리가 너무 느려요.",
     "출력 해상도가 클수록 픽셀 수가 기하급수적으로 늘어 ffmpeg 인코딩과 "
     "디스크 쓰기 시간이 오래 걸립니다. 모델 추론 자체는 입력 해상도에 "
     "비례하지만, 1080p 입력에서 8K 출력이면 추가 cv2 리사이즈도 무거워져요. "
     "빠른 결과가 필요하면 Native(원본 × 2)를 추천합니다."),
]

_FAQ_UPSCALE_EN = [
    ("If I pick 8K target, does the model actually run at 8K?",
     "No. OpenProteus is fixed at 2x — it always produces a 2× version "
     "of the input. If you pick a larger target (e.g., 1080p input + 8K "
     "output), the model first makes 1080p → 2160p, and then cv2 "
     "enlarges that 2160p further to 8K. If you pick a smaller target "
     "(e.g., 1080p input + 1080p output), the model still makes 2160p "
     "and cv2 downsamples it back to 1080p."),

    ("What algorithm does the final resize use?",
     "When enlarging beyond the model's 2× output, OpenCV Lanczos4 "
     "(8×8 kernel, 4 lobes). When shrinking, OpenCV Area interpolation. "
     "Lanczos preserves the model's fine detail well; Area downsamples "
     "cleanly without moiré or ringing. This is the standard combo "
     "most NLEs (DaVinci Resolve, Premiere) ship."),

    ("What sources does OpenProteus suit best?",
     "Clean, low-noise HD/FHD live-action footage going to 4K. No "
     "oversharpening, no ringing. For noisy sources, run the Denoise "
     "tab first, then Upscale."),

    ("Why is 4K/8K output so slow?",
     "Pixel counts grow with the square of resolution, so ffmpeg "
     "encoding and disk writes get heavy at higher targets. Model "
     "inference scales with the input size, but for 1080p → 8K the "
     "additional cv2 enlargement step also costs more. If you want a "
     "fast result, pick Native (source × 2)."),
]


# Batch FAQ — answers the most common "what does this tab even do" and
# "where did my files go" questions. Stays short because the batch tab
# is mostly self-explanatory once you've seen the inpaint/denoise/upscale
# tabs individually.
_FAQ_BATCH_KO = [
    ("배치 탭은 어떨 때 쓰면 되나요?",
     "여러 클립을 한 번에 인페인트 → 디노이즈 → 업스케일로 돌리고 싶을 때 "
     "쓰세요. 단일 탭에서 따로따로 작업하면 클립마다 3번씩 클릭해야 하지만, "
     "배치 탭은 큐에 파일을 넣고 [전체 실행] 한 번이면 끝입니다. 한 클립만 "
     "넣어도 3단계 자동화 용도로 충분히 유용해요."),

    ("배치 뷰어에서도 마스크 미리보기가 되나요?",
     "네. 큐에서 클립을 선택하고 좌측 뷰어에 포커스를 둔 상태에서 **A 키**를 "
     "누르면 그 클립에 적용될 마스크(커스텀 ✱가 있으면 그것, 없으면 Dreamina "
     "자동)가 영상 위에 빨간색 반투명으로 표시됩니다. 인페인트 단일 탭과 동일 "
     "UX이고, 다른 탭에서는 작동하지 않아요."),

    ("Dreamina 자동 마스크가 워터마크를 살짝 못 덮어요.",
     "인페인트 카드의 **「마스크 확장」** 수치 슬라이더로 자동 프리셋 마스크 "
     "외곽을 N픽셀만큼 키울 수 있습니다. 4-8 정도면 가벼운 마진, 15+는 강하게 "
     "확장. 커스텀 마스크(우클릭으로 직접 그린 것)에는 영향이 없어요 — "
     "사용자가 의도해서 그린 그대로 사용됩니다."),

    ("배치 큐의 일부 클립에만 다른 마스크를 적용할 수 있나요?",
     "네. 큐의 클립을 **우클릭 → 「마스크 편집...」** 하면 브러시 / 사각형 / "
     "타원 / 지우개 도구로 그 클립 전용 마스크를 그릴 수 있습니다. 적용한 "
     "클립은 큐 행에 ✱ 표시가 붙고, 인페인트 단계에서 자동 Dreamina 프리셋 "
     "대신 그 마스크가 쓰입니다. 우클릭 → 「커스텀 마스크 제거」로 자동 "
     "프리셋으로 되돌릴 수 있어요. 마스크는 메모리에만 저장되므로 앱을 끄면 "
     "사라집니다(휘발성)."),

    ("마스크 편집기에서 첫 프레임 외 다른 프레임도 확인할 수 있나요?",
     "네, 편집기 하단의 프레임 슬라이더로 영상의 임의 시점을 볼 수 있습니다. "
     "마스크는 그대로 유지되고 배경 프레임만 바뀌어서, 워터마크가 영상 중후반에 "
     "흔들리거나 사라지는지 검증할 때 유용해요. 단일 인페인트 탭의 마스크 "
     "편집기에도 동일하게 적용됩니다."),

    ("세 단계 중 일부만 돌릴 수 있나요?",
     "네. 인페인트/디노이즈/업스케일 각 카드 제목 옆의 체크박스로 단계별 "
     "활성/비활성을 토글할 수 있습니다. 업스케일만 켜면 큐의 클립들을 바로 "
     "업스케일해서 output에 저장하고, 인페인트만 켜면 워터마크 제거 결과를 "
     "바로 output에 저장합니다. 인페인트+디노이즈, 디노이즈+업스케일 같은 "
     "두 단계 조합도 가능해요. 동작 원칙은 「마지막 활성 단계의 출력만 output "
     "폴더로, 그 앞 단계 출력은 cache로 → 사용 즉시 자동 삭제」입니다. 단 "
     "최소 한 단계는 켜야 [전체 실행]이 작동합니다."),

    ("처리 순서가 어떻게 되나요? 클립별로 3단계를 끝낸 다음 다음 클립으로 가나요?",
     "아니에요. 단계별로 일괄 처리합니다 — 큐의 모든 클립이 먼저 인페인트를 "
     "끝낸 다음, 모든 클립이 디노이즈, 그다음 모든 클립이 업스케일을 거칩니다. "
     "이렇게 하면 ProPainter / RAFT / OpenProteus 모델 가중치를 단계마다 "
     "한 번만 로드하면 되니까 클립별 3단계 방식보다 GPU 메모리 효율이 "
     "훨씬 좋아요."),

    ("최종 결과물은 어디에 저장되나요?",
     "기본은 프로젝트 폴더의 output/{yymmdd}_result/ 안입니다. 예를 들어 "
     "2026년 5월 14일에 실행하면 output/260514_result/ 폴더가 생기고 그 안에 "
     "각 클립의 최종본이 들어가요. 우측 상단 [출력 폴더] 입력란에 다른 경로를 "
     "지정하면 그 경로 안에 {yymmdd}_result 서브폴더가 생깁니다 (외장 드라이브 "
     "같은 곳도 OK). 같은 폴더로 두 번째 실행하면 _2, _3 꼬리표가 자동으로 "
     "붙어 이전 결과물을 덮어쓰지 않습니다. 완료되면 탐색기로 폴더가 자동으로 "
     "열립니다."),

    ("중간 단계 결과물(인페인트, 디노이즈)도 저장되나요?",
     "cache/batch_inpaint/, cache/batch_denoise/ 에 임시로 저장되긴 합니다. "
     "다만 다음 단계 시작 직후 자동으로 삭제됩니다 — 디노이즈가 시작되면 "
     "인페인트 캐시가 wipe되고, 업스케일이 시작되면 디노이즈 캐시가 wipe돼요. "
     "이렇게 해야 디스크가 한 단계 분량 이상으로 폭주하지 않습니다. "
     "중간 결과물이 필요하면 단일 탭에서 따로 처리하세요."),

    ("처리 도중 [중지]를 누르면 어떻게 되나요?",
     "현재 진행 중인 클립의 작업이 즉시 중단되고, 그 시점까지 만들어진 "
     "캐시 파일은 다음 실행 시 자동으로 정리됩니다. 이미 완료된 단계의 "
     "중간 결과물은 cache 폴더에 남아 있을 수 있는데, 다음 실행 또는 앱 "
     "재시작 시 sweep됩니다. 중지 후 같은 큐로 다시 실행하면 처음부터 "
     "다시 시작해요 — 이어서 처리는 지원하지 않습니다."),

    ("각 단계의 출력 코덱은 어떻게 정하나요?",
     "단계별로 독립적으로 설정합니다. 기본값은 모두 H.264 (MP4)이고, "
     "ProRes (MOV)를 고르면 그 아래에 프로파일 드롭다운(422 / 422 HQ / 4444 / "
     "4444 XQ 등)이 펼쳐집니다. 중간 단계도 ProRes로 두면 손실이 거의 없지만 "
     "디스크 용량이 훨씬 많이 필요해요. 일반적으로는 셋 다 H.264로 두면 "
     "충분합니다."),

    ("큐의 처리 순서는 어떻게 정해지나요?",
     "파일 이름을 기준으로 자동 정렬됩니다 — 숫자(0-9) → 알파벳(A-Z, "
     "a-z) → 한글(가-하) 순서예요. 'clip_01.mp4', 'clip_02.mp4'처럼 자릿수를 "
     "맞춰서 이름을 지으면 의도한 순서대로 처리됩니다. 큐에서 수동으로 "
     "재배치하는 기능은 아직 없어요."),

    ("뷰어 영역이 좁아서 답답해요.",
     "좌측 뷰어와 우측 큐 영역 사이의 경계선을 드래그하면 비율을 자유롭게 "
     "조절할 수 있습니다. 기본값은 뷰어가 약간 작은 45:55 비율인데, 어도비 "
     "미디어 인코더 같은 도구를 참고했어요."),
]

_FAQ_BATCH_EN = [
    ("When should I use the Batch tab?",
     "When you want to run multiple clips through Inpaint → Denoise → "
     "Upscale at once. Doing it in the individual tabs means 3 clicks "
     "per clip; Batch tab is one [Run all] press. Useful even for a "
     "single clip if you want the full 3-stage automation."),

    ("Does the batch viewer support the mask preview too?",
     "Yes. With a clip selected in the queue and focus inside the "
     "batch viewer, press **A** to toggle a red semi-transparent "
     "overlay showing the mask that will be applied (the custom ✱ "
     "mask if there is one, otherwise the auto-Dreamina). Same UX as "
     "the single Inpaint tab; the shortcut is scoped so it doesn't "
     "fire from other tabs."),

    ("The Dreamina auto-mask doesn't quite cover the watermark edges.",
     "Use the **«Mask expansion»** spinner on the inpaint card to "
     "dilate the auto preset outward by N pixels. 4-8 = light margin, "
     "15+ = aggressive. The setting has no effect on custom masks "
     "(those drawn via right-click → Edit mask) — those are used "
     "exactly as drawn."),

    ("Can a specific clip in the batch queue use its own custom mask?",
     "Yes. **Right-click a queue row → «Edit mask...»** to draw a "
     "custom mask with brush / rect / ellipse / eraser tools. Rows "
     "that have one show a ✱ marker, and the inpaint stage uses that "
     "mask instead of the auto-Dreamina preset. Right-click → «Remove "
     "custom mask» reverts to the auto preset. Custom masks are kept "
     "in memory only — they disappear when the app closes (volatile)."),

    ("Can I scrub through frames in the mask editor, not just frame 0?",
     "Yes — the slider at the bottom of the editor lets you preview "
     "any frame. The mask layer stays put while the background frame "
     "changes, so you can verify the mask still covers the watermark "
     "if it drifts later in the clip. Works in the single-tab inpaint "
     "mask editor too."),

    ("Can I run only a subset of the three stages?",
     "Yes — each stage card title has a checkbox. Turn off the stages "
     "you don't want. Upscale-only, Inpaint-only, Denoise-only, "
     "Inpaint+Denoise, Denoise+Upscale, etc. are all valid. The rule "
     "is: «the last enabled stage's output goes to the output folder; "
     "any earlier active stage routes through cache and gets wiped on "
     "next-stage start.» So Inpaint-only puts the inpainted result "
     "straight into output. At least one stage must be checked for "
     "[Run all] to work."),

    ("What's the processing order — clip-first or stage-first?",
     "Stage-first. All queued clips finish Inpaint first, then all "
     "finish Denoise, then all finish Upscale. This way the ProPainter / "
     "RAFT / OpenProteus weights only need to load once per stage — much "
     "more GPU memory-efficient than cycling through 3 models per clip."),

    ("Where do the final results land?",
     "Default: output/{yymmdd}_result/ under the project folder. e.g. "
     "running on 2026-05-14 creates output/260514_result/ with one file "
     "per clip. The [Output folder] field at the top of the batch tab "
     "lets you point to a custom path (e.g. an external drive) — a "
     "{yymmdd}_result subfolder is still created inside it. Same-day "
     "re-runs into the same folder auto-suffix _2, _3 so earlier "
     "batches aren't overwritten. Explorer opens the folder when the "
     "batch completes."),

    ("Are the intermediate (Inpaint / Denoise) results kept?",
     "They get parked in cache/batch_inpaint/ and cache/batch_denoise/ "
     "during the run, but get wiped as soon as the next stage starts — "
     "Inpaint cache drops when Denoise starts; Denoise cache drops when "
     "Upscale starts. This keeps disk usage bounded to roughly one "
     "stage's worth at any time. If you need intermediates, process them "
     "in the individual tabs."),

    ("What happens if I press [Stop] mid-run?",
     "The current clip's processing aborts immediately. Cache files "
     "produced up to that point get swept on the next launch. Stopping "
     "and re-running with the same queue restarts from the beginning — "
     "resume is not supported."),

    ("How are per-stage output codecs configured?",
     "Each stage has an independent codec selector. Defaults are H.264 "
     "(MP4) everywhere. Selecting ProRes (MOV) reveals a profile combo "
     "(422 / 422 HQ / 4444 / 4444 XQ, etc.). ProRes intermediates "
     "preserve quality but cost a lot more disk; H.264 across all three "
     "is usually fine."),

    ("How is the queue order decided?",
     "Auto-sorted by filename — digits (0-9) first, then alphabet (A-Z, "
     "a-z), then Hangul (가-하). Pad your numbers ('clip_01.mp4', "
     "'clip_02.mp4') for predictable ordering. Manual reordering inside "
     "the queue isn't supported yet."),

    ("The viewer area feels too narrow.",
     "Drag the divider between the left viewer and the right queue / "
     "settings column — you can resize freely. Default is ~45:55 with "
     "the viewer slightly smaller, modeled on Adobe Media Encoder's "
     "layout."),
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

        title = QLabel("CEC All-In-One")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(
            "font-size: 20px; font-weight: 700; "
            "border: none; background: transparent;")
        card_lay.addWidget(title)

        version = QLabel("v1.0.1")
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
        # Three-level structure per the design:
        #   large card (this FAQ wrap)
        #   ↳ medium card: sub-tabs Inpaint | Denoise | Upscale
        #     ↳ small cards: drawer-style _FaqItem rows
        is_ko = get_lang() == "ko"
        sections = [
            ("inpaint", _FAQ_INPAINT_KO if is_ko else _FAQ_INPAINT_EN),
            ("denoise", _FAQ_DENOISE_KO if is_ko else _FAQ_DENOISE_EN),
            ("upscale", _FAQ_UPSCALE_KO if is_ko else _FAQ_UPSCALE_EN),
            ("batch",   _FAQ_BATCH_KO   if is_ko else _FAQ_BATCH_EN),
        ]

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

        header = QLabel("FAQ" if not is_ko else "자주 묻는 질문")
        header.setStyleSheet(
            "font-size: 15px; font-weight: 700; "
            "border: none; background: transparent;"
            "padding-bottom: 4px;")
        wrap_lay.addWidget(header)

        # Sub-tab bar (Inpaint / Denoise / Upscale) — same checkable
        # QPushButton pattern as the main window's top tabs, smaller font.
        sub_tab_qss = (
            "QPushButton {"
            "  background: transparent;"
            "  border: none;"
            "  padding: 4px 10px;"
            "  font-size: 9pt;"
            "  color: rgba(200, 204, 220, 160);"
            "}"
            "QPushButton:hover { color: rgba(220, 224, 240, 240); }"
            "QPushButton:checked {"
            "  color: rgba(220, 224, 240, 255);"
            "  border-bottom: 2px solid #7aa2f7;"
            "  font-weight: 600;"
            "}"
        )
        tab_bar = QHBoxLayout()
        tab_bar.setContentsMargins(0, 0, 0, 4)
        tab_bar.setSpacing(4)
        self._faq_group = QButtonGroup(wrap)
        self._faq_group.setExclusive(True)
        self._faq_stack = QStackedWidget()
        for idx, (key, entries) in enumerate(sections):
            btn = QPushButton(tr(f"about_subtab_{key}"))
            btn.setCheckable(True)
            btn.setChecked(idx == 0)
            btn.setStyleSheet(sub_tab_qss)
            btn.clicked.connect(
                lambda _, i=idx: self._faq_stack.setCurrentIndex(i))
            self._faq_group.addButton(btn)
            tab_bar.addWidget(btn)
            self._faq_stack.addWidget(self._build_faq_list(entries))
        tab_bar.addStretch(1)
        wrap_lay.addLayout(tab_bar)
        wrap_lay.addWidget(self._faq_stack, 1)

        return wrap

    @staticmethod
    def _build_faq_list(entries) -> QWidget:
        """Single sub-tab page: scrollable column of drawer-style FAQ rows."""
        scroll = QScrollArea()
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            "QScrollArea { background: transparent; border: none; }")
        host = QWidget()
        host.setStyleSheet("background: transparent;")
        lay = QVBoxLayout(host)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        for q, a in entries:
            lay.addWidget(_FaqItem(q, a))
        lay.addStretch(1)
        scroll.setWidget(host)
        return scroll

    def refresh_logo(self):
        path = os.path.join(_LOGO_DIR, "aimz_white.png")
        if not os.path.exists(path):
            self._logo_label.setText("AIMZ")
            return
        pix = QPixmap(path)
        scaled = pix.scaledToWidth(280, Qt.SmoothTransformation)
        self._logo_label.setPixmap(scaled)

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

from app._version import __version__
from app.i18n import tr


PROXY_URL = "https://prompt-builder-proxy.khwhite0413.workers.dev"
DEFAULT_MODEL = "meta/llama-4-maverick-17b-128e-instruct"
MAX_HISTORY = 20  # last N turns kept in the request body


# Manual + persona for the chatbot. Korean primary; the model is asked
# to mirror the user's language for general questions.
SYSTEM_PROMPT = f"""\
당신은 "CEC All-In-One" 데스크톱 애플리케이션 안에 내장된 도우미 챗봇입니다.

# 응답 규칙
- 사용자 언어에 맞춰 답하세요 (한국어 질문 → 한국어, English question → English).
- CEC All-In-One 외 일반적인 질문에도 자유롭게 답해도 됩니다.
- 모르는 건 솔직히 모른다고 답하세요.
- 답변은 간결하게. 필요한 경우 마크다운(굵게/리스트)을 사용해도 됩니다.

# CEC All-In-One v{__version__} 개요
AI 영상 후처리 통합 데스크톱 도구 (PyQt5 기반). 네 가지 작업 탭 + About:
1. 인페인트 (Inpaint) — ProPainter로 워터마크/객체 제거
2. 디노이즈 (Denoise) — FlowDenoise로 시간적 광학흐름 기반 노이즈 감소
3. 업스케일 (Upscale) — OpenProteus 2x로 해상도 확대
4. 배치 (Batch) — 위 세 단계를 여러 클립에 자동으로 순차 적용

모든 탭이 동일한 영상 IO 정책을 공유합니다: untagged 영상은 untagged 그대로
출력하고, 24.04처럼 비정수 fps는 24.0 같은 표준값으로 자동 스냅합니다.

# 인페인트 탭 사용법
1. 좌측 영상 영역 클릭 또는 드래그&드롭으로 입력 영상 로드
2. [프리셋 ▼] 버튼 클릭 → "Dreamina (자동)" 선택
   → 영상 가로/세로 비율 자동 감지해서 적절한 마스크 PNG 적용
3. **A 키 누르면 뷰어에서 알파 마스크 오버레이 토글** (인페인트 탭 전용 단축키).
   적용된 마스크가 영상 위에 빨간색 반투명으로 표시되어 어디가 인페인트될지
   실제 픽셀과 함께 한눈에 확인할 수 있어요. 다시 A를 누르면 꺼집니다.
4. (선택) "마스크 확장 (px)" 스피너로 프리셋 외곽 N픽셀 확장 — 워터마크 가장자리
   잔상 방지용 (기본 4)
5. (선택) [마스크 편집] 버튼으로 브러시/사각형/타원/지우개로 직접 마스크 그리기
6. **FPS 강제** — 우측 패널의 "FPS" 라벨에 디노이즈와 동일한 자동 스냅값(24.04→24
   등)이 표시됩니다. 직접 지정하려면 **개발자 모드** 체크 → FPS Override의 Auto
   체크 해제 → 표준 rate(23.976/24/25/29.97/30/50/59.94/60) 드롭다운에서 선택.
   AI 영상 메타데이터가 어중간한 값일 때 강제로 표준 rate로 떨어뜨릴 때 사용.
7. [**미리보기**] (앞 80프레임) — 모달 다이얼로그가 떠서 좌우 와이프로 원본과
   비교 가능. [이전 미리보기 다시 보기] 버튼으로 재실행 없이 다시 열 수 있어요.
8. [전체 처리] 클릭 → output 폴더에 mp4 저장
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
   - **FPS** 라벨: 자동 스냅된 effective fps 표시
   - **FPS 강제**: 개발자 모드 → FPS Override의 Auto 해제 → 표준 rate 드롭다운에서
     선택. 인페인트/디노이즈와 동일한 UX
3. [전체 처리] 클릭
중요: 모델은 항상 입력의 정확히 2배만 처리합니다. 4K나 8K를 골라도 모델은
그 중간(2x) 결과만 만들고 cv2가 마지막에 Lanczos4(키울 때) / Area(줄일 때)로
리사이즈하는 구조입니다.

# 배치 탭 사용법
어도비 미디어 인코더 스타일의 전자동화 파이프라인. 여러 클립을 한 번에
인페인트 → 디노이즈 → 업스케일로 흘려보냅니다.
1. 비디오 파일을 큐에 드래그앤드롭하거나 [추가] 버튼으로 선택. 폴더 통째로
   드롭해도 그 안의 비디오만 자동 추출. 파일은 이름 기준 자동 정렬됩니다
   (숫자 → 알파벳 → 한글 순).
2. 단계별 설정 카드에서 각 단계 옵션 조정:
   - 인페인트: 프리셋(Dreamina 자동 가로/세로 감지) + 마스크 확장(px) + Max batch frames + 출력 코덱
   - 디노이즈: window size + Batch size + 출력 코덱
   - 업스케일: 모델 + 타겟 해상도 + Max batch frames + 출력 코덱
   각 카드 제목 옆 **체크박스로 단계 활성/비활성**을 토글할 수 있어요.
   업스케일만 돌리고 싶으면 인페인트·디노이즈를 끄세요. 인페인트만, 디노이즈만,
   인페인트+디노이즈 등 임의 조합 가능. 단 최소 하나는 켜야 합니다.
   각 출력 코덱은 H.264(MP4) 또는 ProRes(MOV) 선택. ProRes 선택 시 422/4444
   등 프로파일 드롭다운이 펼쳐집니다. 기본값은 셋 다 H.264.
   Batch frames 기본 0(=전체), 디노이즈 Batch size 기본 24. VRAM 부족할 때만 줄이세요.
2-1. **개별 클립에 커스텀 마스크 적용**: 큐의 클립을 우클릭 → "마스크 편집..."
     로 그 클립만의 마스크를 브러시/사각형/타원/지우개로 직접 그릴 수 있어요.
     커스텀 마스크가 적용된 클립은 큐 행에 ✱ 표시가 붙고, 인페인트 단계에서
     자동 Dreamina 프리셋 대신 그 마스크를 사용합니다. 우클릭 → "커스텀 마스크
     제거"로 자동 프리셋으로 되돌릴 수 있어요. 커스텀 마스크는 메모리에만
     저장되므로 앱을 끄면 사라집니다 (휘발성). 마스크 편집기 하단의 프레임
     슬라이더로 영상의 다른 시점도 확인 가능합니다.
2-2. **마스크 미리보기**: 큐의 클립을 선택한 상태에서 좌측 뷰어에 포커스를
     두고 **A 키**를 누르면 적용될 마스크가 영상 위에 빨간색 반투명으로 표시
     됩니다 (커스텀 ✱가 있으면 그 마스크, 없으면 Dreamina 자동 + 확장 적용된
     마스크). 다시 A 누르면 꺼져요. 인페인트 단일 탭의 A 키와 동일 UX.
2-3. **Dreamina 자동 마스크 확장 (px)**: 인페인트 카드의 "마스크 확장" 수치
     슬라이더는 자동 프리셋 마스크의 가장자리를 N픽셀만큼 외곽으로 키웁니다.
     워터마크 경계를 살짝 못 덮어 잔상이 보일 때 4-8 정도로 키우면 깔끔해져요.
     커스텀 마스크(✱)에는 영향 없음 — 사용자가 그린 그대로 사용.
3. [전체 실행] 클릭. 진행률은 통합 바에 단계+클립+프레임 진행도를 합쳐서
   표시합니다.
4. 출력 폴더는 기본적으로 프로젝트의 output/ 안에 yymmdd_result 서브폴더로
   저장되지만, 우측 상단 [출력 폴더] 입력란에서 직접 다른 경로로 지정할 수
   있어요. 외장 드라이브 같은 곳을 지정해도 OK입니다. 지정한 경로 안에도
   여전히 yymmdd_result 서브폴더가 생기고, 같은 날 두 번째 실행하면 _2,
   _3 꼬리표가 붙어 이전 결과물을 덮어쓰지 않습니다. 완료되면 탐색기로
   해당 폴더가 자동으로 열립니다.

처리 순서는 단계별 일괄 처리예요 — 모든 클립이 먼저 인페인트를 끝내고,
다음에 모든 클립이 디노이즈, 마지막으로 모든 클립이 업스케일을 거칩니다.
이렇게 하면 각 단계 모델 가중치를 단 한 번만 로드해서 GPU 메모리가 효율적.
중간 결과물은 cache/batch_inpaint/, cache/batch_denoise/ 에 잠깐 저장된
다음 다음 단계 시작 시 자동 삭제되어 디스크가 폭주하지 않습니다.

단계 토글 동작: **마지막 활성 단계의 출력만 output 폴더로 저장**되고, 그 앞의
활성 단계 출력은 cache 폴더로 가서 다음 단계 끝나면 삭제됩니다. 그래서
인페인트 하나만 켜놓으면 인페인트 결과가 바로 output에 저장되고 cache는
안 쓰여요. 디노이즈만, 업스케일만, 어떤 조합이든 동일하게 마지막 단계만
output에 떨어지는 구조입니다.

# 자주 묻는 질문
Q: 24.04fps 같은 어중간한 fps 영상을 24fps로 강제 출력할 수 있나요?
A: 네. 인페인트 / 디노이즈 / 업스케일 세 탭 모두 동일한 UX로 지원해요.
**개발자 모드** 체크 → 파라미터 패널의 **FPS Override** 행에서 Auto 체크 해제
→ 23.976/24/25/29.97/30/50/59.94/60 중 원하는 표준 rate 선택. 그 값이
reader의 -vf fps= (VFR을 CFR로 리샘플) + writer의 -r 둘 다에 적용돼서 출력
영상이 정확히 그 fps로 떨어집니다. Auto 체크되어 있으면 24.04 같은 값은
자동으로 가장 가까운 표준값(24.0)으로 스냅돼요.

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

Q: 인페인트 탭에서 마스크가 어디에 적용되는지 미리 볼 수 있나요?
A: 네, 인페인트 탭의 뷰어에서 **A 키**를 누르면 알파 마스크 오버레이가
빨간색 반투명으로 영상 위에 표시됩니다. 다시 A를 누르면 꺼져요. 이 단축키는
인페인트 탭에서만 작동합니다 (다른 탭에는 없음). 프리셋이나 브러시로 마스크를
설정한 다음 영상을 재생하면서 모든 프레임에서 마스크 위치가 맞는지 확인할 때
유용합니다.

Q: 배치 탭에서 처리 도중 [중지]를 누르면 어떻게 되나요?
A: 현재 진행 중인 클립의 작업이 즉시 중단되고, 그 시점까지 만들어진 캐시
파일은 다음 실행 또는 앱 재시작 시 자동 정리됩니다. 이어서 처리는 지원하지
않아요 — 다시 [전체 실행]을 누르면 큐 처음부터 다시 시작합니다.

Q: 배치 큐의 처리 순서를 수동으로 바꿀 수 있나요?
A: 현재 버전에서는 자동 정렬(파일명 기준: 숫자 → 알파벳 → 한글 순)만 지원하고
수동 재배치는 안 됩니다. 의도한 순서가 있다면 파일명 앞에 01_, 02_ 같은 접두사를
붙여서 조절하세요.

Q: 배치 중간 결과물(인페인트, 디노이즈)도 보고 싶어요.
A: 배치 탭은 중간 결과물을 자동으로 삭제합니다. 디스크 폭주 방지가 우선이기
때문이에요. 중간 결과물이 필요하면 단일 탭(인페인트 → 디노이즈)에서 따로
처리하거나, 배치 탭에서 **마지막 단계만 활성화**해서 그 결과를 output에
저장하는 방법도 있어요. 예를 들어 인페인트만 켜고 디노이즈/업스케일을
끄면 인페인트 결과가 바로 output에 저장됩니다.

Q: 배치 탭에서 업스케일만 돌릴 수 있나요?
A: 네. 각 단계 카드의 제목 옆 체크박스로 활성/비활성을 토글합니다. 인페인트와
디노이즈를 끄고 업스케일만 켜면 큐의 클립들을 바로 업스케일만 거쳐 output에
저장해요. 마찬가지로 인페인트만, 디노이즈만, 인페인트+업스케일 같은 임의
조합도 가능합니다. 단 최소 하나는 켜야 [전체 실행]이 작동해요.

Q: 배치 큐의 일부 클립에만 다른 마스크를 쓰고 싶어요.
A: 큐에서 그 클립을 **우클릭 → "마스크 편집..."** 하면 브러시 마스크 편집기가
열려요. 그리고 [적용]하면 그 클립만의 커스텀 마스크가 저장되고, 큐 행에 ✱
표시가 붙습니다. 인페인트 실행 시 그 클립은 자동 Dreamina 대신 커스텀 마스크
를 씁니다. 우클릭 → "커스텀 마스크 제거"로 자동 프리셋으로 복귀. 커스텀
마스크는 휘발성(메모리 저장)이라 앱 재시작하면 사라져요.

Q: 마스크 편집기에서 첫 프레임 말고 다른 프레임도 보면서 작업할 수 있나요?
A: 네, 편집기 하단에 프레임 슬라이더가 있어요. 슬라이더를 움직이면 캔버스
배경이 그 프레임으로 바뀌고, 그려둔 마스크는 그대로 유지됩니다. 워터마크가
영상 중간이나 후반에 약간 흔들리는 경우 다른 시점에서도 마스크가 잘 덮는지
확인할 때 유용해요. (단일 인페인트 탭의 마스크 편집기에도 동일하게 적용됩니다.)

# 추가 정보
- 언어 변경: 우측 상단 한국어/English 드롭다운 (변경 후 재시작)
- 로그 파일: <앱폴더>/logs/cec_aio.log
- 모델 가중치: <앱폴더>/weights/ (setup.bat이 자동 다운로드)

# v1.2.0에서 추가/변경된 기능
- **프레임 보간 추가** — RIFE 등 4개 모델로 영상을 2배·4배·8배 부드럽게 만들고,
  원하는 최종 FPS를 직접 고를 수 있음. 단독 탭과 배치 처리 모두 지원.
- **FastDVDnet 디노이즈 추가** — 영상 크기는 그대로 유지하면서 화면 노이즈만 빠르게
  줄이는 선택지를 추가. 기존 Flow Denoise도 계속 선택 가능.
- **장면 전환 보호 개선** — 인페인트·디노이즈·프레임 보간에서 기존 방식 또는
  PySceneDetect Adaptive 방식을 고를 수 있어 컷 사이의 잔상과 섞임을 줄임.
- **비교 뷰어 개선** — 업스케일과 프레임 보간 결과도 Before/After 와이프로 비교 가능.
  FPS가 달라도 두 영상이 같은 시간 위치와 정상 속도로 재생됨.
- **긴 영상과 취소 안정화** — 순차 처리와 진행률 표시를 보강하고, 작업 취소 후 다른
  영상을 실행할 때 이전 프레임 수가 남거나 종료 시 튕기던 문제를 수정.

# v1.1.0에서 추가/변경된 기능
- **인페인트 긴 영상 스트리밍 처리** — 전체 영상을 한 번에 메모리에 올리지 않고
  최대 80프레임 안전 윈도우와 시간 문맥 오버랩으로 순차 처리. 영상 길이가 늘어나도
  CPU/GPU 메모리 사용량이 일정하게 유지되고 배치 경계의 결과 튐을 줄임.
- **배치 최종 출력 완료 멈춤 수정** — 처리 단계를 모두 끄고 Final Output만 사용하는
  포맷 정규화 경로에서 빠른 워커 완료 신호를 놓쳐 무한 대기하던 경쟁 조건 해결.
- **FFmpeg 종료 안정화** — 장시간 포맷 변환의 stderr 버퍼 정체를 제거하고 취소·오류
  시 FFmpeg 프로세스와 임시 리소스를 확실히 정리.

# v1.0.9에서 추가/변경된 기능
- **배치 탭 이미지 입력 및 출력 지원** — 비디오 파일뿐만 아니라 단일 이미지(PNG, JPG,
  JPEG, WebP, BMP 등)를 배치 큐에 추가하여 인페인트, 디노이즈, 업스케일 파이프라인
  처리 가능.
- **이미지 출력 포맷 추가** — 배치 탭의 각 처리 단계 및 최종 출력(Final Output)
  코덱에 PNG 이미지(.png) 및 JPEG 이미지(.jpg) 옵션 추가. 단일 이미지 파일로
  직접 저장 가능.

# v1.0.8에서 추가/변경된 기능
- **FPS 수동 직접 입력 지원** — 인페인트 / 디노이즈 / 업스케일 / 배치 4개 탭 모두
  FPS 강제 드롭다운에 "직접 입력" 항목 신설. 소수점 3자리까지 원하는 float 값(예: 29.970,
  31.250 등)을 자유롭게 입력 가능.
- **자동 업데이트 시스템** — 앱 시작 시 서버에서 최신 버전을 백그라운드로
  조회, 새 버전이 있으면 팝업으로 알림. "지금 업데이트" 클릭 시 소스 파일
  자동 다운로드/교체/재시작. 사용자는 압축 해제나 setup.bat 재실행 안 해도
  됨. 헤더 우측의 "↻ 업데이트 확인" 버튼으로 언제든 수동 체크 가능.
- **버전 문자열 단일화** — `app/_version.py` 가 canonical 소스가 되어
  main_window / about_page / chatbot 이 모두 여기서 import. 다음 릴리즈
  bump 는 _version.py 한 줄만 수정하면 끝.
- **.bat 파일 CRLF 강제** — .gitattributes 로 배포 zipball 이 항상 CRLF
  라인엔딩으로 나오도록. 이전 v1.0.7 auto-update 사이클에서 LF-only .bat
  파일이 cmd.exe 파싱 실패를 유발했던 문제 해결.

# v1.0.7에서 추가/변경된 기능
- **업스케일 모델 15종 추가** — Sirosky Upscale-Hub 의 AniToon(3) / AniSD(3) /
  Ani4K v2(2) / AniScale 2(7) 계열 편입. 업스케일 탭과 배치 탭 모두 "계열 → 세부
  모델" 2단계 드롭다운으로 고름. 세부 모델 항목마다 마우스 올리면 성격 한줄평
  툴팁이 뜸(크기/속도/추천용도).
- **배치 탭 최종 출력 그룹 신설** — 비율 강제·출력 해상도·FPS 강제·최종 코덱
  네 파라미터를 인페인트/디노이즈/업스케일 스테이지와 별개로 상시 활성. 세
  스테이지 다 꺼도 이 섹션만으로 코덱·프레임·비율 정규화 가능(pass-through
  transcode). Adobe Media Encoder 하위 워크플로 대체 목적. Topaz 출력 정리 같은
  케이스에 유용.
- **비율 강제 = Fill Frame (확대 후 크롭)** — Premiere "Fill Frame" 과 동일 시맨틱.
  타겟 프레임을 완전히 덮도록 스케일업 후 초과분 크롭. 정확한 목표 해상도 도달
  + 검은 띠 없음. 이전 letterbox(bar 추가) 동작은 제거.
- **FPS 자동 스냅 커버리지 확장** — 31.24 → 30, 28.992 → 29.97, 30.14 → 30 같은
  Topaz 스타일 비정수 fps 를 자동으로 표준 rate 로 스냅. tolerance 를 0.1 → 2.0
  으로 넓히고 표준 rate 리스트에 12·15·18·48·100·120·240 추가. 45fps 같은 진짜
  특이 값은 그대로 보존.
- **배치 큐 표시 = 원본 fps** — 큐 목록에는 파일 메타데이터 그대로(스냅 X), 스냅
  결과는 최종 출력 섹션 하단 라벨에 별도 표시(선택 클립: 31.24 → 30.00 스냅됨).
- **실행 중 큐 스크롤/선택 복구** — 이전엔 실행 중 큐가 전체 disabled 되어 스크롤·
  프리뷰까지 막혔던 문제. 이제 실행 중에도 큐 스크롤/클릭/프리뷰 다 되고, 파일
  추가·삭제·마스크 편집 같은 mutation 만 차단.

# v1.0.6에서 추가/변경된 기능
- **디노이즈 파이프라인 스트리밍 리팩터** — 이제 프레임 수·해상도 무관하게
  처리 가능. 예전엔 전체 영상을 CPU RAM에 통째로 float32 텐서로 올린
  뒤 처리하다 8000프레임 720p만 넘어가도 페이지파일 스와핑 지옥 또는
  CPU allocator OOM으로 뻗었는데, 이제 200프레임 청크씩 앞뒤 3프레임
  오버랩으로 흘려보내며 처리합니다. 프레임별 결과는 배치 모드와 시각적
  으로 동일합니다.
- **weight_vis 잔재 제거** — temporal_flow_average가 반환하던
  27.6GB(8000×720p 기준) 짜리 시각화 텐서가 실제로는 아무 곳에서도
  안 읽히던 dead code였음. 삭제로 대형 영상에서 발생하던 특정 OOM
  지점 해결.
- **청크 진행률 슬림 바** — 디노이즈 탭·배치 탭 양쪽에, 긴 클립을
  처리할 때 메인 진행률 바 아래에 얇은(6px) 부속 바가 자동으로 나타
  나 청크 단위 진행률을 별도로 표시합니다. 짧은 클립엔 나타나지 않고
  기존과 동일. 배치 탭에서는 디노이즈 페이즈 동안만 표시.

# v1.0.5에서 추가/변경된 기능
- **업스케일 16:9 / 9:16 자동 비율 강제 옵션** — 단일 업스케일 탭과 배치 탭
  업스케일 섹션 양쪽에 같은 토글 추가. 켜면 결과의 긴 변을 기준으로 가로
  영상은 16:9, 세로 영상은 9:16 으로 자동 판단해 가운데 기준 crop 합니다.
  예: 3840×2400 → 3840×2160. 짧은 변만 양쪽 가장자리에서 같은 양만큼 잘림
  (letterbox 검은 띠 없음). 기본값 OFF.
- **배치 탭 출력 폴더 정책 정리** — 사용자가 출력 폴더를 직접 지정하면 그
  폴더에 **바로** 결과물을 저장합니다 (예전엔 그 안에 또 `<yymmdd>_result/`
  하위 폴더를 만들었음). 비워두면 기존대로 `output/<yymmdd>_result/` 로
  자동 생성. 같은 폴더로 재실행해도 같은 이름 파일이 있으면 파일명에 `_2`,
  `_3` … 이 붙어서 덮어쓰기 방지.
- **배치 탭 동명 클립 데이터 손실 수정** — 서로 다른 폴더의 같은 이름 클립
  (예: `A/scene.mp4`, `B/scene.mp4`) 을 큐에 같이 넣으면 인페인트 중간
  캐시에서 두 번째 클립이 첫 번째 클립을 덮어써서 결과가 둘 다 같은 영상
  으로 나오던 문제. 모든 단계에 충돌 회피를 적용해 각 클립이 고유 파일명
  으로 끝까지 흘러갑니다.
- **배치 탭 빈 결과 폴더 정리** — 첫 클립 실패 등으로 결과물이 하나도 안
  만들어졌으면 빈 날짜 폴더를 자동 삭제. 다음 시도 시 `_2`, `_3` 카운터가
  부풀어오르던 문제 해결. (부분적으로 성공한 결과물은 그대로 보존.)
- **휠 스크롤로 파라미터 값이 바뀌던 문제 수정** — 파라미터 패널을 휠로
  스크롤하다가 슬라이더·스핀박스·콤보박스 위에 커서가 닿으면 값이 조용히
  바뀌던 동작 차단. 이제 휠은 패널 스크롤에만 사용됩니다.
- **배치 탭 "오디오 / 오디오 없음" 라벨 정확도 개선** — setup.bat 가 ffmpeg
  와 함께 ffprobe 도 동봉(`./ffmpeg/`). 비-UTF-8 로케일이나 한글 메타데이터
  영상에서 오디오 인식이 가끔 어긋나던 문제 해결. 폴백 경로도 견고화
  (timeout 10→30s, UTF-8 + errors=replace, stdin 차단).

# 안정성 관련 알아두면 좋은 점 (v1.0.6 기준)
- 배치 탭 디노이즈 단계는 단일 디노이즈 탭과 정확히 동일한 파라미터(flow_scale,
  batch_size 등)로 돌아가요. 단일 탭에서 잘 돌던 설정이 배치에서 안 돌면 그건
  버그니까 알려주세요.
- 배치에서 단계 사이마다 PyTorch CUDA 캐시를 OS에 반환합니다. 인페인트 →
  디노이즈 → 업스케일 전환 시 GPU 메모리가 깔끔하게 회수돼서 단편화로 인한
  hang을 방지해요.
- MEMFOF 모델 가중치는 첫 setup.bat 실행 시 HuggingFace에서 받아 weights/memfof
  에 캐시됩니다. 이후 실행은 캐시 활용 + 짧은 HEAD 요청 한 번. 인터넷 안 되면
  명확한 에러 메시지가 떠요 (조용히 hang 안 함).
- ffmpeg 인코더가 실패하면 단순 "Broken pipe" 대신 ffmpeg의 마지막 stderr
  메시지(코덱 / 출력 경로 / 오디오 mux 문제 등)를 함께 표시해서 진짜 원인을
  바로 알 수 있어요.
- 뷰어 줌/팬은 단일 ViewportController 모듈(app/viewport_controller.py)이
  담당. 향후 추가될 뷰어에도 ViewportController(widget) 한 줄이면 동일한
  조작 체계가 자동으로 들어갑니다.
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
                        f"CEC-AllInOne/{__version__} (+khwhite0413)",
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

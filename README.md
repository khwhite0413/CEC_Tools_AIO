# CEC All-In-One

AI 영상 후처리 통합 데스크톱 도구. PyQt5 기반. 인페인트 / 디노이즈 / 업스케일을
한 앱에서 처리하고, **배치 탭**으로 여러 클립을 한 번에 자동 파이프라인 처리.
모든 탭이 동일한 영상 IO 정책(untagged 통과, 비정수 fps 스냅, 시청자 캐시
격리)을 공유합니다.

네 가지 처리 도구 + About:
- **인페인트** — ProPainter 기반 워터마크 / 객체 제거
- **디노이즈** — MEMFOF + RAFT 광학흐름 기반 시간적 가중평균 디노이저
  (FlowDenoise v1.0.0 그대로 통합)
- **업스케일** — Sirosky OpenProteus(SRVGGNetCompact 2x) 기반 슈퍼-레졸루션
- **배치** — Inpaint → Denoise → Upscale 자동 파이프라인 (단계 토글 가능)

## 설치 (Windows)

1. 저장소를 압축 해제
2. `setup.bat` 실행 — 자동으로:
   - Python 3.12 임베디드 다운로드
   - GPU 감지 후 적합한 PyTorch 인덱스(`cu118` / `cu124` / `cu128` / `cpu`) 설치
   - `requirements.txt` 의존성 설치 (PyQt5, opencv, scipy, einops, timm,
     spandrel, huggingface_hub, imageio-ffmpeg 등)
   - **MEMFOF** 패키지 + HuggingFace 가중치 사전 다운로드 (디노이즈용)
   - **ProPainter** 소스 zip 다운로드 + 가중치 3개 (~580MB, 인페인트용)
   - **OpenProteus** SR 가중치 (~2.4MB, 업스케일용)
   - Pretendard 폰트 (OFL)
3. `run.bat` 실행 — 최소화된 cmd로 띄우고 로그는 `logs/cec_aio.log`에 tee

> 기존 시스템 Python과 완전 격리. 디스크는 처음 셋업 시 ~3GB 차지 (Python + 모델
> 가중치 + 의존성).

## 탭 구성

상단 헤더: **인페인트 / 디노이즈 / 업스케일 / 배치 / About** (기본 = 인페인트).
탭 전환해도 우측 하단 챗봇 버블은 항상 떠 있습니다.

### 1. 인페인트 (ProPainter)

워터마크나 작은 객체 제거. 마스크 영역만 잘라서 모델에 통과시키고 결과를
원본에 다시 덮어쓰는 방식 → 마스크 바깥 픽셀 손실 0.

- **마스크 입력 2가지**
  - **프리셋 ▼ → Dreamina (자동)**: 영상 가로/세로 비율 자동 감지 →
    적합한 마스크 PNG 자동 적용 + anchor(top-left) 정렬
  - **마스크 편집**: 모달 다이얼로그로 직접 그리기 — 브러시 / 사각형 / 원형 /
    지우개 + 크기 슬라이더 + 실행취소(Ctrl+Z) + `[`/`]` 크기 ± + **프레임
    슬라이더로 영상의 임의 시점 검증** (마스크 유지, 배경만 변경)
- **A 키 알파 마스크 미리보기** — 뷰어에 포커스 두고 A 누르면 현재 마스크가
  영상 위에 빨간색 반투명으로 표시. 다시 A로 끔
- **미리보기 (앞 80프레임)** — 모달 다이얼로그로 좌우 와이프 비교 (디노이즈와 동일 UX).
  [이전 미리보기 다시 보기] 버튼으로 재실행 없이 다시 열 수 있음
- **전체 처리** — 사용자 지정 출력 경로 + 코덱
- **FPS 표시 + 강제** — 우측 패널 상단의 "FPS" 라벨에 자동 스냅된 effective fps 표시.
  Dev 모드 → FPS Override의 Auto 해제 → 표준 rate(23.976/24/25/29.97/30/50/59.94/60)
  또는 "직접 입력"(소수점 3자리 float) 선택 시 리더(-vf fps=)와 라이터(-r) 둘 다에 강제 적용. 디노이즈/업스케일/배치 탭과 동일
- 파라미터 (기본): 모델, Max batch frames, 마스크 확장 (px)
- 파라미터 (Dev 모드): FP16, neighbor / ref_stride / subvideo / raft_iter, FPS Override

### 2. 디노이즈 (FlowDenoise)

AI 영상 / 라이브액션의 시간적 노이즈 제거. window_size 만큼 이웃 프레임을
참조해 광학흐름으로 정렬한 뒤 가중평균. 가만히 있는 디테일은 보존되고
지글거리는 노이즈만 빠짐.

상세는 [`FlowDenoise` 별도 저장소](https://github.com/khwhite0413/FlowDenoise)
README와 동일.

### 3. 업스케일 (OpenProteus)

HD/FHD 라이브액션 → 4K (또는 그 이상). 오버샤프닝/링잉 없음.

- 모델은 항상 입력의 **2x로 추론** → cv2가 마지막에 사용자 지정 타겟 해상도로
  Lanczos4(키울 때) / Area(줄일 때) 리사이즈
- 출력 해상도 선택: Native (원본×2) / HD / FHD / QHD / 4K / 8K
- 예상 해상도 + 비율 자동 표시 (`예상: 3840×2160 · 16:9`)
- **FPS 표시 + 강제** — 인페인트 / 디노이즈와 동일한 UX. Dev 모드의 FPS Override에서
  표준 rate 선택 가능

### 4. 배치 (전자동화 파이프라인)

여러 클립을 인페인트 → 디노이즈 → 업스케일 순서로 한 번에 처리. Adobe Media
Encoder 스타일 레이아웃: 좌측 뷰어, 우측 큐 + 단계별 설정 카드. QSplitter로
비율 자유 조정.

- **큐 관리** — 파일/폴더 드래그앤드롭, [추가] / [선택 삭제] / [비우기].
  파일명 기준 자동 정렬 (숫자 → 알파벳 → 한글). 각 행에 메타데이터
  (해상도 / fps / 프레임 수 / 길이 / 오디오) 자동 표시
- **단계별 활성/비활성 체크박스** — 각 카드 제목 옆 체크박스로 단계 토글.
  업스케일만 / 인페인트만 / 인페인트+디노이즈 등 임의 조합 가능
- **클립별 커스텀 마스크** — 큐에서 클립 **우클릭 → 마스크 편집...** 으로
  그 클립 전용 마스크 그리기. 적용된 클립은 행에 **✱** 표시. 인페인트 단계
  실행 시 자동 Dreamina 대신 사용. 휘발성(메모리 저장, 앱 재시작 시 사라짐)
- **A 키 마스크 미리보기** — 큐에서 클립 선택 + 뷰어 포커스 → A로 알파 마스크
  오버레이 토글 (단일 인페인트 탭과 동일 UX)
- **Dreamina 마스크 확장 (px)** — 자동 프리셋 마스크 가장자리를 N픽셀 외곽
  확장. 워터마크 경계 잔상 방어용. 커스텀 마스크에는 영향 없음
- **단계별 batch 설정** — 각 카드에 GPU 메모리 튜닝 노브 (Max batch frames /
  Batch size)
- **이미지 & 비디오 동시 지원** — 비디오 파일뿐만 아니라 단일 이미지(PNG, JPG, WebP 등)도 큐에 등록하여 인페인트/디노이즈/업스케일 및 최종 출력 처리 가능
- **단계별 독립 출력 코덱** — 각 단계 및 최종 출력에서 H.264 / ProRes / PNG / JPEG 선택 가능
- **출력 폴더 사용자 지정** — 비워두면 `output/{yymmdd}_result/`, 직접 지정 시
  그 경로 안에 `{yymmdd}_result/` 서브폴더 생성. 같은 폴더 재실행 시 `_2`,
  `_3` 자동 suffix
- **단계별 일괄 처리** — 모든 클립이 인페인트 끝나고 → 모든 클립 디노이즈 →
  모든 클립 업스케일 순. 단계당 모델 가중치 한 번만 로드 → GPU 효율 ↑
- **캐시 즉시 정리** — 마지막 활성 단계 출력만 `output/`, 그 앞 단계 출력은
  `cache/batch_*` 로 가서 다음 단계 시작 시 wipe. 디스크 폭주 방지
- **시각 피드백** — 큐 행은 현재 처리 중(주황 + ASCII 스피너 + 클립별 %),
  단계 완료(초록), 오류(빨강) 색상. 하단 진행률 바는 전체 진행도, 취소/오류
  시 연한 빨강

### 5. About

좌측 명함(제품명 + v1.0.9 + 연락처). 우측 FAQ는 **인페인트 / 디노이즈 / 업스케일
/ 배치** 4개 서브탭으로 나뉘어 각 도구 특화 질문 + 기술적 디테일(예: 업스케일의
Lanczos4/Area 알고리즘 선택, 색역 태그 처리, 배치 단계 토글 동작 등) 정리.

## 출력 포맷 (모든 처리 탭 공통)

매 탭 우측 패널 하단 **출력 포맷** 카드:

| 옵션 | 동작 |
|---|---|
| **Auto (소스 따라감)** | ProRes 소스 → ProRes 출력, 그 외 → H.264. 다단계 워크플로우(인페인트 → 디노이즈 → 업스케일)에서 매번 h264 재압축으로 인한 손실 방지 |
| **H.264 MP4** | libx264 + yuv420p + AAC. 작은 파일, 빠른 디코드 |
| **ProRes MOV** | prores_ks + yuv422p10le(422 계열) 또는 yuv444p10le(4444). 후작업 / 합성용. 서랍에서 프로파일 선택: Proxy / LT / 422 / 422 HQ / **4444 (기본)** / 4444 XQ |

> 디노이즈 탭에는 추가로 **PNG sequence** 옵션 + ProRes 프로파일 카드가 있습니다
> (FlowDenoise 고유 파라미터 패널 그대로 유지).

## 색관리 — 모든 탭에서 동일

mp4/mov는 컨테이너 atom (`colr`/`nclc`)과 H.264 VUI에 컬러 메타
(primaries / transfer / matrix / range)를 박을 수 있는데, **있을 수도 있고
없을 수도 있다**가 핵심입니다. Nuke mov64 reader는 태그가 없으면 Rec.601
휴리스틱으로 풀고, 일부 플레이어는 Rec.709로 풉니다.

### Seedance 2.0 같은 untagged 소스

ffprobe로 소스 태그 유무를 먼저 확인:

- **태그 있음** → primaries/transfer/matrix/range를 그대로 출력에도 박음
  (`-movflags +write_colr`로 컨테이너 atom + VUI 둘 다 작성)
- **태그 없음** → 출력도 **untagged 유지** — 기본값 강제 안 함. 소스와 출력에
  동일한 휴리스틱이 적용되니 어긋날 여지가 없음

세 탭(인페인트/디노이즈/업스케일) 모두 동일한 `fdcore.VideoInfo` 통과 →
동일한 untagged 정책.

## 비정수 fps 자동 스냅

Seedance 같은 AI 영상 생성기는 24.04, 24.06 같은 어중간한 fps로 영상을 뽑는
버릇이 있습니다. NLE 타임라인에 그대로 올리면 시간이 갈수록 한두 프레임씩
밀립니다. 모든 탭이 `_snap_fps()`로 가장 가까운 표준값(23.976 / 24 / 25 /
29.97 / 30 / 50 / 59.94 / 60, ±0.1 tolerance)으로 자동 보정 후 인코딩.

## 도우미 챗봇

우측 하단에 떠 있는 💬 버블 클릭 → 380×520 채팅 패널이 위로 펼쳐짐.

- **NIM 무료 LLM** (`meta/llama-4-maverick-17b-128e-instruct`)을 사용자
  소유 Cloudflare Workers 프록시를 통해 호출 (API 키 노출 없음)
- 시스템 프롬프트에 CEC All-In-One 매뉴얼 + FAQ 박혀있어서 "인페인트 어떻게 써?"
  류 질문에 정확히 답함
- 일반 질문(번역, 영상 후작업 팁 등)도 자유롭게 가능
- 한/영 자동 인식 (사용자 언어로 응답)
- Enter 전송 · Shift+Enter 줄바꿈

## 캐시 / 출력 격리

cv2가 영상 파일에 핸들을 잡으면 Windows에서 원본을 잠금합니다 (이동/리네임 불가).
모든 탭이 처리 직후 결과물을 `cache/<tab>_view.<ext>` (또는 미리보기는
`cache/<tab>_preview.mp4`)로 별도 복사 → 그쪽을 player에 로드 →
`output/`의 진짜 deliverable은 자유롭게 만질 수 있게 유지.

배치 탭 중간 결과물은 `cache/batch_inpaint/`, `cache/batch_denoise/`로 가고,
다음 단계 시작 시 즉시 wipe (디스크 폭주 방지). 마지막 활성 단계 결과만
`output/{yymmdd}_result/`로 직행.

캐시 파일은 앱 종료 시 + 다음 시작 시 자동 sweep (전 세션이 crash해서
OS 락이 안 풀린 케이스 대비 양쪽).

## 아키텍처

```
main.py                       - 로깅 셋업 → torch 선임포트 → QApplication
app/
  main_window.py              - 5탭 셸 + 챗봇 오버레이 부착 + 드래그앤드롭 라우팅
  inpaint_page.py             - 인페인트 탭 UI (프리셋 + 마스크 편집 + ProPainter 파라미터)
  inpaint_worker.py           - QThread 래퍼 (vdcore.InpaintEngine 호출)
  mask_editor_dialog.py       - 브러시/사각형/원형/지우개 + 프레임 슬라이더
  parameter_panel.py          - 디노이즈 파라미터 폼 (FD 그대로)
  workers.py                  - DenoiseWorker / PreviewWorker (FD 그대로)
  upscale_page.py             - 업스케일 탭 UI (OpenProteus + 해상도 선택)
  upscale_worker.py           - QThread 래퍼 (spandrel 모델 로더)
  batch_page.py               - 배치 탭 UI (큐 + 단계별 카드 + 큐 행 색상/스피너)
  batch_orchestrator.py       - 동적 파이프라인 QThread (단계 토글 + 캐시 라우팅)
  about_page.py               - About 카드 + 4탭 서브 FAQ
  chatbot_panel.py            - 플로팅 챗봇 (NIM via Cloudflare 프록시)
  preview_dialog.py           - 디노이즈 좌우 wipe 미리보기 (FD 그대로)
  video_player.py             - cv2 기반 in-app 비디오 플레이어 (+ 알파 마스크 오버레이)
  i18n.py                     - 한/영 번역 사전
fdcore/                       - 디노이즈 코어 (FlowDenoise v1.0.0 그대로)
  pipeline.py / temporal_average.py / flow_models.py / flow_warp.py /
  video_io.py / color_spaces.py / extract_noise.py / progress.py
vdcore/                       - 인페인트 코어 (VDInpaint 발췌)
  inpaint_engine.py           - ProPainter 추론 래퍼
  video_io.py                 - ffmpeg 기반 read_frames_batch (네 탭 공용)
  mask_manager.py / model_downloader.py / watermark_detect.py / config.py
third_party/
  memfof/                     - (선택) MEMFOF 저장소 vendoring — 디노이즈
  ProPainter/                 - ProPainter 저장소 zip 다운로드 — 인페인트
presets/                      - Dreamina 마스크 PNG (가로 / 세로)
assets/{fonts,logo}/          - Pretendard OTF + AIMZ 로고
weights/                      - 모델 가중치 (setup.bat이 채움)
  memfof/                     - HuggingFace 캐시 (디노이즈)
  ProPainter.pth + raft-things.pth + recurrent_flow_completion.pth (인페인트)
  openproteus/2x_OpenProteus_Compact_i2_70K.pth (업스케일)
python/                       - 임베디드 Python (setup이 채움)
cache/ / logs/ / output/      - 런타임 (sweep 대상)
```

## 테스트

디노이즈 코어 단위테스트만 포함:

```
python -m unittest discover -s tests
```

- u8 → tensor → u8 라운드트립 비트정확
- TPDF dither 평균 보존 + 양자화 코드 다양성 증가
- ProRes / h264 인코더 명령에 소스 nclc / VUI 태그가 forward 됨
- 오디오 stream-copy 명령이 `-c:a copy` 사용

## 라이선스

- 본 앱: MIT (`LICENSE` 참고)
- **MEMFOF**: [HuggingFace 페이지](https://huggingface.co/egorchistov/optical-flow-MEMFOF-Tartan-T-TSKH) 확인
- **ProPainter**: [원본 저장소](https://github.com/sczhou/ProPainter) NTU S-Lab License
- **OpenProteus**: [Sirosky/Upscale-Hub](https://github.com/Sirosky/Upscale-Hub/releases/tag/OpenProteus) — Sirosky 본인이 학습한 모델
- 원본 ComfyUI 디노이즈 노드: [AIMZ-GFX/ComfyUI-FlowDenoise](https://github.com/AIMZ-GFX/ComfyUI-FlowDenoise) MIT
- Pretendard 폰트: SIL Open Font License 1.1

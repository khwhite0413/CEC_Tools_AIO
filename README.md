# FlowDenoise (Standalone)

ComfyUI 없이 동작하는 광학 흐름 기반 시간적 비디오 디노이저. PyQt5 데스크톱 앱.

원본 ComfyUI 커스텀 노드 [`ComfyUI-FlowDenoise`](https://github.com/AIMZ-GFX/ComfyUI-FlowDenoise)
의 알고리즘(MEMFOF / RAFT optical flow + 시간적 가중평균)을 독립 실행형
GUI 앱으로 재구성한 것입니다.

## 설치 (Windows)

1. 이 저장소를 압축 해제
2. `setup.bat` 실행 — 자동으로:
   - Python 3.12 임베디드 다운로드
   - GPU 감지 후 적합한 PyTorch 인덱스(cu118 / cu124 / cu128 / cpu) 설치
   - `requirements.txt` 의존성 설치
   - MEMFOF 패키지 + 가중치 사전 다운로드
3. `run.bat` 실행

> 기존 시스템 Python과 격리되어 동작하므로 다른 Python 환경에 영향을 주지 않습니다.

## 사용법

1. 비디오 파일을 메인 윈도우에 드래그 앤 드롭 또는 [파일 선택]
2. 우측 패널에서 출력 포맷 선택 (기본 H.264 MP4), batch_size 조정
3. **미리보기 (앞 80프레임)** — 좌우 wipe 비교 (cv2 기반 in-app 플레이어).
   미리보기는 포맷 무관하게 항상 H.264 mp4로 캐시됩니다 (cv2 호환성).
4. **전체 처리** — 진행률 + ETA. 완료 시 `<원본>_denoised.<ext>` 생성 (오디오 보존)

## 출력 포맷

| 포맷 | 용도 | 컨테이너 / 코덱 | 비고 |
|---|---|---|---|
| H.264 MP4 (기본) | 일반 배포, 빠른 디코드 | `libx264` 8-bit yuv420p, AAC 오디오 | 작은 파일, 어디서나 재생 |
| ProRes MOV | NLE / Nuke deliverable | `prores_ks` 10-bit (yuv422p10le / yuv444p10le) | 색감 보존, 파일 큼 |
| PNG sequence | 비디오 무관 보관 | 폴더 + numbered PNG + meta.json | 코덱 손실 0 |

ProRes 프로파일: `proxy / lt / standard / hq / 4444 / 4444xq` (ProRes 선택 시 노출).

## 색관리 — 소스 태그 상태 그대로 통과

mp4/mov는 컨테이너 atom (`colr`/`nclc`)과 H.264 VUI에 컬러 메타
(primaries / transfer / matrix / range)를 박을 수 있는데, **있을 수도 있고
없을 수도 있다**는 점이 핵심입니다. 다운스트림은 두 경우를 다르게 해석합니다 —
Nuke mov64 reader는 태그가 없으면 Rec.601 휴리스틱으로, 일부 플레이어는
Rec.709로 풀어버립니다.

가장 흔한 사고: **소스는 untagged인데 출력은 bt709로 태그가 박혀 나가는
경우**. Nuke가 소스에는 Rec.601 inverse, 출력에는 bt709 inverse를 적용해
미드톤이 어긋나고, "디노이즈가 색을 바꿔놨다"는 오해로 직결됩니다.

### Seedance 2.0 같은 untagged 소스

Seedance 2.0이 뽑는 mp4에는 nclc atom이 없습니다. FlowDenoise는 ffprobe로
소스의 태그 유무를 먼저 확인합니다:

- **태그가 있으면**: primaries/transfer/matrix/range를 그대로 복사
  (`-movflags +write_colr`로 컨테이너 atom + VUI 둘 다 작성).
- **태그가 없으면**: 출력도 **untagged 유지** — 기본값을 강제하지 않음.
  소스와 출력에 동일한 휴리스틱이 적용되니 어긋날 여지가 없음.

### 그 외

- **디코드**: `FFmpegReader`(ffmpeg subprocess) — Nuke 내부 ffmpeg와 동일
  사슬이라 RGB 표현이 일치.
- **TPDF dither**: 8-bit h264 출력 직전 ±1 LSB triangular dither.
  bgr48le 16-bit 파이프라인을 8-bit로 양자화할 때 어두운 영역 banding 방지.
  ProRes(10-bit) / PNG에는 미적용.
- **양자화 라운딩**: `np.rint`(round-half-to-even)로 u8→float→u8 비트정확.
- **ProRes**: Range tv 고정 + 소스 nclc 보존 (NLE 컨벤션).

## 파라미터

기본값은 단순화된 패널에서 batch_size만 노출되고, **Dev mode** 체크박스를 켜면
전체 파라미터가 노출됩니다.

### Temporal Flow Average

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| window_size | 3 | 좌우 평균할 이웃 프레임 수 |
| weight_decay | 0.80 | 시간 거리에 따른 지수 감쇠 |
| flow_model | memfof | memfof / raft_small / raft_large |
| flow_iterations | 4 | 흐름 정밀화 반복 |
| color_threshold | 0.020 | 색 유사도 임계값 (낮을수록 엄격) |
| scene_threshold | 0.030 | 씬 컷 검출 임계값 |
| batch_size | 24 | MEMFOF 배치. 1080p RTX 5080 16GB ≈ 16, RTX 5090 32GB ≈ 32 |
| precision | bf16 | bf16 / fp32 |
| flow_scale | 0.5 | flow 추정 입력 다운샘플 비율 (warp는 항상 풀 해상도) |
| clamp_output | true | 출력을 [0,1] 범위로 클램프 |

### 출력

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| output_format | h264 | h264 / prores / image_seq |
| prores_profile | 4444 | proxy / lt / standard / hq / 4444 / 4444xq |

### 노이즈 미리보기 (선택)

체크박스를 켜면 처리 후 노이즈 시각화 영상을 우측 미니뷰어에 표시.

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| compute_noise_preview | false | 노이즈 시각화 생성 여부 |
| noise_preview | heatmap | heatmap / signed / gray |
| noise_amplify | 5.0 | 시각화 증폭 |

### Dev mode 전용

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| passthrough | false | 디노이즈 완전 우회. read → write만. ffmpeg 인코드/디코드 사슬 자체의 색 보존을 검증할 때 사용 |

## 아키텍처

```
main.py                 - torch 선임포트 → QApplication
app/                    - PyQt5 UI 레이어
  main_window.py        - 메인 윈도우 + 드래그앤드롭 + preview h264 강제
  parameter_panel.py    - 단순 / Dev mode 토글 파라미터 폼
  preview_dialog.py     - 원본/디노이즈 좌우 wipe 비교
  video_player.py       - cv2 기반 in-app 비디오 플레이어
  workers.py            - QThread (DenoiseWorker, PreviewWorker)
  presets.py            - 프리셋 사전
  i18n.py               - 한/영 토글
fdcore/                 - UI 무지각 코어
  color_spaces.py       - RGB↔YCbCr/HSV/LAB (노이즈 시각화에서만 사용)
  flow_warp.py          - warp / consistency mask
  flow_models.py        - MEMFOF / RAFT lazy 로더
  temporal_average.py   - 시간적 가중평균 (memfof / raft 분기)
  extract_noise.py      - 노이즈 분리 + 시각화 (옵션)
  pipeline.py           - settings dict 기반 통합 파이프라인 + passthrough 분기
  video_io.py           - FFmpegReader / FFmpegWriter / ImageSequenceWriter +
                          nclc/VUI 컬러 태그 패스스루 + TPDF dither
  progress.py           - 진행률 콜백 + 취소 지원
tests/                  - unittest 기반 색 라운드트립 / dither / 태그 보존 검증
third_party/memfof/     - (선택) MEMFOF 저장소 vendoring
weights/memfof/         - HuggingFace 모델 캐시
python/                 - 임베디드 Python (setup이 채움)
```

> 이전 버전에 있던 `selective_denoise.py`는 제거되었습니다. 시간적 가중평균
> 결과가 곧 디노이즈 출력입니다 (chroma/luma 분리 lerp 단계 없음).

## 테스트

```
python -m unittest tests.test_video_io
```

검증 항목:
- u8 → tensor → u8 라운드트립이 비트정확 (`np.rint` 보장)
- TPDF dither 평균 보존 + ±1 LSB 진폭
- 어두운 ramp에서 dither 적용 시 양자화 코드 다양성 증가 (밴딩 방지)
- ProRes / h264 인코더 명령에 소스 nclc / VUI 태그가 forward 됨

## 라이선스

본 앱: MIT (`LICENSE` 참고).
MEMFOF 모델 라이선스: [HuggingFace 페이지](https://huggingface.co/egorchistov/optical-flow-MEMFOF-Tartan-T-TSKH) 확인 필수.
원본 ComfyUI 노드: [AIMZ-GFX/ComfyUI-FlowDenoise](https://github.com/AIMZ-GFX/ComfyUI-FlowDenoise) MIT.

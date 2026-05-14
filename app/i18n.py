"""Lightweight i18n for CEC All-In-One (pure PyQt5).

Translation dict + tr() lookup, persisted via QSettings("CEC_AllInOne").
Scope is intentionally narrow — only the most visible UI strings.
Internal log lines and error dialogs stay in their original language.
"""

from PyQt5.QtCore import QSettings


KO = "ko"
EN = "en"
LANGS = (KO, EN)
LANG_LABELS = {KO: "한국어", EN: "English"}


_STRINGS = {
    # Header / navigation
    "tab_inpaint":    {"ko": "인페인트", "en": "Inpaint"},
    "tab_denoise":    {"ko": "디노이즈", "en": "Denoise"},
    "tab_upscale":    {"ko": "업스케일", "en": "Upscale"},
    "tab_batch":      {"ko": "배치", "en": "Batch"},
    "tab_about":      {"ko": "About",   "en": "About"},

    # Batch page
    "batch_output_folder":    {"ko": "출력 폴더",
                                "en": "Output folder"},
    "batch_output_placeholder": {
        "ko": "비워두면 output/{yymmdd}_result/ 에 저장",
        "en": "Empty → output/{yymmdd}_result/"},
    "batch_output_tooltip":   {
        "ko": ("최종 업스케일 결과물이 저장될 기본 폴더.<br>"
               "비워두면 프로젝트 폴더의 output/ 안에, 직접 지정하면 그 "
               "폴더 안에 <b>{yymmdd}_result</b> 서브폴더가 생기고 그 "
               "안에 클립들이 저장됩니다.<br>"
               "같은 폴더로 두 번째 실행하면 <b>_2</b>, <b>_3</b> "
               "꼬리표가 붙어 이전 결과물을 덮어쓰지 않습니다."),
        "en": ("Base folder for the final upscaled deliverables.<br>"
               "Empty → project's output/. Pick a custom path and a "
               "<b>{yymmdd}_result</b> subfolder is created inside it "
               "with the clips inside.<br>"
               "Re-running into the same folder appends <b>_2</b>, "
               "<b>_3</b> so earlier batches aren't overwritten.")},
    "batch_output_pick":      {"ko": "출력 폴더 선택",
                                "en": "Pick output folder"},
    "batch_output_bad":       {"ko": "출력 폴더를 만들 수 없습니다:",
                                "en": "Cannot create output folder:"},
    "batch_queue_title":      {"ko": "처리 큐 ({n}개)",
                                "en": "Queue ({n} item(s))"},
    "batch_queue_empty":      {"ko": "비디오 파일을 끌어다 놓거나 [추가]를 눌러 큐를 채우세요.",
                                "en": "Drag video files here or press [Add]."},
    "batch_btn_add":          {"ko": "추가", "en": "Add"},
    "batch_btn_remove":       {"ko": "선택 삭제", "en": "Remove"},
    "batch_btn_clear":        {"ko": "비우기", "en": "Clear"},
    "batch_btn_run":          {"ko": "전체 실행", "en": "Run all"},
    "batch_btn_stop":         {"ko": "중지", "en": "Stop"},
    "batch_section_inpaint":  {"ko": "1) 인페인트 (워터마크 제거)",
                                "en": "1) Inpaint (watermark removal)"},
    "batch_section_denoise":  {"ko": "2) 디노이즈 (시간적 광학흐름)",
                                "en": "2) Denoise (temporal optical flow)"},
    "batch_section_upscale":  {"ko": "3) 업스케일 (최종 출력)",
                                "en": "3) Upscale (final output)"},
    "batch_label_preset":     {"ko": "프리셋", "en": "Preset"},
    "batch_label_mask_dilation": {"ko": "마스크 확장 (Dreamina 자동)",
                                    "en": "Mask expansion (Dreamina auto)"},
    "batch_tooltip_mask_dilation": {
        "ko": ("Dreamina 자동 마스크의 가장자리를 N 픽셀만큼 외곽으로 확장.<br>"
               "프리셋이 워터마크 가장자리를 살짝 못 덮을 때 안전 마진을 "
               "추가합니다.<br>"
               "0: 그대로 사용 (기본), 4-8: 가벼운 마진, 15+: 강한 확장.<br>"
               "<b>커스텀 마스크(✱)에는 적용 안 됨</b> — 사용자가 그린 그대로 사용."),
        "en": ("Dilate the Dreamina auto-mask outward by N pixels.<br>"
               "Adds a safety margin when the preset doesn't quite "
               "cover the watermark edges.<br>"
               "0: as-is (default), 4-8: light margin, 15+: aggressive.<br>"
               "<b>Has no effect on custom masks (✱)</b> — they're used as drawn.")},
    "batch_queue_tip":        {
        "ko": "💡 클립 우클릭 → 커스텀 마스크 편집  ·  뷰어에서 A 키 → 알파 마스크 미리보기 토글",
        "en": "💡 Right-click a clip → edit custom mask  ·  Press A in the viewer → toggle alpha mask preview"},
    "batch_label_batch_frames": {"ko": "Max batch frames (0=전체)",
                                  "en": "Max batch frames (0=all)"},
    "batch_label_batch_size": {"ko": "Batch size", "en": "Batch size"},
    "batch_tooltip_batch_frames": {
        "ko": ("한 번에 GPU로 보내는 프레임 수의 상한. "
               "0이면 전체 영상을 한 번에 처리 (가장 빠름).<br>"
               "VRAM 부족할 때만 50/100 같은 값으로 분할하세요."),
        "en": ("Max frames sent to the GPU at once. 0 = whole clip in "
               "one pass (fastest).<br>"
               "Lower this (50/100) only when you hit VRAM limits.")},
    "batch_tooltip_batch_size": {
        "ko": ("디노이즈 모델이 한 번에 처리할 프레임 묶음 크기. "
               "기본 4. VRAM 부족하면 2 또는 1로 줄이세요."),
        "en": ("Frames per GPU batch for the denoise model. Default 4. "
               "Reduce to 2 or 1 if you hit VRAM limits.")},
    "batch_menu_edit_mask":   {"ko": "마스크 편집...",
                                "en": "Edit mask..."},
    "batch_menu_edit_mask_existing": {
        "ko": "마스크 편집... (커스텀 적용됨)",
        "en": "Edit mask... (custom applied)"},
    "batch_menu_remove_mask": {"ko": "커스텀 마스크 제거",
                                "en": "Remove custom mask"},
    "batch_preset_dreamina":  {"ko": "Dreamina (자동 가로/세로 감지)",
                                "en": "Dreamina (auto landscape/portrait)"},
    "batch_label_window":     {"ko": "Window size", "en": "Window size"},
    "batch_label_model":      {"ko": "모델", "en": "Model"},
    "batch_label_target":     {"ko": "출력 해상도", "en": "Target resolution"},
    "batch_target_native":    {"ko": "원본 × 2 (Native)", "en": "Source × 2 (Native)"},
    "batch_label_codec":      {"ko": "출력 코덱", "en": "Output codec"},
    "batch_codec_h264":       {"ko": "H.264 (MP4)", "en": "H.264 (MP4)"},
    "batch_codec_prores":     {"ko": "ProRes (MOV)", "en": "ProRes (MOV)"},
    "batch_label_profile":    {"ko": "ProRes 프로파일", "en": "ProRes profile"},
    "batch_status_idle":      {"ko": "대기 중 — 큐에 파일을 추가하고 [전체 실행]을 누르세요.",
                                "en": "Idle — add files and press Run all."},
    "batch_status_stage":     {"ko": "[{stage}] {fname} ({idx}/{total})",
                                "en": "[{stage}] {fname} ({idx}/{total})"},
    "batch_stage_inpaint":    {"ko": "인페인트", "en": "Inpaint"},
    "batch_stage_denoise":    {"ko": "디노이즈", "en": "Denoise"},
    "batch_stage_upscale":    {"ko": "업스케일", "en": "Upscale"},
    "batch_status_done":      {"ko": "완료 — {n}개 파일 → {out}",
                                "en": "Done — {n} file(s) → {out}"},
    "batch_status_cancelled": {"ko": "취소됨", "en": "Cancelled"},
    "batch_status_error":     {"ko": "오류: {msg}", "en": "Error: {msg}"},
    "batch_item_queued":      {"ko": "대기 중", "en": "Queued"},
    "batch_item_inpaint":     {"ko": "인페인트 중...", "en": "Inpainting..."},
    "batch_item_denoise":     {"ko": "디노이즈 중...", "en": "Denoising..."},
    "batch_item_upscale":     {"ko": "업스케일 중...", "en": "Upscaling..."},
    "batch_item_done_stage":  {"ko": "✓ {stage}", "en": "✓ {stage}"},
    "batch_item_done_all":    {"ko": "✓ 완료", "en": "✓ Done"},
    "batch_item_error":       {"ko": "✗ 오류", "en": "✗ Error"},
    "batch_pick_files":       {"ko": "비디오 파일 선택 (다중)",
                                "en": "Pick video files (multi)"},
    "batch_no_files":         {"ko": "큐가 비어있습니다. 파일을 먼저 추가하세요.",
                                "en": "Queue is empty. Add files first."},
    "batch_no_stages":        {"ko": "실행할 단계가 하나도 없습니다 — 인페인트/디노이즈/업스케일 중 최소 하나를 활성화하세요.",
                                "en": "No stages enabled — turn on at least one of Inpaint / Denoise / Upscale."},
    "batch_confirm_stop":     {"ko": "정말 중지하시겠습니까? 진행 중인 작업이 취소됩니다.",
                                "en": "Stop the batch? Current work will be cancelled."},

    # Upscale page
    "upscale_section_params":  {"ko": "Upscale 파라미터",
                                 "en": "Upscale parameters"},
    "upscale_label_model":     {"ko": "모델", "en": "Model"},
    "upscale_label_batch_frames": {
        "ko": "Max batch frames (0=전체)",
        "en": "Max batch frames (0=all)"},
    "upscale_model_openproteus": {
        "ko": "OpenProteus 2x (Compact)",
        "en": "OpenProteus 2x (Compact)"},
    "upscale_no_preview":      {"ko": "비디오를 불러오면 출력 해상도가 표시됩니다.",
                                 "en": "Load a video to see the output resolution."},
    # Plain-language preview. The user sees the input, the model's
    # intermediate 2x result, and then how the final size is reached —
    # no jargon (Lanczos / Area / "downscale" of a 720→1080 step) here.
    # Internals live in the About > FAQ > Upscale tab instead.
    "upscale_preview_native":   {
        "ko": "{src_w}×{src_h} → 모델이 2배로 키워 {mid_w}×{mid_h} → 그대로 출력",
        "en": "{src_w}×{src_h} → model enlarges 2× to {mid_w}×{mid_h} → final"},
    "upscale_preview_downscale": {
        "ko": "{src_w}×{src_h} → 모델이 2배로 키워 {mid_w}×{mid_h} → "
              "{dst_w}×{dst_h}로 줄여서 출력",
        "en": "{src_w}×{src_h} → model enlarges 2× to {mid_w}×{mid_h} → "
              "resized down to {dst_w}×{dst_h}"},
    "upscale_preview_upscale":   {
        "ko": "{src_w}×{src_h} → 모델이 2배로 키워 {mid_w}×{mid_h} → "
              "{dst_w}×{dst_h}로 더 키워서 출력",
        "en": "{src_w}×{src_h} → model enlarges 2× to {mid_w}×{mid_h} → "
              "enlarged further to {dst_w}×{dst_h}"},
    "upscale_label_target":    {"ko": "출력 해상도", "en": "Target resolution"},
    "upscale_target_native":   {"ko": "원본 × 2 (Native)",
                                 "en": "Source × 2 (Native)"},
    "upscale_target_info_fmt":  {
        "ko": "예상: {w} × {h}  ·  {ar_w}:{ar_h}",
        "en": "Expected: {w} × {h}  ·  {ar_w}:{ar_h}"},
    "upscale_target_info_empty": {
        "ko": "예상: 비디오를 불러오면 표시됩니다",
        "en": "Expected: load a video to see"},
    "upscale_tooltip_target": {
        "ko": ("최종 출력의 긴 변 길이.<br>"
               "<b>Native:</b> 모델 결과 그대로 (1080p → 2160p)<br>"
               "다른 값을 고르면 모델 결과를 그 해상도에 맞춰 출력합니다."),
        "en": ("Final output long-edge length.<br>"
               "<b>Native:</b> use the model's 2× output as-is "
               "(1080p → 2160p)<br>"
               "Any other choice resizes the model's result to that "
               "resolution.")},
    "upscale_placeholder_output": {
        "ko": "비워두면 자동: output/<원본>_upscaled.mp4",
        "en": "Empty → output/<input>_upscaled.mp4"},
    "upscale_tooltip_model": {
        "ko": ("업스케일 모델 선택.<br>"
               "<b>OpenProteus 2x</b>: Sirosky가 공개한 Compact 아키텍처. "
               "고품질·저노이즈 HD 소스의 4K 업스케일에 특화. "
               "오버샤프닝/아티팩트 없음."),
        "en": ("Upscale model.<br>"
               "<b>OpenProteus 2x</b>: Compact-architecture model by Sirosky. "
               "Tuned for clean HD → 4K upscaling without oversharpening "
               "or artifacts.")},
    "upscale_tooltip_batch_frames": {
        "ko": ("한 번에 읽을 최대 프레임 수 (0 = 전체 한번에). "
               "각 프레임은 순차적으로 모델에 들어갑니다.<br>"
               "긴 영상이나 메모리가 부족할 때 분할 처리하세요."),
        "en": ("Max frames read per batch (0 = all at once). Frames go "
               "through the model one by one within the batch.<br>"
               "Split long clips or when memory is tight.")},
    "upscale_tooltip_fp16": {
        "ko": ("CUDA에서 FP16(반정밀) 추론 사용.<br>"
               "<b>켜짐 (기본):</b> VRAM 절반, 1.5–2배 빠름<br>"
               "<b>꺼짐:</b> FP32, 정확도 미세하게 높음, 더 느림<br>"
               "CPU 모드에서는 무시됩니다."),
        "en": ("Use FP16 half-precision inference on CUDA.<br>"
               "<b>On (default):</b> half the VRAM, 1.5–2× faster<br>"
               "<b>Off:</b> FP32, marginally more accurate, slower<br>"
               "Ignored in CPU mode.")},

    # Inpaint page
    "inpaint_preset_btn":   {"ko": "프리셋 ▼", "en": "Preset ▼"},
    "inpaint_preset_dreamina": {"ko": "Dreamina (자동)", "en": "Dreamina (auto)"},
    "inpaint_no_mask":      {"ko": "마스크 없음 — 프리셋 선택",
                              "en": "No mask — pick a preset"},
    "inpaint_mask_set":     {"ko": "마스크 적용됨",
                              "en": "Mask applied"},
    "inpaint_overlay_toggle": {"ko": "마스크 오버레이 표시",
                                "en": "Show mask overlay"},
    "inpaint_overlay_hint":  {"ko": "뷰어에서 A 키로 미리보기",
                               "en": "Press A in the viewer to preview"},
    "inpaint_section_params": {"ko": "ProPainter 파라미터",
                                "en": "ProPainter parameters"},
    "inpaint_feather":      {"ko": "마스크 가장자리 부드럽게",
                              "en": "Feather mask edges"},
    "inpaint_fp16":         {"ko": "FP16 (CUDA만)", "en": "FP16 (CUDA only)"},
    "inpaint_label_neighbor": {"ko": "Neighbor length", "en": "Neighbor length"},
    "inpaint_label_ref_stride": {"ko": "Ref stride", "en": "Ref stride"},
    "inpaint_label_subvideo": {"ko": "Subvideo length", "en": "Subvideo length"},
    "inpaint_label_raft_iter": {"ko": "RAFT iterations", "en": "RAFT iterations"},
    "inpaint_label_dilation": {"ko": "Mask dilation", "en": "Mask dilation"},
    "inpaint_label_batch_frames": {"ko": "Max batch frames (0=전체)",
                                    "en": "Max batch frames (0=all)"},
    "inpaint_placeholder_output": {
        "ko": "비워두면 자동: output/<원본>_inpainted.mp4",
        "en": "Empty → output/<input>_inpainted.mp4"},
    "inpaint_btn_edit_mask": {"ko": "마스크 편집", "en": "Edit mask"},

    # Mask editor dialog
    "mask_editor_title": {"ko": "마스크 편집", "en": "Edit mask"},
    "mask_tool_brush":   {"ko": "브러시",  "en": "Brush"},
    "mask_tool_rect":    {"ko": "사각형",  "en": "Rectangle"},
    "mask_tool_ellipse": {"ko": "원형",    "en": "Ellipse"},
    "mask_tool_eraser":  {"ko": "지우개",  "en": "Eraser"},
    "mask_brush_size":   {"ko": "크기",    "en": "Size"},
    "mask_undo":         {"ko": "실행 취소", "en": "Undo"},
    "mask_clear":        {"ko": "모두 지우기", "en": "Clear all"},
    "mask_apply":        {"ko": "적용",    "en": "Apply"},
    "mask_frame_label":  {"ko": "프레임:",  "en": "Frame:"},
    "inpaint_label_model":      {"ko": "모델", "en": "Model"},
    "inpaint_model_propainter": {"ko": "ProPainter",
                                  "en": "ProPainter"},

    # Inpaint — parameter tooltips (shown on hover)
    "inpaint_tooltip_model": {
        "ko": ("비디오 인페인팅 모델 선택.<br>"
               "<b>ProPainter</b>: 광학흐름 + 변형기 기반. 현재 유일한 모델."),
        "en": ("Video inpainting model.<br>"
               "<b>ProPainter</b>: optical-flow + transformer based. "
               "Currently the only available option.")},
    "inpaint_tooltip_batch_frames": {
        "ko": ("한 번에 처리할 최대 프레임 수 (0 = 전체 한번에).<br>"
               "긴 영상이나 VRAM이 부족할 때 분할 처리하세요. "
               "<b>720p 16GB GPU 기준 ~200, 1080p ~80, 4K ~20</b>."),
        "en": ("Max frames processed per batch (0 = all at once).<br>"
               "Split long clips or when VRAM is tight. "
               "<b>~200 for 720p 16GB / ~80 for 1080p / ~20 for 4K</b>.")},
    "inpaint_tooltip_fp16": {
        "ko": ("CUDA에서 FP16(반정밀) 추론 사용.<br>"
               "<b>켜짐 (기본):</b> VRAM 절반, 1.5–2배 빠름<br>"
               "<b>꺼짐:</b> FP32, 정확도 미세하게 높음, 더 느림<br>"
               "CPU 모드에서는 무시됩니다."),
        "en": ("Use FP16 half-precision inference on CUDA.<br>"
               "<b>On (default):</b> half the VRAM, 1.5–2× faster<br>"
               "<b>Off:</b> FP32, marginally more accurate, slower<br>"
               "Ignored in CPU mode.")},
    "inpaint_tooltip_feather": {
        "ko": ("마스크 가장자리를 약간 부드럽게 처리 (1px 팽창 + 가우시안).<br>"
               "워터마크 경계 잔상이 보이면 켜두는 게 안전합니다."),
        "en": ("Soften mask edges slightly (1-px dilate + Gaussian blur).<br>"
               "Reduces edge artifacts around the inpaint boundary.")},
    "inpaint_tooltip_neighbor": {
        "ko": ("현재 프레임 양옆에서 참조할 이웃 프레임 수 (기본 10).<br>"
               "<b>높이면:</b> 시간적 일관성 ↑, VRAM ↑<br>"
               "<b>낮추면:</b> 빠르지만 깜빡임 위험"),
        "en": ("Neighbor frames consulted around the current frame "
               "(default 10).<br>"
               "<b>Higher:</b> better temporal consistency, more VRAM<br>"
               "<b>Lower:</b> faster but risks flicker")},
    "inpaint_tooltip_ref_stride": {
        "ko": ("장거리 참조 프레임 간격 (기본 10).<br>"
               "<b>작으면:</b> 더 많은 reference, VRAM ↑, 일관성 ↑<br>"
               "<b>크면:</b> 빠르지만 변동 큰 장면에서 끊김"),
        "en": ("Stride between long-range reference frames "
               "(default 10).<br>"
               "<b>Smaller:</b> more references, more VRAM, better "
               "consistency<br>"
               "<b>Larger:</b> faster but jittery on dynamic shots")},
    "inpaint_tooltip_subvideo": {
        "ko": ("한 번에 처리할 서브비디오 길이 (기본 80).<br>"
               "<b>높이면:</b> 시간적 일관성 ↑, VRAM ↑<br>"
               "<b>낮추면:</b> 메모리 절약, 경계에서 미세 변화 가능"),
        "en": ("Sub-video chunk length (default 80).<br>"
               "<b>Higher:</b> better consistency, more VRAM<br>"
               "<b>Lower:</b> saves memory, slight seams at chunk borders")},
    "inpaint_tooltip_raft_iter": {
        "ko": ("RAFT 광학흐름 정밀화 반복 횟수 (기본 20).<br>"
               "<b>높이면:</b> 모션 정확도 ↑, 처리 시간 ↑<br>"
               "<b>낮추면:</b> 빠르지만 흐림/잔상 위험"),
        "en": ("RAFT optical flow refinement iterations (default 20).<br>"
               "<b>Higher:</b> better motion accuracy, slower<br>"
               "<b>Lower:</b> faster but ghosting / blur risk")},
    "inpaint_tooltip_dilation": {
        "ko": ("프리셋 마스크 가장자리를 N 픽셀만큼 외곽으로 확장 (기본 4).<br>"
               "워터마크 가장자리를 살짝 더 가려서 경계 잔상을 줄입니다. "
               "너무 키우면 인페인트 영역이 커져 품질 ↓.<br>"
               "브러시로 그린 커스텀 마스크에는 영향 없음 — "
               "그린 그대로 사용됩니다."),
        "en": ("Expand the preset mask outward by N pixels (default 4).<br>"
               "Helps cover watermark edges to reduce boundary ghosting. "
               "Too large = bigger inpaint area = quality ↓.<br>"
               "No effect on brush-drawn custom masks — those are used "
               "exactly as drawn.")},
    "tooltip_lang":   {"ko": "언어 변경", "en": "Language"},
    "restart_to_apply": {"ko": "언어 변경: 재시작 후 적용됩니다.",
                         "en": "Language: restart to apply."},

    # Denoise page — left column
    "info_no_video": {
        "ko": "입력 비디오 없음 — 파일을 끌어다 놓거나 [파일 선택]",
        "en": "No video loaded — drop a file or click [Choose file]"},
    "btn_pick_file": {"ko": "파일 선택...", "en": "Choose file..."},
    "label_output":  {"ko": "출력 경로",   "en": "Output path"},
    "placeholder_output": {
        "ko": "비워두면 자동: output/<원본>_denoised.<확장자>",
        "en": "Empty → output/<input>_denoised.<ext>"},
    "tooltip_output_path": {
        "ko": "출력 파일 전체 경로. 비워두면 프로젝트의 output/ 폴더에 저장됩니다.",
        "en": "Full output file path. If empty, saved to the project's output/ folder."},
    "btn_browse":    {"ko": "찾아보기...", "en": "Browse..."},
    "btn_preview":   {"ko": "미리보기 (앞 80프레임)",
                      "en": "Preview (first 80 frames)"},
    "btn_show_preview": {"ko": "이전 미리보기 다시 보기",
                          "en": "Show last preview"},
    "btn_run":       {"ko": "전체 처리", "en": "Process"},
    "btn_cancel":    {"ko": "취소",     "en": "Cancel"},
    "label_log":     {"ko": "로그",     "en": "Log"},
    "btn_open_log":  {"ko": "로그 파일 열기", "en": "Open log file"},

    # Right column
    "section_temporal":      {"ko": "Temporal Flow Average",
                              "en": "Temporal Flow Average"},
    "section_noise_preview": {"ko": "노이즈 미리보기 (선택)",
                              "en": "Noise preview (optional)"},
    "checkbox_enable":       {"ko": "활성화", "en": "Enable"},
    "noise_box_title": {
        "ko": "노이즈 미리보기  ·  우클릭 → 다른 이름으로 저장",
        "en": "Noise preview  ·  Right-click → Save As"},

    # About page
    "about_tagline": {"ko": "AI 영상 후처리 통합 도구 — 인페인트 · 디노이즈 · 업스케일 · 배치",
                      "en": "All-in-one AI video post-processing — inpaint, denoise, upscale, batch"},
    "about_contact": {"ko": "Contact", "en": "Contact"},
    "about_subtab_inpaint":  {"ko": "인페인트", "en": "Inpaint"},
    "about_subtab_denoise":  {"ko": "디노이즈", "en": "Denoise"},
    "about_subtab_upscale":  {"ko": "업스케일", "en": "Upscale"},
    "about_subtab_batch":    {"ko": "배치", "en": "Batch"},

    # Output format card — shared by Inpaint + Upscale (Denoise has its
    # own richer card in parameter_panel.py).
    "io_section_format":   {"ko": "출력 포맷", "en": "Output format"},
    "io_label_codec":      {"ko": "코덱", "en": "Codec"},
    "io_format_auto":      {"ko": "Auto (소스 따라감)",
                             "en": "Auto (follow source)"},
    "io_format_h264":      {"ko": "H.264 MP4", "en": "H.264 MP4"},
    "io_format_prores":    {"ko": "ProRes MOV", "en": "ProRes MOV"},
    "io_label_prores_profile": {"ko": "ProRes 프로파일",
                                 "en": "ProRes profile"},
    "io_tooltip_prores_profile": {
        "ko": ("ProRes 프로파일 선택. 위로 갈수록 비트레이트 ↑, 파일 크기 ↑.<br>"
               "<b>Proxy:</b> 최저 비트레이트, 미리보기용<br>"
               "<b>LT:</b> 작은 파일, 일반 후작업<br>"
               "<b>422:</b> 표준 후작업<br>"
               "<b>422 HQ:</b> 고품질 후작업<br>"
               "<b>4444 (기본):</b> 알파/색 보존, 합성용<br>"
               "<b>4444 XQ:</b> 최고 품질"),
        "en": ("ProRes profile. Bitrate / file size grows top to bottom.<br>"
               "<b>Proxy:</b> lowest bitrate, preview only<br>"
               "<b>LT:</b> small files, general post<br>"
               "<b>422:</b> standard post<br>"
               "<b>422 HQ:</b> high-quality post<br>"
               "<b>4444 (default):</b> alpha + color preserved, compositing<br>"
               "<b>4444 XQ:</b> highest quality")},
    "io_tooltip_format": {
        "ko": ("출력 코덱 선택.<br>"
               "<b>Auto:</b> 소스가 ProRes면 ProRes, 아니면 H.264로 출력. "
               "다단계 워크플로우(인페인트 → 디노이즈 → 업스케일)에서 매번 "
               "h264 재압축으로 인한 화질 손실을 막으려면 첫 단계만 ProRes로 "
               "지정하면 이후 Auto가 ProRes를 유지합니다.<br>"
               "<b>H.264 MP4:</b> 작은 파일, 일반 배포용. 재압축마다 손실 발생.<br>"
               "<b>ProRes MOV:</b> 색감 보존, 후작업/체이닝용. 파일 크기 큼."),
        "en": ("Output codec.<br>"
               "<b>Auto:</b> ProRes-tagged sources stay on ProRes, "
               "everything else gets H.264. In multi-step workflows "
               "(Inpaint → Denoise → Upscale), set the first stage to "
               "ProRes and subsequent Auto stages will preserve it — "
               "avoids quality loss from repeated h264 re-encoding.<br>"
               "<b>H.264 MP4:</b> small files, general delivery. Lossy "
               "on each re-encode.<br>"
               "<b>ProRes MOV:</b> color-accurate, post-production / "
               "chaining. Larger files.")},

    # Chatbot
    "chatbot_tooltip":     {"ko": "도우미 챗봇 열기",
                             "en": "Open assistant"},
    "chatbot_title":       {"ko": "도우미 챗봇", "en": "Assistant"},
    "chatbot_placeholder": {"ko": "메시지를 입력하세요... (Enter 전송 / Shift+Enter 줄바꿈)",
                             "en": "Type a message... (Enter to send, Shift+Enter for newline)"},
    "chatbot_send":        {"ko": "전송", "en": "Send"},
    "chatbot_greeting":    {"ko": "안녕하세요! CEC All-In-One 사용법이나 일반 질문 "
                                  "무엇이든 물어보세요.",
                             "en": "Hi! Ask me anything about CEC All-In-One "
                                   "or general questions."},
    "chatbot_error":       {"ko": "오류: {err}", "en": "Error: {err}"},
    "chatbot_thinking":    {"ko": "생각 중...", "en": "Thinking..."},

    # Cancellation summary (appears in eta label after Cancel)
    "cancelled_summary": {
        "ko": "처리됨 {done}/{total} · 소요 {elapsed:.1f}s · 취소됨",
        "en": "Processed {done}/{total} · {elapsed:.1f}s elapsed · cancelled"},

    # Completion summary (replaces "마감 처리 중..." once _on_finished
    # fires, paired with a green progress-bar palette so the user can
    # see at a glance that the run actually finished).
    "finished_summary": {
        "ko": "모든 처리 완료 · {done}/{total} · 소요 {elapsed:.1f}s",
        "en": "All done · {done}/{total} · {elapsed:.1f}s elapsed"},

    # Developer mode toggle (Temporal Flow Average card)
    "label_dev_mode":   {"ko": "개발자 모드", "en": "Dev mode"},
    "tooltip_dev_mode": {
        "ko": "켜면 고정값으로 잠겨 있던 모든 파라미터가 노출됩니다.",
        "en": "Toggle to expose every locked-default parameter."},

    # Advanced parameter tooltips (only seen when dev mode is on)
    "tooltip_weight_decay": {
        "ko": ("이웃 프레임의 시간 가중치 감쇠율. 1.0 = 모든 거리 균등.<br>"
               "<b>높이면 (0.9~1.0):</b> 노이즈 제거 강함, 잔상 위험 ↑<br>"
               "<b>낮추면 (0.5~):</b> 디테일 보존, 노이즈 제거 약함"),
        "en": ("Time weight decay across neighbors. 1.0 = uniform.<br>"
               "<b>Higher (0.9-1.0):</b> stronger denoise, more ghosting risk<br>"
               "<b>Lower (0.5-):</b> detail preservation, weaker denoise")},
    "tooltip_flow_iterations": {
        "ko": ("MEMFOF/RAFT 흐름 정밀화 반복 횟수.<br>"
               "<b>높이면 (8~16):</b> 모션 정확도 ↑, 처리 시간 비례 증가<br>"
               "<b>낮추면 (1~3):</b> 빠르지만 잔상/정렬 오차"),
        "en": ("Optical flow refinement iterations.<br>"
               "<b>Higher (8-16):</b> better motion accuracy, proportionally slower<br>"
               "<b>Lower (1-3):</b> faster but more ghosting / misalignment")},
    "tooltip_color_threshold": {
        "ko": ("정렬된 이웃 프레임을 평균에 합칠 때 색 차이 임계치. "
               "이보다 크면 outlier로 거절.<br>"
               "<b>낮추면 (0.01~):</b> 엄격, 잔상 적음, 디노이즈 약함<br>"
               "<b>높이면 (0.1+):</b> 관대, 디노이즈 강함, 잔상 위험"),
        "en": ("Color similarity threshold for accepting warped neighbors. "
               "Larger differences are rejected as outliers.<br>"
               "<b>Lower (0.01-):</b> strict, less ghosting, weaker denoise<br>"
               "<b>Higher (0.1+):</b> permissive, stronger denoise, ghosting risk")},
    "tooltip_scene_threshold": {
        "ko": ("씬 컷 검출 임계치 (프레임 간 MSE). 이보다 크면 컷으로 간주, "
               "평균 중단.<br>"
               "<b>낮추면 (0.01~):</b> 민감, 잘못된 컷 검출 → 디노이즈 약화<br>"
               "<b>높이면 (0.1+):</b> 둔감, 실제 컷 무시 → 컷 경계 잔상"),
        "en": ("Scene-cut threshold (inter-frame MSE). Above → treated as cut, "
               "averaging stops.<br>"
               "<b>Lower (0.01-):</b> sensitive, false cuts weaken denoise<br>"
               "<b>Higher (0.1+):</b> insensitive, missed cuts cause boundary ghosting")},
    "tooltip_precision": {
        "ko": ("추론 정밀도.<br>"
               "<b>bf16 (기본):</b> VRAM 절반, 1.5–2배 빠름, 시각 차이 거의 없음<br>"
               "<b>fp32:</b> 엄격한 결정론적 결과, 약 2배 느림"),
        "en": ("Inference precision.<br>"
               "<b>bf16 (default):</b> half the VRAM, 1.5–2× faster, "
               "visually indistinguishable<br>"
               "<b>fp32:</b> strictly deterministic, ~2× slower")},
    "tooltip_flow_scale": {
        "ko": ("Flow 추정 시 입력 다운샘플 비율. warp는 항상 풀 해상도.<br>"
               "<b>1.0:</b> 풀 해상도 flow, 가장 정확, 가장 느림<br>"
               "<b>0.5 (기본):</b> 약 3배 빠름, 품질 거의 동등<br>"
               "<b>0.3:</b> 매우 빠름, 정밀 모션엔 부정확"),
        "en": ("Input downsample ratio for flow estimation; warp stays at full res.<br>"
               "<b>1.0:</b> full-res flow, most accurate, slowest<br>"
               "<b>0.5 (default):</b> ~3× faster, near-identical quality<br>"
               "<b>0.3:</b> very fast, inaccurate on fine motion")},
    "label_passthrough": {"ko": "Passthrough", "en": "Passthrough"},
    "tooltip_passthrough": {
        "ko": ("디노이즈를 완전히 끄고 read → write만 합니다 (디버깅용).<br>"
               "<b>켜짐:</b> 입력 BGR을 그대로 인코더에 넘김. float 변환 / "
               "TPDF dither / flow / temporal average 모두 건너뜀. "
               "압축 외 색 변화가 있으면 그건 ffmpeg 인코드/디코드 자체의 문제.<br>"
               "<b>꺼짐 (기본):</b> 정상 디노이즈 동작."),
        "en": ("Skip denoise entirely — pure read → write (debug only).<br>"
               "<b>On:</b> decoded BGR bytes go straight to the encoder. "
               "No float conversion, no TPDF dither, no flow / temporal "
               "averaging. Any color change other than compression is "
               "purely from ffmpeg encode/decode.<br>"
               "<b>Off (default):</b> normal denoise.")},
    "tooltip_clamp_output": {
        "ko": ("출력 픽셀 값을 [0,1] 범위로 클램프할지 여부.<br>"
               "<b>켜짐 (기본):</b> 8-bit 인코딩에 안전한 표준 동작<br>"
               "<b>꺼짐:</b> float 그대로 통과 (HDR / 분석용). "
               "비디오로 저장할 땐 어차피 uint8로 클립되므로 시각 차이 없음."),
        "en": ("Whether to clamp output pixel values to [0,1].<br>"
               "<b>On (default):</b> safe for 8-bit encoding<br>"
               "<b>Off:</b> pass float through unchanged (HDR / inspection). "
               "Final video write still clips to uint8 anyway.")},

    # Output format card
    "section_output":         {"ko": "출력 세팅", "en": "Output Settings"},
    "label_prores_profile":   {"ko": "ProRes 프로파일", "en": "ProRes Profile"},
    # Shown in the Output Settings card — this is the effective fps
    # the pipeline will use end-to-end. With Auto checked it mirrors
    # the auto-detected source fps; with Auto unchecked the user's
    # override value is shown instead. Distinct from the always-on
    # "<file>.mp4 1470×630 @ 24.04fps · ..." info row, which is the
    # raw source metadata.
    "label_input_fps":        {"ko": "FPS", "en": "FPS"},
    "label_fps_override":     {"ko": "FPS 강제", "en": "FPS Override"},
    "label_fps_auto":         {"ko": "Auto", "en": "Auto"},
    "fps_unknown":            {"ko": "(영상 없음)", "en": "(no video)"},
    "tooltip_fps_override": {
        "ko": ("입력 fps를 강제로 지정합니다. <b>Auto</b>는 자동 인식값 사용. "
               "주로 AI 영상(Seedance 등) 메타데이터가 24.04 같은 어중간한 "
               "값으로 나올 때 24.000으로 강제하는 용도. 0(=Auto)로 두면 "
               "자동 인식값이 reader/writer 양쪽에 적용됩니다."),
        "en": ("Force a specific input fps. <b>Auto</b> uses the auto-detected "
               "value. Typical use: AI-generated mp4s (Seedance etc.) report "
               "24.04 in metadata but are really 24.000 — set to 24.000 to "
               "force. 0 (=Auto) uses the detected value for both reader "
               "and writer.")},
    "hint_preview_format_match": {
        "ko": "미리보기도 같은 포맷으로 렌더됩니다",
        "en": "Preview also renders in this format"},
    "tooltip_output_format": {
        "ko": ("본 작업의 출력 코덱.<br>"
               "<b>H.264 MP4</b>: 작은 파일, 빠른 디코드, 일반 배포용<br>"
               "<b>ProRes MOV</b>: 색감 보존, 후작업용 (파일 크기 큼)<br>"
               "<b>PNG Sequence</b>: 폴더에 PNG 프레임 + meta.json. "
               "코덱 손실 0, 비디오 컨테이너 무관"),
        "en": ("Codec for the full run.<br>"
               "<b>H.264 MP4</b>: small files, fast decode, general delivery<br>"
               "<b>ProRes MOV</b>: color-accurate, post-production (large files)<br>"
               "<b>PNG Sequence</b>: folder of PNG frames + meta.json, "
               "zero codec loss, no container concerns")},
    "tooltip_prores_profile": {
        "ko": "Proxy/LT/422/HQ는 yuv422p10le, 4444 계열은 yuv444p10le로 인코딩",
        "en": "Proxy/LT/422/HQ encode as yuv422p10le; 4444 variants as yuv444p10le"},

    # Video player — placeholder + transport tooltips + context menu
    "video_placeholder": {
        "ko": "비디오를 불러오려면 클릭하거나 파일을 끌어다 놓으세요",
        "en": "Click or drop a video file to load"},
    "tooltip_first": {"ko": "처음으로", "en": "First frame"},
    "tooltip_prev":  {"ko": "이전 프레임", "en": "Previous frame"},
    "tooltip_play":  {"ko": "재생 / 일시정지 (Space)",
                      "en": "Play / Pause (Space)"},
    "tooltip_next":  {"ko": "다음 프레임", "en": "Next frame"},
    "tooltip_last":  {"ko": "끝으로", "en": "Last frame"},
    "tooltip_loop":  {"ko": "반복 재생", "en": "Loop playback"},
    "tooltip_mute":  {"ko": "음소거", "en": "Mute"},
    "tooltip_volume": {"ko": "음량", "en": "Volume"},
    "menu_save_as":  {"ko": "다른 이름으로 저장...", "en": "Save As..."},
    "noise_save_label": {"ko": "노이즈 미리보기", "en": "Noise preview"},

    # Parameter tooltips
    "tooltip_window_size": {
        "ko": ("현재 프레임 좌우로 평균낼 이웃 프레임 수.<br>"
               "<b>기본 3 권장</b> — 속도/품질 균형이 좋습니다.<br>"
               "노이즈가 여전히 보이면 <b>5~7</b>로 높이세요.<br>"
               "<i>클수록 처리 시간이 비례 증가하고, "
               "빠르게 움직이는 장면에선 잔상 위험이 커집니다.</i>"),
        "en": ("Number of neighbor frames to average on each side of the "
               "current frame.<br>"
               "<b>Default 3 recommended</b> — balanced speed and quality.<br>"
               "Bump to <b>5–7</b> if noise is still visible.<br>"
               "<i>Larger window = proportionally slower, and risks ghosting "
               "on fast-motion shots.</i>")},
    "tooltip_flow_model": {
        "ko": "memfof = SOTA 2025 (가장 정확). raft_small = 빠름",
        "en": "memfof = SOTA 2025 (most accurate). raft_small = faster"},
    "tooltip_batch_size": {
        "ko": (
            "<b>MEMFOF 배치 크기 가이드</b> "
            "(bf16 + flow_scale 0.5 기준)"
            "<table cellpadding='4'>"
            "<tr>"
            "<td><b>해상도</b></td>"
            "<td><b>RTX 5080<br>16GB</b></td>"
            "<td><b>RTX 5090<br>32GB</b></td>"
            "</tr>"
            "<tr>"
            "<td>720p</td>"
            "<td align='center'>32</td>"
            "<td align='center'>64</td>"
            "</tr>"
            "<tr>"
            "<td>1080p</td>"
            "<td align='center'>16</td>"
            "<td align='center'>32</td>"
            "</tr>"
            "<tr>"
            "<td>4K (UHD)</td>"
            "<td align='center'>4</td>"
            "<td align='center'>8</td>"
            "</tr>"
            "</table>"
            "<i>표는 안전 권장값. 마지노선은 약 1.5–2배까지 가능하지만 "
            "OOM 시 해당 작업만 실패합니다 (앱은 유지). 클수록 빠릅니다.</i>"),
        "en": (
            "<b>MEMFOF batch size guide</b> "
            "(bf16 + flow_scale 0.5)"
            "<table cellpadding='4'>"
            "<tr>"
            "<td><b>Resolution</b></td>"
            "<td><b>RTX 5080<br>16GB</b></td>"
            "<td><b>RTX 5090<br>32GB</b></td>"
            "</tr>"
            "<tr>"
            "<td>720p</td>"
            "<td align='center'>32</td>"
            "<td align='center'>64</td>"
            "</tr>"
            "<tr>"
            "<td>1080p</td>"
            "<td align='center'>16</td>"
            "<td align='center'>32</td>"
            "</tr>"
            "<tr>"
            "<td>4K (UHD)</td>"
            "<td align='center'>4</td>"
            "<td align='center'>8</td>"
            "</tr>"
            "</table>"
            "<i>Recommended safe values. Headroom up to ~1.5–2× before OOM "
            "— OOM aborts only the current job, app stays alive. "
            "Higher = faster.</i>")},
    "tooltip_noise_enable": {
        "ko": "켜면 처리 후 노이즈 시각화 영상을 우측 미니뷰어에 표시",
        "en": "Show noise visualization in the mini-viewer after processing"},
    "tooltip_noise_preview": {
        "ko": "heatmap (turbo 색상) / signed (빨강=+, 파랑=-) / gray (0.5 중심)",
        "en": "heatmap (turbo color) / signed (red=+, blue=-) / gray (centered at 0.5)"},
}


def _settings() -> QSettings:
    return QSettings("AIMZ", "CEC_AllInOne")


def get_lang() -> str:
    val = _settings().value("ui/lang", KO, type=str)
    return val if val in LANGS else KO


def set_lang(lang: str) -> None:
    if lang in LANGS:
        _settings().setValue("ui/lang", lang)


def tr(key: str) -> str:
    lang = get_lang()
    entry = _STRINGS.get(key, {})
    return entry.get(lang) or entry.get(KO) or key

"""Shared streaming scene-boundary detection for temporal video tools."""

from __future__ import annotations

from typing import Iterable, Optional

import cv2
import numpy as np


SCENE_DETECTION_MSE = "mse"
SCENE_DETECTION_PYSCENEDETECT = "pyscenedetect"
DEFAULT_SCENE_DETECTION_MODE = SCENE_DETECTION_MSE
DEFAULT_SCENE_MSE_THRESHOLD = 0.030


def normalize_scene_detection_mode(value) -> str:
    value = str(value or DEFAULT_SCENE_DETECTION_MODE).strip().lower()
    if value in {"pyscenedetect", "adaptive", "pyscenedetect_adaptive"}:
        return SCENE_DETECTION_PYSCENEDETECT
    return SCENE_DETECTION_MSE


def _is_torch_tensor(value) -> bool:
    return value.__class__.__module__.startswith("torch")


def _legacy_mse(left, right, size: Optional[tuple[int, int]]) -> float:
    if _is_torch_tensor(left) and size is None:
        import torch
        return float(torch.mean((right - left) ** 2).item())

    left_np = left.detach().cpu().numpy() if _is_torch_tensor(left) else np.asarray(left)
    right_np = right.detach().cpu().numpy() if _is_torch_tensor(right) else np.asarray(right)
    if size is not None:
        left_np = cv2.resize(left_np, size, interpolation=cv2.INTER_AREA)
        right_np = cv2.resize(right_np, size, interpolation=cv2.INTER_AREA)
    scale = 65535.0 if left_np.dtype == np.uint16 else 255.0
    difference = (left_np.astype(np.float32) - right_np.astype(np.float32)) / scale
    return float(np.mean(difference * difference))


def _to_small_bgr8(frame, target_width: int = 256) -> np.ndarray:
    if _is_torch_tensor(frame):
        import torch.nn.functional as F
        tensor = frame.detach()
        if tensor.ndim == 4:
            tensor = tensor[0]
        height, width = int(tensor.shape[-2]), int(tensor.shape[-1])
        if width > target_width:
            target_height = max(1, round(height * target_width / width))
            tensor = F.interpolate(
                tensor.unsqueeze(0), size=(target_height, target_width),
                mode="area").squeeze(0)
        rgb = (tensor.permute(1, 2, 0).float().clamp(0, 1).cpu().numpy()
               * 255.0 + 0.5).astype(np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    image = np.asarray(frame)
    height, width = image.shape[:2]
    if width > target_width:
        target_height = max(1, round(height * target_width / width))
        image = cv2.resize(image, (target_width, target_height),
                           interpolation=cv2.INTER_AREA)
    if image.dtype == np.uint16:
        image = (image / 257.0 + 0.5).astype(np.uint8)
    elif image.dtype != np.uint8:
        scale = 255.0 if float(np.max(image, initial=0.0)) <= 1.0 else 1.0
        image = np.clip(image * scale, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def _cut_number(value) -> int:
    if hasattr(value, "frame_num"):
        return int(value.frame_num)
    getter = getattr(value, "get_frames", None)
    return int(getter() if callable(getter) else value)


class SceneCutStream:
    """Incremental MSE or PySceneDetect Adaptive boundary detector.

    ``feed`` returns ``(confirmed_through, new_boundaries)``. Adaptive mode
    confirms a pair two frames later, allowing callers to keep a tiny bounded
    frame queue instead of decoding the video a second time.
    """

    def __init__(self, mode=DEFAULT_SCENE_DETECTION_MODE,
                 mse_threshold=DEFAULT_SCENE_MSE_THRESHOLD,
                 mse_size: Optional[tuple[int, int]] = None,
                 adaptive_detector=None):
        self.mode = normalize_scene_detection_mode(mode)
        self.mse_threshold = float(mse_threshold)
        self.mse_size = mse_size
        self._previous = None
        self._last_frame = None
        self._last_index = -1
        self._flushed = False

        self._timecode = int
        self._adaptive = adaptive_detector
        if self.mode == SCENE_DETECTION_PYSCENEDETECT and self._adaptive is None:
            try:
                from scenedetect import FrameTimecode
                from scenedetect.detectors import AdaptiveDetector
            except ImportError as exc:
                raise RuntimeError(
                    "PySceneDetect가 설치되지 않았습니다. setup.bat을 다시 실행하세요.") from exc
            self._adaptive = AdaptiveDetector(
                adaptive_threshold=3.0, min_scene_len=1,
                window_width=2, min_content_val=15.0)
            self._timecode = lambda number: FrameTimecode(number, fps=1000.0)
        self.event_buffer_length = (int(self._adaptive.event_buffer_length)
                                    if self._adaptive is not None else 0)

    def feed(self, frame_index: int, frame) -> tuple[int, set[int]]:
        frame_index = int(frame_index)
        if frame_index <= self._last_index:
            raise ValueError("장면 감지 프레임 번호는 순차적으로 증가해야 합니다.")
        self._last_index = frame_index
        self._last_frame = frame

        if self.mode == SCENE_DETECTION_MSE:
            cuts = set()
            if (self._previous is not None and self.mse_threshold > 0
                    and _legacy_mse(self._previous, frame, self.mse_size)
                    > self.mse_threshold):
                cuts.add(frame_index)
            self._previous = frame
            return frame_index, cuts

        cuts = {
            _cut_number(cut) for cut in self._adaptive.process_frame(
                self._timecode(frame_index), _to_small_bgr8(frame))
        }
        return max(-1, frame_index - self.event_buffer_length), cuts

    def flush(self) -> tuple[int, set[int]]:
        if self._flushed or self._last_frame is None:
            return self._last_index, set()
        self._flushed = True
        cuts = set()
        if self.mode == SCENE_DETECTION_PYSCENEDETECT:
            last_real_index = self._last_index
            for offset in range(1, self.event_buffer_length + 1):
                cuts.update(
                    _cut_number(cut) for cut in self._adaptive.process_frame(
                        self._timecode(last_real_index + offset),
                        _to_small_bgr8(self._last_frame)))
            cuts = {cut for cut in cuts if 0 < cut <= last_real_index}
        return self._last_index, cuts


def detect_scene_boundaries(frames: Iterable, mode=DEFAULT_SCENE_DETECTION_MODE,
                            mse_threshold=DEFAULT_SCENE_MSE_THRESHOLD,
                            mse_size: Optional[tuple[int, int]] = None) -> set[int]:
    detector = SceneCutStream(mode, mse_threshold=mse_threshold,
                              mse_size=mse_size)
    boundaries = set()
    for index, frame in enumerate(frames):
        _confirmed, cuts = detector.feed(index, frame)
        boundaries.update(cuts)
    _confirmed, cuts = detector.flush()
    boundaries.update(cuts)
    return boundaries

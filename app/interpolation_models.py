"""Commercially safe frame-interpolation model registry."""

import os
from collections import OrderedDict


_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_WEIGHTS = os.path.join(_ROOT, "weights", "interpolation")

INTERPOLATION_MODELS = OrderedDict([
    ("rife_426", {
        "rank": 1, "name": "RIFE 4.26", "license": "MIT",
        "commercial_use": True, "filename": "rife426.pth",
        "description": "빠르고 안정적인 기본 선택",
    }),
    ("atm_vfi_lite", {
        "rank": 2, "name": "ATM-VFI Lite", "license": "Apache-2.0",
        "commercial_use": True, "filename": "atm-vfi-lite.pt",
        "description": "큰 움직임에 강한 경량 ATM-VFI",
    }),
    ("sgm_vfi", {
        "rank": 3, "name": "SGM-VFI", "license": "Apache-2.0",
        "commercial_use": True, "filename": "ours-1-2-points.pkl",
        "description": "복잡한 움직임 품질 우선",
    }),
    ("ema_vfi", {
        "rank": 4, "name": "EMA-VFI (ours_t)", "license": "Apache-2.0",
        "commercial_use": True, "filename": "ours_t.pkl",
        "description": "임의 시점 보간을 지원하는 품질 옵션",
    }),
])


def model_path(model_key: str) -> str:
    item = INTERPOLATION_MODELS.get(model_key)
    return os.path.join(_WEIGHTS, item["filename"]) if item else ""


def model_available(model_key: str) -> bool:
    path = model_path(model_key)
    return bool(path) and os.path.isfile(path) and os.path.getsize(path) > 1_000_000

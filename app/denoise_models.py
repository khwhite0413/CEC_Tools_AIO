"""Canonical denoise model registry shared by single and batch UI."""

from collections import OrderedDict


DEFAULT_DENOISE_MODEL = "fastdvdnet"

DENOISE_MODELS = OrderedDict((
    ("fastdvdnet", {
        "label": "FastDVDnet",
        "description_key": "denoise_fastdvdnet_tooltip",
    }),
    ("flow_denoise", {
        "label": "Flow Denoise",
        "description_key": "denoise_flow_tooltip",
    }),
))


def normalize_denoise_model(value: str | None) -> str:
    """Normalize UI/settings values while keeping old settings on Flow."""
    if value in DENOISE_MODELS:
        return value
    return "flow_denoise"

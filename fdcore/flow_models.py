"""Lazy loaders for MEMFOF / RAFT optical flow models.

Imports of `memfof` and `torchvision.models.optical_flow` happen here,
keeping the rest of the core importable without these deps.
"""

import os
import torch

_memfof_cache = None


def _weights_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "weights"))


def get_memfof(device: torch.device):
    """Load MEMFOF (cached on CPU, moved to device on call)."""
    global _memfof_cache
    if _memfof_cache is None:
        from memfof import MEMFOF
        # HuggingFace cache override so model lands inside the bundle
        os.environ.setdefault(
            "HF_HOME", os.path.join(_weights_dir(), "memfof"))
        _memfof_cache = MEMFOF.from_pretrained(
            "egorchistov/optical-flow-MEMFOF-Tartan-T-TSKH"
        ).eval()
    return _memfof_cache.to(device)


def get_raft(model_name: str, device: torch.device):
    """Load RAFT small or large from torchvision."""
    from torchvision.models.optical_flow import (
        raft_large, raft_small, Raft_Large_Weights, Raft_Small_Weights)
    if model_name == "raft_large":
        model = raft_large(weights=Raft_Large_Weights.DEFAULT)
    else:
        model = raft_small(weights=Raft_Small_Weights.DEFAULT)
    return model.to(device).eval()

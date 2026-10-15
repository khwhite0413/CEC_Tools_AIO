"""Small adapters around the four vendored VFI implementations."""

import importlib.util
import os
import sys
import types

import torch
import torch.nn.functional as F

from app.interpolation_models import model_path


_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_FANNOVEL = os.path.join(_ROOT, "third_party", "vfi_fannovel")
_TWEEN = os.path.join(_ROOT, "third_party", "vfi_tween")


def _load_module(name, path, package_paths=None):
    if package_paths is not None and name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = list(package_paths)
        sys.modules[name] = package
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"모듈을 불러올 수 없습니다: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _pad_pair(a, b, divisor):
    h, w = a.shape[-2:]
    ph = (divisor - h % divisor) % divisor
    pw = (divisor - w % divisor) % divisor
    pad = (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2)
    return F.pad(a, pad, mode="replicate"), F.pad(b, pad, mode="replicate"), pad


def _unpad(value, pad):
    left, right, top, bottom = pad
    h_end = value.shape[-2] - bottom if bottom else value.shape[-2]
    w_end = value.shape[-1] - right if right else value.shape[-1]
    return value[..., top:h_end, left:w_end]


class InterpolationBackend:
    def __init__(self, model_key, device, fp16=True):
        self.model_key = model_key
        self.device = device
        self.dtype = (torch.float16 if fp16 and device.type == "cuda"
                      else torch.float32)
        self.model = self._load()

    def _load(self):
        checkpoint = model_path(self.model_key)
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(
                f"프레임 보간 가중치가 없습니다: {checkpoint}\n"
                "setup.bat을 다시 실행하세요.")
        if self.model_key == "rife_426":
            return self._load_rife(checkpoint)
        if self.model_key == "atm_vfi_lite":
            return self._load_atm(checkpoint)
        if self.model_key in ("sgm_vfi", "ema_vfi"):
            return self._load_tween(checkpoint)
        raise ValueError(f"알 수 없는 프레임 보간 모델: {self.model_key}")

    def _install_comfy_device_stub(self):
        comfy = types.ModuleType("comfy")
        mm = types.ModuleType("comfy.model_management")
        mm.get_torch_device = lambda: self.device
        mm.soft_empty_cache = lambda: torch.cuda.empty_cache() if torch.cuda.is_available() else None
        comfy.model_management = mm
        sys.modules["comfy"] = comfy
        sys.modules["comfy.model_management"] = mm

    def _load_rife(self, checkpoint):
        missing = object()
        old_comfy = sys.modules.get("comfy", missing)
        old_mm = sys.modules.get("comfy.model_management", missing)
        try:
            self._install_comfy_device_stub()
            path = os.path.join(_FANNOVEL, "vfi_models", "rife", "rife_arch.py")
            module = _load_module("cec_vfi_rife_arch", path)
        finally:
            if old_comfy is missing:
                sys.modules.pop("comfy", None)
            else:
                sys.modules["comfy"] = old_comfy
            if old_mm is missing:
                sys.modules.pop("comfy.model_management", None)
            else:
                sys.modules["comfy.model_management"] = old_mm
        model = module.IFNet(arch_ver="4.26")
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        incompatible = model.load_state_dict(state, strict=False)
        if incompatible.missing_keys:
            raise RuntimeError(
                "RIFE 체크포인트에 필수 가중치가 없습니다: "
                + ", ".join(incompatible.missing_keys[:5]))
        return model.to(self.device, dtype=self.dtype).eval()

    def _load_atm(self, checkpoint):
        package_dir = os.path.join(_FANNOVEL, "vfi_models", "atm")
        package_name = "cec_vfi_atm"
        if package_name not in sys.modules:
            package = types.ModuleType(package_name)
            package.__path__ = [package_dir]
            sys.modules[package_name] = package
        module = importlib.util.find_spec(package_name + ".network_lite")
        if module is None or module.loader is None:
            raise ImportError("ATM-VFI 런타임을 찾을 수 없습니다.")
        loaded = importlib.util.module_from_spec(module)
        sys.modules[package_name + ".network_lite"] = loaded
        module.loader.exec_module(loaded)
        model = loaded.Network()
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = state.get("model_state_dict", state)
        state = {k: v for k, v in state.items()
                 if "attn_mask" not in k and not k.endswith("HW")}
        model.load_state_dict(state, strict=True)
        model.global_motion = True
        model.ensemble_global_motion = False
        return model.to(self.device).eval()

    def _load_tween(self, checkpoint):
        package_name = "cec_vfi_tween"
        if package_name not in sys.modules:
            package = types.ModuleType(package_name)
            package.__path__ = [_TWEEN]
            sys.modules[package_name] = package
        inference = _load_module(
            package_name + ".inference",
            os.path.join(_TWEEN, "inference.py"),
        )
        cls = inference.SGMVFIModel if self.model_key == "sgm_vfi" else inference.EMAVFIModel
        return cls(checkpoint, device=self.device)

    @torch.inference_mode()
    def midpoint(self, frame0, frame1):
        frame0 = frame0.to(self.device, dtype=self.dtype)
        frame1 = frame1.to(self.device, dtype=self.dtype)
        if self.model_key == "rife_426":
            a, b, pad = _pad_pair(frame0, frame1, 64)
            timestep = a.new_full((a.shape[0], 1, 1, 1), 0.5)
            out = self.model(a, b, timestep, [16, 8, 4, 2, 1], True, False)
            return _unpad(out, pad).float().clamp_(0, 1)
        if self.model_key == "atm_vfi_lite":
            a, b, pad = _pad_pair(frame0.float(), frame1.float(), 64)
            out = self.model(a, b)["I_t"][0].unsqueeze(0)
            return _unpad(out, pad).float().clamp_(0, 1)
        return self.model.interpolate_pair(frame0.float(), frame1.float(), 0.5).float()

    def close(self):
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def recursive_interpolate(backend, frame0, frame1, multiplier):
    """Return both endpoints and all dyadic intermediate tensors in order."""
    if multiplier == 1:
        return [frame0, frame1]
    middle = backend.midpoint(frame0, frame1)
    left = recursive_interpolate(backend, frame0, middle, multiplier // 2)
    right = recursive_interpolate(backend, middle, frame1, multiplier // 2)
    return left[:-1] + right

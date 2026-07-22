"""Environment pre-flight + PyTorch install for CEC All-In-One setup.

Runs after the embedded Python is installed (setup.bat step 1). Owns
the whole PyTorch install / heal path so setup.bat doesn't need its
own GPU-detection heuristic (which historically got RTX 50 + driver 610
wrong: the banner rename from "CUDA Version" → "CUDA UMD Version" broke
the tokens-based parse, and the fallback wheel cu121 has no SM_120
kernels, causing "no kernel image is available" at first GPU launch).

What this does:
    1. Detect NVIDIA GPU + compute capability + driver's CUDA runtime.
    2. Block known-bad combos (RTX 50 with pre-CUDA-12.8 driver,
       pre-Pascal GPUs).
    3. Pick the right PyTorch wheel index for the GPU's compute
       capability (NOT the driver's CUDA runtime — those are different
       concerns).
    4. Compare against whatever torch is currently installed (if any):
        - Missing        → install correct wheel.
        - Right arch     → skip (report OK).
        - Wrong arch     → uninstall + reinstall correct wheel
                           (this is the auto-heal path for anyone who
                           got the pre-fix cu121 pinning).
    5. Post-install verify: torch imports, cuda.is_available(), GPU
       compute cap in torch.cuda.get_arch_list().

Uses only stdlib (subprocess / ctypes / re) — must not depend on any
pip package because pip may not have installed anything yet.

Exit codes:
    0 — OK to proceed (torch installed and matches GPU)
    1 — blocking issue; setup.bat should abort
"""

from __future__ import annotations

import ctypes
import re
import shutil
import subprocess
import sys


# --- ANSI helpers ---------------------------------------------------------

_RESET = "\x1b[0m"
_RED = "\x1b[91m"
_YELLOW = "\x1b[93m"
_GREEN = "\x1b[92m"
_CYAN = "\x1b[96m"
_BOLD = "\x1b[1m"


def _enable_windows_ansi() -> None:
    if sys.platform != "win32":
        return
    try:
        kernel32 = ctypes.windll.kernel32          # type: ignore[attr-defined]
        STD_OUTPUT_HANDLE = -11
        ENABLE_VT_PROCESSING = 0x0004
        handle = kernel32.GetStdHandle(STD_OUTPUT_HANDLE)
        mode = ctypes.c_ulong()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value | ENABLE_VT_PROCESSING)
    except Exception:
        pass


def _banner(color: str, title: str, body: list[str]) -> None:
    line = "=" * 70
    print()
    print(color + line + _RESET)
    print(color + _BOLD + f"  {title}" + _RESET)
    print(color + line + _RESET)
    for entry in body:
        print("  " + entry)
    print(color + line + _RESET)
    print()


# --- nvidia-smi probes ----------------------------------------------------

def _run(cmd: list[str], timeout: int = 15) -> tuple[int, str]:
    try:
        # encoding="utf-8" + errors="replace" so a non-UTF-8 default
        # locale (Windows cp949 on Korean PCs was the trigger) doesn't
        # UnicodeDecodeError while decoding nvidia-smi / pip / torch
        # subprocess output when it contains non-ASCII bytes.
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              encoding="utf-8", errors="replace",
                              timeout=timeout, stdin=subprocess.DEVNULL)
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")
    except Exception as e:
        return -1, str(e)


def _nvidia_smi_present() -> bool:
    return shutil.which("nvidia-smi") is not None


def _query_gpus() -> list[dict]:
    """Return per-GPU dicts with keys: index, name, vram_mib, cap.

    Two separate --query-gpu calls because older drivers ignore unknown
    fields silently — asking for identity separately from compute_cap
    means we still see the GPU list even if compute_cap probe fails.
    """
    gpus: dict[int, dict] = {}
    rc, out = _run(["nvidia-smi",
                    "--query-gpu=index,name,memory.total",
                    "--format=csv,noheader,nounits"])
    if rc == 0:
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            try:
                idx = int(parts[0])
            except ValueError:
                continue
            try:
                vram = int(parts[2])
            except ValueError:
                vram = 0
            gpus[idx] = {"index": idx, "name": parts[1],
                         "vram_mib": vram, "cap": None}
    rc, out = _run(["nvidia-smi",
                    "--query-gpu=index,compute_cap",
                    "--format=csv,noheader"])
    if rc == 0:
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue
            try:
                idx = int(parts[0])
            except ValueError:
                continue
            cap = _parse_cap(parts[1])
            if idx in gpus:
                gpus[idx]["cap"] = cap
            else:
                gpus[idx] = {"index": idx, "name": "unknown",
                             "vram_mib": 0, "cap": cap}
    return sorted(gpus.values(), key=lambda g: g["index"])


def _query_driver_cuda_runtime() -> tuple[int, int] | None:
    """Read driver's max supported CUDA runtime from nvidia-smi output.

    Driver 610+ renamed the banner field from `CUDA Version:` to
    `CUDA UMD Version:` (and the compact banner no longer carries the
    legacy name at all). We match either form. `nvidia-smi -q` verbose
    output is used as a fallback because it still carries both the
    (deprecated) legacy name AND the new one, in a stable colon-aligned
    format that's easier to parse than the pipe-delimited banner.
    """
    rc, out = _run(["nvidia-smi"])
    if rc == 0:
        m = re.search(r"CUDA (?:UMD )?Version:\s*(\d+)\.(\d+)", out)
        if m:
            return int(m.group(1)), int(m.group(2))
    # Fallback to -q verbose format (colon-aligned, more permissive).
    rc, out = _run(["nvidia-smi", "-q"])
    if rc == 0:
        # Prefer new UMD line if present, else legacy.
        m = re.search(r"CUDA UMD Version\s*:\s*(\d+)\.(\d+)", out)
        if m:
            return int(m.group(1)), int(m.group(2))
        m = re.search(r"CUDA Version\s*:\s*(\d+)\.(\d+)", out)
        if m:
            return int(m.group(1)), int(m.group(2))
    return None


def _parse_cap(s: str) -> tuple[int, int] | None:
    m = re.match(r"^\s*(\d+)\.(\d+)\s*$", s or "")
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


# --- wheel selection ------------------------------------------------------
#
# Maps GPU compute capability → PyTorch wheel spec (index_url, is_nightly).
# We drive off compute_cap, NOT driver's CUDA runtime — driver version
# only gates whether the wheel can *run*; the arch kernel presence is
# strictly a function of what was compiled into the wheel.
#
# Rationale for each pick (as of 2026-07):
#   - SM 12.x (Blackwell / RTX 50)  → cu128. sm_120 is only in cu128+.
#   - SM 9.x  (Hopper / H100)       → cu124 stable.
#   - SM 8.x  (Ada / Ampere)        → cu124 stable — covers 8.6 and 8.9.
#   - SM 7.5  (Turing / RTX 20)     → cu121.
#   - SM 6.1  (Pascal / GTX 10)     → cu118. Last stable that still ships
#                                     sm_61 kernels; nightlies increasingly
#                                     drop it.
# Anything below SM 6.0 is refused up-front — modern PyTorch doesn't
# ship those kernels regardless of wheel choice.


def _recommend_wheel(cap: tuple[int, int]) -> str:
    """Return the pip --index-url suffix (cu128 / cu124 / cu121 / cu118 / cpu)."""
    if cap[0] >= 12:
        return "cu128"
    if cap[0] >= 8:
        return "cu124"
    if cap[0] == 7:
        return "cu121"
    if cap == (6, 1):
        return "cu118"
    if cap[0] < 6:
        return "unsupported"
    return "cu118"


def _pip_index_for(wheel: str) -> str:
    return f"https://download.pytorch.org/whl/{wheel}"


# --- torch introspection --------------------------------------------------

def _torch_state() -> dict:
    """Query the currently-installed torch. Returns dict with:
        installed: bool
        version: str | None
        cuda: str | None (e.g. '12.1' or None if CPU wheel)
        archs: list[str] (e.g. ['sm_50', ..., 'sm_120'])
    Runs in a subprocess so a broken torch install can't crash us.
    """
    script = (
        "import json, sys\n"
        "out = {'installed': False}\n"
        "try:\n"
        "    import torch\n"
        "    out['installed'] = True\n"
        "    out['version'] = torch.__version__\n"
        "    out['cuda'] = torch.version.cuda\n"
        "    try:\n"
        "        out['archs'] = list(torch.cuda.get_arch_list())\n"
        "    except Exception:\n"
        "        out['archs'] = []\n"
        "except Exception as e:\n"
        "    out['error'] = str(e)\n"
        "print(json.dumps(out))\n"
    )
    rc, out = _run([sys.executable, "-c", script])
    if rc != 0:
        return {"installed": False}
    try:
        import json as _json
        line = (out or "").strip().splitlines()[-1]
        return _json.loads(line)
    except Exception:
        return {"installed": False}


def _archs_cover(archs: list[str], cap: tuple[int, int]) -> bool:
    """Does `archs` contain a kernel for compute capability `cap`?

    PyTorch's arch list uses `sm_<major><minor>` notation. Exact match
    is the safe test — near-arch fallback via PTX only works when the
    wheel was built with +PTX for that family, which we can't reliably
    tell from the list alone.
    """
    target = f"sm_{cap[0]}{cap[1]}"
    return target in archs


# --- pip driver -----------------------------------------------------------

def _pip(args: list[str]) -> int:
    """Run pip via the current Python. Streams output live to the console
    (no capture) so long-running installs show progress."""
    cmd = [sys.executable, "-m", "pip"] + args
    try:
        return subprocess.call(cmd)
    except Exception as e:
        print(_RED + f"[setup-check] pip 실행 실패: {e}" + _RESET)
        return -1


def _uninstall_torch() -> None:
    print(_CYAN + "[setup-check] 기존 torch / torchvision / torchaudio 제거..." + _RESET)
    _pip(["uninstall", "-y", "torch", "torchvision", "torchaudio"])


def _install_torch(wheel: str) -> int:
    """Install torch + torchvision for the given wheel key.

    cu128 is only carried on the nightly channel today. Everything else
    on stable. When cu128 promotes to stable in a future release we can
    flip the branch — the wheel key stays the same.
    """
    is_nightly = wheel == "cu128"
    index = (f"https://download.pytorch.org/whl/nightly/{wheel}"
             if is_nightly else _pip_index_for(wheel))
    args = ["install"]
    if is_nightly:
        args.append("--pre")
    args += ["torch", "torchvision", "--index-url", index,
             "--no-warn-script-location"]
    print(_CYAN + f"[setup-check] pip install torch torchvision ({wheel}"
          + (" nightly" if is_nightly else "")
          + ")..." + _RESET)
    print(_CYAN + f"           index-url: {index}" + _RESET)
    return _pip(args)


# --- main -----------------------------------------------------------------

def check() -> int:
    _enable_windows_ansi()
    print(_CYAN + "[setup-check] 환경 사전 점검 + PyTorch 설치..." + _RESET)

    # -------------------- 1. GPU / driver detection --------------------
    if not _nvidia_smi_present():
        _banner(_YELLOW,
                "NVIDIA 드라이버 미감지 — CPU 전용 모드로 설치",
                [
                    "nvidia-smi 를 찾을 수 없습니다. PyTorch CPU 빌드로 설치를",
                    "계속합니다.",
                    "",
                    "※ 인페인트/디노이즈/업스케일 은 CPU 에서 실질적으로 사용",
                    "  불가능한 속도입니다. NVIDIA GPU 가 있는데도 이 메시지가",
                    "  뜬다면 드라이버 설치 상태를 확인하세요:",
                    "  https://www.nvidia.com/Download/index.aspx",
                ])
        # Install CPU torch if needed.
        st = _torch_state()
        if st.get("installed") and not st.get("cuda"):
            print(_GREEN + "[setup-check] CPU 빌드 이미 설치됨 — 스킵.\n" + _RESET)
            return 0
        if st.get("installed"):
            _uninstall_torch()
        return 0 if _install_torch("cpu") == 0 else 1

    gpus = _query_gpus()
    driver_cuda = _query_driver_cuda_runtime()

    if not gpus:
        # nvidia-smi returned but we couldn't parse any GPU. Rare —
        # print raw output for the user to share and proceed WITHOUT
        # touching torch (safest — don't blindly overwrite what's there).
        print(_YELLOW + "[setup-check] nvidia-smi 응답에서 GPU 를 파싱하지 못함:"
              + _RESET)
        rc, raw = _run(["nvidia-smi"])
        print(f"  [nvidia-smi rc={rc}]")
        for line in (raw or "").splitlines()[:15]:
            print("  " + line)
        print(_YELLOW + "  체크를 건너뛰고 설치를 진행합니다 (torch 는 별도 처리 X)."
              + _RESET)
        return 0

    # Summary line so users always see what was detected.
    driver_str = (f"{driver_cuda[0]}.{driver_cuda[1]}"
                  if driver_cuda else "미확인")
    print(_CYAN + "[setup-check] 감지된 환경:" + _RESET)
    print(f"  드라이버 CUDA runtime: {driver_str}")
    for g in gpus:
        cap = (f"{g['cap'][0]}.{g['cap'][1]}"
               if g["cap"] is not None else "unknown")
        print(f"  GPU {g['index']}: {g['name']}  "
              f"({g['vram_mib']} MiB, compute {cap})")

    # -------------------- 2. Fatal-block checks --------------------
    for g in gpus:
        cap = g["cap"]
        if cap is None:
            continue
        gname = g["name"] or f"GPU {g['index']}"
        cap_str = f"{cap[0]}.{cap[1]}"

        # Blackwell / RTX 50 with old driver → block.
        if cap[0] >= 12 and driver_cuda is not None and driver_cuda < (12, 8):
            _banner(_RED,
                    "그래픽 드라이버 업데이트 필요 (설치 중단)",
                    [
                        f"감지: {gname}  (compute {cap_str}, Blackwell)",
                        f"현재 드라이버: CUDA {driver_str}",
                        f"필요 드라이버: CUDA 12.8 이상 (RTX 50 시리즈용)",
                        "",
                        "이 상태로 설치하면 첫 GPU 커널 launch 에서 다음 에러가",
                        "발생합니다:",
                        "",
                        '  RuntimeError: CUDA error: no kernel image is',
                        '                available for execution on the device',
                        "",
                        "해결:",
                        "  1. 최신 GeForce 드라이버 설치",
                        "     https://www.nvidia.com/Download/index.aspx",
                        "  2. Windows 재부팅",
                        "  3. setup.bat 다시 실행",
                    ])
            return 1

        # Pre-Pascal → block (modern PyTorch dropped sm_5x).
        if cap[0] < 6:
            _banner(_RED,
                    "GPU 아키텍처 지원 종료 (설치 중단)",
                    [
                        f"감지: {gname}  (compute {cap_str})",
                        "",
                        "최신 PyTorch 는 이 아키텍처의 CUDA 커널을 담고 있지",
                        "않습니다. GPU 를 SM 6.1 (GTX 10) 이상으로 교체하거나",
                        "CPU 모드로만 실행 가능합니다.",
                    ])
            return 1

    # -------------------- 3. Determine target wheel --------------------
    # Use the highest-caps GPU as the target — multi-GPU systems with
    # mixed generations should get the wheel that covers the newest one.
    target_cap = max((g["cap"] for g in gpus if g["cap"] is not None),
                     default=None)
    if target_cap is None:
        print(_YELLOW +
              "[setup-check] compute_cap 을 알 수 없음 — 안전을 위해 torch 는 "
              "건드리지 않고 계속 진행합니다." + _RESET)
        return 0

    wheel = _recommend_wheel(target_cap)
    if wheel == "unsupported":
        # Reachable if cap parses but is <6 (shouldn't happen since we
        # blocked above, but belt-and-suspenders).
        return 1
    target_arch = f"sm_{target_cap[0]}{target_cap[1]}"
    print(_CYAN + f"[setup-check] 목표 wheel: {wheel} (필요 arch {target_arch})"
          + _RESET)

    # -------------------- 4. Compare against installed torch --------------------
    st = _torch_state()
    if st.get("installed"):
        cur_ver = st.get("version") or "?"
        cur_cuda = st.get("cuda") or "cpu"
        archs = st.get("archs") or []
        print(_CYAN +
              f"[setup-check] 현재 설치된 torch: {cur_ver} (cuda={cur_cuda})"
              + _RESET)
        print(f"  archs: {archs}")
        if _archs_cover(archs, target_cap):
            print(_GREEN + f"[setup-check] arch {target_arch} 포함 확인 — 재설치 불필요.\n"
                  + _RESET)
            return 0
        # Mismatch — auto-heal.
        _banner(_YELLOW,
                "torch 재설치 필요 (자동 진행)",
                [
                    f"설치된 torch: {cur_ver} (cuda={cur_cuda})",
                    f"보유 archs:   {archs}",
                    f"필요 arch:    {target_arch}",
                    "",
                    "이 GPU 를 위한 커널이 현재 wheel 에 포함되어 있지 않습니다.",
                    "기존 torch 를 제거하고 올바른 wheel 로 재설치합니다.",
                    "(첫 다운로드는 몇 분 걸릴 수 있음)",
                ])
        _uninstall_torch()

    # -------------------- 5. Install correct wheel --------------------
    rc = _install_torch(wheel)
    if rc != 0:
        print(_RED + f"[setup-check] torch 설치 실패 (rc={rc})" + _RESET)
        return 1

    # -------------------- 6. Post-install verify --------------------
    st = _torch_state()
    if not st.get("installed"):
        print(_RED + "[setup-check] 설치 후에도 torch import 실패." + _RESET)
        return 1
    archs = st.get("archs") or []
    if not _archs_cover(archs, target_cap):
        _banner(_RED,
                "재설치 후에도 arch 불일치 (설치 중단)",
                [
                    f"설치된 wheel: {st.get('version')} (cuda={st.get('cuda')})",
                    f"보유 archs:   {archs}",
                    f"필요 arch:    {target_arch}",
                    "",
                    f"목표 wheel 이었던 {wheel} 이 이 GPU 커널을 담고 있지 않은",
                    "것으로 보입니다. PyTorch 채널 상태를 개발자에게 알려주세요:",
                    "  https://download.pytorch.org/whl/ 목록 확인 필요.",
                ])
        return 1

    print(_GREEN + f"[setup-check] torch {st.get('version')} 설치 완료 "
          f"(archs 에 {target_arch} 포함)." + _RESET)

    # VRAM warning (non-fatal, after successful install)
    for g in gpus:
        if 0 < g["vram_mib"] < 6000:
            _banner(_YELLOW,
                    "VRAM 부족 가능성",
                    [
                        f"GPU {g['index']}: {g['name']}  ({g['vram_mib']} MiB)",
                        "",
                        "인페인트 (ProPainter) / 디노이즈 (MEMFOF) / 업스케일 은",
                        "8 GB 이상 VRAM 을 권장합니다. 현재 용량으로는 720p 이하",
                        "소스만 안정적입니다. OOM 발생 시 각 탭의 batch/max_frames",
                        "파라미터를 낮추면 완화됩니다.",
                    ])

    print(_GREEN + "[setup-check] OK — 다음 단계 진행.\n" + _RESET)
    return 0


if __name__ == "__main__":
    sys.exit(check())

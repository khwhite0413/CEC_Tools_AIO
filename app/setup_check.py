"""Environment pre-flight check for CEC All-In-One setup.

Runs after the embedded Python is installed (setup.bat step 1) but BEFORE
`pip install torch`, so we can catch GPU / driver mismatches that would
otherwise crash on first GPU kernel launch with the notorious:

    RuntimeError: CUDA error: no kernel image is available for execution
                  on the device

Root cause of that error is always the same — the PyTorch wheel that the
existing detection logic (nvidia-smi CUDA-runtime → cuXXX index) picks
was NOT built with kernels for the GPU's actual compute capability. The
classic instance: RTX 5090 (Blackwell, SM_120) with an old driver that
reports CUDA 12.4 → setup picks cu124 → cu124 doesn't include SM_120
kernels → crash.

This module runs BEFORE that install step and blocks known bad cases
with a red banner + concrete instructions, so the user fixes their
driver first and re-runs.

Uses only stdlib (subprocess / ctypes / re) — must not depend on any
pip package because pip hasn't installed anything yet at this point.

Exit codes:
    0 — OK to proceed (may still show yellow warnings)
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
    """Enable virtual-terminal processing on the current stdout handle.

    Windows 10 1607+ supports ANSI escape sequences on cmd.exe, but only
    when the console's ENABLE_VIRTUAL_TERMINAL_PROCESSING mode bit is set.
    Silently no-op on non-Windows or when the console handle refuses.
    """
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
    """Fixed-width ASCII banner. Kept ASCII-only so ancient cmd fonts
    can't mis-render box-drawing chars, and Korean CJK double-width
    issues don't misalign edges."""
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

def _run(cmd: list[str], timeout: int = 10) -> tuple[int, str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout, stdin=subprocess.DEVNULL)
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")
    except Exception as e:
        return -1, str(e)


def _nvidia_smi_present() -> bool:
    return shutil.which("nvidia-smi") is not None


def _query_gpus() -> list[dict]:
    """Return per-GPU info dicts with keys: index, name, vram_mib, cap.

    Split into two nvidia-smi queries because older drivers ignore
    unknown --query-gpu fields silently, and we want the identity fields
    (name, vram) as a floor even when compute_cap fails.
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
    """Read 'CUDA Version: X.Y' from `nvidia-smi` banner.

    The dedicated --query-gpu=driver_version returns the DRIVER version
    (e.g. 546.33), not the CUDA runtime it supports. The banner line is
    the only reliable source without shelling into WMI.
    """
    rc, out = _run(["nvidia-smi"])
    if rc != 0:
        return None
    m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", out)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _parse_cap(s: str) -> tuple[int, int] | None:
    m = re.match(r"^\s*(\d+)\.(\d+)\s*$", s or "")
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


# --- decision matrix ------------------------------------------------------
#
# Only two hard-blocker cases today:
#   1. GPU compute cap >= 12.0 (Blackwell, RTX 50) AND driver CUDA < 12.8
#      → cuXXX index that setup.bat would pick has no SM_120 kernels.
#      Fix: user updates driver, re-runs setup.
#   2. GPU compute cap < 6.0 (Maxwell 1st gen and older)
#      → modern PyTorch wheels dropped SM_5x kernels years ago.
#      Fix: replace GPU or run CPU-only.
#
# Yellow warnings:
#   - SM 6.1 (GTX 10 / Pascal) — near end of life
#   - VRAM < 6 GB — likely OOM on inpaint / denoise / upscale HD sources


def check() -> int:
    _enable_windows_ansi()
    print(_CYAN + "[setup-check] 환경 사전 점검..." + _RESET)

    if not _nvidia_smi_present():
        _banner(_YELLOW,
                "NVIDIA 드라이버 미감지 — CPU-only 모드로 진행",
                [
                    "nvidia-smi 를 찾을 수 없습니다. NVIDIA GPU 가 없거나",
                    "드라이버가 설치되지 않은 것으로 판단, PyTorch CPU 빌드로",
                    "설치를 계속합니다.",
                    "",
                    "※ 인페인트 / 디노이즈 / 업스케일 은 CPU 에서 실질적으로",
                    "  사용 불가능한 속도입니다. NVIDIA GPU 가 있는데도 이",
                    "  메시지가 뜬다면 드라이버 설치 상태를 확인하세요:",
                    "  https://www.nvidia.com/Download/index.aspx",
                ])
        return 0

    gpus = _query_gpus()
    driver_cuda = _query_driver_cuda_runtime()

    if not gpus or driver_cuda is None:
        print(_YELLOW +
              "[setup-check] nvidia-smi 응답을 완전히 파싱하지 못함 — "
              "체크를 건너뛰고 설치를 진행합니다." + _RESET)
        return 0

    # Summary line so users always see what was detected.
    driver_str = f"{driver_cuda[0]}.{driver_cuda[1]}"
    print(_CYAN + "[setup-check] 감지된 환경:" + _RESET)
    print(f"  드라이버 CUDA runtime: {driver_str}")
    for g in gpus:
        cap = (f"{g['cap'][0]}.{g['cap'][1]}"
               if g["cap"] is not None else "unknown")
        print(f"  GPU {g['index']}: {g['name']}  "
              f"({g['vram_mib']} MiB, compute {cap})")

    blocker = False
    warning = False

    for g in gpus:
        cap = g["cap"]
        if cap is None:
            continue
        gname = g["name"] or f"GPU {g['index']}"
        cap_str = f"{cap[0]}.{cap[1]}"

        # --- Blocker 1: Blackwell (RTX 50) with insufficient driver ---
        if cap[0] >= 12 and driver_cuda < (12, 8):
            _banner(_RED,
                    "그래픽 드라이버 업데이트 필요 (설치 중단)",
                    [
                        f"감지: {gname}  (compute {cap_str}, Blackwell)",
                        f"현재 드라이버: CUDA {driver_str}",
                        f"필요 드라이버: CUDA 12.8 이상 (RTX 50 시리즈용)",
                        "",
                        "이 상태로 설치하면 앱 첫 실행 시 인페인트 / 디노이즈 /",
                        "업스케일 의 첫 GPU 커널에서 다음 에러가 뜹니다:",
                        "",
                        '  RuntimeError: CUDA error: no kernel image is',
                        '                available for execution on the device',
                        "",
                        "해결:",
                        "  1. 최신 GeForce 드라이버 다운로드 및 설치",
                        "     https://www.nvidia.com/Download/index.aspx",
                        "  2. Windows 재부팅",
                        "  3. setup.bat 다시 실행",
                    ])
            blocker = True
            continue

        # --- Blocker 2: pre-Pascal GPU (SM < 6.0) ---
        if cap[0] < 6:
            _banner(_RED,
                    "GPU 아키텍처 지원 종료 (설치 중단)",
                    [
                        f"감지: {gname}  (compute {cap_str})",
                        "",
                        "이 GPU 는 최신 PyTorch 에서 CUDA 커널이 포함되지",
                        "않습니다. 설치 후 첫 GPU 커널 실행에서 'no kernel",
                        "image' 에러가 발생합니다.",
                        "",
                        "해결:",
                        "  - GPU 를 compute 6.1 이상 (GTX 10 시리즈 이상)",
                        "    으로 교체하거나",
                        "  - CPU-only 모드로 실행 (매우 느림, 사실상 사용 불가)",
                    ])
            blocker = True
            continue

        # --- Warning: SM 6.1 (Pascal / GTX 10) ---
        if cap == (6, 1):
            _banner(_YELLOW,
                    "Pascal GPU — 지원되지만 EOL 임박",
                    [
                        f"감지: {gname}  (compute 6.1, GTX 10 시리즈)",
                        "",
                        "현재 PyTorch stable 은 지원하지만 향후 릴리즈에서",
                        "지원 종료될 수 있습니다. 인페인트 / 디노이즈 는",
                        "실행되지만 최신 카드 대비 매우 느립니다.",
                    ])
            warning = True

    # --- VRAM warnings (independent of compute cap) ---
    for g in gpus:
        if 0 < g["vram_mib"] < 6000:
            _banner(_YELLOW,
                    "VRAM 부족 가능성",
                    [
                        f"GPU {g['index']}: {g['name']}  ({g['vram_mib']} MiB)",
                        "",
                        "인페인트 (ProPainter) / 디노이즈 (MEMFOF) / 업스케일 은",
                        "8 GB 이상 VRAM 을 권장합니다. 현재 용량으로는 720p 이하",
                        "소스만 안정적입니다.",
                        "",
                        "OOM 발생 시 각 탭의 max_batch_frames / batch_size 파라",
                        "미터를 낮추면 완화됩니다.",
                    ])
            warning = True

    if blocker:
        print(_RED + _BOLD +
              "\n[setup-check] 설치 중단 — 위 문제를 해결한 뒤 setup.bat 을 "
              "다시 실행하세요.\n" + _RESET)
        return 1

    if warning:
        print(_YELLOW + "[setup-check] 경고 있음 — 설치는 진행합니다.\n" + _RESET)
    else:
        print(_GREEN + "[setup-check] OK — 설치 진행.\n" + _RESET)
    return 0


if __name__ == "__main__":
    sys.exit(check())

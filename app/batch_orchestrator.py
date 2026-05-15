"""Sequential Inpaint -> Denoise -> Upscale orchestration for the Batch tab.

Design summary
--------------
The orchestrator is itself a QThread. Inside its run() it walks the queued
files in three stages, batching *by stage* (all files through Inpaint
first, then all through Denoise, then all through Upscale) so the user
sees clear per-stage progress and intermediate caches can be wiped as
each stage completes.

Workers
~~~~~~~
Each stage delegates to the existing per-tab worker class. Those workers
are also QThreads. We start one, wait for it via a local QEventLoop (so
the worker's signals can deliver back to us while we wait — wait() alone
would block the orchestrator's event loop and we'd never observe outcome
signals), then move on. This keeps signal plumbing simple without
refactoring the existing workers to pure functions.

Cache layout
~~~~~~~~~~~~
- cache/batch_inpaint/<original_stem>.<ext>
- cache/batch_denoise/<original_stem>.<ext>
- output/<yymmdd>_result/<original_stem>.<ext>  (collisions append _2, _3, ...)

The two cache folders are wiped *eagerly* — when Denoise starts, the
Inpaint cache is dropped; when Upscale starts, the Denoise cache is
dropped. Final cleanup removes whatever cache survived. This keeps disk
usage bounded to roughly one stage worth of intermediates at any time.
"""

from __future__ import annotations

import os
import shutil
import traceback
from datetime import date
from typing import List, Optional, Sequence

import cv2
import numpy as np
from PyQt5.QtCore import QEventLoop, QObject, QThread, Qt, pyqtSignal

from fdcore.video_io import VideoInfo
from vdcore.config import get_device
from app.inpaint_worker import InpaintWorker
from app.upscale_worker import UpscaleWorker
from app.workers import DenoiseWorker


_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_MODULE_DIR, ".."))
_PRESETS_DIR = os.path.join(_PROJECT_ROOT, "presets")
_CACHE_DIR = os.path.join(_PROJECT_ROOT, "cache")
_OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "output")
BATCH_INPAINT_CACHE = os.path.join(_CACHE_DIR, "batch_inpaint")
BATCH_DENOISE_CACHE = os.path.join(_CACHE_DIR, "batch_denoise")


STAGE_INPAINT = "inpaint"
STAGE_DENOISE = "denoise"
STAGE_UPSCALE = "upscale"


def resolve_output_dir(today: Optional[date] = None,
                       base: Optional[str] = None) -> str:
    """`<base>/yymmdd_result` — suffixes `_2`, `_3` etc. on collision so a
    same-day re-run never overwrites earlier batches.

    `base` defaults to the project's `output/` directory. The caller can
    override this when the user picks a custom location in the Batch tab;
    the orchestrator still wraps the run in a `{yymmdd}_result` subfolder
    so subsequent runs into the same chosen base don't co-mingle.
    """
    today = today or date.today()
    base_dir = base if base else _OUTPUT_DIR
    stem = today.strftime("%y%m%d") + "_result"
    candidate = os.path.join(base_dir, stem)
    if not os.path.exists(candidate):
        return candidate
    n = 2
    while True:
        candidate = os.path.join(base_dir, f"{stem}_{n}")
        if not os.path.exists(candidate):
            return candidate
        n += 1


def _ext_for_codec(codec: str) -> str:
    return ".mov" if codec == "prores" else ".mp4"


def _wipe_dir(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)


def _avoid_collision(path: str) -> str:
    """Return `path` if free, else `<stem>_2<ext>`, `<stem>_3<ext>` etc.

    Used only for the final output folder — same-basename collisions
    can occur when the queue holds two clips with identical filenames
    from different source directories.
    """
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    n = 2
    while True:
        candidate = f"{base}_{n}{ext}"
        if not os.path.exists(candidate):
            return candidate
        n += 1


def _compute_dreamina_mask(video_path: str,
                           dilation: int = 0) -> Optional[np.ndarray]:
    """Resolve the Dreamina preset mask for a given clip.

    Mirrors InpaintPage._apply_preset, but in a UI-free way so the
    orchestrator can stage masks for every queued clip up-front. Returns
    a uint8 H x W mask matching the clip's resolution, or None on failure.

    `dilation` (px, default 0) expands the mask outward by `dilation`
    iterations of a 3x3 ellipse kernel — i.e. roughly `dilation` pixels
    on each side. Used when the Dreamina preset doesn't quite cover the
    watermark and the user wants a safety margin without redrawing.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if vw <= 0 or vh <= 0:
        return None

    is_portrait = vh > vw
    preset_file = ("dreamina_9_16_watermark.png" if is_portrait
                   else "dreamina_ai.png")
    preset_path = os.path.join(_PRESETS_DIR, preset_file)
    preset_mask = cv2.imread(preset_path, cv2.IMREAD_GRAYSCALE)
    if preset_mask is None:
        return None

    ph, pw = preset_mask.shape[:2]
    preset_ar = pw / ph if ph else 1.0
    video_ar = vw / vh if vh else 1.0
    ar_match = abs(preset_ar - video_ar) / max(preset_ar, 1e-9) < 0.01

    if ar_match:
        mask = cv2.resize(preset_mask, (vw, vh),
                          interpolation=cv2.INTER_LINEAR)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    else:
        scale = min(vw / pw, vh / ph)
        new_w, new_h = int(pw * scale), int(ph * scale)
        scaled = cv2.resize(preset_mask, (new_w, new_h),
                            interpolation=cv2.INTER_LINEAR)
        _, scaled = cv2.threshold(scaled, 127, 255, cv2.THRESH_BINARY)
        mask = np.zeros((vh, vw), dtype=np.uint8)
        # Dreamina watermarks anchor top-left (matches InpaintPage default).
        mask[:new_h, :new_w] = scaled

    if dilation > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask, kernel, iterations=int(dilation))

    return mask


class BatchOrchestrator(QThread):
    """Drive the three-stage pipeline across N files.

    Signals
    -------
    overall_progress(stage_str, file_idx, file_total, done, total):
        stage_str ∈ {"inpaint","denoise","upscale"}
        file_idx / file_total: 1-based clip counter within the current stage
        done / total: frame counter inside the current worker
    item_status(item_idx, status_key, payload):
        Per-file status transitions. status_key:
        - "queued" — initial
        - "running:{stage}" — clip currently being processed by stage
        - "stage_done:{stage}" — clip finished this stage successfully
        - "all_done" — clip finished all 3 stages
        - "error" — payload = error message
    log(msg)
    error(msg) — fatal, batch aborts
    cancelled() — user-requested stop
    all_done(output_dir) — full batch completed; output_dir is the final folder
    """

    overall_progress = pyqtSignal(str, int, int, int, int)
    item_status = pyqtSignal(int, str, str)
    log = pyqtSignal(str)
    error = pyqtSignal(str)
    cancelled = pyqtSignal()
    all_done = pyqtSignal(str)

    def __init__(self, files: Sequence[str], inpaint_settings: dict,
                 denoise_settings: dict, upscale_settings: dict,
                 output_dir_base: Optional[str] = None,
                 custom_masks: Optional[dict] = None,
                 parent: Optional[QObject] = None):
        super().__init__(parent)
        self._files: List[str] = list(files)
        self._inpaint_settings = dict(inpaint_settings or {})
        self._denoise_settings = dict(denoise_settings or {})
        self._upscale_settings = dict(upscale_settings or {})
        # Optional override for the final output base folder. When set,
        # `<base>/{yymmdd}_result/` is used; when None, falls back to
        # the project's output/ directory (same as the older default).
        self._output_dir_base = output_dir_base
        # path -> custom mask ndarray. When a queued clip has an entry
        # here, the inpaint stage uses that mask instead of computing
        # the Dreamina preset. Missing entries fall back to auto.
        # The dict is captured by reference (no copy) but masks are
        # treated as immutable — orchestrator only reads.
        self._custom_masks: dict = dict(custom_masks) if custom_masks else {}
        self._cancel = False
        self._errored = False  # set by _run_worker when worker emitted error
        self._current_worker: Optional[QThread] = None
        self._device = None     # set inside _run_inner before any worker
        # Per-worker scratch state read by the bound-method slots
        # (_slot_on_progress / _finished / _error / _cancelled). They
        # live as instance attrs so the slots are pure methods (no
        # closure capture) — PyQt5 holds a strong ref to bound methods
        # via __self__, sidestepping the weakref-disconnect that local
        # lambdas hit on the longer denoise warm-up.
        self._cur_stage: str = ""
        self._cur_file_idx: int = 0
        self._cur_file_total: int = 0
        self._cur_loop: Optional[QEventLoop] = None
        self._cur_result: dict = {"error": None, "cancelled": False}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def cancel(self) -> None:
        self._cancel = True
        w = self._current_worker
        if w is not None:
            try:
                w.cancel()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------
    def run(self) -> None:
        try:
            self._run_inner()
        except Exception as e:
            self.error.emit(f"{e}\n\n{traceback.format_exc()}")

    def _run_inner(self) -> None:
        if not self._files:
            self.error.emit("큐가 비어있습니다.")
            return

        # Dynamic pipeline: each entry is one canonical stage the user
        # opted in to. The last entry's outputs go to out_dir; all
        # earlier entries park their outputs in their own cache dir
        # (which gets wiped as soon as the following stage finishes).
        stages: List[dict] = []
        if self._inpaint_settings.get("enabled", True):
            stages.append({
                "name": STAGE_INPAINT,
                "label": "인페인트",
                "settings": self._inpaint_settings,
                "cache_dir": BATCH_INPAINT_CACHE,
                "build": self._build_inpaint_worker,
            })
        if self._denoise_settings.get("enabled", True):
            stages.append({
                "name": STAGE_DENOISE,
                "label": "디노이즈",
                "settings": self._denoise_settings,
                "cache_dir": BATCH_DENOISE_CACHE,
                "build": self._build_denoise_worker,
            })
        if self._upscale_settings.get("enabled", True):
            stages.append({
                "name": STAGE_UPSCALE,
                "label": "업스케일",
                "settings": self._upscale_settings,
                # Upscale never has its own cache — when it's an
                # intermediate stage (no canonical successor exists,
                # but in our pipeline it's always last when enabled),
                # this branch is unreachable.
                "cache_dir": None,
                "build": self._build_upscale_worker,
            })

        if not stages:
            self.error.emit(
                "실행할 단계가 하나도 없습니다 — 최소 한 개는 활성화하세요.")
            return

        out_dir = resolve_output_dir(base=self._output_dir_base)
        os.makedirs(out_dir, exist_ok=True)
        # Start clean every run so leftovers from a previous failed batch
        # don't get re-piped into the next.
        _wipe_dir(BATCH_INPAINT_CACHE)
        _wipe_dir(BATCH_DENOISE_CACHE)

        self.log.emit(f"배치 시작 — 출력 폴더: {out_dir}")
        self.log.emit("활성 단계: "
                      + " → ".join(s["label"] for s in stages))

        self._device = get_device()
        total = len(self._files)

        # current_inputs[i] tracks where the i-th queued clip "lives"
        # right now — starts as the original source, gets replaced by
        # the latest stage's output path after each stage completes.
        current_inputs: List[str] = list(self._files)

        for stage_idx, stage in enumerate(stages):
            if self._cancel:
                self.cancelled.emit()
                return
            is_last = (stage_idx == len(stages) - 1)

            codec = stage["settings"].get("output_format", "h264")
            ext = _ext_for_codec(codec)

            if is_last:
                target_dir = out_dir
            else:
                target_dir = stage["cache_dir"]
                os.makedirs(target_dir, exist_ok=True)

            new_outputs: List[str] = []
            for i, src in enumerate(current_inputs, start=1):
                if self._cancel:
                    self.cancelled.emit()
                    return
                # Stem comes from the ORIGINAL source so all stages
                # produce a consistently-named file ("scene_01.mp4" all
                # the way through), regardless of which cache-renamed
                # intermediate they're reading from.
                stem = os.path.splitext(
                    os.path.basename(self._files[i - 1]))[0]
                out = os.path.join(target_dir, f"{stem}{ext}")
                if is_last:
                    # Cheap insurance against same-basename collisions
                    # in the final folder (different source dirs, same
                    # filename). Intermediates run in pre-wiped dirs so
                    # no collision possible there.
                    out = _avoid_collision(out)

                self.item_status.emit(
                    i - 1, f"running:{stage['name']}", "")
                self.log.emit(
                    f"[{stage_idx + 1}/{len(stages)} {stage['label']}] "
                    f"{i}/{total} · {os.path.basename(src)}")

                worker = stage["build"](src, out)
                if worker is None:
                    # Worker construction failed — already-emitted error
                    # carries the message. Bail.
                    return
                self._run_worker(worker, stage["name"], i, total)
                if self._errored:
                    return
                if self._cancel:
                    self.cancelled.emit()
                    return
                if not os.path.exists(out):
                    self.item_status.emit(
                        i - 1, "error", f"{stage['label']} 출력 누락")
                    self.error.emit(
                        f"[{os.path.basename(src)}] {stage['label']} "
                        "출력이 생성되지 않았습니다.")
                    return
                new_outputs.append(out)
                if is_last:
                    self.item_status.emit(i - 1, "all_done", "")
                else:
                    self.item_status.emit(
                        i - 1, f"stage_done:{stage['name']}", "")

            # Wipe the PREVIOUS stage's cache as soon as we're done
            # reading from it. The cache is only useful as input to the
            # current stage, which we just finished.
            if stage_idx > 0:
                prev = stages[stage_idx - 1]
                if prev["cache_dir"]:
                    _wipe_dir(prev["cache_dir"])
                    self.log.emit(
                        f"[캐시 정리] {prev['label']} 중간 결과물 삭제됨.")

            current_inputs = new_outputs

        self.all_done.emit(out_dir)

    # ------------------------------------------------------------------
    # Per-stage worker builders. Each returns None on construction
    # failure (after emitting an error) — the run loop bails when None
    # is returned.
    # ------------------------------------------------------------------
    def _build_inpaint_worker(self, src: str, out: str):
        # Mask is computed against the source file the worker will read.
        # Since the canonical pipeline puts Inpaint first when enabled,
        # `src` is always the original clip and resolution matches.
        # Per-clip custom mask wins; auto-Dreamina is the fallback.
        # User-drawn custom masks are taken as-is (no dilation applied
        # — the user already drew exactly what they wanted).
        mask = self._custom_masks.get(src)
        if mask is None:
            mask = _compute_dreamina_mask(
                src,
                dilation=int(
                    self._inpaint_settings.get("dreamina_dilation_px", 0)))
        if mask is None:
            self.error.emit(
                f"[{os.path.basename(src)}] Dreamina 마스크 생성 실패. "
                "프리셋 파일 또는 영상 해상도를 확인하세요.")
            self._errored = True
            return None
        return InpaintWorker(src, mask, self._inpaint_settings,
                             out, self._device)

    def _build_denoise_worker(self, src: str, out: str):
        return DenoiseWorker(src, self._denoise_settings, out)

    def _build_upscale_worker(self, src: str, out: str):
        return UpscaleWorker(src, out, self._upscale_settings, self._device)

    # ------------------------------------------------------------------
    # Worker driver
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Bound-method slots for the per-worker QEventLoop. Used to be
    # local lambdas inside _run_worker; PyQt5 stores connection
    # receivers via weakref for some non-QObject targets, and the
    # local lambdas were apparently being garbage-collected between
    # connect() and the first emit on denoise (which has a longer
    # warm-up than inpaint due to MEMFOF / RAFT loads, so the gap is
    # bigger). Bound methods keep a strong ref through their __self__
    # attribute, so the connection stays live.
    #
    # Per-call context (stage / file_idx / file_total / loop / result)
    # is parked on instance attributes that _run_worker rotates before
    # connecting.
    # ------------------------------------------------------------------
    def _slot_on_progress(self, done: int, total: int) -> None:
        self.overall_progress.emit(
            self._cur_stage, self._cur_file_idx, self._cur_file_total,
            int(done), int(total))

    def _slot_on_finished(self, _path: str) -> None:
        if self._cur_loop is not None:
            self._cur_loop.quit()

    def _slot_on_error(self, msg: str) -> None:
        self._cur_result["error"] = msg
        if self._cur_loop is not None:
            self._cur_loop.quit()

    def _slot_on_cancelled(self) -> None:
        self._cur_result["cancelled"] = True
        if self._cur_loop is not None:
            self._cur_loop.quit()

    def _run_worker(self, worker: QThread, stage: str, file_idx: int,
                    file_total: int) -> None:
        """Start the worker, block on a local QEventLoop until terminal
        signal fires, then return. Records error / cancel into instance
        state so the outer loop can react.
        """
        self._current_worker = worker
        self._cur_stage = stage
        self._cur_file_idx = file_idx
        self._cur_file_total = file_total
        self._cur_result = {"error": None, "cancelled": False}

        loop = QEventLoop()
        self._cur_loop = loop

        # Bound methods keep the receiver alive via __self__ -> self,
        # which is the orchestrator (still referenced by the caller).
        # No more chance of a weakref-induced silent disconnect mid-run.
        worker.progress.connect(self._slot_on_progress, Qt.QueuedConnection)
        worker.finished_signal.connect(
            self._slot_on_finished, Qt.QueuedConnection)
        worker.error_signal.connect(self._slot_on_error, Qt.QueuedConnection)
        worker.cancelled_signal.connect(
            self._slot_on_cancelled, Qt.QueuedConnection)

        worker.start()
        loop.exec_()
        # Make sure the worker thread has truly exited before we drop
        # the reference (otherwise destroying a still-running QThread
        # raises on Windows).
        worker.wait()
        self._current_worker = None
        self._cur_loop = None
        result = self._cur_result

        # Drop python refs that might still hold the worker's model
        # tensors, then ask PyTorch to return its cached VRAM blocks
        # to the OS. Without this, ProPainter's 3-4 GB cache stays
        # parked even after the InpaintEngine releases — and 10
        # consecutive clips followed by MEMFOF + RAFT loading for
        # denoise tips a 16 GB GPU into the cudaMalloc-thrashing
        # regime that froze the batch at "0%". Single-tab workflows
        # never reach this branch (their workers are run directly
        # from the page, not the orchestrator), so this cleanup is
        # scoped to batch chains by construction.
        import gc as _gc
        _gc.collect()
        try:
            import torch as _torch
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
        except Exception:
            pass

        if result["error"]:
            # Emit cleanly to the UI; the outer loop sees _errored and
            # returns WITHOUT also emitting cancelled() (which would
            # double-message the user).
            self.error.emit(result["error"])
            self._errored = True
        elif result["cancelled"]:
            self._cancel = True

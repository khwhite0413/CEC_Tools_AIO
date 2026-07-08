"""Progress callback abstraction.

The core never imports Qt or ComfyUI. Workers pass callables and we wrap
them for safety.

Two channels, distinct semantics:

- `update(done, total)` reports **overall** progress — cumulative frames
  encoded out of the whole clip's total. Monotonically non-decreasing
  over a run.
- `update_chunk(done, chunk_size)` reports progress **within one
  temporal_flow_average call** — how far along the mini-batch of frames
  currently sitting in GPU is. Resets to 0 at the start of every new
  chunk and climbs to `chunk_size` before rolling over. The streaming
  pipeline invokes temporal_flow_average many times per run, so this
  channel is what surfaces the internal MEMFOF batch progress to the UI
  as a secondary "chunk bar" that oscillates while the overall bar
  climbs monotonically.

Both callbacks are optional and None-safe.
"""

from typing import Callable, Optional


class ProgressReporter:
    """Wraps overall + chunk-scoped progress callables."""

    def __init__(self,
                 cb: Optional[Callable[[int, int], None]] = None,
                 chunk_cb: Optional[Callable[[int, int], None]] = None):
        self._cb = cb
        self._chunk_cb = chunk_cb
        self._cancel = False

    def update(self, done: int, total: int) -> None:
        """Emit an overall-progress event (frames encoded / total frames)."""
        if self._cb is not None:
            try:
                self._cb(int(done), int(total))
            except Exception:
                pass

    def update_chunk(self, done: int, chunk_size: int) -> None:
        """Emit a chunk-scoped progress event (frames processed within the
        current temporal_flow_average call / chunk size). No-op when no
        chunk callback was supplied — safe for the legacy single-shot
        code path that only cared about overall progress."""
        if self._chunk_cb is not None:
            try:
                self._chunk_cb(int(done), int(chunk_size))
            except Exception:
                pass

    def request_cancel(self) -> None:
        self._cancel = True

    @property
    def cancelled(self) -> bool:
        return self._cancel


class CancelledError(RuntimeError):
    """Raised when a long-running pipeline is cancelled by the user."""

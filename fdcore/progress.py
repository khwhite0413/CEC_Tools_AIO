"""Progress callback abstraction.

The core never imports Qt or ComfyUI. Workers pass a callable
`progress_cb(done: int, total: int)` and we wrap it for safety.
"""

from typing import Callable, Optional


class ProgressReporter:
    """Wraps a (done, total) callable. None-safe."""

    def __init__(self, cb: Optional[Callable[[int, int], None]] = None):
        self._cb = cb
        self._cancel = False

    def update(self, done: int, total: int) -> None:
        if self._cb is not None:
            try:
                self._cb(int(done), int(total))
            except Exception:
                pass

    def request_cancel(self) -> None:
        self._cancel = True

    @property
    def cancelled(self) -> bool:
        return self._cancel


class CancelledError(RuntimeError):
    """Raised when a long-running pipeline is cancelled by the user."""

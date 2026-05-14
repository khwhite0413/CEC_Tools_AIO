"""ProPainter weight downloader.

Used by setup.bat's step 7 to fetch the three ProPainter checkpoints
(`ProPainter.pth`, `recurrent_flow_completion.pth`, `raft-things.pth`).

The previous version called `progress_callback(...)` on every 8 KB
chunk with a fresh string, and setup.bat passed `progress_callback=print`
— which printed one new line per percentage point on a 580 MB file
(tens of thousands of lines stacked in the console).

This version renders a single self-overwriting line per file via
carriage return: ASCII spinner + bar + percent + size + ETA, throttled
to ~20 fps. The `progress_callback` parameter is kept for backwards
compatibility but is no longer used for per-chunk progress — only the
one-shot info messages from `download_all_weights` go through it.
"""

import os
import sys
import time

import requests

from vdcore.config import WEIGHTS_DIR, WEIGHT_URLS


# Plain ASCII spinner — Windows default cmd codepage 949 (Korean) does
# not render Braille block characters reliably, so we stick with chars
# that exist in every codepage.
_SPINNER = "|/-\\"


class _CliProgressBar:
    """Single-line CLI download progress with spinner + ETA.

    Output is appended via `\\r` so consecutive renders overwrite each
    other; finish() writes the trailing newline so subsequent log lines
    don't clobber the last bar state.
    """

    BAR_WIDTH = 28
    _FRAME_INTERVAL = 0.05  # seconds — ~20 fps

    def __init__(self, label: str, total_bytes: int):
        self.label = label
        self.total = int(total_bytes) if total_bytes > 0 else 0
        # Server may omit Content-Length on chunked transfers. Fall back
        # to a spinner-only "X.X MB downloaded" line in that case.
        self.indeterminate = self.total == 0
        self.done = 0
        self.start = time.time()
        self._spin_idx = 0
        self._last_render = 0.0

    def update(self, nbytes: int):
        self.done += nbytes
        now = time.time()
        if now - self._last_render < self._FRAME_INTERVAL:
            return
        self._last_render = now
        self._render()

    def finish(self):
        if not self.indeterminate:
            self.done = self.total
        self._render(final=True)
        sys.stdout.write("\n")
        sys.stdout.flush()

    def _render(self, final: bool = False):
        spinner = " " if final else _SPINNER[self._spin_idx % len(_SPINNER)]
        self._spin_idx += 1
        mb_done = self.done / 1048576.0
        if self.indeterminate:
            line = (f"\r    [{spinner}] {self.label}  "
                    f"{mb_done:.1f} MB        ")
        else:
            pct = self.done * 100.0 / self.total
            filled = int(self.BAR_WIDTH * self.done / self.total)
            bar = "#" * filled + "-" * (self.BAR_WIDTH - filled)
            mb_total = self.total / 1048576.0
            elapsed = time.time() - self.start
            if self.done > 0 and not final:
                eta = elapsed * (self.total - self.done) / max(1, self.done)
                eta_str = f" ETA {eta:>4.0f}s"
            else:
                eta_str = "           "
            line = (f"\r    [{spinner}] {self.label}  "
                    f"[{bar}]  {pct:5.1f}%  "
                    f"({mb_done:6.1f}/{mb_total:6.1f} MB){eta_str}")
        sys.stdout.write(line)
        sys.stdout.flush()


def download_file(url, dest_path, progress_callback=None):
    """Stream-download `url` to `dest_path` with an inline progress bar.

    `progress_callback` is accepted for backwards compatibility but is
    no longer invoked per chunk — the inline bar handles user feedback
    inside this function. Pre-existing callers can keep passing
    `print` without changing behaviour for the non-chunk paths.
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    name = os.path.basename(dest_path)
    bar = _CliProgressBar(name, total)
    try:
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=65536):
                if not chunk:
                    continue
                f.write(chunk)
                bar.update(len(chunk))
    finally:
        bar.finish()


def download_all_weights(progress_callback=None):
    """Fetch every ProPainter weight that isn't already on disk.

    Per-file progress is rendered inline by download_file(). The
    optional progress_callback only receives short one-shot status
    lines («already present», «complete») — never the noisy
    per-chunk percentage stream that the old version emitted.
    """
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    for name, url in WEIGHT_URLS.items():
        dest = os.path.join(WEIGHTS_DIR, name)
        if os.path.isfile(dest):
            if progress_callback:
                progress_callback(f"    [-] {name} (이미 존재)")
            continue
        download_file(url, dest)
    if progress_callback:
        progress_callback("    [+] 모든 ProPainter 가중치 OK")

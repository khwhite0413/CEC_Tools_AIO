"""Background update check.

Hits the Cloudflare Worker proxy on startup, compares the returned
`version` string against `app._version.__version__`, and emits a signal
with the result. Runs in a QThread so app boot is never blocked; fails
silently on any network / parse issue so an offline PC never gets an
error popup at launch.

The Worker (see cf-worker/worker.js) fronts a private GitHub repo with
a fine-grained PAT so the client itself doesn't need any credentials.
Only the Worker URL is baked in here; changing endpoints is a one-line
edit and a re-release.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Optional

from PyQt5.QtCore import QThread, pyqtSignal

from app._version import __version__


# Update this if the Worker URL ever changes. Kept as a module constant
# so callers can override in tests without touching env vars.
UPDATE_ENDPOINT_BASE = "https://cec-aio-update.khwhite0413.workers.dev"


def _parse_semver(s: str) -> Optional[tuple]:
    """'1.0.7' or 'v1.0.7' -> (1, 0, 7). Returns None on anything weird
    so an odd tag ('v1.0.8-beta', 'nightly') doesn't crash the check —
    we just silently skip the update prompt in that case."""
    if not s:
        return None
    if s.startswith("v"):
        s = s[1:]
    parts = s.split(".")
    try:
        return tuple(int(p) for p in parts)
    except (ValueError, TypeError):
        return None


def _is_newer(latest: str, current: str) -> bool:
    a = _parse_semver(latest)
    b = _parse_semver(current)
    if a is None or b is None:
        return False
    return a > b


class UpdateCheckWorker(QThread):
    """Fetches /latest from the Worker, emits `result(dict)` on success.

    Payload keys:
        is_newer            bool
        current             str  (this app's __version__, no leading 'v')
        latest              str  (Worker-reported version, no leading 'v')
        tag                 str  ('v1.0.8' or similar — for skip-version pref)
        html_url            str  (GitHub release page)
        body                str  (release notes markdown)
        download_endpoint   str  ('/download/v1.0.8' — concat with BASE)

    On failure emits `error(str)` — the automatic startup check leaves
    this unconnected (silent-on-failure by design), the manual "check
    for update" button in the header connects it so the user gets a
    concrete "네트워크 오류" popup instead of a mysterious no-op.
    """

    result = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, parent=None, timeout: float = 5.0):
        super().__init__(parent)
        self._timeout = timeout

    def run(self):
        try:
            req = urllib.request.Request(
                UPDATE_ENDPOINT_BASE + "/latest",
                headers={
                    "User-Agent": f"CEC-AllInOne/{__version__} updater",
                    "Accept": "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                raw = resp.read()
        except Exception as e:
            self.error.emit(f"업데이트 서버 접속 실패: {e}")
            return
        try:
            data = json.loads(raw.decode("utf-8"))
        except Exception as e:
            self.error.emit(f"업데이트 서버 응답 파싱 실패: {e}")
            return
        latest = str(data.get("version") or "")
        payload = {
            "is_newer": _is_newer(latest, __version__),
            "current": __version__,
            "latest": latest,
            "tag": str(data.get("tag_name") or ""),
            "html_url": str(data.get("html_url") or ""),
            "body": str(data.get("body") or ""),
            "download_endpoint": str(data.get("download_endpoint") or ""),
        }
        self.result.emit(payload)

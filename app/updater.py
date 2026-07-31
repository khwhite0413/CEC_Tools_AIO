"""Update download + apply orchestrator.

Runs when the user clicks "지금 업데이트" in the update prompt. Downloads
the release zipball through the Cloudflare Worker proxy (which does the
GitHub auth for our private repo), extracts to a temp staging dir, then
kicks the static `updater.bat` at the project root with PID + src + dst
args. Then the caller quits the app so updater.bat can safely overwrite
source files (Windows locks .py files that are still imported).

Sequence:
    UpdaterWorker.run()            [QThread — download + extract only]
      progress(done, total) → UI
      finished_ok(staging_dir)
        → caller: kick_updater_bat(staging_dir)
        → caller: QApplication.quit()
    updater.bat                     [detached cmd process]
      waits for app PID to exit
      robocopy staging_dir → project root
        (skips weights/, python/, ffmpeg/, cache/, output/, logs/)
      runs setup.bat (idempotent — fetches only missing deps)
      relaunches run.bat
      exits
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

from PyQt5.QtCore import QThread, pyqtSignal

from app.update_checker import UPDATE_ENDPOINT_BASE


_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_MODULE_DIR, ".."))


class UpdaterWorker(QThread):
    """Download release zipball through the Worker + extract to temp.

    Signals
    -------
    progress(done, total)   int bytes; total=0 when Content-Length missing
    finished_ok(src_root)   path to the folder whose contents mirror the
                            project root (i.e. contains app/, fdcore/,
                            setup.bat, run.bat, ...). Ready to hand to
                            updater.bat as the copy source.
    error(msg)              human-readable message; UI shows a modal.
    """

    progress = pyqtSignal(int, int)
    finished_ok = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, download_endpoint: str, parent=None):
        super().__init__(parent)
        # `/download/v1.0.8` style suffix from update_checker's payload;
        # concat with the base to get a full URL.
        self._url = UPDATE_ENDPOINT_BASE + download_endpoint

    def run(self):
        try:
            staging = tempfile.mkdtemp(prefix="cec_aio_update_")
            zip_path = os.path.join(staging, "release.zip")

            # --- download -------------------------------------------------
            req = urllib.request.Request(
                self._url,
                headers={"User-Agent": "CEC-AllInOne updater"},
            )
            # 60 s timeout on connect only; streaming body has its own
            # per-chunk implicit timeout via socket state. Zipball for a
            # source-only release is small (<5 MB) so this is generous.
            with urllib.request.urlopen(req, timeout=60) as resp:
                total = 0
                try:
                    total = int(resp.headers.get("Content-Length") or 0)
                except Exception:
                    total = 0
                done = 0
                with open(zip_path, "wb") as f:
                    while True:
                        chunk = resp.read(64 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        self.progress.emit(done, total or done)

            # --- extract --------------------------------------------------
            extract_dir = os.path.join(staging, "extracted")
            os.makedirs(extract_dir, exist_ok=True)
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(extract_dir)

            # GitHub zipballs wrap everything in a single top-level dir
            # named "<owner>-<repo>-<short_sha>". Descend into it so
            # updater.bat can copy contents directly to project root.
            entries = [e for e in os.listdir(extract_dir)
                       if not e.startswith(".")]
            if (len(entries) == 1 and
                    os.path.isdir(os.path.join(extract_dir, entries[0]))):
                src_root = os.path.join(extract_dir, entries[0])
            else:
                src_root = extract_dir

            # Sanity check — the extracted root should contain at least
            # app/ and setup.bat, otherwise something's wrong and we
            # don't want updater.bat to blindly robocopy random garbage
            # into the install.
            required_markers = ("app", "setup.bat", "run.bat")
            missing = [m for m in required_markers
                       if not os.path.exists(os.path.join(src_root, m))]
            if missing:
                self.error.emit(
                    "다운로드한 ZIP 이 예상 구조가 아닙니다. "
                    f"누락된 항목: {missing}. 업데이트 중단.")
                return

            self.finished_ok.emit(src_root)
        except Exception as e:
            self.error.emit(f"업데이트 다운로드 실패: {e}")


def kick_updater_bat(src_root: str) -> None:
    """Spawn the static updater.bat at project root with PID + paths as
    args, detached so it survives our own process exit. The caller MUST
    call QApplication.quit() (or os._exit) shortly after this — the .bat
    waits for our PID to disappear before touching any source files.

    Raises RuntimeError if updater.bat is missing (release without it,
    or user deleted it) — the caller surfaces that to the user rather
    than silently no-op'ing.
    """
    bat_path = os.path.join(_PROJECT_ROOT, "updater.bat")
    if not os.path.isfile(bat_path):
        raise RuntimeError(
            "updater.bat 이 프로젝트 루트에 없습니다. "
            "GitHub 릴리즈에서 수동으로 최신 ZIP 을 받아 재설치해주세요.")

    pid = os.getpid()

    # `cmd /c start "" ...` opens a new console window (visible, not
    # minimized) so the user can see the update / setup progress. The
    # blank "" is required — start's first quoted arg is treated as
    # window title, so without it the batch path itself becomes the
    # title and start does nothing.
    #
    # DETACHED_PROCESS + CREATE_NEW_PROCESS_GROUP so updater.bat's cmd
    # session survives after this Python process exits (which we're
    # about to do). Without these flags, killing the parent (us) kills
    # the child too.
    flags = (subprocess.DETACHED_PROCESS
             | subprocess.CREATE_NEW_PROCESS_GROUP)

    subprocess.Popen(
        ["cmd", "/c", "start", "",
         bat_path, str(pid), src_root, _PROJECT_ROOT],
        cwd=_PROJECT_ROOT,
        creationflags=flags,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
    )

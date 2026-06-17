"""CEC All-In-One — entry point (pure PyQt5, no third-party UI library).

Layered failure visibility (회고 §8):
  - stdout/stderr go to logs/cec_aio.log (every print/traceback captured)
  - uncaught exceptions also pop a Win32 MessageBox so failures aren't silent
  - the UI exposes a "로그 보기" button that opens the file

torch must be imported BEFORE PyQt5 to avoid Windows DLL load order
conflict (회고 §4).
"""

import os
import sys
import datetime
import traceback


# ---------------------------------------------------------------------------
# Logging setup — must happen before any other imports that might print.
# ---------------------------------------------------------------------------

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

LOG_DIR = os.path.join(ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "cec_aio.log")


class _Tee:
    """Write to a primary stream and also to the original (if any)."""

    def __init__(self, primary, fallback=None):
        self.primary = primary
        self.fallback = fallback

    def write(self, s):
        try:
            self.primary.write(s)
            self.primary.flush()
        except Exception:
            pass
        if self.fallback is not None:
            try:
                self.fallback.write(s)
            except Exception:
                pass

    def flush(self):
        try:
            self.primary.flush()
        except Exception:
            pass


_log_file = open(LOG_PATH, "a", encoding="utf-8", buffering=1)
_log_file.write(f"\n=== CEC All-In-One start {datetime.datetime.now().isoformat()} ===\n")

sys.stdout = _Tee(_log_file, sys.__stdout__)
sys.stderr = _Tee(_log_file, sys.__stderr__)


def _show_error_box(title: str, body: str):
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(0, body, title, 0x10 | 0x1000)
    except Exception:
        pass


def _excepthook(exc_type, exc, tb):
    text = "".join(traceback.format_exception(exc_type, exc, tb))
    sys.stderr.write(text)
    sys.stderr.flush()
    short = f"{exc_type.__name__}: {exc}\n\n로그: {LOG_PATH}"
    _show_error_box("CEC All-In-One — 오류", short)


sys.excepthook = _excepthook


# ---------------------------------------------------------------------------
# Force QMediaPlayer onto Windows Media Foundation. Qt5's default Windows
# backend is DirectShow, which on a clean Windows 11 install can't decode
# h264/aac mp4 (it errors out with VFW_E_NO_TRANSPORT 0x80040266 in
# DirectShowPlayerService::doRender). WMF handles h264/aac natively. Must
# be set BEFORE QtMultimedia is imported anywhere.
# ---------------------------------------------------------------------------
os.environ["QT_MULTIMEDIA_PREFERRED_PLUGINS"] = "windowsmediafoundation"


# ---------------------------------------------------------------------------
# torch first, then PyQt5 (회고 §4)
# ---------------------------------------------------------------------------

try:
    import torch  # noqa: F401
except ImportError:
    pass

from PyQt5.QtWidgets import QApplication, QMessageBox  # noqa: E402
from PyQt5.QtCore import Qt  # noqa: E402
from PyQt5.QtGui import QPalette, QColor, QFont, QFontDatabase, QIcon  # noqa: E402

from app.main_window import MainWindow  # noqa: E402


def _load_bundled_font() -> str:
    """Register Pretendard OTFs from assets/fonts/ and return the family name.

    Fonts are downloaded by setup.bat (SIL Open Font License). If absent or
    blocked, returns "" and the app falls back to the system default.
    """
    fonts_dir = os.path.join(ROOT, "assets", "fonts")
    if not os.path.isdir(fonts_dir):
        return ""
    family = ""
    for name in ("Pretendard-Regular.otf", "Pretendard-Medium.otf",
                 "Pretendard-SemiBold.otf", "Pretendard-Bold.otf"):
        p = os.path.join(fonts_dir, name)
        if not os.path.exists(p):
            continue
        fid = QFontDatabase.addApplicationFont(p)
        if fid < 0:
            print(f"[CEC All-In-One] Failed to register font: {name}")
            continue
        fams = QFontDatabase.applicationFontFamilies(fid)
        if fams and not family:
            family = fams[0]
    return family


def _apply_dark_palette(app: QApplication) -> None:
    p = QPalette()
    p.setColor(QPalette.Window, QColor(30, 30, 46))
    p.setColor(QPalette.WindowText, QColor(205, 214, 244))
    p.setColor(QPalette.Base, QColor(24, 24, 37))
    p.setColor(QPalette.AlternateBase, QColor(30, 30, 46))
    p.setColor(QPalette.ToolTipBase, QColor(49, 50, 68))
    p.setColor(QPalette.ToolTipText, QColor(205, 214, 244))
    p.setColor(QPalette.Text, QColor(205, 214, 244))
    p.setColor(QPalette.Button, QColor(49, 50, 68))
    p.setColor(QPalette.ButtonText, QColor(205, 214, 244))
    p.setColor(QPalette.BrightText, QColor(243, 139, 168))
    p.setColor(QPalette.Link, QColor(137, 180, 250))
    p.setColor(QPalette.Highlight, QColor(137, 180, 250))
    p.setColor(QPalette.HighlightedText, QColor(30, 30, 46))
    app.setPalette(p)


def _qt_excepthook(exc_type, exc, tb):
    text = "".join(traceback.format_exception(exc_type, exc, tb))
    sys.stderr.write(text)
    sys.stderr.flush()
    app = QApplication.instance()
    if app is not None:
        try:
            QMessageBox.critical(None, "CEC All-In-One — 오류",
                                 f"{exc_type.__name__}: {exc}\n\n로그: {LOG_PATH}")
            return
        except Exception:
            pass
    _show_error_box("CEC All-In-One — 오류",
                    f"{exc_type.__name__}: {exc}\n\n로그: {LOG_PATH}")


def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setApplicationName("CEC All-In-One")
    app.setStyle("Fusion")
    _apply_dark_palette(app)

    # Stop the mouse wheel from silently bumping spinbox / combobox /
    # slider values while the user is scrolling the parameter panel.
    # See app/wheel_guard.py for the rationale.
    from app.wheel_guard import install as _install_wheel_guard
    _install_wheel_guard(app)

    # App icon (taskbar / window decoration / Alt-Tab)
    icon_path = os.path.join(ROOT, "assets", "logo", "aimz_icon_v02.png")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    # Bundled Pretendard becomes the app-wide default if available.
    family = _load_bundled_font()
    if family:
        app.setFont(QFont(family, 9))
        app.setStyleSheet(
            f"* {{ font-family: '{family}', 'Segoe UI', sans-serif; "
            f"font-size: 9pt; }}")
        print(f"[CEC All-In-One] Loaded font family: {family}")
    else:
        print("[CEC All-In-One] Pretendard not found, using system default")

    sys.excepthook = _qt_excepthook

    if "--self-test" in sys.argv:
        print("[CEC All-In-One] self-test: GUI bootable")
        return 0

    window = MainWindow()
    window.log_path = LOG_PATH
    window.show()
    return app.exec_()


if __name__ == "__main__":
    rc = main()
    _log_file.write(f"=== CEC All-In-One exit rc={rc} ===\n")
    _log_file.flush()
    os._exit(rc)

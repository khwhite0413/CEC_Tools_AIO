"""Canonical version string for CEC All-In-One.

Single source of truth. update_checker compares this against the latest
release tag reported by the Cloudflare Worker proxy, so this MUST match
whatever git tag was pushed for the current build.

Bumping process (on release):
    1. Edit __version__ below (e.g. "1.0.7" -> "1.0.8")
    2. main_window.py / about_page.py / chatbot_panel.py already pull
       from here — no per-file bumps needed.
    3. README.md still hardcodes the number (markdown can't import) —
       hand-edit that one string.
    4. git commit + git tag v1.0.8 + git push origin main --tags
"""

__version__ = "1.0.9"

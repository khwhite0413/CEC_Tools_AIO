from pathlib import Path

import pytest

from app import updater


def test_kick_uses_downloaded_updater_copy(monkeypatch, tmp_path):
    project_root = tmp_path / "installed"
    source_root = tmp_path / "downloaded"
    helper_root = tmp_path / "helper"
    project_root.mkdir()
    source_root.mkdir()
    helper_root.mkdir()

    (project_root / "updater.bat").write_bytes(b"installed-old")
    (source_root / "updater.bat").write_bytes(b"downloaded-new")

    popen_calls = []
    monkeypatch.setattr(updater, "_PROJECT_ROOT", str(project_root))
    monkeypatch.setattr(updater.tempfile, "mkdtemp",
                        lambda prefix: str(helper_root))
    monkeypatch.setattr(updater.os, "getpid", lambda: 4321)
    monkeypatch.setattr(updater.subprocess, "Popen",
                        lambda *args, **kwargs: popen_calls.append(
                            (args, kwargs)))

    updater.kick_updater_bat(str(source_root))

    helper_bat = helper_root / "updater.bat"
    assert helper_bat.read_bytes() == b"downloaded-new"
    assert popen_calls[0][0][0][4] == str(helper_bat)
    assert popen_calls[0][1]["cwd"] == str(project_root)


def test_kick_rejects_release_without_updater(monkeypatch, tmp_path):
    project_root = tmp_path / "installed"
    source_root = tmp_path / "downloaded"
    project_root.mkdir()
    source_root.mkdir()
    monkeypatch.setattr(updater, "_PROJECT_ROOT", str(project_root))

    with pytest.raises(RuntimeError, match="다운로드한 릴리즈"):
        updater.kick_updater_bat(str(source_root))

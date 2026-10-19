from pathlib import Path

import pytest

from app.updater_files import normalize_root_batch_files


def test_normalize_root_batch_files_is_atomic_and_skips_updater(tmp_path):
    setup = tmp_path / "setup.bat"
    run = tmp_path / "run.bat"
    updater = tmp_path / "updater.bat"
    nested = tmp_path / "nested"
    nested.mkdir()
    nested_batch = nested / "nested.bat"

    setup.write_bytes(b"@echo off\necho setup\n")
    run.write_bytes(b"@echo off\r\necho run\n")
    updater_bytes = b"@echo off\necho updater\n"
    updater.write_bytes(updater_bytes)
    nested_batch.write_bytes(b"@echo off\necho nested\n")

    changed = normalize_root_batch_files(tmp_path)

    assert changed == (setup, run)
    assert setup.read_bytes() == b"@echo off\r\necho setup\r\n"
    assert run.read_bytes() == b"@echo off\r\necho run\r\n"
    assert updater.read_bytes() == updater_bytes
    assert nested_batch.read_bytes() == b"@echo off\necho nested\n"
    assert not tuple(tmp_path.glob(".*.tmp"))


def test_normalize_failure_preserves_original(monkeypatch, tmp_path):
    setup = tmp_path / "setup.bat"
    original = b"@echo off\necho setup\n"
    setup.write_bytes(original)

    def fail_replace(_source: Path, _destination: Path) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr("app.updater_files.os.replace", fail_replace)

    with pytest.raises(OSError, match="simulated replace failure"):
        normalize_root_batch_files(tmp_path)

    assert setup.read_bytes() == original
    assert not tuple(tmp_path.glob(".*.tmp"))


def test_normalize_accepts_quote_in_install_path(tmp_path):
    root = tmp_path / "Bob's Apps" / "CEC AIO"
    root.mkdir(parents=True)
    setup = root / "setup.bat"
    setup.write_bytes(b"@echo off\necho setup\n")

    normalize_root_batch_files(root)

    assert setup.read_bytes() == b"@echo off\r\necho setup\r\n"

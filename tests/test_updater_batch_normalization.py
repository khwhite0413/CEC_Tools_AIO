from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPDATER_BAT = PROJECT_ROOT / "updater.bat"


def _normalizer_code() -> str:
    marker = '"%DST%\\python\\python.exe" -c "'
    suffix = '" >> "%LOG%" 2>&1'

    for line in UPDATER_BAT.read_text(encoding="utf-8").splitlines():
        if line.startswith(marker) and line.endswith(suffix):
            return line[len(marker) : -len(suffix)]

    raise AssertionError("updater.bat normalization command was not found")


def test_batch_normalizer_preserves_content_and_skips_running_updater(tmp_path):
    setup = tmp_path / "setup.bat"
    run = tmp_path / "run.bat"
    updater = tmp_path / "updater.bat"
    untouched = tmp_path / "notes.txt"

    setup.write_bytes(b"@echo off\necho setup\n")
    run.write_bytes(b"@echo off\r\necho run\n")
    updater_bytes = b"@echo off\necho updater stays unchanged\n"
    updater.write_bytes(updater_bytes)
    untouched.write_bytes(b"keep\nthis\n")

    code = _normalizer_code().replace("%DST%", tmp_path.as_posix())
    exec(code, {})

    assert setup.read_bytes() == b"@echo off\r\necho setup\r\n"
    assert run.read_bytes() == b"@echo off\r\necho run\r\n"
    assert updater.read_bytes() == updater_bytes
    assert untouched.read_bytes() == b"keep\nthis\n"


def test_updater_sets_utf8_console_before_korean_output():
    lines = UPDATER_BAT.read_text(encoding="utf-8").splitlines()

    assert any(line.strip().lower().startswith("chcp 65001")
               for line in lines[:5])

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPDATER_BAT = PROJECT_ROOT / "updater.bat"


def test_updater_invokes_atomic_batch_normalizer():
    text = UPDATER_BAT.read_text(encoding="utf-8")

    assert ('"%DST%\\python\\python.exe" '
            '"%DST%\\app\\updater_files.py" "%DST%"') in text
    assert "open(f,'wb').write(open(f,'rb').read()" not in text


def test_updater_sets_utf8_console_before_korean_output():
    lines = UPDATER_BAT.read_text(encoding="utf-8").splitlines()

    assert any(line.strip().lower().startswith("chcp 65001")
               for line in lines[:5])

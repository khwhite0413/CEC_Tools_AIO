"""File operations used by the detached Windows updater."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile


_EXCLUDED_BATCH_NAMES = frozenset({"updater.bat"})


def _to_crlf(data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")


def normalize_root_batch_files(root: str | Path) -> tuple[Path, ...]:
    """Atomically normalize root-level batch files to CRLF.

    The installed ``updater.bat`` is excluded because older installations
    may still invoke that copy directly while it is being read by cmd.exe.
    """
    root_path = Path(root)
    changed: list[Path] = []

    for path in sorted(root_path.glob("*.bat"),
                       key=lambda item: item.name.casefold()):
        if path.name.casefold() in _EXCLUDED_BATCH_NAMES:
            continue

        original = path.read_bytes()
        normalized = _to_crlf(original)
        if normalized == original:
            continue

        fd, temp_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
        temp_path = Path(temp_name)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(normalized)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_path, path)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

        changed.append(path)

    return tuple(changed)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: updater_files.py <install-root>", file=sys.stderr)
        return 2

    changed = normalize_root_batch_files(args[0])
    print(f"Normalized {len(changed)} batch file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

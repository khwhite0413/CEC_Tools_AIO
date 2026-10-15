"""Pinned, repeatable installer for frame-interpolation sources and weights."""

import os
import hashlib
import shutil
import tempfile
import urllib.request
import zipfile


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
THIRD_PARTY = os.path.join(ROOT, "third_party")
WEIGHTS = os.path.join(ROOT, "weights", "interpolation")

SOURCES = (
    ("vfi_fannovel", "Fannovel16/ComfyUI-Frame-Interpolation",
     "26545cc2dd95bc3d27f056016300673bdeee78f5",
     "18febf83edca39068a7947050672f056c8b70aaf0fcbb2380a4f0d88d90346b0"),
    ("vfi_tween", "ethanfel/ComfyUI-Tween",
     "15b88d16898c14aa0812672f86186a82f4769594",
     "5289c0500be4a34b6796653708c9da26559cf125f399cf1c8b3a64c2547e0f24"),
)

DIRECT_WEIGHTS = (
    ("rife426.pth", "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation/releases/download/models/rife426.pth",
     "606421fe2148a9fdeca14e58d94dc339e87b87e0ebcc5dec84a50d6f488cfe7b"),
    ("atm-vfi-lite.pt", "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation/releases/download/models/atm-vfi-lite.pt",
     "451d8fde75b7994f49f9ee31dd7ff3e2d973962dc4f76c44f4af27fc2b6b3b05"),
)

DRIVE_WEIGHTS = (
    ("ours_t.pkl", "1mL4ht5DYFMA1CYiGDFbej-ufh5QoqfwN",
     "c09e82d193e303d7b91f1e36bf535886b820cb56105574355211d60a6b557a99"),
    ("ours-1-2-points.pkl", "11LlBA8gbLGuXrpNGwfLX9UsJiYAT378A",
     "58c239cbfeabcbaa6921df58865f98926abca12afd37471d77984573f7612803"),
)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_atomic(url, destination, minimum_size=1_000_000, sha256=None):
    if (os.path.isfile(destination)
            and os.path.getsize(destination) >= minimum_size
            and (sha256 is None or _sha256(destination) == sha256)):
        return
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    part = destination + ".part"
    try:
        urllib.request.urlretrieve(url, part)
        if os.path.getsize(part) < minimum_size:
            raise RuntimeError(f"다운로드 파일이 너무 작습니다: {destination}")
        if sha256 is not None and _sha256(part) != sha256:
            raise RuntimeError(f"SHA-256 검증 실패: {destination}")
        os.replace(part, destination)
    finally:
        if os.path.exists(part):
            os.remove(part)


def _install_source(name, repository, commit, sha256, progress):
    destination = os.path.join(THIRD_PARTY, name)
    marker = os.path.join(destination, ".cec_commit")
    if os.path.isfile(marker):
        with open(marker, "r", encoding="utf-8") as handle:
            if handle.read().strip() == commit:
                progress(f"  {name}: source OK")
                return
    url = f"https://codeload.github.com/{repository}/zip/{commit}"
    with tempfile.TemporaryDirectory(prefix="cec_vfi_") as temp:
        archive = os.path.join(temp, "source.zip")
        _download_atomic(url, archive, minimum_size=100_000, sha256=sha256)
        extract = os.path.join(temp, "extract")
        with zipfile.ZipFile(archive) as zf:
            extract_root = os.path.abspath(extract) + os.sep
            for member in zf.infolist():
                target = os.path.abspath(os.path.join(extract, member.filename))
                if not target.startswith(extract_root):
                    raise RuntimeError(
                        f"{name} 소스 압축에 안전하지 않은 경로가 있습니다.")
            zf.extractall(extract)
        roots = [os.path.join(extract, item) for item in os.listdir(extract)]
        roots = [item for item in roots if os.path.isdir(item)]
        if len(roots) != 1:
            raise RuntimeError(f"{name} 소스 압축 구조가 예상과 다릅니다.")
        if os.path.isdir(destination):
            shutil.rmtree(destination)
        shutil.copytree(roots[0], destination)
    with open(marker, "w", encoding="utf-8") as handle:
        handle.write(commit + "\n")
    progress(f"  {name}: source installed ({commit[:10]})")


def _download_drive_weight(filename, file_id, sha256, progress):
    destination = os.path.join(WEIGHTS, filename)
    if (os.path.isfile(destination) and os.path.getsize(destination) > 1_000_000
            and _sha256(destination) == sha256):
        progress(f"  {filename}: weight OK")
        return
    import gdown
    os.makedirs(WEIGHTS, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cec_vfi_gdrive_", dir=WEIGHTS) as temp:
        match = os.path.join(temp, filename)
        result = gdown.download(id=file_id, output=match, quiet=False,
                               use_cookies=False)
        if not result or not os.path.isfile(match):
            raise RuntimeError(f"Google Drive에서 {filename}을 받지 못했습니다.")
        if os.path.getsize(match) <= 1_000_000:
            raise RuntimeError(f"가중치 파일이 너무 작습니다: {filename}")
        if _sha256(match) != sha256:
            raise RuntimeError(f"SHA-256 검증 실패: {filename}")
        os.replace(match, destination)
    progress(f"  {filename}: weight installed")


def install_interpolation_assets(progress=print):
    os.makedirs(THIRD_PARTY, exist_ok=True)
    os.makedirs(WEIGHTS, exist_ok=True)
    for source in SOURCES:
        _install_source(*source, progress)
    # Tween declares Apache-2.0 in pyproject/README but its snapshot omits
    # the standard license text. Keep a copy beside the redistributed code.
    tween_license = os.path.join(THIRD_PARTY, "vfi_tween", "LICENSE")
    _download_atomic("https://www.apache.org/licenses/LICENSE-2.0.txt",
                     tween_license, minimum_size=5_000)
    for filename, url, sha256 in DIRECT_WEIGHTS:
        _download_atomic(url, os.path.join(WEIGHTS, filename), sha256=sha256)
        progress(f"  {filename}: weight OK")
    for filename, file_id, sha256 in DRIVE_WEIGHTS:
        _download_drive_weight(filename, file_id, sha256, progress)


if __name__ == "__main__":
    install_interpolation_assets()

"""Download and validate the official FastDVDnet inference checkpoint."""

from __future__ import annotations

import hashlib
import os
import urllib.request


_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WEIGHT_DIR = os.path.join(_ROOT, "weights", "fastdvdnet")
WEIGHT_PATH = os.path.join(WEIGHT_DIR, "model.pth")
MODEL_URL = (
    "https://raw.githubusercontent.com/m-tassano/fastdvdnet/"
    "c8fdf6182a0340e89dd18f5df25b47337cbede6f/model.pth"
)
MODEL_SHA256 = "9d9d8413c33e3d9d961d07c530237befa1197610b9d60602ff42fd77975d2a17"


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def weights_available() -> bool:
    return os.path.isfile(WEIGHT_PATH) and _sha256(WEIGHT_PATH) == MODEL_SHA256


def download_fastdvdnet_weights() -> str:
    """Download atomically; never retain an incomplete/corrupt checkpoint."""
    if weights_available():
        print(f"    FastDVDnet weights OK: {WEIGHT_PATH}")
        return WEIGHT_PATH
    os.makedirs(WEIGHT_DIR, exist_ok=True)
    partial = WEIGHT_PATH + ".part"
    try:
        if os.path.exists(partial):
            os.remove(partial)
        print("    Downloading FastDVDnet model.pth (~10 MB)...")
        urllib.request.urlretrieve(MODEL_URL, partial)
        actual = _sha256(partial)
        if actual != MODEL_SHA256:
            raise RuntimeError(
                "FastDVDnet checkpoint checksum mismatch: "
                f"expected {MODEL_SHA256}, got {actual}")
        os.replace(partial, WEIGHT_PATH)
        return WEIGHT_PATH
    finally:
        if os.path.exists(partial):
            try:
                os.remove(partial)
            except OSError:
                pass


if __name__ == "__main__":
    download_fastdvdnet_weights()

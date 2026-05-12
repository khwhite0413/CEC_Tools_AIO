import os
import requests
from vdcore.config import WEIGHTS_DIR, WEIGHT_URLS


def download_file(url, dest_path, progress_callback=None):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    downloaded = 0

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if progress_callback and total > 0:
                pct = int(downloaded / total * 100)
                name = os.path.basename(dest_path)
                progress_callback(f"다운로드 중: {name} ({pct}%)")


def download_all_weights(progress_callback=None):
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    for name, url in WEIGHT_URLS.items():
        dest = os.path.join(WEIGHTS_DIR, name)
        if os.path.isfile(dest):
            if progress_callback:
                progress_callback(f"이미 존재: {name}")
            continue
        if progress_callback:
            progress_callback(f"다운로드 시작: {name}")
        download_file(url, dest, progress_callback)
    if progress_callback:
        progress_callback("모든 가중치 다운로드 완료")

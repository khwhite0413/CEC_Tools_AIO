import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
THIRD_PARTY_DIR = os.path.join(BASE_DIR, "third_party")
PROPAINTER_DIR = os.path.join(THIRD_PARTY_DIR, "ProPainter")

WEIGHT_URLS = {
    "ProPainter.pth": "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/ProPainter.pth",
    "recurrent_flow_completion.pth": "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/recurrent_flow_completion.pth",
    "raft-things.pth": "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/raft-things.pth",
}

DEFAULT_SETTINGS = {
    "fp16": True,
    "subvideo_length": 80,
    "neighbor_length": 10,
    "ref_stride": 10,
    "raft_iter": 20,
    "mask_dilation": 4,
    "max_batch_frames": 0,  # 0=자동 안전 청크(80f). 양수=최대 윈도우 크기
}

VIDEO_EXTENSIONS = "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm)"


def get_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def weights_available():
    for name in WEIGHT_URLS:
        if not os.path.isfile(os.path.join(WEIGHTS_DIR, name)):
            return False
    return True

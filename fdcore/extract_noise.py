"""Extract and visualize noise (chroma / luma) between original and clean."""

import torch

from .color_spaces import to_color_space


def _turbo_colormap(t: torch.Tensor) -> torch.Tensor:
    """Piecewise turbo-like colormap. Input (B,1,H,W) or (B,H,W,1).
    Returns BHWC RGB [0,1]."""
    if t.dim() == 4 and t.shape[1] == 1:
        t = t.permute(0, 2, 3, 1)
    t = t.squeeze(-1).clamp(0, 1)
    r = (1.5 - torch.abs(t * 4.0 - 3.0)).clamp(0, 1)
    g = (1.5 - torch.abs(t * 4.0 - 2.0)).clamp(0, 1)
    b = (1.5 - torch.abs(t * 4.0 - 1.0)).clamp(0, 1)
    return torch.stack([r, g, b], dim=-1)


def visualize_noise(noise_bhwc: torch.Tensor, amplify: float, mode: str) -> torch.Tensor:
    if mode == "gray":
        return (noise_bhwc * amplify + 0.5).clamp(0, 1)
    if mode == "heatmap":
        mag = torch.mean(noise_bhwc.abs(), dim=-1, keepdim=True)
        for i in range(mag.shape[0]):
            fmax = mag[i].max()
            if fmax > 1e-8:
                mag[i] = mag[i] / fmax
        return _turbo_colormap(mag)
    if mode == "signed":
        lum = torch.mean(noise_bhwc, dim=-1)
        for i in range(lum.shape[0]):
            amax = lum[i].abs().max()
            if amax > 1e-8:
                lum[i] = lum[i] / amax
        pos = lum.clamp(min=0)
        neg = (-lum).clamp(min=0)
        return torch.stack([pos, torch.zeros_like(lum), neg], dim=-1).clamp(0, 1)
    return (noise_bhwc * amplify + 0.5).clamp(0, 1)


def extract_noise(original: torch.Tensor, clean: torch.Tensor, settings: dict):
    """Returns (noise_total_vis, noise_chroma_vis, noise_luma_vis), all BHWC [0,1]."""
    color_space = settings.get("noise_color_space", "YCbCr")
    amplify = float(settings.get("noise_amplify", 5.0))
    preview = settings.get("noise_preview", "heatmap")

    noise_rgb = original - clean

    orig_bchw = original.permute(0, 3, 1, 2)
    clean_bchw = clean.permute(0, 3, 1, 2)
    orig_cs = to_color_space(orig_bchw, color_space)
    clean_cs = to_color_space(clean_bchw, color_space)

    noise_cs = orig_cs - clean_cs
    noise_luma_cs = torch.zeros_like(noise_cs)
    noise_luma_cs[:, 0:1] = noise_cs[:, 0:1]
    noise_chroma_cs = torch.zeros_like(noise_cs)
    noise_chroma_cs[:, 1:] = noise_cs[:, 1:]

    noise_total = visualize_noise(noise_rgb, amplify, preview)
    noise_luma_mono = noise_luma_cs[:, 0:1].expand(-1, 3, -1, -1).permute(0, 2, 3, 1)
    noise_luma_vis = visualize_noise(noise_luma_mono, amplify, preview)
    noise_chroma_bhwc = noise_chroma_cs.permute(0, 2, 3, 1)
    noise_chroma_vis = visualize_noise(noise_chroma_bhwc, amplify, preview)

    return noise_total, noise_chroma_vis, noise_luma_vis

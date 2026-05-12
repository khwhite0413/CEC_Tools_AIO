"""RGB <-> YCbCr / HSV / LAB conversions on BCHW tensors in [0,1].

Lifted from ComfyUI-FlowDenoise/nodes.py — pure tensor math, no ComfyUI imports.
"""

import torch


def _rgb_to_hsv(rgb: torch.Tensor) -> torch.Tensor:
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    maxc, _ = rgb.max(dim=1, keepdim=True)
    minc, _ = rgb.min(dim=1, keepdim=True)
    diff = maxc - minc

    s = torch.where(maxc > 0, diff / (maxc + 1e-8), torch.zeros_like(maxc))
    v = maxc

    h = torch.zeros_like(maxc)
    mask_r = (maxc == r) & (diff > 0)
    mask_g = (maxc == g) & (diff > 0)
    mask_b = (maxc == b) & (diff > 0)
    h[mask_r] = (((g - b) / (diff + 1e-8)) % 6)[mask_r]
    h[mask_g] = (((b - r) / (diff + 1e-8)) + 2)[mask_g]
    h[mask_b] = (((r - g) / (diff + 1e-8)) + 4)[mask_b]
    h = h / 6.0

    return torch.cat([h, s, v], dim=1)


def _rgb_to_ycbcr(rgb: torch.Tensor) -> torch.Tensor:
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 0.5 + (-0.168736 * r - 0.331264 * g + 0.5 * b)
    cr = 0.5 + (0.5 * r - 0.418688 * g - 0.081312 * b)
    return torch.cat([y, cb, cr], dim=1)


def _rgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    linear = torch.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    r, g, b = linear[:, 0:1], linear[:, 1:2], linear[:, 2:3]
    x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b
    x = x / 0.95047
    z = z / 1.08883

    epsilon = 0.008856
    kappa = 903.3

    def f(t):
        return torch.where(t > epsilon, t ** (1.0 / 3.0), (kappa * t + 16.0) / 116.0)

    fx, fy, fz = f(x), f(y), f(z)
    l_star = 116.0 * fy - 16.0
    a_star = 500.0 * (fx - fy)
    b_star = 200.0 * (fy - fz)

    return torch.cat([l_star / 100.0,
                      (a_star + 128.0) / 256.0,
                      (b_star + 128.0) / 256.0], dim=1)


def _ycbcr_to_rgb(ycbcr: torch.Tensor) -> torch.Tensor:
    y, cb, cr = ycbcr[:, 0:1], ycbcr[:, 1:2], ycbcr[:, 2:3]
    r = y + 1.402 * (cr - 0.5)
    g = y - 0.344136 * (cb - 0.5) - 0.714136 * (cr - 0.5)
    b = y + 1.772 * (cb - 0.5)
    return torch.cat([r, g, b], dim=1)


def _hsv_to_rgb(hsv: torch.Tensor) -> torch.Tensor:
    h, s, v = hsv[:, 0:1], hsv[:, 1:2], hsv[:, 2:3]
    h6 = h * 6.0
    i = torch.floor(h6)
    f = h6 - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i_mod = i.long() % 6

    r = torch.where(i_mod == 0, v, torch.where(i_mod == 1, q, torch.where(
        i_mod == 2, p, torch.where(i_mod == 3, p, torch.where(i_mod == 4, t, v)))))
    g = torch.where(i_mod == 0, t, torch.where(i_mod == 1, v, torch.where(
        i_mod == 2, v, torch.where(i_mod == 3, q, torch.where(i_mod == 4, p, p)))))
    b = torch.where(i_mod == 0, p, torch.where(i_mod == 1, p, torch.where(
        i_mod == 2, t, torch.where(i_mod == 3, v, torch.where(i_mod == 4, v, q)))))
    return torch.cat([r, g, b], dim=1)


def _lab_to_rgb(lab: torch.Tensor) -> torch.Tensor:
    l_star = lab[:, 0:1] * 100.0
    a_star = lab[:, 1:2] * 256.0 - 128.0
    b_star = lab[:, 2:3] * 256.0 - 128.0

    fy = (l_star + 16.0) / 116.0
    fx = a_star / 500.0 + fy
    fz = fy - b_star / 200.0

    epsilon = 0.008856
    kappa = 903.3

    def f_inv(t):
        t3 = t * t * t
        return torch.where(t3 > epsilon, t3, (116.0 * t - 16.0) / kappa)

    x = f_inv(fx) * 0.95047
    y = f_inv(fy)
    z = f_inv(fz) * 1.08883

    r_lin = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z
    g_lin = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z
    b_lin = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z

    linear = torch.cat([r_lin, g_lin, b_lin], dim=1).clamp(min=0)
    srgb = torch.where(linear > 0.0031308,
                       1.055 * linear.clamp(min=1e-10) ** (1.0 / 2.4) - 0.055,
                       12.92 * linear)
    return srgb.clamp(0, 1)


def to_color_space(rgb_bchw: torch.Tensor, space: str) -> torch.Tensor:
    if space == "HSV":
        return _rgb_to_hsv(rgb_bchw)
    if space == "YCbCr":
        return _rgb_to_ycbcr(rgb_bchw)
    if space == "LAB":
        return _rgb_to_lab(rgb_bchw)
    raise ValueError(f"Unknown color space: {space}")


def from_color_space(cs_bchw: torch.Tensor, space: str) -> torch.Tensor:
    if space == "HSV":
        return _hsv_to_rgb(cs_bchw)
    if space == "YCbCr":
        return _ycbcr_to_rgb(cs_bchw)
    if space == "LAB":
        return _lab_to_rgb(cs_bchw)
    raise ValueError(f"Unknown color space: {space}")

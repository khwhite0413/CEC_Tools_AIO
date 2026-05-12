"""Flow-based warping and consistency masks. Lifted from nodes.py."""

import torch
import torch.nn.functional as F


def warp_with_flow(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Warp img (B,C,H,W) using flow (B,2,H,W). flow[b,0]=dx, flow[b,1]=dy in pixels."""
    B, C, H, W = img.shape
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=img.device, dtype=img.dtype),
        torch.arange(W, device=img.device, dtype=img.dtype),
        indexing='ij'
    )
    grid_x = grid_x.unsqueeze(0).expand(B, -1, -1)
    grid_y = grid_y.unsqueeze(0).expand(B, -1, -1)

    new_x = grid_x + flow[:, 0]
    new_y = grid_y + flow[:, 1]
    new_x = 2.0 * new_x / (W - 1) - 1.0
    new_y = 2.0 * new_y / (H - 1) - 1.0

    grid = torch.stack([new_x, new_y], dim=-1)
    return F.grid_sample(img, grid, mode='bilinear', padding_mode='border',
                         align_corners=True)


def color_similarity_mask(warped: torch.Tensor, ref: torch.Tensor,
                          threshold: float) -> torch.Tensor:
    """Soft color similarity mask. Returns (B,1,H,W) weights in [0,1]."""
    color_diff = torch.mean((warped - ref) ** 2, dim=1, keepdim=True)
    mask = torch.exp(-color_diff / (2.0 * threshold ** 2))
    return F.avg_pool2d(mask, kernel_size=5, stride=1, padding=2)


def forward_backward_occlusion_mask(flow_fwd: torch.Tensor, flow_bwd: torch.Tensor,
                                    threshold: float = 1.0) -> torch.Tensor:
    """Forward-backward consistency check. 1=consistent, 0=occluded."""
    bwd_at_fwd = warp_with_flow(flow_bwd, flow_fwd)
    fb_diff = torch.sum((flow_fwd + bwd_at_fwd) ** 2, dim=1, keepdim=True)
    fb_norm = torch.sum(flow_fwd ** 2 + bwd_at_fwd ** 2, dim=1, keepdim=True)
    mask = (fb_diff < threshold ** 2 * (fb_norm + 0.5)).float()
    return F.avg_pool2d(mask, kernel_size=5, stride=1, padding=2)

"""Minimal FastDVDnet inference model.

Adapted from https://github.com/m-tassano/fastdvdnet at commit
c8fdf6182a0340e89dd18f5df25b47337cbede6f.
Copyright (c) 2024 Matias Tassano. Distributed under the MIT License.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.convblock(x)


class InputCvBlock(nn.Module):
    def __init__(self, num_in_frames, out_ch):
        super().__init__()
        interm_ch = 30
        self.convblock = nn.Sequential(
            nn.Conv2d(num_in_frames * 4, num_in_frames * interm_ch, 3,
                      padding=1, groups=num_in_frames, bias=False),
            nn.BatchNorm2d(num_in_frames * interm_ch), nn.ReLU(inplace=True),
            nn.Conv2d(num_in_frames * interm_ch, out_ch, 3,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.convblock(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            CvBlock(out_ch, out_ch),
        )

    def forward(self, x):
        return self.convblock(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.convblock = nn.Sequential(
            CvBlock(in_ch, in_ch),
            nn.Conv2d(in_ch, out_ch * 4, 3, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.convblock(x)


class OutputCvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        )

    def forward(self, x):
        return self.convblock(x)


class DenBlock(nn.Module):
    def __init__(self, num_input_frames=3):
        super().__init__()
        self.inc = InputCvBlock(num_input_frames, 32)
        self.downc0 = DownBlock(32, 64)
        self.downc1 = DownBlock(64, 128)
        self.upc2 = UpBlock(128, 64)
        self.upc1 = UpBlock(64, 32)
        self.outc = OutputCvBlock(32, 3)

    def forward(self, in0, in1, in2, noise_map):
        x0 = self.inc(torch.cat(
            (in0, noise_map, in1, noise_map, in2, noise_map), dim=1))
        x1 = self.downc0(x0)
        x2 = self.downc1(x1)
        x2 = self.upc2(x2)
        x1 = self.upc1(x1 + x2)
        return in1 - self.outc(x0 + x1)


class FastDVDnet(nn.Module):
    def __init__(self, num_input_frames=5):
        super().__init__()
        self.num_input_frames = num_input_frames
        self.temp1 = DenBlock(3)
        self.temp2 = DenBlock(3)

    def forward(self, x, noise_map):
        frames = tuple(x[:, 3 * i:3 * i + 3]
                       for i in range(self.num_input_frames))
        x20 = self.temp1(frames[0], frames[1], frames[2], noise_map)
        x21 = self.temp1(frames[1], frames[2], frames[3], noise_map)
        x22 = self.temp1(frames[2], frames[3], frames[4], noise_map)
        return self.temp2(x20, x21, x22, noise_map)


def infer_frame(model, five_frames, sigma_map):
    """Infer one centered frame, padding spatial dimensions to /4."""
    height, width = five_frames.shape[-2:]
    pad_h = (-height) % 4
    pad_w = (-width) % 4
    if pad_h or pad_w:
        padding = (0, pad_w, 0, pad_h)
        five_frames = F.pad(five_frames, padding, mode="reflect")
        sigma_map = F.pad(sigma_map, padding, mode="reflect")
    output = model(five_frames, sigma_map).clamp_(0.0, 1.0)
    return output[..., :height, :width]

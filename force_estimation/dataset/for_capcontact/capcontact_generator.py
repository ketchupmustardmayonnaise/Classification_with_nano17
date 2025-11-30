import torch
from torch import nn
import math


class Generator(nn.Module):
    """
        Generator for 8x cross-modality super-resolution
    """

    def __init__(self):
        super(Generator, self).__init__()

        self.in_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=9, stride=1, padding=4),
            nn.PReLU()
        )

        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(channels=64) for _ in range(5)],
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )

        self.upscaling_blocks = nn.Sequential(
            *[UpscalingBlock(64, upscale_factor=2) for _ in range(3)]
        )

        self.out_layer = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=9, padding=4),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.in_layer(x)
        x = self.residual_blocks(x) + x
        x = self.upscaling_blocks(x)
        return self.out_layer(x)


class UpscalingBlock(nn.Module):

    def __init__(self, channels, upscale_factor=2):
        super(UpscalingBlock, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels * (upscale_factor ** 2), kernel_size=3, stride=1,
                      padding=1),
            nn.PixelShuffle(upscale_factor),
            nn.PReLU(),
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return x + self.net(x)
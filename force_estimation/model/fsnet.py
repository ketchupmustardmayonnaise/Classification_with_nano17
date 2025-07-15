import torch
import torch.nn as nn

class FSNet(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        for i in range(5):
            x = self.block(x)

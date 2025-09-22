from sys import exception


from model.lstm import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut Connection을 위한 1x1 Conv 레이어
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # Shortcut Connection 적용
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class CFModel(nn.Module):
    def __init__(self, decoder_layer, decoder_dimension, decoder_output):

        super().__init__()

        # encoder: convNext
        self.conv = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1, stride=1)
        self.resblock1 = ResBlock(32, 32)
        self.resblock2 = ResBlock(32, 64)
        self.fc_layer1 = nn.Linear(64 * 32 * 32, 128)

        self.fc_layer2 = nn.Linear(256, 128)
        self.fc_layer3 = nn.Linear(128, 3)

    def forward(self, input):
        # input.shape = (batch, 2, height, width)
        # x.shape = (batch, 2, 1, height, width)
        x = input.unsqueeze(2)
        b, t, c, h, w = x.shape
        
        x1 = self.conv(x[:, 0])
        x1 = self.resblock1(x1)
        x1 = self.resblock2(x1)
        x1 = torch.flatten(x1, start_dim=1)
        x1 = self.fc_layer1(x1)

        x2 = self.conv(x[:, 0])
        x2 = self.resblock1(x2)
        x2 = self.resblock2(x2)
        x2 = torch.flatten(x2, start_dim=1)
        x2 = self.fc_layer1(x2)

        x = torch.cat((x1, x2), dim=1)
        x = self.fc_layer2(x)
        x = self.fc_layer3(x)

        return x
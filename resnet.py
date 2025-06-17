import torch
from torch import nn

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, inner_channels, stride=1, projection=None):
        super().__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(),
            nn.Conv2d(inner_channels, inner_channels * self.expansion, 3, padding=1, bias=False),
            nn.BatchNorm2d(inner_channels * self.expansion),
        )

        self.projection = projection
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = self.residual(x)

        if self.projection is not None:
            skip_connection = self.projection(x)
        else:
            skip_connection = x

        out = self.relu(residual + skip_connection)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_block_list, n_classes=1000):
        super().__init__()
        assert len(num_block_list) == 4

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.in_channels = 1
        self.stage1 = self.make_stage(block, 32, num_block_list[0], stride=1)
        self.stage2 = self.make_stage(block, 64, num_block_list[1], stride=2)
        self.stage3 = self.make_stage(block, 128, num_block_list[2], stride=2)
        self.stage4 = self.make_stage(block, 256, num_block_list[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256 * 2, 125)

    def make_stage(self, block, inner_channels, num_blocks, stride=1):
        if stride != 1 or self.in_channels != inner_channels * block.expansion:
            projection = nn.Sequential(
                nn.Conv2d(self.in_channels, inner_channels * block.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(inner_channels * block.expansion),
            )
        else:
            projection = None

        layers = []
        for idx in range(num_blocks):
            if idx == 0:
                layers.append(block(self.in_channels, inner_channels, stride, projection))
                self.in_channels = inner_channels * block.expansion
            else:
                layers.append(block(self.in_channels, inner_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.maxpool(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        # x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def resnet_learning(**kwargs):
    return ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
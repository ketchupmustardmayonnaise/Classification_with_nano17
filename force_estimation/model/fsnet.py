import torch
import torch.nn as nn

class FSNet(nn.Module):
    def __init__(self, n_classes=32, image_type='origin'):
        super().__init__()
        self.encoder_residual = []
        self.decoder_residual = []
        self.in_channels = 3 if image_type.lower() == 'rgb' else 1

        self.encoder_block1 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.encoder_block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.encoder_block3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.encoder_block4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.t_conv1 = nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.t_conv2 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.t_conv3 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1,
                                          output_padding=1)
        self.t_conv4 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1,
                                          output_padding=1)

        self.decoder_block1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.decoder_block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.decoder_block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.decoder_block4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(in_features=256, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=n_classes)

    def forward(self, x):
        # encoder
        x = self.encoder_block1(x)
        self.encoder_residual.append(x)
        x = self.maxpool(x)

        x = self.encoder_block2(x)
        self.encoder_residual.append(x)
        x = self.maxpool(x)

        x = self.encoder_block3(x)
        self.encoder_residual.append(x)
        x = self.maxpool(x)

        x = self.encoder_block4(x)
        self.encoder_residual.append(x)
        x = self.maxpool(x)

        x = self.bottleneck(x)
        x_copy = x.clone()

        # decoder
        x = self.t_conv1(x)
        x = torch.cat((x, self.encoder_residual[3]), dim=1)
        x = self.decoder_block1(x)
        self.decoder_residual.append(x)

        x = self.t_conv2(x)
        x = torch.cat((x, self.encoder_residual[2]), dim=1)
        x = self.decoder_block2(x)
        self.decoder_residual.append(x)

        x = self.t_conv3(x)
        x = torch.cat((x, self.encoder_residual[1]), dim=1)
        x = self.decoder_block3(x)
        self.decoder_residual.append(x)

        x = self.t_conv4(x)
        x = torch.cat((x, self.encoder_residual[0]), dim=1)
        x = self.decoder_block4(x)
        self.decoder_residual.append(x)


        x_copy = self.gap(x_copy)
        x1 = self.gap(self.decoder_residual[0])
        x2 = self.gap(self.decoder_residual[1])
        x3 = self.gap(self.decoder_residual[2])
        x4 = self.gap(self.decoder_residual[3])
        x = torch.cat((x1, x2, x3, x4), dim=1)

        x_copy = torch.flatten(x_copy, 1)
        x = torch.flatten(x, 1)

        x = torch.cat((x_copy, x), dim=1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        self.encoder_residual = []
        self.decoder_residual = []

        return x

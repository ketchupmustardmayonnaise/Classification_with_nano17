from sys import exception


from model.lstm import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class CFModel(nn.Module):
    def __init__(self, decoder_layer, decoder_dimension, decoder_output):

        super().__init__()

        # encoder: convNext
        pretrained_model = models.convnext_tiny(pretrained=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        new_stem = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=3, stride=1, padding=1),
            # LayerNorm은 채널 수(96)와 이미지 크기(8x10)를 알아야 합니다.
            nn.LayerNorm([96, 10, 8]))
        pretrained_model.features[0] = new_stem
        self.encoder = nn.Sequential(
            pretrained_model.features[0],
            pretrained_model.features[1]
        )
        for param in self.encoder[1].parameters():
            param.requires_grad = False
        self.decoder_input_size = 96

        
        # Decoder
        self.decoder = LSTM(input_size=self.decoder_input_size, hidden_size=2, output_size=3, h_RNN_layers=128, bidirectional=True)


    def forward(self, input):
        # input.shape = (batch, target_index, channel, height, width)
        b, t, c, h, w = input.shape
        x = input.view((b * t, c, h, w))
        
        x = self.encoder(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = x.reshape((b, t, -1))
            
        x = self.decoder(x) # x.shape = (batch_size, target_index, 3(force_vector))
        return x
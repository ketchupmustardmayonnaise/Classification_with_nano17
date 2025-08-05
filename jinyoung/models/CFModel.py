from email.utils import encode_rfc2231
from sys import exception

from models import fsnet
from models.gru import *
from models.lstm import *
from models.fsnet import *
from models.convNeXt import ConvNeXtEncoder
import torch.nn.functional as F
from torchvision import models

# encoder(encoder), lstm, fully-connected layer 모두를 합한 모델

# encoder의 종류: resnet-18, resnet-34, resnet-50, resnet-101, resnet-121, densenet-121, FS-Net
# decoder(decoder)의 종류: lstm, bidirectional-lstm, gru
# fully-connected layer는, fc_num_layer 개수만큼 stacking (최소 1, 최대 4)
class CFModel(nn.Module):
    def __init__(self, encoder, encoder_output,
                 decoder, decoder_output, 
                 frame_length):
        
        super().__init__()
        self.frame_length = frame_length
        self.encoder_output = encoder_output
        self.decoder_output = decoder_output

        # Encoder
        if encoder.lower() == 'resnet18':
            self.encoder = models.resnet18(pretrained=True)
        elif encoder.lower() == 'resnet34':
            self.encoder = models.resnet34(pretrained=True)
        elif encoder.lower() == 'resnet50':
            self.encoder = models.resnet50(pretrained=True)
        elif encoder.lower() == 'densenet121':
            self.encoder = models.densenet121(pretrained=True)
        elif encoder.lower() == 'alexnet':
            self.encoder = models.alexnet(pretrained=True)
        elif encoder.lower() == 'vgg16':
            self.encoder = models.vgg16(pretrained=True)
        elif encoder.lower() == "convNeXt":
            self.encoder = models.convnext_tiny(pretrained=True)

        for param in self.encoder.parameters():
            param.requires_grad = False


        # Decoder
        if decoder == 'lstm':
            self.decoder = LSTM(input_size=encoder_output, output_size=decoder_output, hidden_size=256, bidirectional=False)
        elif decoder == 'bidirectional_lstm':
            self.decoder = LSTM(input_size=encoder_output, output_size=decoder_output, hidden_size=256, bidirectional=True)
        else: raise Exception('Unknown decoder type')



    def forward(self, input):
        input, size = (frame, channel, height, width)
        frame, channel, height, width = input.shape

        x = self.encoder(input)
        # x = input.view((frame, -1))
        x = self.decoder(x, self.frame_length) # x input shape: ( 시퀀스 길이, 배치 크기, 입력 feature )
                            # x output shape: ( 시퀀스 길이, 배치 크기, 3차원 힘벡터 )


        return x

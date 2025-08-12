from sys import exception

from model import fsnet
from model.gru import *
from model.lstm import *
from model.fsnet import *
import torch.nn.functional as F
from torchvision import models

# encoder(cnn), lstm, fully-connected layer 모두를 합한 모델

# encoder의 종류: resnet-18, resnet-34, resnet-50, resnet-101, resnet-121, densenet-121, FS-Net
# decoder(rnn)의 종류: lstm, bidirectional-lstm, gru
# fully-connected layer는, fc_num_layer 개수만큼 stacking (최소 1, 최대 4)
class CFModel(nn.Module):
    def __init__(self, encoder,
                 decoder, decoder_layer, decoder_dimension,decoder_output,  fc_num_layer,
                 target_index, fc_hidden_dim, dropout):
        assert fc_num_layer > 0

        super().__init__()
        self.fc_num_layer = fc_num_layer
        self.target_indices = target_index


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
        elif encoder.lower() == 'convnext':
            self.encoder = models.convnext_base(pretrained=True)
        else: raise Exception('Unknown encoder type')
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Decoder
        if decoder == 'lstm':
            self.decoder = LSTM(input_size=1000, hidden_size = decoder_dimension, output_size=decoder_output, h_RNN_layers=decoder_layer, fc_dim=fc_hidden_dim,drop_out= dropout,bidirectional=False)
        elif decoder == 'bidirectional_lstm':
            self.decoder = LSTM(input_size=1000, hidden_size = decoder_dimension, output_size=decoder_output, h_RNN_layers=decoder_layer, bidirectional=True)
        elif decoder == 'gru':
            self.decoder = GRU(input_size=1000, output_size=decoder_output, h_RNN_layers=decoder_layer)
        
        else: raise Exception('Unknown rnn type')

        
        

    def forward(self, input):
        # input.shape = (batch, target_index, channel, height, width)
        b, t, c, h, w = input.shape
        x = input.view((b * t, c, h, w))
        
        x = self.encoder(x)
        # x.shape = (batch_size(8) * target_index(45), 1000(alexnet))
        x = x.reshape((b, t, -1))
        # x.shape = (batch_size, target_index, 1000(alexnet))
        x = self.decoder(x)
        # x.shape = (batch_size, target_index, 3(force_vector))
        return x
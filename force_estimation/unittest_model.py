import torch
import torch.nn as nn
from sys import exception

from model import fsnet
from model.gru import *
from model.lstm import *
from model.fsnet import *
import torch.nn.functional as F
from torchvision import models
# LSTM 레이어 정의 (은닉 상태 크기 = 10)

class CFModel(nn.Module):
    def __init__(self, encoder,
                 decoder, decoder_output, rnn_layer, fc_num_layer,
                 target_index, drop_p,
                 fc_hidden_dim = None):
        assert fc_num_layer > 0

        super().__init__()
        self.fc_num_layer = fc_num_layer
        self.target_indices = target_index

        # Encoder
        # Encoder
        # torch.size [10 , 8] -> [224,224]
        # torch.size [1, 45, 3,224*224] 
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
        else: raise Exception('Unknown encoder type')
        # torch.size [1, 45, 1000]
        for param in self.encoder.parameters():
            param.requires_grad = False

        # torch,size[1, 45, 1000] - > [1, 1, 256]
        # Decoder
        if decoder == 'lstm':
            self.decoder = LSTM(input_size=1000, output_size=decoder_output, h_RNN_layers=rnn_layer, bidirectional=False)
        elif decoder == 'bidirectional_lstm':
            self.decoder = LSTM(input_size=1000, output_size=decoder_output, h_RNN_layers=rnn_layer, bidirectional=True)
        elif decoder == 'gru':
            self.decoder = GRU(input_size=1000, output_size=decoder_output, h_RNN_layers=rnn_layer)
        
        else: raise Exception('Unknown rnn type')

        fc_input_size = decoder_output * 2 if decoder == 'bidirectional_lstm' else decoder_output

        
        self.input = nn.Linear(fc_input_size, fc_hidden_dim) # [1,1,256] -> [1000]
        self.hidden_layer = nn.Linear(fc_hidden_dim, fc_hidden_dim)
        self.output = nn.Linear(fc_hidden_dim, 3)

        self.bn1 = torch.nn.BatchNorm1d(fc_hidden_dim)
        self.bn2 = torch.nn.BatchNorm1d(fc_hidden_dim)
        
        self.dropout1 = nn.Dropout(drop_p)
        self.dropout2 = nn.Dropout(drop_p)

    def forward(self, input):
        # input size = (batch, target_index, channel, height, width)
        b, t, c, h, w = input.shape
        print(f"X (inputs) shape: {input.shape}")
        x = input.view((b * t, c, h, w))
        
        x = self.encoder(x)
        print(f"X (encoder) shape: {x.shape}")
        x = x.reshape((b, t, -1))
        print(f"X (reshape) shape: {x.shape}")
        x = self.decoder(x)
        print(f"X (decorder) shape: {x.shape}")
        x = self.input(x)
        print(f"X (Linear) shape: {x.shape}")
        x = self.bn1(x)
        print(f"X (batch_n1) shape: {x.shape}")
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.hidden_layer(x)
        print(f"X (hidden_layer) shape: {x.shape}")
        x = self.bn2(x)
        print(f"X (batch_n2) shape: {x.shape}")
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.output(x)
        print(f"X (output) shape: {x.shape}")
        return x

import torch.nn as nn
import torch

class LSTM(nn.Module):
    def __init__(self, input_size=256, output_size=256, h_RNN_layers=2, bidirectional=False):
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.h_RNN_layers = h_RNN_layers  # RNN hidden layers
        self.output_size = output_size  # RNN hidden nodes
        self.bidirectional = bidirectional

        self.LSTM = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.output_size,
            num_layers=self.h_RNN_layers,
            batch_first=True,   # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            bidirectional=bidirectional
        )

    def forward(self, x):
        # x.shape = (chunk_size, frame, input_size)
        self.LSTM.flatten_parameters()
        RNN_out, (h, c) = self.LSTM(x)

        """ h_n shape (n_layers, batch, hidden_size), h_c shape (n_layers, batch, hidden_size) """
        """ None represents zero initial hidden state. RNN_out has shape=(batch, time_step, output_size) """
        return RNN_out
        # if not self.bidirectional: return RNN_out[:, -1, :]
        # else:
        #     h_f = h[-2]
        #     h_b = h[-1] # RNN_out[:,0,0:hidden_size]
        #     return torch.concat((h_f, h_b), -1)

# model = CFModel(encoder= 'alexnet', decoder='lstm', decoder_output=256, rnn_layer = 2, fc_num_layer = 3, target_index = 45, drop_p = 0.3)

# # model = nn.LSTM(
# #             input_size=1000,
# #             hidden_size=256,
# #             num_layers=2,
# #             batch_first=True,   # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
# #             bidirectional=True
# #         )

# # 입력 데이터 생성 (시퀀스 길이=45, 배치 크기=1, 입력 특징=5)
input_seq = torch.randn(8, 45, 3, 224,224)

# model.flatten_parameters()

# # LSTM 실행
# outputs, (h,c) = model(input_seq, None)

# print(outputs.shape)

encoder = models.alexnet(pretrained=True)

for param in encoder.parameters():
    param.requires_grad = False

decoder = LSTM(input_size=1000, output_size=256, h_RNN_layers=2,bidirectional=True)

fc_input_size = 256 * 2 

        
input = nn.Linear(fc_input_size, 1000) # [1,1,256] -> [1000]
hidden_layer = nn.Linear(1000, 1000)
output = nn.Linear(1000, 3)

bn1 = torch.nn.BatchNorm1d(1000)
bn2 = torch.nn.BatchNorm1d(1000)

dropout1 = nn.Dropout(0.3)
dropout2 = nn.Dropout(0.3)

b, t, c, h, w = input_seq.shape
print(f"X (input_seqs) shape: {input_seq.shape}")
x = input_seq.view((b * t, c, h, w))

x = encoder(x)
print(f"X (encoder) shape: {x.shape}")
x = x.reshape((b, t, -1))
print(f"X (reshape) shape: {x.shape}")
x = decoder(x)
print(f"X (decoder) shape: {x.shape}")
x = input(x)
print(f"X (Linear) shape: {x.shape}")
x = x.permute(0, 2, 1)
x = bn1(x)
x = x.permute(0, 2, 1)
print(f"X (batch_n1) shape: {x.shape}")
x = F.relu(x)
x = dropout1(x)
x = hidden_layer(x)
print(f"X (hidden_layer) shape: {x.shape}")
x = x.permute(0, 2, 1)
x = bn2(x)
x = x.permute(0, 2, 1)
print(f"X (batch_n2) shape: {x.shape}")
x = F.relu(x)
x = dropout2(x)
x = output(x)
print(f"X (output) shape: {x.shape}")
from sys import exception

import force_estimation.model.resnet as resnet
from force_estimation.model import fsnet
from force_estimation.model.densenet import *
from force_estimation.model.gru import *
from force_estimation.model.lstm import *
from force_estimation.model.fsnet import *
import torch.nn.functional as F
from torchvision import models

# encoder(cnn), lstm, fully-connected layer 모두를 합한 모델

# encoder의 종류: resnet-18, resnet-34, resnet-50, resnet-101, resnet-121, densenet-121, FS-Net
# decoder(rnn)의 종류: lstm, bidirectional-lstm, gru
# fully-connected layer는, fc_num_layer 개수만큼 stacking (최소 1, 최대 4)
class CFModel(nn.Module):
    def __init__(self, encoder, encoder_output,
                 decoder, decoder_output, rnn_layer, fc_num_layer,
                 target_index, image_type, drop_p,
                 fc_hidden_dim = None):
        assert fc_num_layer > 0

        super().__init__()
        self.fc_num_layer = fc_num_layer
        self.target_index = target_index
        self.encoder_output = encoder_output
        self.decoder_output = decoder_output

        # Encoder
        if encoder[0].lower() == 'resnet':
            if encoder[1].lower() == "basicblock":
                self.cnn = resnet.ResNet(resnet.BasicBlock, encoder[2], n_classes=encoder_output, image_type=image_type)
            elif encoder[1].lower() == "bottleneck":
                self.cnn = resnet.ResNet(resnet.Bottleneck, encoder[2], n_classes=encoder_output, image_type=image_type)
        elif encoder[0].lower() == 'densenet':
            self.cnn = DenseNet(block_list = encoder[2], growth_rate = int(encoder[1]), n_classes=encoder_output, image_type=image_type)
        elif encoder[0].lower() == 'fsnet':
            self.cnn = FSNet(n_classes=encoder_output, image_type=image_type)
        else: raise Exception('Unknown encoder type')

        # Decoder
        if decoder == 'lstm':
            self.rnn = LSTM(input_size=encoder_output, output_size=decoder_output, h_RNN_layers=rnn_layer, bidirectional=False)
        elif decoder == 'gru':
            self.rnn = GRU(input_size=encoder_output, output_size=decoder_output, h_RNN_layers=rnn_layer)
        elif decoder == 'bidirectional_lstm':
            self.rnn = LSTM(input_size=encoder_output, output_size=decoder_output, h_RNN_layers=rnn_layer, bidirectional=True)
        else: raise Exception('Unknown rnn type')

        self.fc_input_size = decoder_output * 2 if decoder == 'bidirectional_lstm' else decoder_output

        if fc_num_layer == 1:
            self.bn = torch.nn.BatchNorm1d(self.fc_input_size)
            self.dropout = nn.Dropout(drop_p)
            self.output = nn.Linear(self.fc_input_size, 1)
        elif fc_num_layer == 2:
            assert fc_hidden_dim is not None
            self.input = nn.Linear(self.fc_input_size, fc_hidden_dim)
            self.hidden_layer = nn.Linear(fc_hidden_dim, fc_hidden_dim)
            self.bn = torch.nn.BatchNorm1d(fc_hidden_dim)
            self.dropout = nn.Dropout(drop_p)
            self.output = nn.Linear(fc_hidden_dim, 1)
        elif fc_num_layer == 3:
            assert fc_hidden_dim is not None
            self.input = nn.Linear(self.fc_input_size, fc_hidden_dim)
            self.hidden_layer = nn.Linear(fc_hidden_dim, fc_hidden_dim)
            self.bn1 = torch.nn.BatchNorm1d(fc_hidden_dim)
            self.bn2 = torch.nn.BatchNorm1d(fc_hidden_dim)
            self.dropout1 = nn.Dropout(drop_p)
            self.dropout2 = nn.Dropout(drop_p)
            self.output = nn.Linear(fc_hidden_dim, 1)
        elif fc_num_layer == 4:
            assert fc_hidden_dim is not None
            self.input = nn.Linear(self.fc_input_size, fc_hidden_dim)
            self.hidden_layer1 = nn.Linear(fc_hidden_dim, fc_hidden_dim)
            self.hidden_layer2 = nn.Linear(fc_hidden_dim, fc_hidden_dim)
            self.bn1 = torch.nn.BatchNorm1d(fc_hidden_dim)
            self.bn2 = torch.nn.BatchNorm1d(fc_hidden_dim)
            self.bn3 = torch.nn.BatchNorm1d(fc_hidden_dim)
            self.dropout1 = nn.Dropout(drop_p)
            self.dropout2 = nn.Dropout(drop_p)
            self.dropout3 = nn.Dropout(drop_p)
            self.output = nn.Linear(fc_hidden_dim, target_index)

    def forward(self, input):
        # input size = (batch_size, frame, channel, height, width)
        batch_size, frame, channel, height, width = input.shape
        x = input.view((batch_size * frame, channel, height, width)) # (batch_size * chunk * frame, channel, height, width)

        x = self.cnn(x) # output shape: ( batch_size * frame, encoder_output )
        x = x.view((batch_size, frame, self.encoder_output))
        x = self.rnn(x) # output shape: ( batch_size, frame, decoder_output )
        x = x.contiguous().view((batch_size * frame, self.fc_input_size)) # output shape = ( batch_size * frame, decoder_output )

        if self.fc_num_layer == 1:
            x = self.bn(x)
            x = self.dropout(x)
            x = self.output(x)

        elif self.fc_num_layer == 2:
            x = self.input(x)
            x = self.bn(x)
            x = F.relu(x)
            x = self.dropout(x)

            x = self.output(x)

        elif self.fc_num_layer == 3:
            x = self.input(x)
            x = self.bn1(x)
            x = F.relu(x)
            x = self.dropout1(x)

            x = self.hidden_layer(x)
            x = self.bn2(x)
            x = F.relu(x)
            x = self.dropout2(x)

            x = self.output(x)

        elif self.fc_num_layer == 4:
            x = self.input(x)
            x = self.bn1(x)
            x = F.relu(x)
            x = self.dropout1(x)

            x = self.hidden_layer1(x)
            x = self.bn2(x)
            x = F.relu(x)
            x = self.dropout2(x)

            x = self.hidden_layer2(x)
            x = self.bn3(x)
            x = F.relu(x)
            x = self.dropout3(x)

            x = self.output(x)

        x = x.view((batch_size, frame)) # output shape = ( batch_size, frame )

        return x
from email.utils import encode_rfc2231
from sys import exception

import models.resnet as resnet
from models import fsnet
from models.densenet import *
from models.gru import *
from models.lstm import *
from models.fsnet import *
from models.convNeXt import ConvNeXtEncoder
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
        if encoder.lower() == 'resnet18':
            self.cnn = models.resnet18(pretrained=True)
        elif encoder.lower() == 'resnet34':
            self.cnn = models.resnet34(pretrained=True)
        elif encoder.lower() == 'resnet50':
            self.cnn = models.resnet50(pretrained=True)
        elif encoder.lower() == 'densenet121':
            self.cnn = models.densenet121(pretrained=True)
        elif encoder.lower() == 'alexnet':
            self.cnn = models.alexnet(pretrained=True)
        elif encoder.lower() == 'vgg16':
            self.cnn = models.vgg16(pretrained=True)
        elif encoder.lower() == "convNeXt":
            self.cnn = ConvNeXtEncoder(pretrained=True)
        else:
            self.cnn = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            )
        #modules = list(self.cnn.children())[:-1]
        #self.cnn = nn.Sequential(*modules)

        for param in self.cnn.parameters():
            param.requires_grad = False

        self.enc_fc = nn.Linear(1000, encoder_output)

        # Decoder
        if decoder == 'lstm':
            self.rnn = LSTM(input_size=encoder_output, output_size=decoder_output, h_RNN_layers=rnn_layer, bidirectional=False, target_index=target_index)
        elif decoder == 'gru':
            self.rnn = GRU(input_size=encoder_output, output_size=decoder_output, h_RNN_layers=rnn_layer)
        elif decoder == 'bidirectional_lstm':
            self.rnn = LSTM(input_size=encoder_output, output_size=decoder_output, h_RNN_layers=rnn_layer, bidirectional=True, target_index=target_index)
        else: raise Exception('Unknown rnn type')

        self.fc_input_size = decoder_output * 2 if decoder == 'bidirectional_lstm' else decoder_output

        if fc_num_layer == 1:
            # self.bn = torch.nn.BatchNorm1d(self.fc_input_size)
            self.dropout = nn.Dropout(drop_p)
            self.output = nn.Linear(self.fc_input_size, 3)
        elif fc_num_layer == 2:
            assert fc_hidden_dim is not None
            self.input = nn.Linear(self.fc_input_size, fc_hidden_dim)
            self.hidden_layer = nn.Linear(fc_hidden_dim, fc_hidden_dim)
            # self.bn = torch.nn.BatchNorm1d(fc_hidden_dim)
            self.dropout = nn.Dropout(drop_p)
            self.output = nn.Linear(fc_hidden_dim, 3)
        elif fc_num_layer == 3:
            assert fc_hidden_dim is not None
            self.input = nn.Linear(self.fc_input_size, fc_hidden_dim)
            self.hidden_layer = nn.Linear(fc_hidden_dim, fc_hidden_dim)
            # self.bn1 = torch.nn.BatchNorm1d(fc_hidden_dim)
            # self.bn2 = torch.nn.BatchNorm1d(fc_hidden_dim)
            self.dropout1 = nn.Dropout(drop_p)
            self.dropout2 = nn.Dropout(drop_p)
            self.output = nn.Linear(fc_hidden_dim, 3)
        elif fc_num_layer == 4:
            assert fc_hidden_dim is not None
            self.input = nn.Linear(self.fc_input_size, fc_hidden_dim)
            self.hidden_layer1 = nn.Linear(fc_hidden_dim, fc_hidden_dim)
            self.hidden_layer2 = nn.Linear(fc_hidden_dim, fc_hidden_dim)
            # self.bn1 = torch.nn.BatchNorm1d(fc_hidden_dim)
            # self.bn2 = torch.nn.BatchNorm1d(fc_hidden_dim)
            # self.bn3 = torch.nn.BatchNorm1d(fc_hidden_dim)
            self.dropout1 = nn.Dropout(drop_p)
            self.dropout2 = nn.Dropout(drop_p)
            self.dropout3 = nn.Dropout(drop_p)
            self.output = nn.Linear(fc_hidden_dim, 3)

    def forward(self, input):
        # input size = (frame, channel, height, width)
        # frame, channel, height, width = input.shape

        # x = self.cnn(input)
        # # x = input.view((frame, -1))
        # x = self.enc_fc(x)  # output shape: ( frame, encoder_output )
        # x = self.rnn(x) # x output shape: ( chunk size, target_index, decoder output )
        # # x.reshape((-1, self.decoder_output))

        # if self.fc_num_layer == 1:
        #     # x = self.bn(x)
        #     x = self.dropout(x)
        #     x = self.output(x)

        # elif self.fc_num_layer == 2:
        #     x = self.input(x)
        #     # x = self.bn(x)
        #     x = F.relu(x)
        #     x = self.dropout(x)

        #     x = self.output(x)

        # elif self.fc_num_layer == 3:
            # x = self.input(x)
            # # x = self.bn1(x)
            # x = F.relu(x)
            # x = self.dropout1(x)

            # x = self.hidden_layer(x)
            # # x = self.bn2(x)
            # x = F.relu(x)
            # x = self.dropout2(x)

            # x = self.output(x)

        # elif self.fc_num_layer == 4:
        #     x = self.input(x)
        #     # x = self.bn1(x)
        #     x = F.relu(x)
        #     x = self.dropout1(x)

        #     x = self.hidden_layer1(x)
        #     # x = self.bn2(x)
        #     x = F.relu(x)
        #     x = self.dropout2(x)

        #     x = self.hidden_layer2(x)
        #     # x = self.bn3(x)
        #     x = F.relu(x)
        #     x = self.dropout3(x)

        #     x = self.output(x)

        # return x
         # 입력 텐서의 shape 저장
        s, c, h, w = input.shape
        
        # (S, C, H, W) -> (B * S, C, H, W)
        # ConvNeXt에 한 번에 입력하기 위해 배치와 시퀀스 차원을 합칩니다.
        input = input.view(s, c, h, w)
        
        # ConvNeXt Encoder를 통과시켜 특징 추출
        # output shape: (S, 768)
        features = self.cnn(input)
        
        # LSTM을 통과시켜 시퀀스 정보 처리
        # lstm_out shape: (S, lstm_hidden_size)
        # self.lstm은 (output, (hidden_state, cell_state))를 반환합니다.
        x = self.rnn(features)
        
        x = self.input(x)
        # x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.hidden_layer(x)
        # x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        x = self.output(x)
    
        
        return x
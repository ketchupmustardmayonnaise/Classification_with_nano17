from torch import nn
from torchvision import models
import pytorch_lightning as pl
import torch.nn.functional as F


class Prediction3DForceModel(pl.LightningModule):
    def __init__(self,
                 type_of_encoder, type_of_decoder,
                 num_of_decoder_layers=2,
                 dim_of_decoder_hidden_layer=128,
                 num_of_fc_layers=1,
                 dim_of_fc_hidden_layer=128,
                 size_of_image=[6, 6]
                 ):

        assert dim_of_fc_hidden_layer > 0

        super().__init__()
        self.type_of_encoder = type_of_encoder
        self.type_of_decoder = type_of_decoder
        self.num_of_decoder_layers = num_of_decoder_layers
        self.dim_of_decoder_hidden_layer = dim_of_decoder_hidden_layer
        self.num_of_fc_layers = num_of_fc_layers
        self.dim_of_fc_hidden_layer = dim_of_fc_hidden_layer

        self.extractor = False
        self.size_of_decoder_input = 1000
        self.size_of_image = size_of_image

        # ================================================
        # Encoder
        # ================================================

        if self.type_of_encoder.lower() == 'resnet18':
            self.encoder = models.resnet18(pretrained=True)
            for param in self.encoder.parameters():
                param.requires_grad = False

        elif self.type_of_encoder.lower() == 'densenet121':
            self.encoder = models.densenet121(pretrained=True)
            for param in self.encoder.parameters():
                param.requires_grad = False

        elif self.type_of_encoder.lower() == 'alexnet':
            self.encoder = models.alexnet(pretrained=True)
            for param in self.encoder.parameters():
                param.requires_grad = False

        elif self.type_of_encoder.lower() == 'vgg16':
            self.encoder = models.vgg16(pretrained=True)
            for param in self.encoder.parameters():
                param.requires_grad = False

        elif self.type_of_encoder.lower() == 'convnext':
            self.encoder = models.convnext_tiny(pretrained=True)
            for param in self.encoder.parameters():
                param.requires_grad = False

        elif self.type_of_encoder.lower() == 'convnext_changed':
            self.extractor = True
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            pretrained_model = models.convnext_tiny(pretrained=True)

            new_stem = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=96, kernel_size=3, stride=1, padding=1),
                # LayerNorm은 채널 수(96)와 이미지 크기(8x10)를 알아야 합니다.
                nn.LayerNorm([96, self.size_of_image[0], self.size_of_image[1]]))

            pretrained_model.features[0] = new_stem
            self.encoder = nn.Sequential(
                pretrained_model.features[0],
                pretrained_model.features[1]
            )
            for param in self.encoder[1].parameters():
                param.requires_grad = False
            self.decoder_input_size = 96

        else:
            raise Exception('Unknown encoder type')

        # ================================================
        # Decoder
        # ================================================

        if self.decoder_type == 'lstm':
            self.decoder = LSTM(input_size=self.decoder_input_size, hidden_size=self.dim_of_decoder_hidden_layer,
                                output_size=decoder_output, h_RNN_layers=decoder_layer, fc_dim=fc_hidden_dim,
                                bidirectional=False)
        elif self.decoder_type == 'bidirectional_lstm':
            self.decoder = LSTM(input_size=self.decoder_input_size, hidden_size=decoder_dimension,
                                output_size=decoder_output, h_RNN_layers=decoder_layer, bidirectional=True)
        elif self.decoder_type == 'transformer':
            self.decoder = TransformerModel(input_size=self.decoder_input_size,  # encoder 출력 크기
                                            output_size=decoder_output,
                                            h_RNN_layers=decoder_layer,  # 트랜스포머 레이어 수
                                            nhead=transformer_nhead,  # 어텐션 헤드 수
                                            dim_feedforward=transformer_ff_dim,  # 트랜스포머 내부 FF 레이어 차원
                                            dropout=dropout,
                                            max_len=target_index)  # 시퀀스 최대 길이
        else:
            raise Exception('Unknown decoder type')

    def forward(self, input):
        # in lightning, foward defines the prediction/inference actions

    def training_step(self, batch, batch_idx):
        # input.shape = (batch, target_index, channel, height, width)
        input, _ = batch
        b, t, c, h, w = input.shape
        x = input.view((b * t, c, h, w))

        x = self.encoder(x)
        if self.extractor:
            x = self.pool(x)
        x = torch.flatten(x, 1)
        x = x.reshape((b, t, -1))

        x = self.decoder(x) # x.shape = (batch_size, target_index, 3(force_vector))
        return x


class LSTM(nn.Module):
    def __init__(self, input_size=256, hidden_size=256, output_size=3, h_RNN_layers=2, fc_dim=1000,
                 bidirectional=False):
        super(LSTM, self).__init__()

        # self.linear_input = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.linear_dim = fc_dim
        self.LSTM = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=h_RNN_layers,
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            bidirectional=bidirectional
        )

        self.linear_output = nn.Linear(in_features=hidden_size * (2 if bidirectional else 1), out_features=output_size)
        # self.bn1 = nn.BatchNorm1d(self.linear_dim)

        # # 3개의 히든 레이어를 각각 정의
        # self.hidden_layer1 = nn.Linear(self.linear_dim, self.linear_dim)
        # self.bn2 = nn.BatchNorm1d(self.linear_dim)

        # self.dropout = nn.Dropout(p=drop_out)
        # self.output = nn.Linear(self.linear_dim, output_size)

    def forward(self, x):
        # x.shape = (batch_size, sequence_length, 1000(Alexnet))
        self.LSTM.flatten_parameters()

        outputs, h = self.LSTM(x)
        # outputs.shape = if bidirectional = (batch, sequence_length, hidden_size * 2) else (batch, sequence_length, hidden_size)
        outputs = self.linear_output(outputs)
        # outputs = outputs.permute(0, 2, 1) # BN을 위해 (N, C, L) 형태로 변경
        # outputs = self.bn1(outputs)
        # outputs = outputs.permute(0, 2, 1) # 원래 형태로 복원
        # outputs = relu(outputs)
        # outputs = self.dropout(outputs)

        # # Hidden Layer 1
        # outputs = self.hidden_layer1(outputs)
        # outputs = outputs.permute(0, 2, 1); outputs = self.bn2(outputs); outputs = outputs.permute(0, 2, 1)
        # outputs = relu(outputs)
        # outputs = self.dropout(outputs)

        # outputs = self.output(outputs)
        # #outputs.shape = (batch_size, sequence_length, output_size)
        return outputs


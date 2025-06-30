import torch
import torch.nn as nn
import torch.nn.functional as F
from data_load import TARGET_INDICES, SIZE

encoder_output = 256
decoder_output = 64
regression_FC_dim = 64
classification_FC_dim = 128

num_hidden_of_regression = 0
num_hidden_of_classification = 3

class Encoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=[32, 32, 64], kernel_size=3, padding=0):
        super(Encoder, self).__init__()

        self.layer1 = nn.Conv2d(in_channels=in_channels,
                                out_channels=out_channels[0],
                                kernel_size=kernel_size,
                                padding=padding)

        self.layer2 = nn.Conv2d(in_channels=out_channels[0],
                                out_channels=out_channels[1],
                                kernel_size=kernel_size,
                                padding=padding)


        self.layer3 = nn.Conv2d(in_channels=out_channels[1],
                               out_channels=out_channels[2],
                               kernel_size=kernel_size,
                               padding=padding)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = x.flatten()
        return x

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
        self.stage1 = self.make_stage(block, 8, num_block_list[0], stride=1)
        self.stage2 = self.make_stage(block, 16, num_block_list[1], stride=2)
        self.stage3 = self.make_stage(block, 32, num_block_list[2], stride=2)
        self.stage4 = self.make_stage(block, 64, num_block_list[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * 2, 4)

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
        #x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)

        return x

def resnet_learning(**kwargs):
    return ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)


class DecoderRNN(nn.Module):
    def __init__(self, input_size=32, h_RNN_layers=1, output_size=decoder_output, drop_p=0.3):
        super(DecoderRNN, self).__init__()

        self.input_size = input_size
        self.h_RNN_layers = h_RNN_layers  # RNN hidden layers
        self.output_size = output_size  # RNN hidden nodes
        self.drop_p = drop_p

        self.GRU = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.output_size,
            num_layers=h_RNN_layers,
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.LSTM = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.output_size,
            num_layers=h_RNN_layers,
            batch_first=True,   # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            # bidirectional=True
        )

    def forward(self, x_RNN):
        self.GRU.flatten_parameters()
        RNN_out, _ = self.GRU(x_RNN, None)
        """ h_n shape (n_layers, batch, hidden_size), h_c shape (n_layers, batch, hidden_size) """
        """ None represents zero initial hidden state. RNN_out has shape=(batch, time_step, output_size) """

        return RNN_out[:, -1, :]


'''
데이터 라벨 분류 => Classification
'''
class PredictLabelModel(nn.Module):
    def __init__(self, num_classes=8, input_size=decoder_output, FCdim=classification_FC_dim, num_hidden_layer=num_hidden_of_classification, drop_p=0.3):
        super(PredictLabelModel, self).__init__()
        self.drop_p = drop_p
        self.num_hidden_layer = num_hidden_layer

        self.input = nn.Linear(input_size, FCdim)
        self.hidden_layer = nn.Linear(FCdim, FCdim)
        self.output = nn.Linear(FCdim, num_classes)

    def forward(self, input):
        x = self.input(input)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        for index in range(self.num_hidden_layer):
            x = self.hidden_layer(x)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)

        x = self.output(x)

        return x

'''
Shear 값 예측 => Regression
'''
class PredictSheerModel(nn.Module):
    def __init__(self, dim=1, input_size=decoder_output, FCdim=regression_FC_dim, num_hidden_layer=num_hidden_of_regression, drop_p=0.3):
        super(PredictSheerModel, self).__init__()
        self.drop_p = None
        self.num_hidden_layer = num_hidden_layer

        self.input = nn.Linear(input_size, FCdim)
        self.hidden_layer = nn.Linear(FCdim, FCdim)
        self.output = nn.Linear(64, dim)

        self.bn = torch.nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(drop_p)
        self.dropout2 = nn.Dropout(drop_p)

    def forward(self, input):
        '''
        x = self.input(input)
        x = self.bn(x)
        x = F.relu(x)
        x = self.dropout1(x)

        for index in range(self.num_hidden_layer):
            x = self.hidden_layer(x)
            x = self.bn(x)
            x = F.relu(x)
            x = self.dropout2(x)

        x = self.output(x)
        '''

        x = self.bn(input)
        return self.output(input)
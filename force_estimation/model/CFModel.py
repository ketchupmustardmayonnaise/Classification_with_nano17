from force_estimation.model.resnet import *
from force_estimation.model.densenet import *
from force_estimation.model.gru import *
from force_estimation.model.lstm import *

class CFModel(nn.Module):
    def __init__(self, encoder, encoder_output, decoder, rnn_layer, bidirectional,
                 use_only_output_layer, drop_p, target_indices, image_type, fc_hidden_layer = None, fc_hidden_dim = None):
        super().__init__()
        self.fc_hidden_layer = fc_hidden_layer
        self.use_only_output_layer = use_only_output_layer
        self.target_indices = target_indices

        # Encoder
        if encoder == 'resnet':
            self.cnn = ResNet(BasicBlock, [2, 2, 2, 2], n_classes=encoder_output, image_type=image_type)
        else:
            self.cnn = DenseNet(block_list = [6, 6, 6, 6], growth_rate = 16, n_classes=encoder_output, image_type=image_type)

        # Decoder
        if decoder == 'lstm':
            self.rnn = LSTM(input_size=encoder_output, h_RNN_layers=rnn_layer, bidirectional=bidirectional)
        else:
            self.rnn = GRU(input_size=encoder_output, h_RNN_layers=rnn_layer)

        fc_input_size = encoder_output * 2 if bidirectional else encoder_output
        if not use_only_output_layer:
            assert fc_hidden_layer is not None and fc_hidden_dim is not None
            self.fc_hidden_layer = fc_hidden_layer

            self.input = nn.Linear(fc_input_size, fc_hidden_dim)
            self.hidden_layer = nn.Linear(fc_hidden_dim, fc_hidden_dim)
            self.bn = torch.nn.BatchNorm1d(fc_hidden_dim)
            self.dropout = nn.Dropout(drop_p)
            self.output = nn.Linear(fc_hidden_dim, 1)
        else:
            self.bn = torch.nn.BatchNorm1d(fc_input_size)
            self.dropout = nn.Dropout(drop_p)
            self.output = nn.Linear(fc_input_size, 1)

    def forward(self, input):
        # input size = (batch, frame, channel, height, width)
        x = input.reshape((input.shape[0] * input.shape[1], input.shape[2], input.shape[3], input.shape[4]))

        x = self.cnn(x)
        x = x.reshape((int(x.shape[0] / self.target_indices), self.target_indices, -1))
        x = self.rnn(x)

        if not self.use_only_output_layer:
            x = self.input(x)

            for index in range(self.fc_hidden_layer):
                x = self.bn(x)
                x = F.relu(x)
                x = self.dropout(x)
                x = self.hidden_layer(x)

            x = self.bn(x)
            x = F.relu(x)
            x = self.dropout(x)
            x = self.output(x)

        else:
            x = self.bn(x)
            x = self.dropout(x)
            x = self.output(x)

        return x
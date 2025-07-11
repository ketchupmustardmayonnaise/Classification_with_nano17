from force_estimation.model.resnet import *
from force_estimation.model.densenet import *
from force_estimation.model.gru import *
from force_estimation.model.lstm import *

class CFModel(nn.Module):
    def __init__(self, encoder, encoder_output, decoder, rnn_layer,
                 use_only_output_layer, use_all_sequence, drop_p, target_indices, image_type, resnet_block, densenet_block, densenet_growth,
                 fc_hidden_layer = None, fc_hidden_dim = None):
        super().__init__()
        self.fc_hidden_layer = fc_hidden_layer
        self.use_only_output_layer = use_only_output_layer
        self.target_indices = target_indices
        self.use_all_sequence = use_all_sequence

        # Encoder
        if encoder == 'resnet':
            self.cnn = ResNet(BasicBlock, resnet_block, n_classes=encoder_output, image_type=image_type)
        else:
            self.cnn = DenseNet(block_list = densenet_block, growth_rate = densenet_growth, n_classes=encoder_output, image_type=image_type)

        # Decoder
        if decoder == 'lstm':
            self.rnn = LSTM(input_size=encoder_output, h_RNN_layers=rnn_layer, bidirectional=False)
        elif decoder == 'gru':
            self.rnn = GRU(input_size=encoder_output, h_RNN_layers=rnn_layer)
        elif decoder == 'bidirectional_lstm':
            self.rnn = LSTM(input_size=encoder_output, h_RNN_layers=rnn_layer, bidirectional=True)

        fc_input_size = encoder_output * 2 if decoder == 'bidirectional_lstm' else encoder_output
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
        x = x.reshape((int(x.shape[0] / self.target_indices), self.target_indices, -1)) if self.use_all_sequence is False else x
        x = self.rnn(x)

        if not self.use_all_sequence:
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
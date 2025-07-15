from force_estimation.model.resnet import *
from force_estimation.model.densenet import *
from force_estimation.model.gru import *
from force_estimation.model.lstm import *
from torchvision import models

class CFModel(nn.Module):
    def __init__(self, encoder, encoder_output,
                 decoder, rnn_layer, fc_num_layer, use_all_sequence,
                 target_indices, image_type, drop_p,
                 fc_hidden_dim = None):
        assert fc_num_layer > 0

        super().__init__()
        self.fc_num_layer = fc_num_layer
        self.target_indices = target_indices
        self.use_all_sequence = use_all_sequence

        # Encoder
        if encoder[0].lower() == 'resnet':
            if encoder[1].lower() == "basicblock":
                self.cnn = ResNet(BasicBlock, encoder[2], n_classes=encoder_output, image_type=image_type)
            elif encoder[1].lower() == "bottleneck":
                self.cnn = ResNet(Bottleneck, encoder[2], n_classes=encoder_output, image_type=image_type)
        elif encoder[0].lower() == 'densenet':
            self.cnn = DenseNet(block_list = encoder[2], growth_rate = int(encoder[1]), n_classes=encoder_output, image_type=image_type)

        # Decoder
        if decoder == 'lstm':
            self.rnn = LSTM(input_size=encoder_output, h_RNN_layers=rnn_layer, use_all_sequence=self.use_all_sequence, bidirectional=False)
        elif decoder == 'gru':
            self.rnn = GRU(input_size=encoder_output, h_RNN_layers=rnn_layer, use_all_sequence=self.use_all_sequence)
        elif decoder == 'bidirectional_lstm':
            self.rnn = LSTM(input_size=encoder_output, h_RNN_layers=rnn_layer, bidirectional=True)

        fc_input_size = encoder_output * 2 if decoder == 'bidirectional_lstm' else encoder_output
        if fc_num_layer > 1:
            assert fc_hidden_dim is not None
            self.input = nn.Linear(fc_input_size, fc_hidden_dim)
            self.hidden_layer = nn.Linear(fc_hidden_dim, fc_hidden_dim)
            self.bn = torch.nn.BatchNorm1d(fc_hidden_dim)
            self.dropout = nn.Dropout(drop_p)
            self.output = nn.Linear(fc_hidden_dim, 1)
        elif fc_num_layer == 1:
            self.bn = torch.nn.BatchNorm1d(fc_input_size)
            self.dropout = nn.Dropout(drop_p)
            self.output = nn.Linear(fc_input_size, 1)

    def forward(self, input):
        # input size = (batch, frame, channel, height, width)
        x = input.reshape((input.shape[0] * input.shape[1], input.shape[2], input.shape[3], input.shape[4]))

        x = self.cnn(x)
        x = x.reshape((int(x.shape[0] / self.target_indices), self.target_indices, -1)) if input.shape[0] != 1 else x
        x = self.rnn(x)

        if self.fc_num_layer > 1:
            x = self.input(x)

            for index in range(self.fc_num_layer - 1):
                if self.use_all_sequence: x = self.bn(x)
                x = F.relu(x)
                x = self.dropout(x)
                x = self.hidden_layer(x)

            if self.use_all_sequence: x = self.bn(x)
            x = F.relu(x)
            x = self.dropout(x)
            x = self.output(x)

        else:
            if self.use_all_sequence: x = self.bn(x)
            x = self.dropout(x)
            x = self.output(x)

        return x
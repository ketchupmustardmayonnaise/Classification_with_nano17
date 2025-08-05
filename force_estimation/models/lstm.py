import torch.nn as nn
import torch
from torch.nn.functional import relu
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_rnn_layer=2, bidirectional=True, dropout=0.4):
        super(LSTM, self).__init__()

        self.linear_input = nn.Linear(in_features=input_size, out_features=hidden_size)
        
        self.LSTM = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_rnn_layer,
            batch_first=True,   # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            bidirectional=bidirectional
        )

        self.linear_output = nn.Linear(in_features=hidden_size * (2 if bidirectional else 1), out_features=output_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, seq_lengths=None, h=None):
        
        x= self.linear_input(x)
        
        x = self.dropout(relu(x))

        output, h = self.LSTM(x, h)

        output = self.linear_output(output)

        return output, h
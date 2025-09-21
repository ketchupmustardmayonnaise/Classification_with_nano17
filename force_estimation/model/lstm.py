import torch.nn as nn
from torch.nn.functional import relu

class LSTM(nn.Module):
    def __init__(self, input_size=256, hidden_size = 128, output_size=3, rnn_layer=2, bidirectional=True):
        super(LSTM, self).__init__()

        self.LSTM = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=rnn_layer,
            batch_first=True,   # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            bidirectional=bidirectional
        )
        
        self.linear_output = nn.Linear(in_features=hidden_size * (2 if bidirectional else 1), out_features=output_size)

    def forward(self, x):
        # x.shape = (batch_size, sequence_length, 1000(Alexnet))
        self.LSTM.flatten_parameters()
        
        outputs, h = self.LSTM(x)
        outputs = self.linear_output(outputs)

        return outputs
        

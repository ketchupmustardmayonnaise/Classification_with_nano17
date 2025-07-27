import torch.nn as nn
import torch

class LSTM(nn.Module):
    def __init__(self, input_size=256, output_size=256, h_RNN_layers=2, bidirectional=False, drop_p=0.3, target_index=12):
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.h_RNN_layers = h_RNN_layers  # RNN hidden layers
        self.output_size = output_size  # RNN hidden nodes
        self.drop_p = drop_p
        self.target_index = target_index

        self.LSTM = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.output_size,
            num_layers=self.h_RNN_layers,
            batch_first=True,   # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            bidirectional=bidirectional
        )

        # 학습 가능한 초기 hidden & cell state
        self.h0 = nn.Parameter(
            torch.zeros(self.h_RNN_layers * 2 if bidirectional else self.h_RNN_layers, self.output_size))
        self.c0 = nn.Parameter(
            torch.zeros(self.h_RNN_layers * 2 if bidirectional else self.h_RNN_layers, self.output_size))

    def forward(self, x):
        frame, input_size = x.shape
        self.LSTM.flatten_parameters()

        output = None
        h, c = self.h0, self.c0

        for f in range(frame - (self.target_index - 1)):
            xi = x[f : f + self.target_index]

            out, (h, c) = self.LSTM(xi, (h, c))
            # h, c = h.detach(), c.detach()

            if output is None:
                output = out.unsqueeze(0)
            else:
                output = torch.cat((output, out.unsqueeze(0)), 0)

        """ h_n shape (n_layers, batch, hidden_size), h_c shape (n_layers, batch, hidden_size) """
        """ None represents zero initial hidden state. RNN_out has shape=(batch, time_step, output_size) """

        return output
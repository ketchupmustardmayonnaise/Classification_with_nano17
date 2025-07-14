import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size=256, h_RNN_layers=2, bidirectional=False, drop_p=0.3, use_all_sequence=False):
        super(LSTM, self).__init__()

        self.input_size = int(input_size)
        self.h_RNN_layers = int(h_RNN_layers)  # RNN hidden layers
        self.output_size = int(input_size)  # RNN hidden nodes
        self.drop_p = float(drop_p)
        self.use_all_sequence = bool(use_all_sequence)

        self.LSTM = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.output_size,
            num_layers=self.h_RNN_layers,
            batch_first=True,   # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            bidirectional=bidirectional
        )

    def forward(self, x_RNN):
        self.LSTM.flatten_parameters()
        RNN_out, _ = self.LSTM(x_RNN, None)
        """ h_n shape (n_layers, batch, hidden_size), h_c shape (n_layers, batch, hidden_size) """
        """ None represents zero initial hidden state. RNN_out has shape=(batch, time_step, output_size) """

        if self.use_all_sequence:
            return RNN_out
        else: return RNN_out[:, -1, :]
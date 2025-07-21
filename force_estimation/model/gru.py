import torch.nn as nn

class GRU(nn.Module):
    def __init__(self, input_size=256, h_RNN_layers=2, drop_p=0.3):
        super(GRU, self).__init__()

        self.input_size = int(input_size)
        self.h_RNN_layers = int(h_RNN_layers)  # RNN hidden layers
        self.output_size = int(input_size)  # RNN hidden nodes
        self.drop_p = float(drop_p)

        self.GRU = nn.GRU(
            input_size=self.input_size,
            hidden_size=int(self.output_size),
            num_layers=h_RNN_layers,
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

    def forward(self, x_RNN):
        self.GRU.flatten_parameters()
        RNN_out, _ = self.GRU(x_RNN, None)
        """ h_n shape (n_layers, batch, hidden_size), h_c shape (n_layers, batch, hidden_size) """
        """ None represents zero initial hidden state. RNN_out has shape=(batch, time_step, output_size) """

        return RNN_out[:, -1, :]
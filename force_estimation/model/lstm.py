import torch.nn as nn
from torch.nn.functional import relu

class LSTM(nn.Module):
    def __init__(self, input_size=256, hidden_size = 256, output_size=3, h_RNN_layers=2, bidirectional=False):
        super(LSTM, self).__init__()

        # self.linear_input = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.linear_dim = 1000
        self.LSTM = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=h_RNN_layers,
            batch_first=True,   # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            bidirectional=bidirectional
        )
        
        self.linear_output = nn.Linear(in_features=hidden_size * (2 if bidirectional else 1), out_features=3)
        # self.bn1 = nn.BatchNorm1d(self.linear_dim)
        
        # # 3개의 히든 레이어를 각각 정의
        # self.hidden_layer1 = nn.Linear(self.linear_dim, self.linear_dim)
        # self.bn2 = nn.BatchNorm1d(self.linear_dim)
        # self.hidden_layer2 = nn.Linear(self.linear_dim, self.linear_dim)
        # self.bn3 = nn.BatchNorm1d(self.linear_dim)
        # self.hidden_layer3 = nn.Linear(self.linear_dim, self.linear_dim)
        # self.bn4 = nn.BatchNorm1d(self.linear_dim)

        # self.dropout = nn.Dropout(p=0.3)
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

        # # Hidden Layer 2
        # outputs = self.hidden_layer2(outputs)
        # outputs = outputs.permute(0, 2, 1); outputs = self.bn3(outputs); outputs = outputs.permute(0, 2, 1)
        # outputs = relu(outputs)
        # outputs = self.dropout(outputs)

        # # Hidden Layer 3
        # outputs = self.hidden_layer3(outputs)
        # outputs = outputs.permute(0, 2, 1); outputs = self.bn4(outputs); outputs = outputs.permute(0, 2, 1)
        # outputs = relu(outputs)
        # outputs = self.dropout(outputs)

        # outputs = self.output(outputs)
        # # outputs.shape = (batch_size, sequence_length, output_size)
        return outputs
        

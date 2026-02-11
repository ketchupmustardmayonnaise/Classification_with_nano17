import math
import torch
import torch.nn as nn

# =====================================
# Long Short-Term Memory
# =====================================

class LSTM(nn.Module):
    def __init__(self,
                 size_of_input : int = 96,
                 size_of_hidden_layer_features : int = 128,
                 num_layer : int = 2,
                 is_bidirectional : bool = True,
                 export_multistep : bool = True # 모든 timestep의 결과를 출력하도록 할 것인지?
    ):
        super().__init__()

        self._export_multistep = export_multistep
        self._create_networks(size_of_input, size_of_hidden_layer_features, num_layer, is_bidirectional)
        self._size_of_output = size_of_input


    def _create_networks(self,
                        size_of_input : int,
                        size_of_hidden_layer_features : int,
                        num_layer : int,
                        is_bidirectional : bool):

        self.LSTM = nn.LSTM(
            input_size=size_of_input,
            hidden_size=size_of_hidden_layer_features,
            num_layers=num_layer,
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            bidirectional=is_bidirectional
        )

    @property
    def export_multistep(self):
        return self._export_multistep


    @export_multistep.setter
    def export_multistep(self, value):
        if not isinstance(value, bool):
            raise ValueError("export_multistep must be boolean")
        self._export_multistep = value


    @property
    def size_of_output(self): # output은, input과 똑같은 dimension을 갖는 feature
        return self._size_of_output


    def forward(self, x):
        # x.shape = (batch_size, timesteps, size_of_input)
        self.LSTM.flatten_parameters()
        outputs, h = self.LSTM(x)

        # outputs.shape = if bidirectional = (batch, sequence_length, hidden_size * 2) else (batch, sequence_length, hidden_size)
        if self._export_multistep:
            return outputs
        else: return outputs[:, -1, :] # 마지막 timestep 결과만 출력



# =====================================
# Transformer
# =====================================

class PositionalEncoding(nn.Module):
    def __init__(self,
                 size_of_input : int = 256,
                 dropout : float = 0.1,
                 length_of_sequence : float = 50):

        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(length_of_sequence, size_of_input)
        position = torch.arange(0, length_of_sequence, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, size_of_input, 2).float() * (-math.log(10000.0) / size_of_input))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)



class Transformer(nn.Module):

    def __init__(self,
                 size_of_input : int,
                 num_rnn_layer : int,
                 num_of_heads : int = 2,

                 dropout : float = 0.0,
                 length_of_sequence : int = 30,

                 dim_feedforward : int = None,
                 export_multistep : bool = True # 모든 timestep의 결과를 출력하도록 할 것인지?
    ):
        super().__init__()

        # size_of_hidden_layer_features 가 None이라면, 2배로 늘리는 것을 default로 함.
        if dim_feedforward is None or dim_feedforward == 'default':
            dim_feedforward = size_of_input * 2

        self._create_networks(size_of_input, dim_feedforward, num_rnn_layer, num_of_heads, dropout)
        self._length_of_sequence = length_of_sequence
        self._size_of_output = size_of_input
        self._export_multistep = export_multistep

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(size_of_input=size_of_input, dropout=dropout, length_of_sequence=length_of_sequence)


    def _create_networks(self,
                         size_of_input: int,
                         dim_feedforward: int,
                         num_rnn_layer : int,
                         num_of_heads: int = 2,
                         dropout : float = 0.0
                         ):

        # Transformer Encoder Layer 정의
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=size_of_input,
            nhead=num_of_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # 입력 텐서의 차원 순서를 (Batch, Seq, Feature)로 설정
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_rnn_layer)


    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)

        # 1. 위치 정보 추가
        x = self.pos_encoder(x)

        # 2. 트랜스포머 통과
        x = self.transformer_encoder(x)

        if self._export_multistep:
            # 3. 각 타임스텝의 출력에 대해 Linear 레이어를 적용하여 최종 Force 예측
            # CFModel의 기존 출력 형태 (batch, seq, output)를 맞추기 위함
            # x = self.output_layer(x)

            return x

        else:
            x = x[:, -1,  :]
            x = self.output_layer(x)
            return x


    @property
    def size_of_output(self): # output은, input과 똑같은 dimension을 갖는 feature
        return self._size_of_output

# transformer.py 또는 CFModel.py 상단에 추가
import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=50):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, input_size, num_rnn_layer, nhead, dim_feedforward, dropout=0.0, max_len=50,
                 num_classes=8,
                 regression_output_dim=3):
        super().__init__()

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model=input_size, dropout=dropout, max_len=max_len)
        self.num_classes = num_classes

        # Transformer Encoder Layer 정의
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=input_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # 입력 텐서의 차원 순서를 (Batch, Seq, Feature)로 설정
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_rnn_layer)

        # 최종 출력을 위한 Linear 레이어
        # 각 시퀀스 토큰마다 Force를 예측하도록 설정
        self.regression_output_layer = nn.Linear(input_size, regression_output_dim)
        self.classification_output_layer = nn.Linear(input_size * max_len, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)

        # 1. 위치 정보 추가
        x = self.pos_encoder(x)

        # 2. 트랜스포머 통과
        x = self.transformer_encoder(x)

        # 3. 각 타임스텝의 출력에 대해 Linear 레이어를 적용하여 최종 Force 예측
        # CFModel의 기존 출력 형태 (batch, seq, output)를 맞추기 위함
        r_output = self.regression_output_layer(x)
        c_output = self.classification_output_layer(torch.flatten(x, 1))

        return c_output, r_output


class TransformerModel_each_label(nn.Module):
    def __init__(self, input_size, num_rnn_layer, nhead, dim_feedforward, dropout=0.0, max_len=50,
                 num_classes=8,
                 regression_output_dim=3):
        super().__init__()

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model=input_size, dropout=dropout, max_len=max_len)
        self.num_classes = num_classes
        self.regression_output_dim = regression_output_dim

        # Transformer Encoder Layer 정의
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=input_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # 입력 텐서의 차원 순서를 (Batch, Seq, Feature)로 설정
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_rnn_layer)

        # 최종 출력을 위한 Linear 레이어
        # 각 시퀀스 토큰마다 Force를 예측하도록 설정
        # class별 regression head
        self.regression_output_layers = nn.ModuleList([
            nn.Linear(input_size, regression_output_dim)
            for _ in range(self.num_classes)
        ])

        self.classification_output_layer = nn.Linear(input_size * max_len, self.num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)

        # 1. 위치 정보 추가
        x = self.pos_encoder(x)

        # 2. 트랜스포머 통과
        x = self.transformer_encoder(x)

        # 3. 각 타임스텝의 출력에 대해 Linear 레이어를 적용하여 최종 Force 예측
        # CFModel의 기존 출력 형태 (batch, seq, output)를 맞추기 위함

        # classification
        c_logits = self.classification_output_layer(torch.flatten(x, 1))
        c_pred = torch.argmax(c_logits, dim=1)  # (batch,)

        # regression output 저장용
        batch_size, seq_len, _ = x.shape
        r_output = torch.zeros(
            batch_size, seq_len, self.regression_output_dim,
            device=x.device
        )

        # class별로 해당 regression head 적용
        for cls in range(self.num_classes):
            mask = (c_pred == cls)
            if mask.any():
                r_output[mask] = self.regression_output_layers[cls](x[mask])

        return c_logits, r_output
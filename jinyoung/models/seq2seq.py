import torch
import torch.nn as nn
import torchvision.models as models
from torchsummary import summary

class Seq2SeqModel(nn.Module):
    """
    사전 학습된 CNN 인코더와 LSTM을 사용하는 Sequence-to-Sequence 모델입니다.
    """
    def __init__(self, cnn_encoder, lstm_hidden_size=64, force_vector_dim=6):
        super(Seq2SeqModel, self).__init__()
        
        self.cnn_encoder = cnn_encoder
        
        # AlexNet features의 최종 출력 크기는 (256, 6, 6) 입니다.
        # Flatten하면 256 * 6 * 6 = 9216이 됩니다.
        lstm_input_size = 256 * 6 * 6
        
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(0.5)
        self.fc_out = nn.Linear(lstm_hidden_size, force_vector_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_length, C, H, W)
        batch_size, seq_length, C, H, W = x.shape
        
        # TimeDistributed 구현: (N, S, C, H, W) -> (N*S, C, H, W)
        x_reshaped = x.view(batch_size * seq_length, C, H, W)
        
        # CNN 인코더 적용
        encoded = self.cnn_encoder(x_reshaped)
        
        # Flatten
        encoded_flat = torch.flatten(encoded, start_dim=1)
        
        # 다시 시퀀스 형태로 복원: (N*S, features) -> (N, S, features)
        encoded_sequence = encoded_flat.view(batch_size, seq_length, -1)
        
        # LSTM 적용
        lstm_out, _ = self.lstm(encoded_sequence)
        
        # Dropout 적용
        lstm_out = self.dropout(lstm_out)
        
        # 최종 출력 레이어 적용
        output_sequence = self.fc_out(lstm_out)
        
        return output_sequence


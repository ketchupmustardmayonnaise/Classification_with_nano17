import torch
import torch.nn as nn
import torchvision.models as models

class DecoderLSTM(nn.Module):
    """
    사용자가 제공한 이미지 아키텍처 기반의 힘 예측 모델
    """
    def __init__(self, cnn_encoder, cnn_output_size, pos_data_dim, force_vector_dim=1):
        super(DecoderLSTM, self).__init__()
        
        self.cnn_encoder = cnn_encoder
        
        # 1. Feature Vector Concatenation 이후의 FC 레이어들
        # FC 1056 -> FC 512 -> FC 128
        # 입력 크기는 CNN 출력과 위치 데이터 크기의 합입니다.
        self.initial_fc = nn.Sequential(
            nn.Linear(cnn_output_size + pos_data_dim, 1056),
            nn.ReLU(inplace=True),
            nn.Linear(1056, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True)
        )
        
        # 2. LSTM 레이어들 (2개 스택)
        # units: 32 -> 16
        self.lstm = nn.LSTM(
            input_size=128,          # initial_fc의 최종 출력 크기
            hidden_size=32,          # 첫 번째 LSTM의 hidden size
            num_layers=1,              # 첫 번째 LSTM
            batch_first=True
        )
        self.lstm2 = nn.LSTM(
            input_size=32,           # 첫 번째 LSTM의 출력 크기
            hidden_size=16,          # 두 번째 LSTM의 hidden size
            num_layers=1,              # 두 번째 LSTM
            batch_first=True
        )

        # 3. 최종 출력을 위한 FC 레이어들
        # FC 16 -> FC 8 -> FC 1
        self.final_fc = nn.Sequential(
            nn.Linear(16, 16), # 오타일 수 있으나, 그림대로 FC16으로 시작
            nn.ReLU(inplace=True),
            nn.Linear(16, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, force_vector_dim)
        )

    def forward(self, image_sequence, pos_sequence):
        # image_sequence shape: (batch_size, seq_length, C, H, W)
        # pos_sequence shape:   (batch_size, seq_length, pos_data_dim)
        
        batch_size, seq_length, C, H, W = image_sequence.shape
        
        # --- 각 타임스텝별 특징 추출 및 결합 ---
        # TimeDistributed CNN 적용
        img_reshaped = image_sequence.view(batch_size * seq_length, C, H, W)
        img_features = self.cnn_encoder(img_reshaped)
        img_features_flat = torch.flatten(img_features, start_dim=1)
        img_features_seq = img_features_flat.view(batch_size, seq_length, -1)
        
        # 이미지 특징과 위치 데이터 결합
        combined_features = torch.cat([img_features_seq, pos_sequence], dim=2)
        
        # --- Temporal Feature Modeling ---
        # TimeDistributed FC 적용
        # PyTorch의 Linear는 자동으로 마지막 차원에 대해 연산하므로 reshape 트릭이 필요없습니다.
        z_sequence = self.initial_fc(combined_features)
        
        # LSTM 적용
        lstm_out, _ = self.lstm(z_sequence)
        lstm_out2, _ = self.lstm2(lstm_out)
        
        # Sequence-to-Vector: 시퀀스의 마지막 출력만 사용
        last_time_step_out = lstm_out2[:, -1, :]
        
        # 최종 FC 레이어를 통해 힘 예측
        final_force = self.final_fc(last_time_step_out)
        
        return final_force


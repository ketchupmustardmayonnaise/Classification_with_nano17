import torch
import torch.nn as nn
import torchvision.models as models


class ConvNeXtEncoder(nn.Module):
    """
    이미지를 입력받아 특징 벡터를 추출하는 인코더.
    """
    def __init__(self, pretrained=True, freeze=True):
        super().__init__()
        # 사전 훈련된 ConvNeXt 모델 불러오기
        self.encoder = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
        
        # 마지막 분류 레이어를 제거하여 특징 추출기로 사용
        self.encoder.classifier[2] = nn.Identity()

        # 가중치 동결 설정
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
    
    def forward(self, image):
        # image: [B, 3, 224, 224] -> features: [B, 768]
        features = self.encoder(image)
        return features



class DecoderLSTM(nn.Module):

    def __init__(self, input_dim=768, output_dim=3):
        super().__init__()
        
        # 1. Feature Vector Concatenation 이후의 FC 레이어들
        # FC 1056 -> FC 512 -> FC 128
        # 입력 크기는 CNN 출력과 위치 데이터 크기의 합입니다.
        self.initial_fc = nn.Sequential(
            nn.Linear(input_dim, 512),
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
            nn.Linear(8, output_dim)
        )

    def forward(self, feature):
        # image_sequence shape: (batch_size, seq_length, C, H, W)
        # pos_sequence shape:   (batch_size, seq_length, pos_data_dim)

        z_sequence = self.initial_fc(feature)
        
        # LSTM 적용
        lstm_out, _ = self.lstm(z_sequence)
        lstm_out2, _ = self.lstm2(lstm_out)
        
        # Sequence-to-Vector: 시퀀스의 마지막 출력만 사용
        last_time_step_out = lstm_out2[:, -1, :]
        
        # 최종 FC 레이어를 통해 힘 예측
        final_force = self.final_fc(last_time_step_out)
        
        return final_force


class ImageToVectorModel(nn.Module):
    def __init__(self,lstm_input, lstm_output):
        super().__init__()
        self.encoder = ConvNeXtEncoder()
        self.decoder = DecoderLSTM(lstm_input, lstm_output)

    def forward(self, image):
        features = self.encoder(image)
        output = self.decoder(features)
        return output

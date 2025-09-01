# /model/CFModel_XY_Z.py

import torch
import torch.nn as nn
from torchvision import models

# 기존에 사용하시던 디코더 모델들을 모두 import 합니다.
# 파일 경로는 실제 프로젝트 구조에 맞게 수정해주세요.
from .lstm import LSTM
from .transformer import TransformerModel

class CFModelXYZ(nn.Module):
    def __init__(self, encoder,
                 decoder, decoder_layer, decoder_dimension,
                 target_index, fc_hidden_dim, dropout, image_size,
                 # Transformer용 하이퍼파라미터
                 transformer_nhead=4, transformer_ff_dim=256):
        
        super().__init__()
        self.target_indices = target_index
        self.extractor = False
        self.input_size = 1000 # 기본값

        # =================================================================
        # 1. 공통 인코더 (Shared Encoder) - 기존 CFModel과 동일
        # =================================================================
        if encoder.lower() == 'convnext':
            self.extractor = True
            pretrained_model = models.convnext_tiny(pretrained=True)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            new_stem = nn.Sequential(
                        nn.Conv2d(in_channels=3, out_channels=96, kernel_size=3, stride=1, padding=1),
                        nn.LayerNorm([96, 10, 8]))
            pretrained_model.features[0] = new_stem
            self.encoder = nn.Sequential(
                pretrained_model.features[0],
                pretrained_model.features[1]
            )
            for param in self.encoder[1].parameters():
                param.requires_grad = False
            self.input_size = 96
        else:
            raise Exception('Unknown encoder type')

        # =================================================================
        # 2. 독립적인 디코더 2개 (XY용, Z용)
        # =================================================================
        
        # 주어진 decoder 타입과 output 크기에 따라 디코더 인스턴스를 생성하는 헬퍼 함수
        def create_decoder(output_size):
            if decoder == 'lstm':
                return LSTM(input_size=self.input_size, hidden_size=decoder_dimension, output_size=output_size, h_RNN_layers=decoder_layer, fc_dim=fc_hidden_dim, drop_out=dropout, bidirectional=False)
            elif decoder == 'bidirectional_lstm':
                return LSTM(input_size=self.input_size, hidden_size=decoder_dimension, output_size=output_size, h_RNN_layers=decoder_layer, bidirectional=True)
            elif decoder == 'transformer':
                return TransformerModel(input_size=self.input_size, output_size=output_size, h_RNN_layers=decoder_layer, nhead=transformer_nhead, dim_feedforward=transformer_ff_dim, dropout=dropout, max_len=target_index)
            else:
                raise Exception('Unknown decoder type')

        # XY를 함께 예측하는 디코더 (출력 차원=2)
        self.decoder_xy = create_decoder(output_size=2)
        
        # Z만 예측하는 디코더 (출력 차원=1)
        self.decoder_z = create_decoder(output_size=1)

    def forward(self, input):
        # input shape: (batch, seq_len, channel, height, width)
        b, t, c, h, w = input.shape
        x = input.view((b * t, c, h, w))

        # 1. 공통 인코더를 통과시켜 특징 추출
        features = self.encoder(x)
        if self.extractor:
            features = self.pool(features)
        
        features = torch.flatten(features, 1)
        features = features.reshape((b, t, -1)) # (batch, seq_len, feature_dim)
        
        # 2. 추출된 특징을 2개의 독립적인 디코더에 각각 전달
        out_xy = self.decoder_xy(features) # (batch, seq_len, 2)
        out_z = self.decoder_z(features)  # (batch, seq_len, 1)
        
        # 3. 각 디코더의 출력을 마지막 차원에서 합쳐서 (batch, seq_len, 3) 형태로 만듦
        output = torch.cat([out_xy, out_z], dim=-1)
        
        return output
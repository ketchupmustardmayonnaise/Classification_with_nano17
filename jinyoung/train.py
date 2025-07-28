import yaml
import torch
import torch.nn as nn
import torchvision.models as models
from models.cnnLstm import *

with open("config.yaml", 'r+', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

# 1. 사전 학습된 인코더 준비
alexnet_full = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
encoder = alexnet_full.features

for param in encoder.parameters():
    param.requires_grad = True
encoder.eval()

# 2. 모델 파라미터 정의


# 3. 최종 모델 생성
model = DecoderLSTM(
    cnn_encoder=encoder,
    cnn_output_size=config.CNN_OUTPUT_SIZE,
    pos_data_dim=config.POS_DATA_DIM,
    force_vector_dim=config.FORCE_VECTOR_DIM
)

print(model)

# 4. Dummy 데이터로 테스트
batch_size = 4
sequence_length = 10 # T=10
# 이미지는 전처리를 거쳐 224x224 크기가 되었다고 가정
dummy_image_seq = torch.randn(batch_size, sequence_length, 3, 224, 224)
dummy_pos_seq = torch.randn(batch_size, sequence_length, config.POS_DATA_DIM)

output = model(dummy_image_seq, dummy_pos_seq)
print(f"\nInput image shape:  {dummy_image_seq.shape}")
print(f"Input pos shape:    {dummy_pos_seq.shape}")
print(f"Output force shape: {output.shape}")
import torch
import torch.nn as nn
from torchvision.models import *

# 1. 사전 학습된 ConvNeXt Tiny 모델 불러오기
# ConvNeXt 모델 중 가장 가벼운 버전으로 시작하는 것이 좋습니다.
pretrained_model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)

# 2. ConvNeXt의 구조 확인 (features와 classifier로 구성)
#    features의 첫 번째 블록(stem)이 입력 크기를 처리합니다.
print("Original ConvNeXt Stem (features[0]):\n", pretrained_model.features[0])

# 3. 8x10 이미지에 맞게 새로운 Stem 레이어 정의
#    ConvNeXt의 Stem은 Conv2d와 LayerNorm으로 구성됩니다.
#    입력 채널=1, 커널=3, 스트라이드=1, 패딩=1 로 수정합니다.
new_stem = nn.Sequential(
    nn.Conv2d(1, 96, kernel_size=3, stride=1, padding=1),
    # LayerNorm은 채널 수(96)와 이미지 크기(8x10)를 알아야 합니다.
    nn.LayerNorm([96, 8, 10]) 
)

# 4. 모델의 기존 Stem을 새로운 Stem으로 교체
pretrained_model.features[0] = new_stem

# 5. 필요한 만큼의 특징 추출 블록만 선택하여 새로운 모델 만들기
#    - features[0]: 우리가 수정한 Stem
#    - features[1]: 첫 번째 ConvNeXt 블록
#    - 과적합을 피하기 위해 여기까지만 사용하는 것을 추천합니다.
feature_extractor = nn.Sequential(
    pretrained_model.features[0],
    pretrained_model.features[1]
)

# 6. 가중치 동결 (새로 만든 Stem은 학습되도록 놔둠)
#    - pretrained_model.features[1] (첫 번째 블록)의 가중치만 동결합니다.
for param in pretrained_model.features[1].parameters():
    param.requires_grad = False

# 7. 수정된 feature_extractor 구조 확인
print("\nCustom ConvNeXt Feature Extractor:\n", feature_extractor)

# 8. 이제 이 가벼워진 feature_extractor에 이미지를 통과시킵니다.
dummy_input = torch.randn(1, 1, 8, 10) # (batch, channel, height, width)
output = feature_extractor(dummy_input)

print("\nOutput shape after feature extraction:", output.shape)
# ConvNeXt의 첫 블록은 크기를 줄이지 않으므로 입력과 비슷한 크기를 가집니다.
# 예상 출력: torch.Size([1, 96, 8, 10])

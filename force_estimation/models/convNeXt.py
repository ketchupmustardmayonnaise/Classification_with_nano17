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
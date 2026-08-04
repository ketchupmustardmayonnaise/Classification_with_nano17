import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class _AlexNetWithNewHead(nn.Module):
    """
    Pretrained AlexNet은 그대로 두고(제거/슬라이싱 없이),
    classifier의 4096-dim 출력(마지막 1000-way Linear 이전) 위에
    새로운 Linear(4096, 96) 레이어만 덧붙여서 그 레이어만 학습시킵니다.
    """

    def __init__(self, pretrained_model):
        super().__init__()
        self.pretrained_model = pretrained_model  # 통째로 보존 (마지막 레이어 포함, frozen)
        self.new_head = nn.Linear(4096, 96)  # 이 레이어만 학습

    def forward(self, x):
        x = self.pretrained_model.features(x)
        x = self.pretrained_model.avgpool(x)
        x = torch.flatten(x, 1)
        # classifier의 마지막 Linear(4096, 1000) 직전까지만 통과시켜 4096-dim feature를 얻음
        x = self.pretrained_model.classifier[:-1](x)
        x = self.new_head(x)
        return x


class VisionFeatureExtractor(nn.Module):
    """
    각 이미지의 feature를 extract하는 코드

    입력: (batch, timesteps, channels, height, width)
    출력: (batch, timesteps, _size_of_output)
    """

    def __init__(self,
                 cnn_name : str,
                 image_size : (int, int)):
        super().__init__()

        if cnn_name is not None and image_size is not None:
            self._create_networks(cnn_name, image_size)
        else:
            raise Exception('Encoder로 사용할 모델명과 이미지 사이즈를 정확히 입력하여야 합니다.')


    def _create_networks(self,
                         cnn_name : str,
                         image_size : (int, int)):

        self._resize_to = None  # 대부분 모델은 resize 불필요; AlexNet은 224로 설정됨

        if cnn_name.lower() == "resnet18":
            self._size_of_output = 64

            pretrained_model = models.resnet18(pretrained=True)

            # 이미지가 6x6으로 매우 작기 때문에, 기존의 stem 대신 우리 방식대로 customize된 stem을 사용합니다.
            customized_stem = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
                # LayerNorm은 채널 수(96)와 이미지 크기(8x10)를 알아야 합니다.
                nn.LayerNorm([64, image_size[0], image_size[1]])
            )

            # customized stem -> Stage 1 of pretrained model -> AdaptiveAvgPool2d
            self.model = nn.Sequential(
                customized_stem,
                pretrained_model.layer1,  # stage 1 of pretrained model
                nn.AdaptiveAvgPool2d((1, 1))  # AdaptiveAvgPool2d
            )

            # stem과 pooling 제외하고, pretrained model에서 떼어 온 부분을 동결합니다.
            for param in self.model[1].parameters():
                param.requires_grad = False


        elif cnn_name.lower() == 'densenet121':
            self._size_of_output = 256

            pretrained_model = models.densenet121(pretrained=True)

            # 이미지가 6x6으로 매우 작기 때문에, 기존의 stem 대신 우리 방식대로 customize된 stem을 사용합니다.
            customized_stem = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
                # LayerNorm은 채널 수(96)와 이미지 크기(8x10)를 알아야 합니다.
                nn.LayerNorm([64, image_size[0], image_size[1]])
            )

            # customized stem -> Stage 1 of pretrained model -> AdaptiveAvgPool2d
            self.model = nn.Sequential(
                customized_stem,
                pretrained_model.features.denseblock1,  # stage 1 of pretrained model
                nn.AdaptiveAvgPool2d((1, 1))  # AdaptiveAvgPool2d
            )

            # stem과 pooling 제외하고, pretrained model에서 떼어 온 부분을 동결합니다.
            for param in self.model[1].parameters():
                param.requires_grad = False


        elif cnn_name.lower() == 'alexnet':
            self._size_of_output = 96
            self._resize_to = 224  # AlexNet은 224x224 입력 필요

            pretrained_model = models.alexnet(pretrained=True)
            # pretrained_model은 그대로 유지(제거 없음)하고, 동결한 뒤
            # 4096-dim feature 위에 새로운 Linear(4096, 96) 레이어만 덧붙여 학습합니다.
            for param in pretrained_model.parameters():
                param.requires_grad = False

            self.model = _AlexNetWithNewHead(pretrained_model)

        elif cnn_name.lower() == 'vgg16':
            self.model = models.vgg16(pretrained=True)
            for param in self.encoder.parameters():
                param.requires_grad = False

        elif cnn_name.lower() == 'lightweight_convnexttiny':
            self._size_of_output = 96

            # Pretrained ConvNeXT-tiny 모델을 불러옵니다.
            pretrained_model = models.convnext_tiny(pretrained=True)

            # 이미지가 6x6으로 매우 작기 때문에, 기존의 stem 대신 우리 방식대로 customize된 stem을 사용합니다.
            customized_stem = nn.Sequential(
                        nn.Conv2d(in_channels= 3, out_channels= 96, kernel_size=3, stride=1, padding=1),
                        # LayerNorm은 채널 수(96)와 이미지 크기(8x10)를 알아야 합니다.
                        nn.LayerNorm([96, image_size[0], image_size[1]])
            )

            # customized stem -> Stage 1 of pretrained model -> AdaptiveAvgPool2d
            self.model = nn.Sequential(
                customized_stem,
                pretrained_model.features[1], # stage 1 of pretrained model
                nn.AdaptiveAvgPool2d((1, 1)) # AdaptiveAvgPool2d
            )

            # stem과 pooling 제외하고, pretrained model에서 떼어 온 부분을 동결합니다.
            for param in self.model[1].parameters():
                param.requires_grad = False

        else: raise Exception('Unknown encoder type')


    def forward(self, x):
        # x.shape = (batch, timesteps, channels, height, width)
        if len(x.shape) == 4:
            b, c, h, w = x.shape
            if self._resize_to is not None:
                x = F.interpolate(x, size=(self._resize_to, self._resize_to),
                                  mode='bilinear', align_corners=False)
            x = self.model(x)
            x = torch.flatten(x, 1)

        elif len(x.shape) == 5:
            b, t, c, h, w = x.shape
            x = x.reshape(b * t, c, h, w)
            if self._resize_to is not None:
                x = F.interpolate(x, size=(self._resize_to, self._resize_to),
                                  mode='bilinear', align_corners=False)
            x = self.model(x)
            x = torch.flatten(x, 1)
            x = x.reshape((b, t, -1))

        return x


    @property
    def size_of_output(self):
        if self._size_of_output is None: raise Exception("네트워크가 아직 생성되지 않았습니다.")
        return self._size_of_output
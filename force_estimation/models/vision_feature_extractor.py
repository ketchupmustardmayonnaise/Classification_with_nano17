import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

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
            self._size_of_output = 4096
            self._resize_to = 224  # AlexNet은 224x224 입력 필요

            pretrained_model = models.alexnet(pretrained=True)
            # 마지막 1000-way Linear 제거 → 4096-dim feature extractor로 사용
            pretrained_model.classifier = nn.Sequential(
                *list(pretrained_model.classifier.children())[:-1]
            )
            self.model = pretrained_model
            for param in self.model.parameters():
                param.requires_grad = False

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
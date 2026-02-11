import torch
import torch.nn as nn
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

        if cnn_name.lower() == "resnet18":
            self.model = models.resnet18(pretrained=True)
            for param in self.model.parameters():
                param.requires_grad = False

        elif cnn_name.lower() == 'densenet121':
            self.model = models.densenet121(pretrained=True)
            for param in self.model.parameters():
                param.requires_grad = False

        elif cnn_name.lower() == 'alexnet':
            self.model = models.alexnet(pretrained=True)
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
                        nn.LayerNorm([96, image_size[0], image_size[1]]))

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
        b, t, c, h, w = x.shape

        x = x.view((b * t, c, h, w))
        x = self.model(x)
        x = torch.flatten(x, 1)
        x = x.reshape((b, t, -1))

        return x


    @property
    def size_of_output(self):
        if self._size_of_output is None: raise Exception("네트워크가 아직 생성되지 않았습니다.")
        return self._size_of_output
from sys import exception


from model.lstm import *
from model.transformer import *
import torch.nn.functional as F
from torchvision import models

# encoder(cnn), lstm, fully-connected layer 모두를 합한 모델

# encoder의 종류: resnet-18, resnet-34, resnet-50, resnet-101, resnet-121, densenet-121, FS-Net
# decoder(rnn)의 종류: lstm, bidirectional-lstm, gru
# fully-connected layer는, fc_num_layer 개수만큼 stacking (최소 1, 최대 4)
class CFModel(nn.Module):
    def __init__(self, encoder,
                 decoder, decoder_layer, decoder_dimension,decoder_output,  fc_num_layer,
                 target_index, fc_hidden_dim, dropout, image_size, transformer_nhead=4, transformer_ff_dim=256):
        assert fc_num_layer > 0

        super().__init__()
        self.fc_num_layer = fc_num_layer
        self.target_indices = target_index
        self.extractor = False
        self.decoder_input_size = 1000
        if encoder.lower() == 'resnet18':
            self.extractor = True
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.input_size = 64
            pretrained_model = models.resnet18(pretrained=True)
            pretrained_model.conv1 = nn.Conv2d(in_channels =3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
            self.encoder = nn.Sequential(
                pretrained_model.conv1,
                pretrained_model.bn1,
                pretrained_model.relu,
                pretrained_model.maxpool,
                # pretrained_model.layer1,
                # pretrained_model.layer2,  # layer1, layer2 정도만 사용하는 것을 추천 (경량화)
                # pretrained_model.layer3, # 필요에 따라 더 깊은 레이어를 추가할 수 있습니다.
                # pretrained_model.layer4,
            )
            for name, param in self.encoder.named_parameters():
                if 'conv1' not in name :
                    param.requires_grad = False
                
        elif encoder.lower() == 'resnet34':
            self.encoder = models.resnet34(pretrained=True)
        elif encoder.lower() == 'resnet50':
            self.encoder = models.resnet50(pretrained=True)
        elif encoder.lower() == 'densenet121':
            self.encoder = models.densenet121(pretrained=True)
        elif encoder.lower() == 'alexnet':
            self.encoder = models.alexnet(pretrained=True)
        elif encoder.lower() == 'vgg16':
            self.encoder = models.vgg16(pretrained=True)
        elif encoder.lower() == 'convnext':
            self.extractor = True
            pretrained_model = models.convnext_tiny(pretrained=True)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            new_stem = nn.Sequential(
                        nn.Conv2d(in_channels= 3, out_channels= 96, kernel_size=3, stride=1, padding=1),
                        # LayerNorm은 채널 수(96)와 이미지 크기(8x10)를 알아야 합니다.
                        nn.LayerNorm([96, 10, 8]))
            pretrained_model.features[0] = new_stem
            self.encoder = nn.Sequential(
                pretrained_model.features[0],
                pretrained_model.features[1]
            )
            for param in self.encoder[1].parameters():
                param.requires_grad = False
            self.decoder_input_size = 96
        elif encoder.lower() == 'alexnetchange':
            self.extractor = True
            pretrained_alexnet = models.alexnet(weights='DEFAULT')
            self.decoder_input_size = 64
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            if image_size == (10,8) :
                feature_layers = list(pretrained_alexnet.features[:3].children())
                feature_layers[0] = nn.Conv2d(in_channels=3, 
                              out_channels=64, 
                              kernel_size=3, 
                              stride=1, 
                              padding=1)
                feature_layers[2] = nn.MaxPool2d(kernel_size=2, stride=2)
                self.encoder = nn.Sequential(*feature_layers)
            else : 
                self.encoder = pretrained_alexnet.features[:3]
            for name, param in self.encoder.named_parameters():
                if not name.startswith('0.') :
                    param.requires_grad = False
        else: raise Exception('Unknown encoder type')
        # for param in self.encoder.parameters():
        #     param.requires_grad = False

        
        # Decoder
        if decoder == 'lstm':
            self.decoder = LSTM(input_size=self.decoder_input_size, hidden_size = decoder_dimension, output_size=decoder_output, h_RNN_layers=decoder_layer, fc_dim=fc_hidden_dim,drop_out= dropout,bidirectional=False)
        elif decoder == 'bidirectional_lstm':
            self.decoder = LSTM(input_size=self.decoder_input_size, hidden_size = decoder_dimension, output_size=decoder_output, h_RNN_layers=decoder_layer, bidirectional=True)
        elif decoder == 'transformer' :
            self.decoder = TransformerModel(input_size=self.decoder_input_size, # encoder 출력 크기
                                             output_size=decoder_output,
                                             h_RNN_layers=decoder_layer, # 트랜스포머 레이어 수
                                             nhead=transformer_nhead, # 어텐션 헤드 수
                                             dim_feedforward=transformer_ff_dim, # 트랜스포머 내부 FF 레이어 차원
                                             dropout=dropout,
                                             max_len=target_index) # 시퀀스 최대 길이
        
        else: raise Exception('Unknown decoder type')

        
        

    def forward(self, input):
        # input.shape = (batch, target_index, channel, height, width)
        b, t, c, h, w = input.shape
        x = input.view((b * t, c, h, w))
        
        x = self.encoder(x)
        if self.extractor :
            x = self.pool(x)
            x = torch.flatten(x, 1)
                
        # x.shape = (batch_size(8) * target_index(45), 1000(alexnet))
        x = x.reshape((b, t, -1))
        # x.shape = (batch_size, target_index, 1000(alexnet))
        x = self.decoder(x)
        # x.shape = (batch_size, target_index, 3(force_vector))
        return x
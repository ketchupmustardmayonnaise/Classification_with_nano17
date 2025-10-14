from sys import exception


from model.lstm import *
from model.transformer import *
from model.convlstm import *
import torch.nn.functional as F
from torchvision import models

class ConvLSTMDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, kernel_sizes, num_layers, output_size):
        super().__init__()
        if not isinstance(hidden_dims, list) or len(hidden_dims) != num_layers:
            raise ValueError("hidden_dims must be a list with length equal to num_layers")
        if not isinstance(kernel_sizes, list) or len(kernel_sizes) != num_layers:
            raise ValueError("kernel_sizes must be a list with length equal to num_layers")
        
        self.convlstm = ConvLSTM(input_dim=input_dim,
                                 hidden_dim=hidden_dims,
                                 kernel_size=kernel_sizes,
                                 num_layers=num_layers,
                                 batch_first=True,
                                 bias=True)
        
        # 최종 예측을 위한 부분
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # ConvLSTM의 마지막 레이어의 hidden_dim을 받아 최종 output_size로 변환
        self.fc = nn.Linear(hidden_dims[-1], output_size)

    def forward(self, x):
        # x shape: (b, t, c, h, w)
        
        # ConvLSTM 실행
        # layer_outputs는 모든 레이어의 출력을 담은 리스트
        layer_outputs, last_states = self.convlstm(x)
        
        # 우리는 마지막 레이어의 출력만 필요함
        # last_layer_output shape: (b, t, hidden_dim, h, w)
        last_layer_output = layer_outputs[-1]
        
        b, t, c, h, w = last_layer_output.shape
        
        # FC 레이어에 넣기 위해 reshape
        # (b, t, c, h, w) -> (b*t, c, h, w)
        x_reshaped = last_layer_output.view(b * t, c, h, w)
        
        # 공간적 특징을 평균 풀링
        x_pooled = self.avgpool(x_reshaped) # (b*t, c, 1, 1)
        x_flattened = torch.flatten(x_pooled, 1) # (b*t, c)
        
        # 최종 예측
        output = self.fc(x_flattened) # (b*t, output_size)
        
        # 원래 시퀀스 형태로 복원
        # (b*t, output_size) -> (b, t, output_size)
        output = output.view(b, t, -1)
        
        return output
    
# encoder(cnn), lstm, fully-connected layer 모두를 합한 모델

# encoder의 종류: resnet-18, resnet-34, resnet-50, resnet-101, resnet-121, densenet-121, FS-Net
# decoder(rnn)의 종류: lstm, bidirectional-lstm, gru
# fully-connected layer는, fc_num_layer 개수만큼 stacking (최소 1, 최대 4)
class CFModel(nn.Module):
    def __init__(self, encoder,
                 decoder, decoder_layer, decoder_dimension,decoder_output,  fc_num_layer,
                 target_index, fc_hidden_dim, dropout, image_size, transformer_nhead=2, transformer_ff_dim=192,
                 convlstm_hidden_dims=[64, 32], convlstm_kernel_sizes=[(3,3), (3,3)]
                 ):
        assert fc_num_layer > 0

        super().__init__()
        self.fc_num_layer = fc_num_layer
        self.target_indices = target_index
        self.extractor = False
        self.decoder_input_size = 1000
        self.decoder_type = decoder

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
            )
            for name, param in self.encoder.named_parameters():
                if 'conv1' not in name :
                    param.requires_grad = False
                
        elif encoder.lower() == 'resnet34':
            self.encoder = models.resnet34(pretrained=True)
            for param in self.encoder.parameters():
                param.requires_grad = False

        elif encoder.lower() == 'resnet50':
            self.encoder = models.resnet50(pretrained=True)
            for param in self.encoder.parameters():
                param.requires_grad = False

        elif encoder.lower() == 'densenet121':
            self.encoder = models.densenet121(pretrained=True)
            for param in self.encoder.parameters():
                param.requires_grad = False

        elif encoder.lower() == 'alexnet':
            self.encoder = models.alexnet(pretrained=True)
            for param in self.encoder.parameters():
                param.requires_grad = False

        elif encoder.lower() == 'vgg16':
            self.encoder = models.vgg16(pretrained=True)
            for param in self.encoder.parameters():
                param.requires_grad = False

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

        
        # Decoder
        if self.decoder_type == 'lstm':
            self.decoder = LSTM(input_size=self.decoder_input_size, hidden_size = decoder_dimension, output_size=decoder_output, h_RNN_layers=decoder_layer, fc_dim=fc_hidden_dim,drop_out= dropout,bidirectional=False)
        elif self.decoder_type == 'bidirectional_lstm':
            self.decoder = LSTM(input_size=self.decoder_input_size, hidden_size = decoder_dimension, output_size=decoder_output, h_RNN_layers=decoder_layer, bidirectional=True)
        elif self.decoder_type == 'transformer' :
            self.decoder = TransformerModel(input_size=self.decoder_input_size, # encoder 출력 크기
                                             output_size=decoder_output,
                                             h_RNN_layers=decoder_layer, # 트랜스포머 레이어 수
                                             nhead=transformer_nhead, # 어텐션 헤드 수
                                             dim_feedforward=transformer_ff_dim, # 트랜스포머 내부 FF 레이어 차원
                                             dropout=dropout,
                                             max_len=target_index) # 시퀀스 최대 길이
        elif self.decoder_type == 'convlstm':
            self.decoder = ConvLSTMDecoder(
                input_dim=self.input_size, # encoder의 출력 채널 수
                hidden_dims=convlstm_hidden_dims,
                kernel_sizes=convlstm_kernel_sizes,
                num_layers=len(convlstm_hidden_dims),
                output_size=decoder_output
            )
        
        else: raise Exception('Unknown decoder type')

        
        

    def forward(self, input):
        # input.shape = (batch, target_index, channel, height, width)
        b, t, c, h, w = input.shape
        x = input.view((b * t, c, h, w))
        
        x = self.encoder(x)
        if self.decoder_type in ['lstm', 'gru', 'transformer', 'bidirectional_lstm']:
            # 기존 디코더들은 Flatten된 벡터를 입력으로 받음
            if self.extractor:
                x = self.pool(x)
            x = torch.flatten(x, 1)
            x = x.reshape((b, t, -1))
        
        elif self.decoder_type == 'convlstm':
            # ConvLSTM은 2D 특징 맵을 그대로 입력으로 받음
            # x shape: (b*t, c_feat, h_feat, w_feat)
            c_feat, h_feat, w_feat = x.shape[1], x.shape[2], x.shape[3]
            x = x.reshape((b, t, c_feat, h_feat, w_feat))
            
        x = self.decoder(x)
        # x.shape = (batch_size, target_index, 3(force_vector))
        return x
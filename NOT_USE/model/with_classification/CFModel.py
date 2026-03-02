from NOT_USE.model.with_classification.transformer import *
from torchvision import models

transformer_nhead = 2
transformer_ff_dim = 192

class CFModel(nn.Module):
    def __init__(self, encoder, decoder, num_decoder_layer, dim_decoder_hidden,
                 target_index, image_size,
                 num_classes = 8):

        super().__init__()
        self.extractor = False
        self.decoder_input_size = 1000
        self.num_classes = num_classes

        if encoder.lower() == 'resnet18':
            self.encoder = models.resnet18(pretrained=True)
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

        elif encoder.lower() == 'convnexttiny':
            self.encoder = models.convnext_tiny(pretrained=True)
            for param in self.encoder.parameters():
                param.requires_grad = False

        elif encoder.lower() == 'lightweight_convnexttiny':
            self.extractor = True
            pretrained_model = models.convnext_tiny(pretrained=True)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            new_stem = nn.Sequential(
                        nn.Conv2d(in_channels= 3, out_channels= 96, kernel_size=3, stride=1, padding=1),
                        # LayerNorm은 채널 수(96)와 이미지 크기(8x10)를 알아야 합니다.
                        nn.LayerNorm([96, image_size[0], image_size[1]]))
            pretrained_model.features[0] = new_stem
            self.encoder = nn.Sequential(
                pretrained_model.features[0],
                pretrained_model.features[1]
            )
            for param in self.encoder[1].parameters():
                param.requires_grad = False
            self.decoder_input_size = 96

        else: raise Exception('Unknown encoder type')

        
        # Decoder
        if decoder.lower() == 'lstm':
            self.decoder = LSTM(input_size=self.decoder_input_size, hidden_size=dim_decoder_hidden, output_size=3, num_rnn_layer=num_decoder_layer, bidirectional=False)
        elif decoder.lower() == 'bidirectional_lstm':
            self.decoder = LSTM(input_size=self.decoder_input_size, hidden_size=dim_decoder_hidden, output_size=3, num_rnn_layer=num_decoder_layer, bidirectional=True)
        elif decoder.lower() == 'transformer' :
            self.decoder = TransformerModel(input_size=self.decoder_input_size, # encoder 출력 크기
                                            num_classes=self.num_classes,
                                            regression_output_dim=3,
                                             num_rnn_layer=num_decoder_layer, # 트랜스포머 레이어 수
                                             nhead=transformer_nhead, # 어텐션 헤드 수
                                             dim_feedforward=transformer_ff_dim, # 트랜스포머 내부 FF 레이어 차원
                                             max_len=target_index) # 시퀀스 최대 길이
        
        else: raise Exception('Unknown decoder type')

        
        

    def forward(self, input):
        # input.shape = (batch, target_index, channel, height, width)
        b, t, c, h, w = input.shape
        x = input.view((b * t, c, h, w))
        
        x = self.encoder(x)
        if self.extractor:
            x = self.pool(x)
        x = torch.flatten(x, 1)
        x = x.reshape((b, t, -1))
            
        x = self.decoder(x)
        # x.shape = (batch_size, target_index, 3(force_vector))
        return x
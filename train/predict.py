from train.parameters import *
from train.model import *
from train.model_densenet import *
from torchinfo import summary
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from train.scheduler import *
from train.model_densenet import *

class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = DenseNet121()
        self.rnn = DecoderRNN()
        self.fc = PredictSheerModel()

    def forward(self, input):
        x = self.cnn(input)

        if USE_RNN:
            x = x.reshape((int(input.shape[0] / TARGET_INDICES), TARGET_INDICES, -1))
            x = self.rnn(x)
        else:
            x = x.reshape((x.shape[0], -1))

        x = self.fc(x)
        return x

def initialize_model(device, train=True):
    # 모델
    cnn = DenseNet121()
    cnn = cnn.to(device)
    rnn = DecoderRNN()
    rnn = rnn.to(device)
    fc = PredictSheerModel()
    fc = fc.to(device)

    md = model()

    C = 3 if USE_RGB_IMG else 1
    # 입력
    if FLAG_RESIZE:
        INPUT_SIZE = (batch_size * TARGET_INDICES, C, SIZE[0], SIZE[1])
    else:
        INPUT_SIZE = (batch_size * TARGET_INDICES, C, 10, 8)

    summary(md, input_size=INPUT_SIZE)

    # parameter
    cnn_params = list(cnn.parameters())
    rnn_params = list(rnn.parameters())
    fc_params = list(fc.parameters())

    # Optimizer
    optim = torch.optim.Adam(cnn_params + rnn_params + fc_params, learning_rate, weight_decay=weight_decay)

    # Scheduler
    scheduler = CosineAnnealingWarmUpRestarts(optim, T_0=150, eta_max=0.1, T_up=10, gamma=0.5)

    if train:
        return [cnn, rnn, fc], [cnn_params, rnn_params, fc_params], optim, scheduler
    else:
        return [cnn, rnn, fc], [cnn_params, rnn_params, fc_params]


def prediction(input, model, train=True):
    encoder, decoder, regression_model = model[0], model[1], model[2]

    C = 3 if USE_RGB_IMG else 1
    if FLAG_RESIZE:
        X = input.reshape((input.shape[0] * TARGET_INDICES, C, SIZE[0], SIZE[1]))
    else:
        X = input.reshape((input.shape[0] * TARGET_INDICES, C, 10, 8))
    X = encoder(X)

    if USE_RNN:
        X = X.reshape((input.shape[0], TARGET_INDICES, -1))
        X = decoder(X)
    else:
        X = X.reshape((input.shape[0], -1))

    pred_force = regression_model(X)

    return pred_force
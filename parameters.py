TARGET_INDICES = 1
STRIDE = 1
SIZE = (28, 28)

FLAG_RESIZE = True
INPUT_SIZE = ()
USE_RNN = True

USE_RGB_IMG = False
CHANNEL = 3

encoder_output = 1
if USE_RNN: decoder_output = encoder_output
else: decoder_output = encoder_output * TARGET_INDICES
regression_FC_dim = 64
classification_FC_dim = 128

num_hidden_of_regression = 0
num_hidden_of_classification = 3

epochs = 100000
batch_size = 64
learning_rate = 1e-7
weight_decay = 0.005
log_interval = 8
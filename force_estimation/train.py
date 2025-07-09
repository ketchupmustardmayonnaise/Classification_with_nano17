import os
import time
import argparse
import torchinfo
import yaml

from force_estimation.dataset import CFDataset
from torch.utils.data import DataLoader
from model import *
from apply_img_augmentation import *

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class ForceEstimation:
    def __init__(self, cfg):
        parser = argparse.ArgumentParser()
        parser.add_argument('--target_indices', default=5, type=int)
        parser.add_argument('--stride', default=1, type=int)
        parser.add_argument('--image_resize', default=False, type=str2bool)
        parser.add_argument('--image_size', default=None, type=tuple)

        parser.add_argument('--encoder', default=None, type=str)
        parser.add_argument('--encoder_output', default=None, type=str)
        parser.add_argument('--decoder', default=None, type=str)
        parser.add_argument('--rnn_layer', default=2, type=int)
        parser.add_argument('--bidirectional', default=False, type=str2bool)

        parser.add_argument('--use_only_output_layer', default=False, type=str2bool)
        parser.add_argument('--dropout', default=0.0, type=float)
        parser.add_argument('--fc_hidden_layer', default=None, type=int)
        parser.add_argument('--fc_hidden_dim', default=None, type=int)
        parser.add_argument('--optimizer', default=None, type=str)

        parser.add_argument('--image_type', default=None, type=str)
        parser.add_argument('--train_mode', default=True, type=str2bool)

        parser.add_argument('--batch_size', default=None, type=int)
        parser.add_argument("--num_epoch", default=None, type=int)
        parser.add_argument("--learning_rate", default=None, type=float)
        parser.add_argument("--weight_decay", default=None, type=float)

        parser.add_argument('--loss', default=None, type=str)
        args = parser.parse_args()

        # parameters
        target_indices = args.target_indices if args.target_indices is not None else cfg['target_indices']
        stride = args.stride if args.stride is not None else cfg['stride']
        image_resize = args.image_resize if args.image_resize is not None else cfg['image_resize']
        image_size = args.image_size if args.image_size is not None else cfg['image_size']
        encoder = args.encoder if args.encoder is not None else cfg['encoder']
        encoder_output = args.encoder_output if args.encoder_output is not None else cfg['encoder_output']
        decoder = args.decoder if args.decoder is not None else cfg['decoder']
        bidirectional = args.bidirectional if args.bidirectional is not None else cfg['bidirectional']
        use_only_output_layer = args.use_only_output_layer if args.use_only_output_layer is not None else cfg['use_only_output_layer']
        fc_hidden_layer = args.fc_hidden_layer if args.fc_hidden_layer is not None else cfg['fc_hidden_layer']
        fc_hidden_dim = args.fc_hidden_dim if args.fc_hidden_dim is not None else cfg['fc_hidden_dim']
        optimizer = args.optimizer if args.optimizer is not None else cfg['optimizer']
        dropout = args.dropout if args.dropout is not None else cfg['dropout']
        rnn_layer = args.rnn_layer if args.rnn_layer is not None else cfg['rnn_layer']
        self.image_type = args.image_type if args.image_type is not None else cfg['image_type']
        train_mode = args.train_mode if args.train_mode is not None else cfg['train_mode']
        batch_size = args.batch_size if args.batch_size is not None else cfg['batch_size']
        loss = args.loss if args.loss is not None else cfg['loss']

        # model configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CFModel(encoder, encoder_output, decoder, rnn_layer, bidirectional,
                             use_only_output_layer, dropout, target_indices, self.image_type,
                             fc_hidden_layer=fc_hidden_layer, fc_hidden_dim=fc_hidden_dim)
        self.model = self.model.to(self.device)
        torchinfo.summary(self.model)

        # test dataset
        self.test_dataset = CFDataset(target_indices, stride, image_type=self.image_type, image_resize=image_resize, image_size=image_size, path=cfg['test_data_dir'])
        self.test_dataloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)

        if train_mode:
            self.num_epoch = args.num_epoch if args.num_epoch is not None else cfg['num_epoch']
            self.learning_rate = args.learning_rate if args.learning_rate is not None else cfg['learning_rate']
            self.learning_rate = float(self.learning_rate)
            weight_decay = args.weight_decay if args.weight_decay is not None else cfg['weight_decay']

            # create directories
            time_suffix = time.strftime('_%m-%d_%H-%M-%S', time.localtime())
            self.save_dir = cfg['save_dir'] + "/" + encoder + "_" + decoder + time_suffix
            self.log_dir = self.save_dir + '/log'
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)

            # force_estimation dataset
            self.train_dataset = CFDataset(target_indices, stride, image_type=self.image_type, image_resize=image_resize, image_size=image_size, path=cfg['train_data_dir'])
            self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=False)

            # force_estimation dataset
            self.val_dataset = CFDataset(target_indices, stride, image_type=self.image_type, image_resize=image_resize, image_size=image_size, path=cfg['val_data_dir'])
            self.val_dataloader = torch.utils.data.DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)

            # optimizer
            if optimizer == 'SGD':
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=float(self.learning_rate))
            else:
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.learning_rate), betas=(0.9, 0.999),
                                            eps=1e-08, weight_decay=weight_decay, amsgrad=False)

            # scheduler
            #self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1.0,
            #                                       end_factor=args.end_factor, total_iters=args.total_iters)

            # loss
            self.loss = torch.nn.MSELoss() if loss == "MSE" else torch.nn.L1Loss()
            self.train()

        else:
            self.save_dir = cfg['save_dir']
            self.model.load_state_dict(torch.load(self.save_dir, model_location=self.device))

    def test(self, dataloader):
        self.model.eval()
        MSE = 0
        MAE = 0

        with torch.no_grad():
            for X, y, force in dataloader:
                X, y, force = X.float().to(self.device), y.to(self.device), force.to(self.device)

                pred_force = self.model(X)

                MSE += torch.nn.functional.mse_loss(pred_force.reshape((len(force))).float(), force.float(), reduction='mean').item()
                MAE += torch.nn.functional.l1_loss(pred_force.reshape((len(force))).float(), force.float(), reduction='mean').item()

        MSE /= len(dataloader)
        MAE /= len(dataloader)
        return MSE, MAE

    def train(self):
        msg = '[Epoch: {}/{}], Lr: {:.6f}, Train MSE: {:.3f}, Train MAE: {:.3f}' \
              ' Test MSE: {:.3f}, Test MAE: {:.3f}, {}'

        test_best_loss = 1e10

        for epoch in range(self.num_epoch):
            self.model.train()
            MSE = []
            MAE = []
            for batch_idx, (X, y, force) in enumerate(self.train_dataloader):
                self.model.train()

                X = apply_all_frames_RGB_image(X) if self.image_type == 'RGB' else apply_all_frames(X)
                X, y, force = X.float().to(self.device), y.to(self.device), force.to(self.device)

                self.optimizer.zero_grad()
                pred_force = self.model(X)

                MSEloss = torch.nn.functional.mse_loss(pred_force.reshape((len(force))).float(), force.float())
                L1_loss = torch.nn.functional.l1_loss(pred_force.reshape((len(force))).float(), force.float())

                MSE.append(MSEloss.item())
                MAE.append(L1_loss.item())

                loss = self.loss(pred_force.reshape((len(force))).float(), force.float())
                loss.backward()
                self.optimizer.step()

            #self.scheduler.step()

            # test the model
            test_MSE, test_MAE = self.test(self.val_dataloader)

            # best model select
            if test_MAE < test_best_loss:
                test_best_loss = test_MAE
                torch.save(self.model.state_dict(), self.save_dir + "/epoch_" + str(epoch + 1) + ".pt")
                improve = "******"
            else: improve = " "

            print(msg.format(
                epoch + 1,  # {}/{} 중 첫 번째
                self.num_epoch,  # {}/{} 중 두 번째
                float(self.learning_rate),
                float(np.mean(MSE)),  # {:.3f}
                float(np.mean(MAE)),  # {:.3f}
                float(test_MSE),  # {:.3f}
                float(test_MAE),  # {:.3f}
                improve
            ))

if __name__ == '__main__':
    f = open("config.yaml", 'r+', encoding='utf-8')
    config = yaml.load(f, Loader=yaml.FullLoader)
    force_estimation = ForceEstimation(config)
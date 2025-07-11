import os
import sys
import time
import argparse
import torchinfo
import yaml
import wandb
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from dataset import CFDataset
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
        parser.add_argument('--train_mode', default=None, type=str2bool)

        parser.add_argument('--encoder', default=None, type=str)
        parser.add_argument('--resnet_block', default=None, type=tuple)
        parser.add_argument('--densenet_block', default=None, type=tuple)
        parser.add_argument('--densenet_growth', default=None, type=int)
        parser.add_argument('--encoder_output', default=None, type=str)

        parser.add_argument('--decoder', default=None, type=str)
        parser.add_argument('--rnn_layer', default=None, type=int)
        parser.add_argument('--decoder_output', default=None, type=int)

        parser.add_argument('--fc_num_layer', default=None, type=int)
        parser.add_argument('--fc_hidden_dim', default=None, type=int)

        parser.add_argument('--use_all_sequence', default=None, type=str2bool)
        parser.add_argument('--target_indices', default=None, type=int)
        parser.add_argument('--stride', default=None, type=int)

        parser.add_argument('--image_type', default=None, type=str)
        parser.add_argument('--image_resize', default=None, type=str2bool)
        parser.add_argument('--image_size', default=None, type=tuple)

        parser.add_argument('--optimizer', default=None, type=str)
        parser.add_argument('--batch_size', default=None, type=int)
        parser.add_argument("--num_epoch", default=None, type=int)
        parser.add_argument("--learning_rate", default=None, type=float)
        parser.add_argument("--weight_decay", default=None, type=float)
        parser.add_argument('--dropout', default=None, type=float)
        parser.add_argument('--loss', default=None, type=str)
        args = parser.parse_args()

        # parameters
        train_mode = args.train_mode if args.train_mode is not None else cfg['train_mode']
        target_indices = args.target_indices if args.target_indices is not None else cfg['target_indices']
        stride = args.stride if args.stride is not None else cfg['stride']
        self.image_type = args.image_type if args.image_type is not None else cfg['image_type']
        image_resize = args.image_resize if args.image_resize is not None else cfg['image_resize']
        image_size = args.image_size if args.image_size is not None else cfg['image_size']

        encoder = args.encoder if args.encoder is not None else cfg['encoder']
        resnet_block = args.resnet_block if args.resnet_block is not None else cfg['resnet_block']
        densenet_block = args.densenet_block if args.densenet_block is not None else cfg['densenet_block']
        densenet_growth = args.densenet_growth if args.densenet_growth is not None else cfg['densenet_growth']
        encoder_output = args.encoder_output if args.encoder_output is not None else cfg['encoder_output']

        decoder = args.decoder if args.decoder is not None else cfg['decoder']
        rnn_layer = args.rnn_layer if args.rnn_layer is not None else cfg['rnn_layer']
        decoder_output = args.decoder_output if args.decoder_output is not None else cfg['decoder_output']

        self.use_all_sequence = args.use_all_sequence if args.use_all_sequence is not None else cfg['use_all_sequence']
        fc_num_layer = args.fc_num_layer if args.fc_num_layer is not None else cfg['fc_num_layer']
        fc_hidden_dim = args.fc_hidden_dim if args.fc_hidden_dim is not None else cfg['fc_hidden_dim']

        optimizer = args.optimizer if args.optimizer is not None else cfg['optimizer']
        self.num_epoch = args.num_epoch if args.num_epoch else cfg['num_epoch']
        self.learning_rate = args.learning_rate if args.learning_rate is not None else cfg['learning_rate']
        weight_decay = args.weight_decay if args.weight_decay is not None else cfg['weight_decay']
        dropout = args.dropout if args.dropout is not None else cfg['dropout']
        self.batch_size = args.batch_size if args.batch_size is not None else cfg['batch_size']
        self.batch_size = self.batch_size if self.use_all_sequence is False else 1
        self.loss_type = args.loss if args.loss is not None else cfg['loss']

        # model configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CFModel(encoder, resnet_block, densenet_block, densenet_growth, encoder_output,
                             decoder, rnn_layer, fc_num_layer,
                             target_indices, self.image_type, dropout,
                             fc_hidden_dim=fc_hidden_dim)
        self.model = self.model.to(self.device)
        torchinfo.summary(self.model, input_size=(self.batch_size, target_indices, 3 if self.image_type == "RGB" else 1, image_size[0], image_size[1]))

        # test dataset
        self.test_dataset = CFDataset(target_indices, stride, use_all_sequence=self.use_all_sequence, image_type=self.image_type, image_resize=image_resize, image_size=image_size, path=cfg['test_data_dir'])
        self.test_dataloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

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
            self.train_dataset = CFDataset(target_indices, stride, use_all_sequence=self.use_all_sequence, image_type=self.image_type, image_resize=image_resize, image_size=image_size, path=cfg['train_data_dir'])
            self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False)

            # force_estimation dataset
            self.val_dataset = CFDataset(target_indices, stride, use_all_sequence=self.use_all_sequence, image_type=self.image_type, image_resize=image_resize, image_size=image_size, path=cfg['val_data_dir'])
            self.val_dataloader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

            # optimizer
            if optimizer == 'SGD':
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=float(self.learning_rate), weight_decay=weight_decay)
            else:
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.learning_rate), betas=(0.9, 0.999),
                                            eps=1e-08, weight_decay=weight_decay, amsgrad=False)

            # scheduler
            #self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1.0,
            #                                       end_factor=args.end_factor, total_iters=args.total_iters)

            # loss
            self.loss = torch.nn.MSELoss() if self.loss_type == "MSE" else torch.nn.L1Loss()
            self.write_basic_information( self.use_all_sequence, target_indices, stride,
                                          self.image_type, image_resize, image_size,
                                         encoder, resnet_block, densenet_block, densenet_growth, encoder_output,
                                         fc_num_layer, fc_hidden_dim,
                                         decoder, rnn_layer, decoder_output,
                                         optimizer, dropout, self.batch_size, self.loss_type, self.learning_rate, weight_decay)

            # wandb
            wandb_config = {
                "encoder": encoder,
                "resnet_block": resnet_block,
                "densenet_block": densenet_block,
                "densenet_growth": densenet_growth,
                "encoder_output": encoder_output,

                "decoder": decoder,
                "rnn_layer": rnn_layer,
                "decoder_output": decoder_output,

                "fc_num_layer": fc_num_layer,
                "fc_hidden_dim": fc_hidden_dim,

                "use_all_sequence": self.use_all_sequence,
                "target_indices": target_indices,
                "stride": stride,

                "image_type": self.image_type,
                "image_resize": image_resize,
                "image_size": image_size,

                "optimizer": optimizer,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "weight_decay": weight_decay,
                "dropout": dropout,
                "loss": self.loss_type,
            }

            run_name = encoder + '_' + decoder + '_' + optimizer + '_lr-' + f'{self.learning_rate:.4f}'
            self.run_log = wandb.init(project="force_estimation", config=wandb_config, name=run_name)

            self.train()

        else:
            self.save_dir = cfg['save_dir']
            self.model.load_state_dict(torch.load(self.save_dir))
            self.test()

    def write_basic_information(self, use_all_sequence, target_indices, stride,
                                image_type, image_resize, image_size,
                                encoder, resnet_block, densenet_block, densenet_growth, encoder_output,
                                fc_num_layer, fc_hidden_dim,
                                decoder, rnn_layer, decoder_output,
                                optimizer, dropout, batch_size, loss, learning_rate, weight_decay):
        basic_txt_path = self.save_dir + "/basic.txt"
        basic_txt_file = open(basic_txt_path, 'a')
        basic_txt_file.write(f"encoder: {encoder}\n")
        basic_txt_file.write(f"resnet_block: {resnet_block}")
        basic_txt_file.write(f"densenet_block: {densenet_block}")
        basic_txt_file.write(f"densenet_growth: {densenet_growth}")
        basic_txt_file.write(f"encoder_output: {encoder_output}\n")

        basic_txt_file.write(f"use_all_sequence: {use_all_sequence}\n")
        basic_txt_file.write(f"target_indices: {target_indices}\n")
        basic_txt_file.write(f"stride: {stride}\n")

        basic_txt_file.write(f"image_type: {image_type}\n")
        basic_txt_file.write(f"image_resize: {image_resize}\n")
        basic_txt_file.write(f"image_size: {image_size}\n")

        basic_txt_file.write(f"decoder: {decoder}\n")
        basic_txt_file.write(f"rnn_layer: {rnn_layer}\n")
        basic_txt_file.write(f"decoder_output: {decoder_output}\n")

        basic_txt_file.write(f"fc_num_layer: {fc_num_layer}\n")
        basic_txt_file.write(f"fc_hidden_dim: {fc_hidden_dim}\n")

        basic_txt_file.write(f"optimizer: {optimizer}\n")
        basic_txt_file.write(f"dropout: {dropout}\n")

        basic_txt_file.write(f"loss: {loss}\n")
        basic_txt_file.write(f"batch_size: {batch_size}\n")
        basic_txt_file.write(f"learning_rate: {learning_rate}\n")
        basic_txt_file.write(f"weight_decay: {weight_decay}\n")
        basic_txt_file.close()

    def validate(self, dataloader):
        self.model.eval()
        MSE = 0
        MAE = 0

        with torch.no_grad():
            for X, y, force in dataloader:
                X, y, force = X.float().to(self.device), y.to(self.device), force.to(self.device)

                pred_force = self.model(X)

                if self.use_all_sequence:
                    MSE += torch.nn.functional.mse_loss(pred_force.float(), force.reshape((force.shape[1])).float())
                    MAE += torch.nn.functional.l1_loss(pred_force.float(), force.reshape((force.shape[1])).float())
                else:
                    MSE += torch.nn.functional.mse_loss(pred_force.reshape((len(force))).float(), force.float(), reduction='mean').item()
                    MAE += torch.nn.functional.l1_loss(pred_force.reshape((len(force))).float(), force.float(), reduction='mean').item()

        MSE /= len(dataloader)
        MAE /= len(dataloader)
        return MSE, MAE

    def train(self):
        msg = '[Epoch: {}/{}], Lr: {:.6f}, Train MSE: {:.3f}, Train MAE: {:.3f}' \
              ' Test MSE: {:.3f}, Test MAE: {:.3f}, {}'

        val_best_loss = 1e10
        train_txt_path = self.save_dir + "/train.txt"
        train_txt_file = open(train_txt_path, 'a')

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

                if self.use_all_sequence:
                    MSEloss = torch.nn.functional.mse_loss(pred_force.float(), force.squeeze().float())
                    L1_loss = torch.nn.functional.l1_loss(pred_force.float(), force.squeeze().float())
                    loss = self.loss(pred_force.float(), force.reshape((force.shape[1])).float())
                else:
                    MSEloss = torch.nn.functional.mse_loss(pred_force.squeeze().float(), force.float(), reduction='mean').item()
                    L1_loss = torch.nn.functional.l1_loss(pred_force.squeeze().float(), force.float(), reduction='mean').item()
                    loss = self.loss(pred_force.reshape((len(force))).float(), force.float())

                MSE.append(MSEloss.item())
                MAE.append(L1_loss.item())

                loss.backward()
                self.optimizer.step()

            #self.scheduler.step()

            # test the model
            val_MSE, val_MAE = self.validate(self.val_dataloader)

            # best model select
            if self.loss_type == "MAE" and val_MAE < val_best_loss:
                val_best_loss = val_MAE
                torch.save(self.model.state_dict(), self.save_dir + "/epoch_" + str(epoch + 1) + ".pt")
                improve = "******"
            elif self.loss_type == "MSE" and val_MSE < val_best_loss:
                val_best_loss = val_MSE
                torch.save(self.model.state_dict(), self.save_dir + "/epoch_" + str(epoch + 1) + ".pt")
                improve = "******"
            else: improve = " "

            print(msg.format(
                epoch + 1,  # {}/{} 중 첫 번째
                self.num_epoch,  # {}/{} 중 두 번째
                float(self.learning_rate),
                float(np.mean(MSE)),  # {:.3f}
                float(np.mean(MAE)),  # {:.3f}
                float(val_MSE),  # {:.3f}
                float(val_MAE),  # {:.3f}
                improve
            ))

            train_txt_file.write(msg.format(
                epoch + 1,  # {}/{} 중 첫 번째
                self.num_epoch,  # {}/{} 중 두 번째
                float(self.learning_rate),
                float(np.mean(MSE)),  # {:.3f}
                float(np.mean(MAE)),  # {:.3f}
                float(val_MSE),  # {:.3f}
                float(val_MAE),  # {:.3f}
                improve
            ) + '\n')

            self.run_log.log({"train_MSE": np.mean(MSE), "train_MAE": np.mean(MAE),
                              "test_MSE": val_MSE, "test_MAE": val_MAE})

        self.run_log.finish()


    def test(self):
        self.model.eval()
        MSE = 0
        MAE = 0

        f = []
        pf = []

        with torch.no_grad():
            for X, y, force in self.test_dataloader:
                X, y, force = X.float().to(self.device), y.to(self.device), force.to(self.device)
                _force = force.detach().cpu().numpy()[0].squeeze()
                f.append(_force)

                pred_force = self.model(X)
                _pred_force = pred_force.detach().cpu().numpy().squeeze()
                pf.append(_pred_force)

                if self.use_all_sequence:
                    MSE += torch.nn.functional.mse_loss(pred_force.float(), force.reshape((force.shape[1])).float())
                    MAE += torch.nn.functional.l1_loss(pred_force.float(), force.reshape((force.shape[1])).float())
                else:
                    MSE += torch.nn.functional.mse_loss(pred_force.reshape((len(force))).float(), force.float(),
                                                        reduction='mean').item()
                    MAE += torch.nn.functional.l1_loss(pred_force.reshape((len(force))).float(), force.float(),
                                                       reduction='mean').item()

        for index in range(12):
            _f = f[index]
            _pf = pf[index]
            plt.subplot(3, 4, index + 1)
            plt.plot(np.array(_f), color='r', label='Force')
            plt.plot(np.array(_pf), color='b', label='Predicted Force')

        MSE /= len(self.test_dataloader)
        MAE /= len(self.test_dataloader)
        print("Train: \tMSE loss: {:.5f}\tL1 loss: {:.5f}".format(MSE, MAE))
        plt.legend()
        plt.show()
        return MSE, MAE

if __name__ == '__main__':
    f = open("config.yaml", 'r+', encoding='utf-8')
    config = yaml.load(f, Loader=yaml.FullLoader)
    force_estimation = ForceEstimation(config)
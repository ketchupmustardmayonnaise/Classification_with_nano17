import os
import sys
import time
import argparse
import torchinfo
import yaml
import ast
import shutil
from torch.optim import lr_scheduler
import numpy as np
import wandb
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from util import *
from Classification_with_nano17.force_estimation.dataPreprocessing import CFDataset
from model import *
from apply_img_augmentation import *


class ForceEstimation:
    def __init__(self, cfg):
        parser = argparse.ArgumentParser()
        parser.add_argument('--train_mode', default=None, type=str2bool)
        parser.add_argument('--use_wandb', default=None, type=str2bool)

        parser.add_argument('--encoder', default=None, type=str)
        parser.add_argument('--encoder_output', default=None, type=str)

        parser.add_argument('--decoder', default=None, type=str)
        parser.add_argument('--rnn_layer', default=None, type=int)
        parser.add_argument('--decoder_output', default=None, type=int)

        parser.add_argument('--fc_num_layer', default=None, type=int)
        parser.add_argument('--fc_hidden_dim', default=None, type=int)

        parser.add_argument('--target_indices', default=None, type=int)
        parser.add_argument('--stride', default=None, type=int)
        parser.add_argument('--augmentation_angle', default=None, type=int)

        parser.add_argument('--image_type', default=None, type=str)
        parser.add_argument('--image_resize', default=None, type=str2bool)
        parser.add_argument('--image_size', default=None, type=ast.literal_eval)

        parser.add_argument('--save_dir', default=None, type=str)
        parser.add_argument('--test_data_dir', default=None, type=str)
        parser.add_argument('--train_data_dir', default=None, type=str)
        parser.add_argument('--val_data_dir', default=None, type=str)

        parser.add_argument('--optimizer', default=None, type=str)
        parser.add_argument('--scheduler', default=None, type=str)
        parser.add_argument('--batch_size', default=None, type=int)
        parser.add_argument("--num_epoch", default=None, type=int)
        parser.add_argument("--learning_rate", default=None, type=float)
        parser.add_argument("--weight_decay", default=None, type=float)
        parser.add_argument('--dropout', default=None, type=float)
        parser.add_argument('--loss', default=None, type=str)
        args = parser.parse_args()

        # parameters
        self.mode = args.train_mode if args.train_mode is not None else cfg['train_mode'] # true일 때만 학습 (bool)
        self.use_wandb = args.use_wandb if args.use_wandb is not None else cfg['use_wandb'] # wandb에 올릴 것인지 (bool)
        target_indices = int(args.target_indices) if args.target_indices is not None else cfg['target_indices'] # 몇 frame으로 자를 것인지 (bool)
        stride = int(args.stride) if args.stride is not None else cfg['stride'] # 몇 frame만큼 띄워서 학습할 것인지 (int)
        self.image_type = args.image_type if args.image_type is not None else cfg['image_type'] # origin, rgb, diff (str)
        image_resize = args.image_resize if args.image_resize is not None else cfg['image_resize'] # 이미지를 확대할 것인지 (bool)
        image_size = args.image_size if args.image_size is not None else cfg['image_size'] # 확대한다면, 이미지를 얼마나 확대할 것인지 (int, int)
        image_size = [int(image_size[0]), int(image_size[1])]
        self.augmentation_angle = int(args.augmentation_angle) if args.augmentation_angle is not None else cfg['augmentation_angle'] # online augmentation 시, 각도 최대치 (int)

        encoder = args.encoder if args.encoder is not None else cfg['encoder'] # 어떤 cnn을 쓸 것인지 (str)
        encoder_output = int(args.encoder_output) if args.encoder_output is not None else cfg['encoder_output'] # cnn에서 결과적으로 나온 embedding_vector의 차원 (int)

        # decoder: (bool, str)
        # 첫 번째 parameter: use_all_sequence (하나의 데이터 시퀀스를 모두 순차적으로 학습할 것인지) (bool)
        # 두 번째 parameter: 어떤 시계열 모델을 사용할 것인지 (str)
        decoder = args.decoder if args.decoder is not None else cfg['decoder']
        rnn_layer = int(args.rnn_layer) if args.rnn_layer is not None else cfg['rnn_layer'] # 시계열 모델 내 layer의 개수 (int)
        decoder_output = int(args.decoder_output) if args.decoder_output is not None else cfg['decoder_output'] # 시계열 모델에서 결과적으로 나온 embedding_vector의 차원 (int)

        self.use_all_sequence = decoder[0] # 하나의 데이터 시퀀스를 모두 순차적으로 학습할 것인지 (bool)
        decoder = decoder[1] # 어떤 시계열 모델을 사용할 것인지

        fc_num_layer = int(args.fc_num_layer) if args.fc_num_layer is not None else cfg['fc_num_layer'] # fully-connected layer의 개수 (int)
        fc_hidden_dim = int(args.fc_hidden_dim) if args.fc_hidden_dim is not None else cfg['fc_hidden_dim'] # fully-connected layer의 hidden layer의 dimension 개수

        optimizer = args.optimizer if args.optimizer is not None else cfg['optimizer']
        scheduler = args.scheduler if args.scheduler is not None else cfg['scheduler']
        self.num_epoch = int(args.num_epoch) if args.num_epoch else cfg['num_epoch']
        dropout = float(args.dropout) if args.dropout is not None else cfg['dropout']
        self.batch_size = args.batch_size if args.batch_size is not None else cfg['batch_size']
        # use_all_sequence = True라면 batch_size=1로 고정됨. 각 시퀀스마다 길이가 다른데, tensor는 같은 크기의 데이터만 담을 수 있기 때문
        self.batch_size = int(self.batch_size) if self.use_all_sequence is False else 1
        self.loss_type = args.loss if args.loss is not None else cfg['loss']

        # model configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CFModel(encoder, encoder_output,
                             decoder, rnn_layer, fc_num_layer, self.use_all_sequence,
                             target_indices, self.image_type, dropout,
                             fc_hidden_dim=fc_hidden_dim)
        self.model = self.model.to(self.device)
        torchinfo.summary(self.model, input_size=(self.batch_size, target_indices, 3 if self.image_type.lower() == "rgb" else 1, image_size[0], image_size[1]))

        # test dataset
        self.test_dataset = CFDataset(target_indices, stride, use_all_sequence=self.use_all_sequence, image_type=self.image_type, image_resize=image_resize, image_size=image_size, path=cfg['test_data_dir'])
        self.test_dataloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

        if train_mode:
            self.best_epoch = 0
            self.num_epoch = args.num_epoch if args.num_epoch is not None else cfg['num_epoch']
            self.learning_rate = args.learning_rate if args.learning_rate is not None else cfg['learning_rate']
            self.learning_rate = float(self.learning_rate)
            weight_decay = args.weight_decay if args.weight_decay is not None else cfg['weight_decay']

            # create directories
            time_suffix = time.strftime('_%m-%d_%H-%M-%S', time.localtime())
            self.save_dir = cfg['save_dir'] + "/" + encoder[0] + "_" + decoder + time_suffix
            self.log_dir = self.save_dir + '/log'
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            copy_config(self.save_dir)

            # train dataset
            self.train_dataset = CFDataset(target_indices, stride, use_all_sequence=self.use_all_sequence, image_type=self.image_type, image_resize=image_resize, image_size=image_size, path=cfg['train_data_dir'])
            self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False)

            # validation dataset
            self.val_dataset = CFDataset(target_indices, stride, use_all_sequence=self.use_all_sequence, image_type=self.image_type, image_resize=image_resize, image_size=image_size, path=cfg['val_data_dir'])
            self.val_dataloader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

            # optimizer
            if optimizer.lower() == 'adam':
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, betas=(0.9, 0.999),
                                            eps=1e-08, weight_decay=weight_decay, amsgrad=False)
            elif optimizer.lower() == 'rmsprop':
                self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.learning_rate, weight_decay=weight_decay)
            else:
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate,
                                                 weight_decay=weight_decay)

            # scheduler
            if scheduler.lower() == 'step':
                self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.99)
            else: self.scheduler = None

            #self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1.0,
            #                                       end_factor=args.end_factor, total_iters=args.total_iters)

            # loss
            if self.loss_type != "MSE+MAE":
                self.loss = torch.nn.MSELoss() if self.loss_type == "MSE" else torch.nn.L1Loss()
            self.write_basic_information( self.use_wandb, self.use_all_sequence, target_indices, stride,
                                          self.image_type, image_resize, image_size, self.augmentation_angle,
                                         encoder, encoder_output,
                                         fc_num_layer, fc_hidden_dim,
                                         decoder, rnn_layer, decoder_output,
                                         optimizer, scheduler, dropout, self.batch_size, self.loss_type, self.learning_rate, weight_decay)

            if self.use_wandb:
                # wandb
                wandb_config = {
                    "encoder": encoder,
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
                    "augmentation_angle": self.augmentation_angle,

                    "optimizer": optimizer,
                    "scheduler": scheduler,
                    "batch_size": self.batch_size,
                    "learning_rate": self.learning_rate,
                    "weight_decay": weight_decay,
                    "dropout": dropout,
                    "loss": self.loss_type,
                }

                run_name = encoder[0] + '_' + decoder + '_' + optimizer + '_lr-' + f'{time_suffix}'
                self.run_log = wandb.init(project="force_estimation", config=wandb_config, name=run_name)



    def write_basic_information(self, use_wandb, use_all_sequence, target_indices, stride,
                                image_type, image_resize, image_size, augmentation_angle,
                                encoder, encoder_output,
                                fc_num_layer, fc_hidden_dim,
                                decoder, rnn_layer, decoder_output,
                                optimizer, scheduler, dropout, batch_size, loss, learning_rate, weight_decay):
        basic_txt_path = self.save_dir + "/basic.txt"
        basic_txt_file = open(basic_txt_path, 'a')
        basic_txt_file.write(f"use_wandb: {use_wandb}\n\n") # wandb를 사용할 것인지?
        basic_txt_file.write(f"encoder: {encoder}\n")
        basic_txt_file.write(f"encoder_output: {encoder_output}\n\n")

        basic_txt_file.write(f"use_all_sequence: {use_all_sequence}\n")
        basic_txt_file.write(f"target_indices: {target_indices}\n")
        basic_txt_file.write(f"stride: {stride}\n\n")

        basic_txt_file.write(f"image_type: {image_type}\n")
        basic_txt_file.write(f"image_resize: {image_resize}\n")
        basic_txt_file.write(f"image_size: {image_size}\n")
        basic_txt_file.write(f"augmentation_angle: {augmentation_angle}\n\n")

        basic_txt_file.write(f"decoder: {decoder}\n")
        basic_txt_file.write(f"rnn_layer: {rnn_layer}\n")
        basic_txt_file.write(f"decoder_output: {decoder_output}\n\n")

        basic_txt_file.write(f"fc_num_layer: {fc_num_layer}\n")
        basic_txt_file.write(f"fc_hidden_dim: {fc_hidden_dim}\n\n")

        basic_txt_file.write(f"optimizer: {optimizer}\n")
        basic_txt_file.write(f"scheduler: {scheduler}\n")
        basic_txt_file.write(f"dropout: {dropout}\n\n")

        basic_txt_file.write(f"loss: {loss}\n")
        basic_txt_file.write(f"batch_size: {batch_size}\n")
        basic_txt_file.write(f"learning_rate: {learning_rate}\n")
        basic_txt_file.write(f"weight_decay: {weight_decay}\n\n")
        basic_txt_file.close()

    def validation(self, dataloader):
        self.model.eval()
        MSE = 0
        MAE = 0

        with torch.no_grad():
            for X, y, force in dataloader:
                X, y, force = X.float().to(self.device), y.to(self.device), force.to(self.device)

                pred_force = self.model(X)

                if self.use_all_sequence:
                    MSE += torch.nn.functional.mse_loss(pred_force.squeeze().float(), force.squeeze().float())
                    MAE += torch.nn.functional.l1_loss(pred_force.squeeze().float(), force.squeeze().float())
                else:
                    MSE += torch.nn.functional.mse_loss(pred_force.squeeze().float(), force.squeeze().float(), reduction='mean')
                    MAE += torch.nn.functional.l1_loss(pred_force.squeeze().float(), force.squeeze().float(), reduction='mean')

        MSE /= len(dataloader)
        MAE /= len(dataloader)
        return MSE, MAE
    
    

    def train(self):
        msg = '[Epoch: {}/{}], Lr: {:.6f}, Train MSE: {:.3f}, Train MAE: {:.3f}' \
              ' Test MSE: {:.3f}, Test MAE: {:.3f}, {}'

        val_best_loss = 1e10

        for epoch in range(self.num_epoch):
            self.model.train()
            MSE = []
            MAE = []
            for batch_idx, (X, y, force) in enumerate(self.train_dataloader):
                self.model.train()

                X = apply_augmentation_to_all_frame(X, self.augmentation_angle)
                X, y, force = X.float().to(self.device), y.to(self.device), force.to(self.device)

                self.optimizer.zero_grad()
                pred_force = self.model(X)

                if self.use_all_sequence:
                    MSEloss = torch.nn.functional.mse_loss(pred_force.squeeze().float(), force.squeeze().float(), reduction='mean')
                    L1_loss = torch.nn.functional.l1_loss(pred_force.squeeze().float(), force.squeeze().float(), reduction='mean')
                    if self.loss_type == 'MSE+MAE':
                        loss = (MSEloss + L1_loss) / 2
                    else: loss = self.loss(pred_force.squeeze().float(), force.squeeze().float())
                else:
                    MSEloss = torch.nn.functional.mse_loss(pred_force.squeeze().float(), force.squeeze().float(), reduction='mean')
                    L1_loss = torch.nn.functional.l1_loss(pred_force.squeeze().float(), force.squeeze().float(), reduction='mean')
                    if self.loss_type == 'MSE+MAE':
                        loss = (MSEloss + L1_loss) / 2
                    else:
                        loss = self.loss(pred_force.squeeze().float(), force.squeeze().float())

                MSE.append(MSEloss.item())
                MAE.append(L1_loss.item())

                loss.backward()
                self.optimizer.step()

            if self.scheduler is not None: self.scheduler.step()

            # test the model
            val_MSE, val_MAE = self.validate(self.val_dataloader)

            # best model select
            if self.loss_type == "MAE" and val_MAE < val_best_loss:
                self.best_epoch = epoch + 1
                val_best_loss = val_MAE
                torch.save(self.model.state_dict(), self.save_dir + "/epoch_" + str(epoch + 1) + ".pt")
                improve = "******"
            elif self.loss_type == "MSE" and val_MSE < val_best_loss:
                self.best_epoch = epoch + 1
                val_best_loss = val_MSE
                torch.save(self.model.state_dict(), self.save_dir + "/epoch_" + str(epoch + 1) + ".pt")
                improve = "******"
            elif self.loss_type == "MSE+MAE" and (val_MSE + val_MAE) / 2 < val_best_loss:
                self.best_epoch = epoch + 1
                val_best_loss = (val_MSE + val_MAE) / 2
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

            train_txt_path = self.save_dir + "/train.txt"
            with open(train_txt_path, 'a') as train_txt_file:
                train_txt_file.write(msg.format(
                    epoch + 1,
                    self.num_epoch,
                    float(self.learning_rate),
                    float(np.mean(MSE)),
                    float(np.mean(MAE)),
                    float(val_MSE),
                    float(val_MAE),
                    improve
                ) + '\n')

            if self.use_wandb:
                try:
                    wandb.log({"train_MSE": np.mean(MSE), "train_MAE": np.mean(MAE),
                                  "validation_MSE": val_MSE, "validation_MAE": val_MAE})
                except:
                    wandb.finish()

        self.model.load_state_dict(torch.load(self.save_dir + "/epoch_{}.pt".format(self.best_epoch)))
        self.test(self.save_dir)
        if self.use_wandb: wandb.finish()


    def test(self, save_dir):
        self.model.eval()
        MSE = 0
        MAE = 0

        f = []
        pf = []

        with torch.no_grad():
            for X, y, force in self.test_dataloader:
                X, y, force = X.float().to(self.device), y.to(self.device), force.to(self.device)
                _force = force.detach().cpu().numpy().squeeze().tolist()
                f.append(_force)

                pred_force = self.model(X)
                _pred_force = pred_force.detach().cpu().numpy().squeeze().tolist()
                pf.append(_pred_force)

                if self.use_all_sequence:
                    MSE += torch.nn.functional.mse_loss(pred_force.float(), force.reshape((force.shape[1])).float())
                    MAE += torch.nn.functional.l1_loss(pred_force.float(), force.reshape((force.shape[1])).float())
                else:
                    MSE += torch.nn.functional.mse_loss(pred_force.reshape((len(force))).float(), force.float(),
                                                        reduction='mean').item()
                    MAE += torch.nn.functional.l1_loss(pred_force.reshape((len(force))).float(), force.float(),
                                                       reduction='mean').item()

        if self.use_all_sequence:
            f = sum(f, [])
            pf = sum(pf, [])

        for start in range(0, 36, 12):
            i = 0
            end = start + 12

            for index in range(40):
                _f = f[i : i + self.test_dataset.frame[index]]
                _pf = pf[i : i + self.test_dataset.frame[index]]
                if index > start and index <= end:
                    plt.subplot(3, 4, index - start)
                    plt.plot(np.array(_f), color='r', label='Force')
                    plt.plot(np.array(_pf), color='b', label='Predicted Force')
                i += self.test_dataset.frame[index]

            plt.legend()
            plt.savefig(save_dir + f'/test_{int(start / 12) + 1}.png', dpi=300, facecolor='white', edgecolor='black',
                format='png', bbox_inches='tight', pad_inches=0.1, transparent=False)
            plt.show()

        MSE /= len(self.test_dataloader)
        MAE /= len(self.test_dataloader)
        print("Train: \tMSE loss: {:.5f}\tL1 loss: {:.5f}".format(MSE, MAE))
        return MSE, MAE
    
    def setup_run(self) : 
        time_suffix = time.strftime('_%m-%d_%H-%M-%S', time.localtime())
        self.save_dir = self.cfg['save_dir'] + "/" + self.encoder[0] + "_" + self.decoder + time_suffix
        self.log_dir = self.save_dir + '/log'
        
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        copy_config(self.save_dir)
        self.write_basic_information(...) # self에 저장된 파라미터 사용

        if self.use_wandb:
            wandb_config = { ... } # self에 저장된 파라미터로 config 생성
            run_name = ...
            self.run_log = wandb.init(project="force_estimation", config=wandb_config, name=run_name)

    
    def run(self) :
        if self.mode == 'Train':
            self.setup_run() # 훈련 전 준비 작업 실행
            self.train()
        elif self.mode == 'test' :
            self.test()
        else:
            model_path = self.cfg['model_dir']
            self.model.load_state_dict(torch.load(model_path))
            epoch_idx = model_path.find("epoch")
            self.test(model_path[:epoch_idx-1])

if __name__ == '__main__':
    with open("config.yaml", 'r+', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    force_estimation = ForceEstimation(config)
    
    force_estimation.run()
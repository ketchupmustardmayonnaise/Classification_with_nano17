import os
import time
import torchinfo
import yaml
import ast
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
    def __init__(self, cfg, args):
        # parameters
        self.cfg = cfg
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.best_epoch = 0

        # 1. 파라미터 초기화를 위한 헬퍼 메소드 정의
        def _init_param(key, p_type=None):
            """args에 값이 있으면 가져오고, 없으면 cfg에서 가져옵니다. 필요한 경우 타입도 변환합니다."""
            value = getattr(self.args, key, None)
            if value is None:
                value = self.cfg.get(key)
            
            if p_type is not None:
                # str2bool 같은 커스텀 함수도 처리 가능
                value = p_type(value)
                
            setattr(self, key, value)

        # 2. 공통 파라미터 리스트 정의 (파라미터명, 타입)
        # 타입이 None이면 변환 없이 그대로 사용
        common_params = [
            ('train_mode', str2bool), # train일 때만 학습 (bool)
            ('use_wandb', str2bool), # wandb에 올릴 것인지 (bool)
            ('image_resize', str2bool), # 이미지를 확대할 것인지 (bool)
            ('target_indices', int), # 몇 frame으로 자를 것인지 (bool)
            ('stride', int), # 몇 frame만큼 띄워서 학습할 것인지 (int)
            ('augmentation_angle', int), # online augmentation 시, 각도 최대치 (int)
            ('num_epoch', int),
            ('encoder_output', int), # cnn에서 결과적으로 나온 embedding_vector의 차원 (int)
            ('rnn_layer', int), # 시계열 모델 내 layer의 개수 (int)
            ('decorder', str) # 어떤 시계열 모델을 사용할 것인지 (str)
            ('decoder_output', int), # 시계열 모델에서 결과적으로 나온 embedding_vector의 차원 (int)
            ('fc_num_layer', int), # fully-connected layer의 개수 (int)
            ('fc_hidden_dim', int), # fully-connected layer의 hidden layer의 dimension 개수
            ('learning_rate', float), 
            ('weight_decay', float), 
            ('dropout', float),
            ('batch_size', int),
            ('image_type', str), # origin, rgb, diff (str)
            ('encoder', str), # 어떤 cnn을 쓸 것인지 (str)
            ('optimizer_type', str),
            ('scheduler_type', str), 
            ('loss_type', str), 
            ('image_size', ast.literal_eval) # 확대한다면, 이미지를 얼마나 확대할 것인지 (int, int)
        ]

        # 3. 반복문을 통해 공통 파라미터 한 번에 초기화
        for key, p_type in common_params:
            _init_param(key, p_type)     

        image_size = [int(image_size[0]), int(image_size[1])]

        # model configuration
        self.model = CFModel(self.encoder, self.encoder_output,
                             self.decoder, self.rnn_layer, self.fc_num_layer, self.use_all_sequence,
                             self.target_indices, self.image_type, self.dropout,
                             fc_hidden_dim=self.fc_hidden_dim)
        self.model = self.model.to(self.device)
        torchinfo.summary(self.model, input_size=(self.batch_size, self.target_indices, 3 if self.image_type.lower() == "rgb" else 1, image_size[0], image_size[1]))

        # test dataset
        self.test_dataset = CFDataset(self.target_indices, self.stride, use_all_sequence=self.use_all_sequence, image_type=self.image_type, image_resize=self.image_resize, image_size=image_size, path=cfg['test_data_dir'])
        self.test_dataloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)



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

        # train dataset
        self.train_dataset = CFDataset(self.target_indices, self.stride, use_all_sequence=self.use_all_sequence, image_type=self.image_type, image_resize=self.image_resize, image_size=self.image_size, path=self.cfg['train_data_dir'])
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False)

        # validation dataset
        self.val_dataset = CFDataset(self.target_indices, self.stride, use_all_sequence=self.use_all_sequence, image_type=self.image_type, image_resize=self.image_resize, image_size=self.image_size, path=self.cfg['val_data_dir'])
        self.val_dataloader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

        # optimizer
        if self.optimizer_type.lower() == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, betas=(0.9, 0.999),
                                        eps=1e-08, weight_decay=self.weight_decay, amsgrad=False)
        elif self.optimizer_type.lower() == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate,
                                                weight_decay=self.weight_decay)

        # scheduler
        if self.scheduler_type.lower() == 'step':
            self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.99)
        else: self.scheduler = None

        #self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1.0,
        #                                       end_factor=args.end_factor, total_iters=args.total_iters)

        # loss
        if self.loss_type != "MSE+MAE":
            self.loss = torch.nn.MSELoss() if self.loss_type == "MSE" else torch.nn.L1Loss()
        
        
        time_suffix = time.strftime('_%m-%d_%H-%M-%S', time.localtime())
        self.save_dir = self.cfg['save_dir'] + "/" + self.encoder[0] + "_" + self.decoder + time_suffix
        self.log_dir = self.save_dir + '/log'
        
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        copy_config(self.save_dir)
        write_basic_information( self.save_dir, self.use_wandb, self.use_all_sequence, self.target_indices, self.stride,
                                          self.image_type, self.image_resize, self.image_size, self.augmentation_angle,
                                         self.encoder, self.encoder_output,
                                         self.fc_num_layer, self.fc_hidden_dim,
                                         self.decoder, self.rnn_layer, self.decoder_output,
                                         self.optimizer_type, self.scheduler_type, self.dropout, self.batch_size, self.loss_type, self.learning_rate, self.weight_decay)

        if self.use_wandb:
                # wandb
                wandb_config = {
                    "encoder": self.encoder,
                    "encoder_output": self.encoder_output,

                    "decoder": self.decoder,
                    "rnn_layer": self.rnn_layer,
                    "decoder_output": self.decoder_output,

                    "fc_num_layer": self.fc_num_layer,
                    "fc_hidden_dim": self.fc_hidden_dim,

                    "use_all_sequence": self.use_all_sequence,
                    "target_indices": self.target_indices,
                    "stride": self.stride,

                    "image_type": self.image_type,
                    "image_resize": self.image_resize,
                    "image_size": self.image_size,
                    "augmentation_angle": self.augmentation_angle,

                    "optimizer": self.optimizer,
                    "scheduler": self.scheduler,
                    "batch_size": self.batch_size,
                    "learning_rate": self.learning_rate,
                    "weight_decay": self.weight_decay,
                    "dropout": self.dropout,
                    "loss": self.loss_type,
                }

                run_name = self.encoder[0] + '_' + self.decoder + '_' + self.optimizer + '_lr-' + f'{time_suffix}'
                self.run_log = wandb.init(project="force_estimation", config=wandb_config, name=run_name)

    
    def run(self) :
        if self.train_mode == 'Train':
            self.setup_run() # 훈련 전 준비 작업 실행
            self.train()
        elif self.train_mode == 'test' :
            self.test()
        else:
            model_path = self.cfg['model_dir']
            self.model.load_state_dict(torch.load(model_path))
            epoch_idx = model_path.find("epoch")
            self.test(model_path[:epoch_idx-1])

if __name__ == '__main__':
    args = parse_args()
        
    with open("config.yaml", 'r+', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    force_estimation = ForceEstimation(config, args)
    
    force_estimation.run()
import os
import time
import torchinfo
import yaml
import ast
import dataset.npz_dataset_method  # 먼저 메서드 붙임이 일어남
from dataset.npz_dataset import RFTDataset
from torch.optim import lr_scheduler
from utils.util import *
from utils.apply_img_augmentation import *
import wandb
from torch.utils.data import DataLoader #, Subset, ConcatDataset
#from sklearn.model_selection import KFold
from models.CFModel import CFModel

class ForceEstimation:
    def __init__(self, cfg):
        self.cfg = cfg
        self.best_epoch = 0

        # 1. 파라미터 초기화를 위한 헬퍼 메소드 정의
        def _init_param(key, p_type=None):
            value = self.cfg.get(key)
            
            if value is not None and p_type is not None:
                # --- 수정된 부분 ---
                # p_type이 ast.literal_eval이고, 실제 값이 문자열일 때만 변환을 수행합니다.
                if p_type is ast.literal_eval:
                    if isinstance(value, str):
                        value = p_type(value)
                else: # int, float, str2bool 등 다른 타입 변환은 그대로 수행
                    value = p_type(value)
                # --- 수정 끝 ---
                    
            setattr(self, key, value)

        # 2. 공통 파라미터 리스트 정의 (파라미터명, 타입)
        # 타입이 None이면 변환 없이 그대로 사용
        common_params = [
            ('train_mode', str), # train일 때만 학습 (bool)
            ('use_wandb', str2bool), # wandb에 올릴 것인지 (bool)
            ('image_resize', str2bool), # 이미지를 확대할 것인지 (bool)
            ('frame_length', int), # 몇 frame으로 자를 것인지 (bool)
            ('stride', int), # 몇 frame만큼 띄워서 학습할 것인지 (int)
            ('augmentation_angle', int), # online augmentation 시, 각도 최대치 (int)
            ('num_epoch', int),
            ('encoder_output', int), # cnn에서 결과적으로 나온 embedding_vector의 차원 (int)
            ('rnn_layer', int), # 시계열 모델 내 layer의 개수 (int)
            ('decoder', str), # 어떤 시계열 모델을 사용할 것인지 (str)
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
            ('plot_index', int) # 몇 번째 data를 가지고 plot할 것인지 (-1이면 mean으로 처리)
        ]

        # 3. 반복문을 통해 공통 파라미터 한 번에 초기화
        for key, p_type in common_params:
            _init_param(key, p_type)
            
        image_size = [int(image_size[0]), int(image_size[1])]

        time_suffix = time.strftime('_%m-%d_%H-%M-%S', time.localtime())
        self.save_dir = self.cfg['save_dir'] + "/" + self.encoder[0] + "_" + self.decoder + time_suffix
        self.log_dir = self.save_dir + '/log'
        
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        copy_config(self.save_dir)
        
        self.model = CFModel(self.encoder, self.encoder_output,
                             self.decoder, self.decoder_output, self.rnn_layer, self.fc_num_layer,
                             self.target_index, self.image_type, self.dropout,
                             fc_hidden_dim=self.fc_hidden_dim)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        torchinfo.summary(self.model, input_size=(self.batch_size, self.target_index, 3 if self.image_type.lower() == "rgb" else 1, image_size[0], image_size[1]))
        
        self.train_dataset = RFTDataset(image_type=self.image_type, image_size=image_size, path=cfg['train_data_dir'])
        self.train_dataset.preprocess()
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False)
        
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

        # loss
        self.loss = torch.nn.MSELoss() if self.loss_type == "MSE" else torch.nn.L1Loss()

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


    def validate(self, dataloader):
        self.model.eval()
        MSE = 0
        MAE = 0

        with torch.no_grad():
            for X, y, force in dataloader:
                if X.shape[1] < self.target_index: continue

                X = X.squeeze()
                force = force.squeeze()
                _force = []
                for start in range(0, len(force) - (self.target_index - 1)):
                    _force.append(force[start:start + self.target_index].tolist())
                _force = torch.Tensor(_force)
                X, y, force = X.float().to(self.device), y.to(self.device), _force.to(self.device)

                pred_force = self.model(X)

                MSE += torch.nn.functional.mse_loss(pred_force.squeeze().float(), force.float(),
                                                    reduction='mean')
                MAE += torch.nn.functional.l1_loss(pred_force.squeeze().float(), force.float(),
                                                   reduction='mean')

        MSE /= len(dataloader)
        MAE /= len(dataloader)
        return MSE, MAE

    def train(self):
        msg = '[Epoch: {}/{}], Lr: {:.6f}, Train MSE: {:.3f}, Train MAE: {:.3f}' \
              ' Validation MSE: {:.3f}, Validation MAE: {:.3f}, {}'

        val_best_loss = 1e10

        for epoch in range(self.num_epoch):
            self.model.train()
            MSE = []
            MAE = []
            for batch_idx, (X, y, force) in enumerate(self.train_dataloader):
                if X.shape[1] < self.target_index: continue
                X = X.squeeze()

                # X = apply_augmentation_to_all_frame(X, self.augmentation_angle)

                force = force.squeeze()
                _force = []
                for start in range(0, len(force) - (self.target_index - 1)):
                    _force.append(force[start:start + self.target_index].tolist())
                _force = torch.Tensor(_force)
                X, y, force = X.float().to(self.device), y.to(self.device), _force.to(self.device)

                self.optimizer.zero_grad()
                pred_force = self.model(X)

                loss = self.loss(pred_force.squeeze().float(), force.squeeze().float())

                loss.backward(retain_graph=True)
                self.optimizer.step()

            if self.scheduler is not None: self.scheduler.step()

            # test the model
            val_MSE, val_MAE = self.validate(self.val_dataloader)

            # best model select
            if self.loss_type == "MAE" and val_MAE < val_best_loss:
                if os.path.exists(self.save_dir + "/epoch_" + str(self.best_epoch) + ".pt"):
                    os.remove(self.save_dir + "/epoch_" + str(self.best_epoch) + ".pt")
                self.best_epoch = epoch + 1
                val_best_loss = val_MAE
                torch.save(self.model.state_dict(), self.save_dir + "/epoch_" + str(epoch + 1) + ".pt")
                improve = "******"
            elif self.loss_type == "MSE" and val_MSE < val_best_loss:
                if os.path.exists(self.save_dir + "/epoch_" + str(self.best_epoch) + ".pt"):
                    os.remove(self.save_dir + "/epoch_" + str(self.best_epoch) + ".pt")
                self.best_epoch = epoch + 1
                val_best_loss = val_MSE
                torch.save(self.model.state_dict(), self.save_dir + "/epoch_" + str(epoch + 1) + ".pt")
                improve = "******"
            elif self.loss_type == "MSE+MAE" and (val_MSE + val_MAE) / 2 < val_best_loss:
                if os.path.exists(self.save_dir + "/epoch_" + str(self.best_epoch) + ".pt"):
                    os.remove(self.save_dir + "/epoch_" + str(self.best_epoch) + ".pt")
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

        MSE_for_axis = []
        MAE_for_axis = []

        f = []
        pf = []

        with torch.no_grad():
            for X, y, force in self.test_dataloader:
                if X.shape[1] < self.target_index: continue
                X = X.squeeze()

                force_origin = force.squeeze()
                _force = []
                for start in range(0, len(force_origin) - (self.target_index - 1)):
                    _force.append(force_origin[start:start + self.target_index].tolist())
                _force = torch.Tensor(_force)

                X, y, force = X.float().to(self.device), y.to(self.device), _force.to(self.device)
                _force = force.detach().cpu().numpy().squeeze().tolist()

                pred_force = self.model(X)
                _pred_force = pred_force.detach().cpu().numpy().squeeze().tolist()

                # plot을 위해, 시퀀스를 합침
                if self.plot_index <= 0 or self.plot_index > self.target_index:
                    _f, _pf = lists_for_plot_mean(force_origin, _pred_force)
                else: _f, _pf = lists_for_plot(force_origin, _pred_force, self.target_index, self.plot_index)

                f.append(_f)
                pf.append(_pf)

                # 평균 MSE, MAE 계산
                MSE += torch.nn.functional.mse_loss(pred_force.squeeze().float(), force.float(),
                                                    reduction='mean').item()
                MAE += torch.nn.functional.l1_loss(pred_force.squeeze().float(), force.float(),
                                                   reduction='mean').item()

                # 각 X, Y, Z별 MSE, MAE 계산
                mse = torch.mean(torch.nn.functional.mse_loss(pred_force.float(), force.float(),
                                                    reduction='none'), dim=0).tolist()
                mae = torch.mean(torch.nn.functional.l1_loss(pred_force.float(), force.float(),
                                                    reduction='none'), dim=0).tolist()

                MSE_for_axis.append(mse)
                MAE_for_axis.append(mae)

        plot(f, pf, -1, save_dir)
        start = 15
        for i in range(start, self.target_index+1, 5):
            plot(f, pf, i, save_dir)

        MSE /= len(self.test_dataloader)
        MAE /= len(self.test_dataloader)
        print("Test: \tMSE loss: {:.5f}\tL1 loss: {:.5f}".format(MSE, MAE))

        MSE_for_axis = torch.Tensor(MSE_for_axis)
        MSE_for_axis = torch.mean(MSE_for_axis, dim=(0,1))
        MAE_for_axis = torch.Tensor(MAE_for_axis)
        MAE_for_axis = torch.mean(MAE_for_axis, dim=(0,1))

        print(f'(TEST MSE) X: {MSE_for_axis[0]}\tY: {MSE_for_axis[1]}\tZ: {MSE_for_axis[2]}')
        print(f'(TEST MAE) X: {MAE_for_axis[0]}\tY: {MAE_for_axis[1]}\tZ: {MAE_for_axis[2]}')

        train_txt_path = save_dir + "/train.txt"
        with open(train_txt_path, 'a') as train_txt_file:
            train_txt_file.write("Test: \tMSE loss: {:.5f}\tL1 loss: {:.5f}".format(MSE, MAE) + '\n')
            train_txt_file.write(f'(TEST MSE) X: {MSE_for_axis[0]}\tY: {MSE_for_axis[1]}\tZ: {MSE_for_axis[2]}\n')
            train_txt_file.write(f'(TEST MAE) X: {MAE_for_axis[0]}\tY: {MAE_for_axis[1]}\tZ: {MAE_for_axis[2]}\n')

        return MSE, MAE

if __name__ == '__main__':
        
    with open("config.yaml", 'r+', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    force_estimation = ForceEstimation(config)
    
    write_basic_information(force_estimation.save_dir, force_estimation.use_wandb, force_estimation.use_all_sequence, force_estimation.target_indices, force_estimation.stride,
                                          force_estimation.image_type, force_estimation.image_resize, force_estimation.image_size, force_estimation.augmentation_angle,
                                         force_estimation.encoder, force_estimation.encoder_output,
                                         force_estimation.fc_num_layer, force_estimation.fc_hidden_dim,
                                         force_estimation.decoder, force_estimation.rnn_layer, force_estimation.decoder_output,
                                         force_estimation.optimizer_type, force_estimation.scheduler_type, force_estimation.dropout, force_estimation.batch_size, force_estimation.loss_type, force_estimation.learning_rate, force_estimation.weight_decay)
    
    if force_estimation.train_mode == 'Train':
        force_estimation.train()
    elif force_estimation.train_mode == 'Test' :
        model_path = force_estimation.cfg['best_model_dir']
        force_estimation.model.load_state_dict(torch.load(model_path))
        force_estimation.test()
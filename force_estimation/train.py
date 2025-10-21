import pathlib
import sys
import time
import torchinfo
import yaml
from tqdm import tqdm
from utils import *
from dataset import *
from torch.optim import lr_scheduler
from utils.earlystopping import *
import wandb
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from model import *

class ForceEstimation:
    def __init__(self, cfg):
        # parameters
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
                else:  # int, float, str2bool 등 다른 타입 변환은 그대로 수행
                    value = p_type(value)
                # --- 수정 끝 ---

            setattr(self, key, value)

        # 2. 공통 파라미터 리스트 정의 (파라미터명, 타입)
        # 타입이 None이면 변환 없이 그대로 사용
        common_params = [
            ('train_mode', bool),  # train일 때만 학습 (bool)
            ('use_wandb', bool),  # wandb에 올릴 것인지 (bool)
            ('resume', bool),

            ('image_type', str),  # origin, rgb, diff (str)
            ('image_size', ast.literal_eval),  # 이미지를 몇 size로 할 것인지 (int, int)
            ('target_index', int),  # 몇 frame으로 자를 것인지 (bool)
            ('stride', int),  # 몇 frame만큼 띄워서 학습할 것인지 (int)
            ('xflip', bool)
            # ('augmentation_angle', int),  # online augmentation 시, 각도 최대치 (int)

            ('encoder', str),  # 어떤 cnn을 쓸 것인지 (str)

            ('decoder', str),  # 어떤 시계열 모델을 사용할 것인지 (str)
            ('decoder_layer', int),  # 시계열 모델 내 layer의 개수 (int)
            ('decoder_dimension', int),
            ('decoder_output', int),  # 시계열 모델에서 결과적으로 나온 embedding_vector의 차원 (int)

            ('fc_num_layer', int),  # fully-connected layer의 개수 (int)
            ('fc_hidden_dim', int),  # fully-connected layer의 hidden layer의 dimension 개수

            ('batch_size', int),
            ('learning_rate', float),
            ('weight_decay', float),
            ('num_epoch', int),

            ('optimizer_type', str),
            ('scheduler_type', str),
            ('loss_type', str),
            ('loss_weights', list),

            ('model_dir', str),
            ('save_dir', str)
        ]

        # 3. 반복문을 통해 공통 파라미터 한 번에 초기화
        for key, p_type in common_params:
            _init_param(key, p_type)

        # model configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CFModel(self.encoder,
                             self.decoder,
                             self.decoder_layer,
                             self.decoder_dimension, 
                             self.decoder_output,                               
                             self.fc_num_layer,
                             self.target_index,
                             self.fc_hidden_dim,
                             self.image_size,
                             )
        self.model = self.model.to(self.device)
        torchinfo.summary(self.model, input_size=(self.batch_size, self.target_index, 3, self.image_size[0], self.image_size[1]))

        # test dataset
        self.test_dataset = RFTDataset(target_index=self.target_index, stride=self.stride, path=cfg['test_data_dir'], dataset_type='test')
        self.test_dataloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=1, shuffle=False)

        # optimizer
        if self.optimizer_type.lower() == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, betas=(0.9, 0.999),
                                              eps=1e-08, weight_decay=self.weight_decay, amsgrad=False)
        elif self.optimizer_type.lower() == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.learning_rate,
                                                 weight_decay=self.weight_decay)
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate,
                                             weight_decay=self.weight_decay)
            self.optimizer_type = 'sgd'

        # train
        if self.train_mode:
            # train dataset
            self.train_dataset = RFTDataset(target_index=self.target_index, stride=self.stride, path=cfg['train_data_dir'], dataset_type='train', xflip=self.xflip)
            self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size,
                                                                shuffle=True)

            # validation dataset
            self.val_dataset = RFTDataset(target_index=self.target_index, stride=self.stride, path=cfg['val_data_dir'], dataset_type='val')
            self.val_dataloader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size,
                                                              shuffle=False)
            
            self.learning_rate = float(self.learning_rate)

            
            # scheduler
            if self.scheduler_type.lower() == 'step':
                self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.99)
            else:
                self.scheduler = None

            # self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1.0,
            #                                       end_factor=args.end_factor, total_iters=args.total_iters)


            # create directories
            time_suffix = time.strftime('%m-%d_%H-%M-%S', time.localtime())
            self.save_dir = cfg['save_dir'] + "/" + self.encoder + "_" + self.decoder + time_suffix

            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            copy_config(self.save_dir)

            # loss
            if self.loss_type != "MSE+MAE":
                self.loss = torch.nn.MSELoss(reduction='none') if self.loss_type == "MSE" else torch.nn.L1Loss(reduction='none')

            # wandb
            if self.use_wandb:
                # wandb
                wandb_config = {
                    "encoder": self.encoder,

                    "decoder": self.decoder,
                    "decoder_layer": self.decoder_layer,
                    "decoder_output": self.decoder_output,

                    "fc_num_layer": self.fc_num_layer,
                    "fc_hidden_dim": self.fc_hidden_dim,

                    "target_index": self.target_index,
                    "stride": self.stride,
                    "xflip": self.xflip,

                    "image_type": self.image_type,
                    "image_size": self.image_size,

                    "optimizer": self.optimizer_type,
                    "scheduler": self.scheduler_type,
                    "batch_size": self.batch_size,
                    "learning_rate": self.learning_rate,
                    "weight_decay": self.weight_decay,
                    "loss": self.loss_type,
                    "loss_weights" : self.loss_weights,
                }

                run_name = f'MTM_{self.encoder}_{self.decoder}_{self.target_index}frames_{time_suffix}_{self.fc_num_layer}layer_{self.loss_type}'
                self.run_log = wandb.init(project="compare_model", config=wandb_config, name=run_name)

                self.train()

        # test
        else:
            self.save_dir = pathlib.Path(cfg['model_dir']).parent.__str__()
            checkpoint = torch.load(cfg['model_dir'])
            self.model.load_state_dict(checkpoint)
            self.test()

    def validate(self, dataloader):
        self.model.eval()
        MSE = 0
        MAE = 0

        with torch.no_grad():
            for X, force, y in dataloader:
                X, y, force = X.float().to(self.device), y.to(self.device), force.to(self.device)

                pred_force = self.model(X)

                MSE += torch.nn.functional.mse_loss(pred_force.float(), force.float(),
                                                    reduction='mean')
                if self.loss_type != "huber" :
                    MAE += torch.nn.functional.l1_loss(pred_force.float(), force.float(),
                                                    reduction='mean')
                else :
                    MAE += torch.nn.functional.huber_loss(pred_force.float(), force.float(),
                                                    reduction='mean')

        MSE /= len(dataloader)
        MAE /= len(dataloader)
        return MSE, MAE

    def train(self, start_epoch=0):
        msg = '[Epoch: {}/{}], Lr: {:.6f}, Train MSE: {:.3f}, Train MAE: {:.3f}' \
              ' Validation MSE: {:.3f}, Validation MAE: {:.3f}'

        val_best_loss = 1e10
        patience = 20
        best_model_path = os.path.join(self.save_dir, 'best_model.pt')
        loss_weights = torch.tensor(self.loss_weights).to(self.device)
        early_stopping = EarlyStopping(patience = patience, verbose=True, path = best_model_path)
        for epoch in range(start_epoch, self.num_epoch):
            self.model.train()

            MSE = []
            MAE = []

            # tqdm으로 배치 진행 상황을 보여줌
            progress_bar = tqdm(self.train_dataloader, desc="Training", leave=False)
            running_loss = 0.0

            # X.shape =
            for batch in progress_bar:
                X, force, y = batch
                # X = apply_augmentation_to_all_frame(X, self.augmentation_angle)
                X, y, force = X.float().to(self.device), y.to(self.device), force.to(self.device)

                self.optimizer.zero_grad()
                pred_force = self.model(X)

                MSEloss = torch.nn.functional.mse_loss(pred_force.float(), force.float(),
                                                       reduction='none')
                L1_loss = torch.nn.functional.l1_loss(pred_force.float(), force.float(),
                                                      reduction='none')
                if self.loss_type == 'MSE+MAE':
                    loss = (MSEloss + L1_loss) / 2
                else:
                    loss = self.loss(pred_force.float(), force.float())

                loss = loss * loss_weights
                
                loss = loss.mean()
                
                loss.backward()
                self.optimizer.step()
                
                MSE.append(MSEloss.mean().item())
                MAE.append(L1_loss.mean().item())

                # 현재 평균 loss를 tqdm description에 표시
                running_loss += loss.item()
                progress_bar.set_postfix(loss=running_loss / (progress_bar.n + 1))

                

            if self.scheduler is not None: self.scheduler.step()
            # validate the model
            val_MSE, val_MAE = self.validate(self.val_dataloader)

            
            if self.loss_type == 'MSE+MAE':
                val_loss_to_monitor = (val_MSE + val_MAE) / 2
            elif self.loss_type == 'MSE':
                val_loss_to_monitor = val_MSE
            else: # MAE or default
                val_loss_to_monitor = val_MAE

            early_stopping(val_loss_to_monitor, self.model)
            
            if early_stopping.early_stop:
                print("Early stopping")
                break
            avg_train_mse = np.mean(MSE)
            avg_train_mae = np.mean(MAE)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(msg.format(
                epoch + 1,
                self.num_epoch,
                current_lr,
                avg_train_mse,
                avg_train_mae,
                val_MSE,
                val_MAE
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
                    float(val_MAE)
                ) + '\n')

            if self.use_wandb:
                try:
                    wandb.log({"train_MSE": np.mean(MSE), "train_MAE": np.mean(MAE),
                                  "validation_MSE": val_MSE, "validation_MAE": val_MAE})
                except:
                    wandb.finish()

        # --- 학습 종료 후 최종 작업 ---
        print("Loading best model weights...")
        # EarlyStopping에 의해 저장된 가장 좋은 모델의 가중치를 불러옵니다.
        self.model.load_state_dict(torch.load(best_model_path))
        
        print("Running test with best model...")
        self.test()
        
        if self.use_wandb:
            wandb.finish()

    def test(self):
        self.model.eval()

        f = [] # (# of sequences, full sequence, 3)
        pf = [] # (# of sequences, # of chunks, 30)

        MSE = [] # shape: (num of chunks, target_index, 3)
        MAE = [] # shape: (num of chunks, target_index, 3)
        
        
        with torch.no_grad():
            for _X, _force, y in self.test_dataloader: # 한 시퀀스씩 추정
                _X = _X.squeeze().tolist()
              
                _force = _force.squeeze()
                _force = _force.tolist()
                
                
                f.append(_force) # 리스트에 각 시퀀스마다의 force label을 전체 저장

                # 각 시퀀스를 sliding window씩 자름
                X = []
                force = []
                for i in range(len(_X) - self.target_index + 1):
                    X.append(_X[i:i+self.target_index])
                    force.append(_force[i:i+self.target_index])
                
                
                pf.append([])

                # 모델을 통해 예측하기 위해 batch size마다 잘라서 predict
                batch_start = 0
                while True:
                    if batch_start + self.batch_size < len(X):
                        X_batch, force_batch, y_batch = torch.Tensor(X[batch_start:batch_start+self.batch_size]),\
                                                        torch.Tensor(force[batch_start:batch_start+self.batch_size]),\
                                                        torch.Tensor(y[batch_start:batch_start+self.batch_size])
                    else:
                        X_batch, force_batch, y_batch = torch.Tensor(X[batch_start:len(X)]),\
                                                        torch.Tensor(force[batch_start:len(X)]),\
                                                        torch.Tensor(y[batch_start:len(X)])

                    X_batch, force_batch, y_batch = X_batch.to(self.device), force_batch.to(self.device), y_batch.to(self.device)

                    pred_force = self.model(X_batch) # shape: (batch size, sliding window size, 3) = (8, 30, 3)
                    
                    _pred_force_detach = pred_force.detach().cpu().numpy().tolist()
                    pf[len(pf)-1].extend(_pred_force_detach)
                    # 각 X, Y, Z별 MSE, MAE 계산
                    MSE.extend(torch.nn.functional.mse_loss(pred_force, force_batch,
                                                            reduction='none').tolist())
                    MAE.extend(torch.nn.functional.l1_loss(pred_force, force_batch,
                                                            reduction='none').tolist())

                    batch_start += self.batch_size
                    if batch_start >= len(X): break

        train_txt_path = self.save_dir + "/train.txt"
        plot_for_lstm(f, pf, self.target_index, self.save_dir, train_txt_path)

        MSE = torch.Tensor(MSE)
        MAE = torch.Tensor(MAE)

        MSE_for_axis = torch.mean(MSE, dim=(0, 1))
        MAE_for_axis = torch.mean(MAE, dim=(0, 1))

        print("TEST: \tMSE loss: {:.5f}\tL1 loss: {:.5f}".format(torch.mean(MSE), torch.mean(MAE)))
        print(f'(TEST MSE) X: {MSE_for_axis[0]}\tY: {MSE_for_axis[1]}\tZ: {MSE_for_axis[2]}')
        print(f'(TEST MAE) X: {MAE_for_axis[0]}\tY: {MAE_for_axis[1]}\tZ: {MAE_for_axis[2]}')
        std_MSE = torch.std(MSE)
        std_MAE = torch.std(MAE)
        print(f'(STD) MSE: {std_MSE}, MAE: {std_MAE}')

        with open(train_txt_path, 'a') as train_txt_file:
            train_txt_file.write("Test: \tMSE loss: {:.5f}\tL1 loss: {:.5f}".format(torch.mean(MSE), torch.mean(MAE)) + '\n')
            train_txt_file.write(f'(TEST MSE) X: {MSE_for_axis[0]}\tY: {MSE_for_axis[1]}\tZ: {MSE_for_axis[2]}\n')
            train_txt_file.write(f'(TEST MAE) X: {MAE_for_axis[0]}\tY: {MAE_for_axis[1]}\tZ: {MAE_for_axis[2]}\n')
            train_txt_file.write(f'(STD) MSE: {std_MSE}, MAE: {std_MAE}')

        return MSE, MAE


if __name__ == '__main__':
    with open("config.yaml", 'r+', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    force_estimation = ForceEstimation(config)
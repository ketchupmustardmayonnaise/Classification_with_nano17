import os
import time
import torchinfo
import yaml
import ast
import wandb
import matplotlib.pyplot as plt
import npz_dataset_method  # 먼저 메서드 붙임이 일어남
import lightning as pl
from npz_dataset import RFTDataset
from torch.optim import lr_scheduler
from sklearn.model_selection import KFold
from torch.utils.data import Subset, ConcatDataset
from util import *
from models import ImageToVectorModel
from apply_img_augmentation import *


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
            ('target_index', int), # 몇 frame으로 자를 것인지 (bool)
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
        ]

        # 3. 반복문을 통해 공통 파라미터 한 번에 초기화
        for key, p_type in common_params:
            _init_param(key, p_type)     

        image_size = [int(self.image_size[0]), int(self.image_size[1])]

        time_suffix = time.strftime('_%m-%d_%H-%M-%S', time.localtime())
        self.save_dir = self.cfg['save_dir'] + "/" + self.encoder[0] + "_" + self.decoder + time_suffix
        self.log_dir = self.save_dir + '/log'
        
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        copy_config(self.save_dir)
        
        self.model = ImageToVectorModel(self.encoder_output, 3)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        torchinfo.summary(self.model, input_size=(self.batch_size, self.target_index, 3 if self.image_type.lower() == "rgb" else 1, image_size[0], image_size[1]))
        
        # train dataset
        self.train_dataset = RFTDataset(image_type=self.image_type, image_size=image_size, path=cfg['train_data_dir'])
        self.train_dataset.preprocess()
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False)

        # validation dataset
        self.val_dataset = RFTDataset(image_type=self.image_type, image_size=image_size, path=cfg['val_data_dir'])
        self.val_dataset.preprocess()
        self.val_dataloader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        
        self.dataset = ConcatDataset([self.train_dataset, self.val_dataset])

        # test dataset
        # self.test_dataset = RFTDataset(image_type=self.image_type, image_size=image_size, path=cfg['test_data_dir'])
        # self.test_dataset.preprocess()
        # self.test_dataloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

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
        
        
        # model configuration
 

        
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
        


    def validation(self, dataloader):
        self.model.eval()
        val_mse_losses = []
        val_mae_losses = []
        with torch.no_grad():
            for X, y, force in dataloader:
                X, force = X.float().to(self.device), force.to(self.device)
                pred_force = self.model(X)
                val_mse_losses.append(torch.nn.functional.mse_loss(pred_force.squeeze(), force.squeeze()).item())
                val_mae_losses.append(torch.nn.functional.l1_loss(pred_force.squeeze(), force.squeeze()).item())
                
        return np.mean(val_mse_losses), np.mean(val_mae_losses)

    
    

    def train_fold(self, train_dataloader, val_dataloader,fold_index):
        """하나의 폴드에 대한 학습 및 검증을 수행하는 내부 메서드"""
                
        val_best_loss = float('inf')

        for epoch in range(self.num_epoch):
            self.model.train()
            epoch_mse_losses = []
            epoch_mae_losses = []

            for batch_idx, (X, y, force) in enumerate(train_dataloader):
                # X = apply_augmentation_to_all_frame(X, self.augmentation_angle) # 필요시 증강 사용
                X, y, force = X.float().to(self.device), y.to(self.device), force.to(self.device)

                self.optimizer.zero_grad()
                pred_force = self.model(X)
                
                # 손실 계산
                mse_loss = torch.nn.functional.mse_loss(pred_force.squeeze(), force.squeeze())
                l1_loss = torch.nn.functional.l1_loss(pred_force.squeeze(), force.squeeze())
                
                if self.loss_type == 'MSE+MAE':
                    loss = (mse_loss + l1_loss) / 2
                else:
                    loss = self.loss(pred_force.squeeze(), force.squeeze())

                epoch_mse_losses.append(mse_loss.item())
                epoch_mae_losses.append(l1_loss.item())

                loss.backward()
                self.optimizer.step()

            if self.scheduler is not None: self.scheduler.step()

            # 검증
            val_MSE, val_MAE = self.validation(val_dataloader)

            # 최고 모델 선택 및 저장
            loss_criteria = {"MAE": val_MAE, "MSE": val_MSE, "MSE+MAE": (val_MSE + val_MAE) / 2}
            current_val_loss = loss_criteria.get(self.loss_type)

            improve = ""
            if current_val_loss < val_best_loss:
                self.best_epoch = epoch + 1
                val_best_loss = current_val_loss
                # 파일명에 폴드 번호를 추가하여 덮어쓰기 방지
                save_path = os.path.join(self.save_dir, f"fold_{fold_index}_best_model.pt")
                torch.save(self.model.state_dict(), save_path)
                improve = "******"
            
            # 로그 메시지 포맷 및 출력
            msg = '[Fold {} | Epoch {}/{}] Lr: {:.6f} | Train MSE: {:.3f}, MAE: {:.3f} | Val MSE: {:.3f}, MAE: {:.3f} {}'
            print_msg = msg.format(fold_index, epoch + 1, self.num_epoch, self.learning_rate, 
                                   np.mean(epoch_mse_losses), np.mean(epoch_mae_losses), 
                                   val_MSE, val_MAE, improve)
            print(print_msg)

            # 로그 파일 및 wandb 기록
            train_txt_path = os.path.join(self.save_dir, "train_log.txt")
            with open(train_txt_path, 'a') as f:
                f.write(print_msg + '\n')
            
            if self.use_wandb:
                wandb.log({
                    f"Fold-{fold_index}/Train MSE": np.mean(epoch_mse_losses),
                    f"Fold-{fold_index}/Train MAE": np.mean(epoch_mae_losses),
                    f"Fold-{fold_index}/Validation MSE": val_MSE,
                    f"Fold-{fold_index}/Validation MAE": val_MAE,
                    f"Fold-{fold_index}/Best Loss": val_best_loss,
                })
        
        return val_best_loss
    
    def train(self):
        """K-Fold 교차 검증을 총괄하는 메인 학습 메서드"""
        kfold = KFold(n_splits=self.k_folds, shuffle=True, random_state=self.seed)
        fold_results = []

        if self.use_wandb:
            # wandb.init(...) # wandb 초기화
            pass

        for fold, (train_ids, val_ids) in enumerate(kfold.split(self.dataset)):
            print(f'{"="*25} FOLD {fold + 1}/{self.k_folds} {"="*25}')
            
            train_subset = Subset(self.dataset, train_ids)
            val_subset = Subset(self.dataset, val_ids)

            train_loader = torch.utils.data.DataLoader(train_subset, batch_size=self.batch_size, shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_subset, batch_size=self.batch_size)

            # 현재 폴드에 대한 학습 실행
            best_loss_for_fold = self.train_fold(train_loader, val_loader, fold + 1)
            fold_results.append(best_loss_for_fold)
        
        # 최종 결과 출력
        print(f'\n{"="*25} K-FOLD FINAL RESULTS {"="*25}')
        print(f'{self.k_folds}-Fold CV best loss: {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}')

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

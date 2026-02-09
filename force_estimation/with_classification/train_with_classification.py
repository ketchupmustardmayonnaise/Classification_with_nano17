import pathlib
import sys
import time

import torchinfo
import yaml
from tqdm import tqdm
from force_estimation.utils import *
from torch.optim import lr_scheduler
from force_estimation.utils.earlystopping import *
from torch.utils.data import DataLoader
from force_estimation.metrics import *

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from force_estimation.model.CFModel import *


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
            ('wandb_project', str),

            ('image_size', list),  # 이미지를 몇 size로 할 것인지 (int, int)
            ('num_classes', int),
            ('target_index', int),  # 몇 frame으로 자를 것인지 (bool)
            ('xflip', bool),

            ('encoder', str),  # 어떤 cnn을 쓸 것인지 (str)

            ('decoder', str),  # 어떤 시계열 모델을 사용할 것인지 (str)
            ('num_decoder_layer', int),  # 시계열 모델 내 layer의 개수 (int)
            ('dim_decoder_hidden', int),

            ('batch_size', int),
            ('learning_rate', float),
            ('weight_decay', float),
            ('num_epoch', int),

            ('optimizer_type', str),
            ('scheduler_type', str),
            ('loss_type', str),

            ('model_dir', str),
            ('save_dir', str),
            ('dataset_dir', str)
        ]

        # 3. 반복문을 통해 공통 파라미터 한 번에 초기화
        for key, p_type in common_params:
            _init_param(key, p_type)

        # model configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CFModel(self.encoder,
                             self.decoder,
                             self.num_decoder_layer,
                             self.dim_decoder_hidden,
                             self.target_index,
                             self.image_size)
        self.model = self.model.to(self.device)
        torchinfo.summary(self.model, input_size=(self.batch_size, self.target_index, 3, self.image_size[0], self.image_size[1]))

        # train
        if self.train_mode:

            # create directories
            self.time_suffix = time.strftime('%m-%d_%H-%M-%S', time.localtime())
            self.save_dir = cfg['save_dir'] + "/" + self.encoder + "_" + self.decoder + "_" + self.time_suffix

            if not os.path.exists(self.save_dir): os.makedirs(self.save_dir)
            copy_config(self.save_dir)

            # scheduler
            if self.scheduler_type.lower() == 'step':
                self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.99)
            else:
                self.scheduler = None

            # optimizer
            if self.optimizer_type.lower() == 'adam':
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, betas=(0.9, 0.999),
                                                  eps=1e-08, weight_decay=self.weight_decay, amsgrad=False)
            elif self.optimizer_type.lower() == 'rmsprop':
                self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            else:
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
                self.optimizer_type = 'sgd'

            # loss
            self.loss = torch.nn.MSELoss(reduction='mean') if self.loss_type == "MSE" else torch.nn.L1Loss(reduction='mean')

            self.learning_rate = float(self.learning_rate)

            # wandb 사용한다면
            if self.use_wandb:
                # wandb
                wandb_config = {
                    "encoder": self.encoder,

                    "decoder": self.decoder,
                    "num_decoder_layer": self.num_decoder_layer,
                    "dim_decoder_hidden": self.dim_decoder_hidden,
                    "num_classes": self.num_classes,

                    "target_index": self.target_index,
                    "xflip": self.xflip,
                    "image_size": self.image_size,

                    "optimizer": self.optimizer_type,
                    "scheduler": self.scheduler_type,
                    "batch_size": self.batch_size,
                    "learning_rate": self.learning_rate,
                    "weight_decay": self.weight_decay,

                    "loss": self.loss_type
                }

                run_name = f'MTM_{self.encoder}_{self.decoder}_{self.target_index}frames_{self.time_suffix}'
                self.run_log = wandb.init(project="compare_model", job_type="train_model", config=wandb_config, name=run_name)
                self.model_artifact = wandb.Artifact(f"model_{self.run_log.id}", type="model")

            test_data_dir = f"{cfg['dataset_dir']}/test_dataset.h5"
            train_data_dir = f"{cfg['dataset_dir']}/train_dataset.h5"
            val_data_dir = f"{cfg['dataset_dir']}/val_dataset.h5"

            # test dataset
            self.test_dataset = RFTDataset(target_index=self.target_index, path=test_data_dir, dataset_type='test')
            self.test_dataloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=1, shuffle=False)

            # train dataset
            self.train_dataset = RFTDataset(target_index=self.target_index, path=train_data_dir, dataset_type='train', xflip=self.xflip)
            self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

            # validation dataset
            self.val_dataset = RFTDataset(target_index=self.target_index, path=val_data_dir, dataset_type='val')
            self.val_dataloader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

            self.train()

        # test
        else:
            self.use_wandb = False
            self.test_dataset = RFTDataset(target_index=self.target_index, path=f"{cfg['dataset_dir']}/test_dataset.h5", dataset_type='test')
            self.test_dataloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=1, shuffle=False)

            self.save_dir = pathlib.Path(cfg['model_dir']).parent.__str__()
            checkpoint = torch.load(cfg['model_dir'])
            self.model.load_state_dict(checkpoint)
            self.test()




    def validate(self, dataloader):
        self.model.eval()

        metrics = TrainMetrics(is_classification = True)
        with torch.no_grad():
            for X, force, y in dataloader:
                X, y, force = X.float().to(self.device), y.to(self.device), force.to(self.device)

                pred_class, pred_force = self.model(X)
                pred_class = torch.nn.functional.softmax(pred_class, dim=1)

                metrics.calculate_regression_metrics(pred_force, force)
                metrics.calculate_classification_metrics(pred_class, y)

        mse, mae, ce, accuracy = metrics.report_metrics()
        return mse, mae, ce, accuracy



    def train(self, start_epoch=0):
        msg = '[Epoch: {}/{}], Lr: {:.6f}, Train CE: {:.3f}, Train Accuracy: {:.3f}, Train MSE: {:.3f}, Train MAE: {:.3f}' \
              ' Validation CE: {:.3f}, Validation Accuracy: {:.3f}, Validation MSE: {:.3f}, Validation MAE: {:.3f}'

        patience = 20
        best_model_path = os.path.join(self.save_dir, 'best_model.pt')
        early_stopping = EarlyStopping(patience=patience, verbose=True, path=best_model_path)

        metrics = TrainMetrics(is_classification = True)
        for epoch in range(start_epoch, self.num_epoch):
            self.model.train()
            metrics.initialize_metrics()

            # tqdm으로 배치 진행 상황을 보여줌
            progress_bar = tqdm(self.train_dataloader, desc="Training", leave=False)
            running_loss = 0.0

            # X.shape =
            for X, force, y in progress_bar:
                X, y, force = X.to(self.device), y.to(self.device), force.to(self.device)
                self.optimizer.zero_grad()

                pred_class, pred_force = self.model(X)
                pred_class = torch.nn.functional.softmax(pred_class, dim=0)

                metrics.calculate_regression_metrics(pred_force, force)
                metrics.calculate_classification_metrics(pred_class, y)

                loss = self.loss(pred_force.float(), force.float())
                # loss_clf = torch.nn.functional.cross_entropy(pred_class, y.float(), reduction='mean')

                # loss = (loss_clf + loss_reg) / 2
                loss.backward()

                self.optimizer.step()

                # 현재 평균 loss를 tqdm description에 표시
                running_loss += loss.item()
                progress_bar.set_postfix(loss=running_loss / (progress_bar.n + 1))

            mse, mae, ce, accuracy = metrics.report_metrics()

            if self.scheduler is not None: self.scheduler.step()
            # validate the model
            mse_val, mae_val, ce_val, acc_val = self.validate(self.val_dataloader)


            # ===============================================
            # Early Stopping & best model save

            val_loss_to_monitor = mse_val if self.loss_type.lower() == 'mse' else mae_val

            is_best_loss = early_stopping(val_loss_to_monitor, self.model)
            # wandb 에 업로드
            if self.use_wandb:
                new_version_artifact = wandb.Artifact(
                    f"{self.encoder}_{self.decoder}_{self.time_suffix}_{self.target_index}frames", type="model")
                if is_best_loss:
                    torch.save(self.model.state_dict(), os.path.join(wandb.run.dir, "best_model.pt"))
                    new_version_artifact.add_file(os.path.join(wandb.run.dir, "best_model.pt"))

                else:
                    new_version_artifact.add_file(os.path.join(wandb.run.dir, "best_model.pt"))

                torch.save(self.model.state_dict(), os.path.join(wandb.run.dir, "current_model.pt"))
                new_version_artifact.add_file(os.path.join(wandb.run.dir, "current_model.pt"))
                wandb.log_artifact(new_version_artifact)

            current_lr = self.optimizer.param_groups[0]['lr']

            # ===============================================
            print(msg.format(epoch + 1, self.num_epoch, current_lr,
                ce, accuracy, mse, mae,
                ce_val, acc_val, mse_val, mae_val))

            train_txt_path = self.save_dir + "/train.txt"
            with open(train_txt_path, 'a') as train_txt_file:
                train_txt_file.write(msg.format(epoch + 1, self.num_epoch, current_lr,
                    ce, accuracy, mse, mae,
                    ce_val, acc_val, mse_val, mae_val) + '\n')

            if self.use_wandb:
                try:
                    wandb.log({"train_CE": ce, "train_accuracy": accuracy, "train_MSE": mse, "train_MAE": mae,
                                "validation_CE": ce_val, "validation_accuracy": acc_val, "validation_MSE": mse_val, "validation_MAE": mae_val})
                except:
                    wandb.finish()


            if early_stopping.early_stop:
                print("Early stopping")
                break

        # --- 학습 종료 후 최종 작업 ---
        print("Loading best model weights...")
        # EarlyStopping에 의해 저장된 가장 좋은 모델의 가중치를 불러옵니다.
        self.model.load_state_dict(torch.load(best_model_path))

        print("Running test with best model...")
        self.test()

        if self.use_wandb: wandb.finish()




    def test(self):
        self.model.eval()
        metrics = TestMetrics(is_classification=True)

        # 소요 시간 측정
        self.batch_size = 1
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()

        upper_bound_of_force = np.array(self.cfg.get('upper_bound_of_force')).astype('float32')

        with torch.no_grad():
            for _X, _force, _label in self.test_dataloader:  # 한 시퀀스씩 추정
                _X, _label, _force = _X.squeeze().tolist(), _label.squeeze().tolist(), _force.squeeze().tolist()

                metrics.init_new_sequence()
                metrics.save_sequence(_force) # 리스트에 각 시퀀스마다의 force label을 전체 저장

                # ===============================================
                # 각 시퀀스를 sliding window씩 자름
                X, force, label  = [], [], []
                for i in range(len(_X) - self.target_index + 1):
                    X.append(_X[i:i + self.target_index])
                    force.append(_force[i:i + self.target_index])

                    window_labels = torch.Tensor(_label[i:i+self.target_index])
                    label.append(RFTDataset.majority_last_onehot_torch(window_labels).tolist())
                # ===============================================


                # ===============================================
                # 모델을 통해 예측하기 위해 batch size마다 잘라서 predict
                batch_start = 0
                while True:
                    if batch_start + self.batch_size < len(X):
                        X_batch, force_batch, y_batch = torch.Tensor(X[batch_start:batch_start + self.batch_size]), \
                                                        torch.Tensor(force[batch_start:batch_start + self.batch_size]), \
                                                        torch.Tensor(label[batch_start:batch_start + self.batch_size])
                    else:
                        X_batch, force_batch, y_batch = torch.Tensor(X[batch_start:len(X)]), \
                                                        torch.Tensor(force[batch_start:len(X)]), \
                                                        torch.Tensor(label[batch_start:len(X)])

                    X_batch, force_batch, y_batch = X_batch.to(self.device), force_batch.to(self.device), y_batch.to(self.device)

                    # ===============================================
                    # 소요 시간 측정
                    start.record()
                    pred_class, pred_force = self.model(X_batch)  # shape: (batch size, sliding window size, 3) = (8, 30, 3)
                    end.record()
                    torch.cuda.synchronize()
                    # ===============================================

                    # ===============================================
                    # Classification - Bodypart Classification
                    pred_class = torch.nn.functional.softmax(pred_class, dim=1)  # 각 레이블별 확률값
                    # ===============================================

                    metrics.save_inference_time(start.elapsed_time(end))
                    metrics.save_regression_result_of_window(pred_force.detach().cpu().numpy().tolist(),
                                                            force_batch.detach().cpu().numpy().tolist())
                    metrics.save_classification_data_of_window(pred_class, y_batch)

                    batch_start += self.batch_size
                    if batch_start >= len(X): break
                # ===============================================


        # wandb에 test 이미지를 저장할 artifact를 만듦
        # artifact_for_save_test = wandb.Artifact("")

        train_txt_path = self.save_dir + "/train.txt"
        metrics.set_mean_of_test(self.test_dataset.get_mean_of_test_dataset())
        metrics.set_upper_bound_of_test(upper_bound_of_force)
        message, mse, mae, r2_score, nrmse_for_axis = metrics.get_regression_metrics(export_r2_score=True,
                                                                                     export_nrmse=True)

        print(message)

        # ===============================================
        Visualization.plot_for_lstm(metrics.ground_truth_for_sequence, metrics.prediction_for_window,
                      self.target_index, self.save_dir, train_txt_path,
                      use_wandb=self.use_wandb, label=self.test_dataset.return_sequence_labels())
        # classification 결과의 accuracy
        Visualization.save_accuracy_metrics(self.save_dir, metrics.ground_truth_label, metrics.prediction_label)
        # ===============================================


        # =======================================================
        # 각 subject별 통계
        subjects = self.test_dataset.return_unique_names()
        for s in subjects:
            filter_list = (self.test_dataset.name == s)
            mse, mae, rmse = metrics.get_metrics_for_filter(filter_list)
            print(f"({s}: mse: {mse}, mae: {mae}, rmse: {rmse})")
        # =======================================================

        metrics.get_metrics_for_label()


        # ===============================================
        # 결과 출력
        print(message)
        with open(train_txt_path, 'a') as train_txt_file:
            train_txt_file.write(message)
        # ===============================================


if __name__ == '__main__':
    with open("config.yaml", 'r+', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    force_estimation = ForceEstimation(config)
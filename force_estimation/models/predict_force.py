import json

import torch
import matplotlib.pyplot as plt
from torch.nn.functional import cross_entropy
from wandb.integration.sklearn.calculate import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, precision_recall_fscore_support

try:
    from force_estimation.utils.plot_sequence import *
except ModuleNotFoundError:
    from utils.plot_sequence import *
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
import pandas as pd
import wandb
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datetime import datetime
from typing import List, Dict, Any
from torchmetrics import MetricCollection, MeanMetric
from torchmetrics.classification import Accuracy
import os
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError, R2Score
from typing import Any, Dict, Tuple
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
from .cross_entropy import CrossEntropyMetrics
try:
    from force_estimation.utils.bodypart_label import bodypart_label
except ModuleNotFoundError:
    from utils.bodypart_label import bodypart_label


class PredictionForce(pl.LightningModule):

    def __init__(self,
                 # === 모델 구조 관련 파라미터 ===
                 vision_feature_extractor,
                 temporal_feature_extractor,

                 # === 입력 관련 파라미터 ===
                 dim_of_regression : int = 3,
                 classes : dict = None,
                 task : str = 'regression',     # "regression" | "classification" | "finetune"

                 refer_recent_frames_for_classification: int = 5,
                 classification_type: str = 'MTM', # MTM, OTO
                 classification_head_type: str = 'linear', # linear | gru | lstm | tcn
                 classification_head_hidden_size: int = 128,
                 classification_head_num_layers: int = 1,
                 classification_head_dropout: float = 0.1,
                 classification_tcn_kernel_size: int = 3,
                 freeze_vf_net_for_classification: bool = True,
                 freeze_tf_net_for_classification: bool = True,

                 # === Hyperparameter ===
                 learning_rate : float = 1e-4,
                 weight_decay : float = 1e-3,

                 directory = None
                 ):

        super().__init__()

        self.save_hyperparameters(ignore=['vision_feature_extractor', 'temporal_feature_extractor'])

        self.vf_net = vision_feature_extractor
        self.tf_net = temporal_feature_extractor

        self._task = task
        self.learning_rate = learning_rate

        self.regression_head = nn.Linear(self.vf_net.size_of_output, dim_of_regression)
        self.classification_head = None
        self.classification_rnn = None
        self.classification_tcn = None
        self.classification_output_head = None

        if classes is not None:
            self.classes = classes
        if self._task == "classification":
            if classes is None:
                raise ValueError("Classification task에서는 classes 인자가 필요합니다.")
            self._build_classification_head(len(self.classes))
            self._apply_classification_freeze_policy()

        self.directory = directory
        self.log_file = f'{self.directory}/train.txt'


        # ================= Metrics =================
        if self._task == "regression":
            self.criterion = torch.nn.MSELoss(reduction='mean')

            metrics = MetricCollection({
                "rmse": MeanSquaredError(squared=False),
                "mae": MeanAbsoluteError(),
                "r2": R2Score() })

            self.train_metrics = metrics.clone(prefix="train/")
            self.val_metrics = metrics.clone(prefix="val/")
            self.test_metrics = metrics.clone(prefix="test/")

        elif self._task == "classification":
            self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')

            metrics = MetricCollection({
                "accuracy": Accuracy(task="multiclass", num_classes=len(classes)),
                "cross_entropy": CrossEntropyMetrics() # cross entropy 평균용
            })

            self.train_metrics = metrics.clone(prefix="train/")
            self.val_metrics = metrics.clone(prefix="val/")
            self.test_metrics = metrics.clone(prefix="test/")

        else:
            raise Exception("정확한 Task 기입")


        # ================= Results =================

        self.train_results = self._init_result_dict()
        self.val_results = self._init_result_dict()
        self.test_results = self._init_result_dict()



    def forward(self, x):
        batch_size, timestep, channels, height, width = x.shape

        if self._task == "regression":
            x = self.vf_net(x)
            x = self.tf_net(x)
            x = self.regression_head(x)
            return x

        elif self._task == "classification":
            # [기존] x = x[:, -self.hparams.refer_recent_frames_for_classification:] # 마지막 5 프레임 timestep (batch_size, 5, channels, height, width)
            x = x[:, -self.hparams.refer_recent_frames_for_classification:] # 마지막 R 프레임 timestep (batch_size, R, channels, height, width)
            # [기존] x = self.vf_net(x) # (batch_size, 5, dim_hidden_features=96)
            x = self.vf_net(x) # (batch_size, R, dim_hidden_features=96)
            head_type = self.hparams.classification_head_type.lower()

            if head_type == "linear":
                if self.hparams.classification_type == "MTM":
                    # [기존] x = self.classification_head(x) # (batch_size, 5, 8)
                    x = self.classification_head(x) # (batch_size, R, num_classes)
                elif self.hparams.classification_type == "OTO":
                    # [기존] x = torch.flatten(x, start_dim=1) # (batch_size, dim_hidden_features * 5)
                    x = torch.flatten(x, start_dim=1) # (batch_size, dim_hidden_features * R)
                    # [기존] x = self.classification_head(x)  # (batch_size, 8)
                    x = self.classification_head(x)  # (batch_size, num_classes)
            elif head_type in {"gru", "lstm"}:
                x, _ = self.classification_rnn(x) # (batch_size, R, hidden_size)
                if self.hparams.classification_type == "MTM":
                    x = self.classification_output_head(x) # (batch_size, R, num_classes)
                elif self.hparams.classification_type == "OTO":
                    x = self.classification_output_head(x[:, -1, :]) # (batch_size, num_classes)
            elif head_type == "tcn":
                x = x.transpose(1, 2) # (batch_size, dim_hidden_features, R)
                x = self.classification_tcn(x) # (batch_size, hidden_size, R)
                if self.hparams.classification_type == "MTM":
                    x = self.classification_output_head(x.transpose(1, 2)) # (batch_size, R, num_classes)
                elif self.hparams.classification_type == "OTO":
                    x = self.classification_output_head(x[:, :, -1]) # (batch_size, num_classes)
            else:
                raise ValueError(f"Unknown classification_head_type: {head_type}")

            return x


    # =================== Methods ===================
    def _init_result_dict(self):
        return {
            "prediction": [], # shape: (num of overall sliding window, timestep, 3) or (N, 8)
            "target": [], # shape: (num of overall sliding window, timestep, 3) or (N,)
            "sequence_id": [], # shape: (num of overall sliding window, 1)
            "subject_id": [],  # shape: (num of overall sliding window, 1)
            "bodypart": [] # shape: (num of overall sliding window, 1)
        }

    def _save_result(self, storage, preds, target, sequence_index, subject_id, bodypart):

        storage["prediction"].extend(preds.detach().cpu().tolist())
        storage["target"].extend(target.detach().cpu().tolist())
        storage["sequence_id"].extend(sequence_index.detach().cpu().tolist())
        storage["subject_id"].extend(self._to_python_list(subject_id))
        storage["bodypart"].extend(self._to_python_list(bodypart))

    @staticmethod
    def _to_python_list(value):
        if torch.is_tensor(value):
            value = value.detach().cpu()
            return value.tolist() if value.ndim > 0 else [value.item()]

        if isinstance(value, np.ndarray):
            return value.tolist() if value.ndim > 0 else [value.item()]

        if isinstance(value, (list, tuple)):
            result = []
            for v in value:
                if torch.is_tensor(v):
                    vv = v.detach().cpu()
                    result.append(vv.item() if vv.ndim == 0 else vv.tolist())
                elif isinstance(v, np.ndarray):
                    result.append(v.tolist())
                else:
                    result.append(v)
            return result

        return [value]

    @staticmethod
    def _format_bodypart_name(raw_label):
        # numpy/tensor scalar를 python scalar로 변환
        if torch.is_tensor(raw_label):
            raw_label = raw_label.detach().cpu().item() if raw_label.ndim == 0 else raw_label.detach().cpu().tolist()
        elif isinstance(raw_label, np.generic):
            raw_label = raw_label.item()

        # int 라벨이면 매핑 테이블 사용
        if isinstance(raw_label, (int, np.integer)):
            return bodypart_label.get(int(raw_label), str(raw_label))

        # 문자열 라벨이면 그대로 사용, 세부 라벨(FINGERPAD-THUMB 등)은 상위 라벨로 축약
        if isinstance(raw_label, str):
            if raw_label in bodypart_label.values():
                return raw_label
            coarse_label = raw_label.split("-")[0]
            if coarse_label in bodypart_label.values():
                return coarse_label
            return raw_label

        return str(raw_label)

    def _build_classification_head(self, num_classes: int):
        self.classification_head = None
        self.classification_rnn = None
        self.classification_tcn = None
        self.classification_output_head = None

        head_type = self.hparams.classification_head_type.lower()
        classification_type = self.hparams.classification_type
        vf_dim = self.vf_net.size_of_output
        refer_frames = self.hparams.refer_recent_frames_for_classification

        if head_type == "linear":
            if classification_type == "MTM":
                self.classification_head = nn.Linear(vf_dim, num_classes)
            elif classification_type == "OTO":
                self.classification_head = nn.Linear(vf_dim * refer_frames, num_classes)
            else:
                raise ValueError(f"Unknown classification_type: {classification_type}")
            return

        if head_type in {"gru", "lstm"}:
            hidden_size = self.hparams.classification_head_hidden_size
            num_layers = self.hparams.classification_head_num_layers
            dropout = self.hparams.classification_head_dropout if num_layers > 1 else 0.0

            rnn_cls = nn.GRU if head_type == "gru" else nn.LSTM
            self.classification_rnn = rnn_cls(
                input_size=vf_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                batch_first=True
            )
            self.classification_output_head = nn.Linear(hidden_size, num_classes)
            return

        if head_type == "tcn":
            hidden_size = self.hparams.classification_head_hidden_size
            kernel_size = self.hparams.classification_tcn_kernel_size
            if kernel_size % 2 == 0:
                raise ValueError("classification_tcn_kernel_size는 홀수여야 합니다. (예: 3, 5)")
            padding = kernel_size // 2
            dropout = self.hparams.classification_head_dropout

            self.classification_tcn = nn.Sequential(
                nn.Conv1d(vf_dim, hidden_size, kernel_size=kernel_size, padding=padding),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size, padding=padding),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.classification_output_head = nn.Linear(hidden_size, num_classes)
            return

        raise ValueError(f"Unknown classification_head_type: {head_type}")

    def _classification_parameters(self):
        head_type = self.hparams.classification_head_type.lower()

        if head_type == "linear":
            return list(self.classification_head.parameters())
        if head_type in {"gru", "lstm"}:
            return list(self.classification_rnn.parameters()) + list(self.classification_output_head.parameters())
        if head_type == "tcn":
            return list(self.classification_tcn.parameters()) + list(self.classification_output_head.parameters())

        raise ValueError(f"Unknown classification_head_type: {head_type}")

    def _apply_classification_freeze_policy(self):
        if self._task != "classification":
            return

        if self.hparams.freeze_vf_net_for_classification:
            for p in self.vf_net.parameters():
                p.requires_grad = False
            self.vf_net.eval()
        else:
            for p in self.vf_net.parameters():
                p.requires_grad = True

        if self.hparams.freeze_tf_net_for_classification:
            for p in self.tf_net.parameters():
                p.requires_grad = False
            self.tf_net.eval()
        else:
            for p in self.tf_net.parameters():
                p.requires_grad = True

    def train(self, mode: bool = True):
        super().train(mode)

        # Lightning이 model.train()을 호출해도 freeze된 backbone은 eval 상태를 유지
        if mode and self._task == "classification":
            if self.hparams.freeze_vf_net_for_classification:
                self.vf_net.eval()
            if self.hparams.freeze_tf_net_for_classification:
                self.tf_net.eval()

        return self


    # task 변경
    def set_task(self, value):

        if value == "classification":
            self._build_classification_head(len(self.classes))
            self._apply_classification_freeze_policy()

            metrics = MetricCollection({
                "accuracy": Accuracy(task="multiclass", num_classes=len(self.classes)),
                "cross_entropy": CrossEntropyMetrics()  # cross entropy 평균용
            })

            self.train_metrics = metrics.clone(prefix="train/")
            self.val_metrics = metrics.clone(prefix="val/")
            self.test_metrics = metrics.clone(prefix="test/")

            self.configure_optimizers()

        elif value == "finetuning":
            self.learning_rate = self.hparams.learning_rate * 0.1

        else:
            raise Exception("정확한 Task 기입")

        self._task = value


    def configure_optimizers(self):

        # trainer.fit() 이 시작될 때 딱 한 번 실행됨.
        if self._task == "regression":
            params = (
                    list(self.vf_net.parameters()) +
                    list(self.tf_net.parameters()) +
                    list(self.regression_head.parameters())
            )

        elif self._task == "classification":
            params = list(self._classification_parameters())  # head
            if not self.hparams.freeze_vf_net_for_classification:
                params += list(self.vf_net.parameters())
            if not self.hparams.freeze_tf_net_for_classification:
                params += list(self.tf_net.parameters())

        elif self._task == "finetune":
            params = self.parameters()  # 전체

        return torch.optim.Adam(params, lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)


    # =================== EACH STEP ===================
    def training_step(self, batch, batch_idx):
        # forward
        prediction = self(batch['img'])

        if self._task == "regression":
            # Calculate overall loss
            loss = self.criterion(prediction, batch['force'])

            # per-axis losses
            loss_fx = self.criterion(prediction[:, :, 0], batch['force'][:, :, 0])
            loss_fy = self.criterion(prediction[:, :, 1], batch['force'][:, :, 1])
            loss_fz = self.criterion(prediction[:, :, 2], batch['force'][:, :, 2])

            # logging
            self.log("train/loss", loss, batch_size=batch['img'].size(0), on_step=False, on_epoch=True, prog_bar=True)
            self.log("train/loss_fx", loss_fx, batch_size=batch['img'].size(0), on_step=False, on_epoch=True, prog_bar=False)
            self.log("train/loss_fy", loss_fy, batch_size=batch['img'].size(0), on_step=False, on_epoch=True, prog_bar=False)
            self.log("train/loss_fz", loss_fz, batch_size=batch['img'].size(0), on_step=False, on_epoch=True, prog_bar=False)

            # (B, T, 3) -> (B*T, 3)
            preds_flat = prediction.detach().reshape((-1, 3))  # (B*T,3)
            targets_flat = batch['force'].detach().reshape((-1, 3))  # (B*T,3)

            self.train_metrics.update(preds_flat, targets_flat)
            self._save_result(self.train_results, prediction.detach(), batch['force'].detach(),
                              batch['sequence_id'].detach(), batch['subject_id'], batch['bodypart'])


        elif self._task == "classification":
            # Calculate loss
            # [기존] ground_truth = batch['label'] # (N, 5)
            ground_truth = batch['label'] # (N, R)

            if self.hparams.classification_type == "OTO": # ONE-TO-ONE
                # [기존] ground_truth = torch.mode(ground_truth, dim=1).values.to(torch.long) # (N,)
                ground_truth = ground_truth[:, -1].to(torch.long) # (N,) : 현재(마지막) 프레임 라벨
                prediction = prediction # (N, 8)

            else: # MANY-TO-MANY
                N, T, C = prediction.shape

                ground_truth = ground_truth.reshape(N*T) # (N, 5)
                prediction = prediction.reshape(N*T, C) # (N, 5, 8)

            loss = self.criterion(prediction, ground_truth)
            self.train_metrics.update(prediction, ground_truth)

            self.log("train/loss", loss, batch_size=batch['img'].size(0), on_step=False, on_epoch=True)
            self._save_result(self.train_results, prediction.detach(), ground_truth.detach(),
                              batch['sequence_id'].detach(), batch['subject_id'], batch['bodypart'])

        return {"loss": loss}



    def on_train_epoch_end(self):
        metrics = self.train_metrics.compute()
        self.log_dict(metrics)

        if self._task == "regression":
            with open(f'{self.directory}/train.txt', "a") as f:
                f.write(
                    f"train_rmse={metrics['train/rmse']:.6f}, train_mae={metrics['train/mae']:.6f}, train_r2={metrics['train/r2']:.6f}\n")

            prediction = torch.Tensor(self.train_results['prediction']).reshape((-1, 3))
            ground_truth = torch.Tensor(self.train_results['target']).reshape((-1, 3))

            # 축별 R2, RMSE, MAE를 수동으로 계산하고 로깅
            r2_fx = R2Score()(prediction[:, 0], ground_truth[:, 0])
            r2_fy = R2Score()(prediction[:, 1], ground_truth[:, 1])
            r2_fz = R2Score()(prediction[:, 2], ground_truth[:, 2])

            rmse_fx = MeanSquaredError(squared=False)(prediction[:, 0], ground_truth[:, 0])
            rmse_fy = MeanSquaredError(squared=False)(prediction[:, 1], ground_truth[:, 1])
            rmse_fz = MeanSquaredError(squared=False)(prediction[:, 2], ground_truth[:, 2])

            mae_fx = MeanAbsoluteError()(prediction[:, 0], ground_truth[:, 0])
            mae_fy = MeanAbsoluteError()(prediction[:, 1], ground_truth[:, 1])
            mae_fz = MeanAbsoluteError()(prediction[:, 2], ground_truth[:, 2])

            self.log("train/r2_fx", r2_fx, on_epoch=True, logger=True)
            self.log("train/r2_fy", r2_fy, on_epoch=True, logger=True)
            self.log("train/r2_fz", r2_fz, on_epoch=True, logger=True)
            self.log("train/rmse_fx", rmse_fx, on_epoch=True, logger=True)
            self.log("train/rmse_fy", rmse_fy, on_epoch=True, logger=True)
            self.log("train/rmse_fz", rmse_fz, on_epoch=True, logger=True)
            self.log("train/mae_fx", mae_fx, on_epoch=True, logger=True)
            self.log("train/mae_fy", mae_fy, on_epoch=True, logger=True)
            self.log("train/mae_fz", mae_fz, on_epoch=True, logger=True)

        elif self._task == "classification":
            with open(f'{self.directory}/train.txt', "a") as f:
                f.write(
                    f"train_ce={metrics['train/cross_entropy']:.6f}, train_acc={metrics['train/accuracy']:.6f}\n")

            prediction = torch.Tensor(self.train_results['prediction'])  # (N, C)
            ground_truth = torch.Tensor(self.train_results['target']).to(torch.long)  # (N,)

            acc = Accuracy(task="multiclass", num_classes=len(self.classes))(prediction, ground_truth)
            ce = cross_entropy(prediction, ground_truth)

            self.log(f"train/accuracy", acc, on_epoch=True, logger=True)
            self.log(f"train/cross_entropy", ce, on_epoch=True, logger=True)

        self.train_metrics.reset()
        self.train_results = self._init_result_dict()



    def validation_step(self, batch, batch_idx):
        # forward
        prediction = self(batch['img'])

        if self._task == "regression":
            # Calculate overall loss
            loss = self.criterion(prediction, batch['force'])

            # per-axis losses
            loss_fx = self.criterion(prediction[:, :, 0], batch['force'][:, :, 0])
            loss_fy = self.criterion(prediction[:, :, 1], batch['force'][:, :, 1])
            loss_fz = self.criterion(prediction[:, :, 2], batch['force'][:, :, 2])

            # logging
            self.log("loss_for_monitor", loss, batch_size=batch['img'].size(0), on_step=False, on_epoch=True, prog_bar=False)
            self.log("val/loss", loss, batch_size=batch['img'].size(0), on_step=False, on_epoch=True, prog_bar=True)
            self.log("val/loss_fx", loss_fx, batch_size=batch['img'].size(0), on_step=False, on_epoch=True, prog_bar=False)
            self.log("val/loss_fy", loss_fy, batch_size=batch['img'].size(0), on_step=False, on_epoch=True, prog_bar=False)
            self.log("val/loss_fz", loss_fz, batch_size=batch['img'].size(0), on_step=False, on_epoch=True, prog_bar=False)

            # (B, T, 3) -> (B*T, 3)
            preds_flat = prediction.detach().reshape((-1, 3))  # (B*T,3)
            targets_flat = batch['force'].detach().reshape((-1, 3))  # (B*T,3)

            self.val_metrics.update(preds_flat, targets_flat)
            self._save_result(self.val_results, prediction.detach(), batch['force'].detach(),
                              batch['sequence_id'].detach(), batch['subject_id'], batch['bodypart'])

        elif self._task == "classification":
            # Calculate loss
            # [기존] ground_truth = batch['label']  # (N, 5)
            ground_truth = batch['label']  # (N, R)

            if self.hparams.classification_type == "OTO":  # ONE-TO-ONE
                # [기존] ground_truth = torch.mode(ground_truth, dim=1).values.to(torch.long)  # (N,)
                ground_truth = ground_truth[:, -1].to(torch.long)  # (N,) : 현재(마지막) 프레임 라벨
                prediction = prediction  # (N, 8)

            else:  # MANY-TO-MANY 에서도 하나의 값을 추론하도록

                N, T, C = prediction.shape
                ground_truth = torch.mode(ground_truth, dim=1).values.to(torch.long)  # (N,)
                prediction = prediction.mean(dim=1)  # (N, 8)

            loss = self.criterion(prediction, ground_truth)
            self.val_metrics.update(prediction, ground_truth)

            # logging
            self._save_result(self.val_results, prediction.detach(), ground_truth.detach(),
                              batch['sequence_id'].detach(), batch['subject_id'], batch['bodypart'])

        return {"loss": loss}



    def on_validation_epoch_end(self):
        metrics = self.val_metrics.compute()
        self.log_dict(metrics)

        # =================================================
        # Regression
        # =================================================
        if self._task == "regression":
            with open(f'{self.directory}/train.txt', "a") as f:
                f.write(
                    f"[{self.current_epoch} Epoch] val_rmse={metrics['val/rmse']:.6f}, val_mae={metrics['val/mae']:.6f}, val_r2={metrics['val/r2']:.6f} ")


            prediction = torch.Tensor(self.val_results['prediction']).reshape((-1, 3))
            ground_truth = torch.Tensor(self.val_results['target']).reshape((-1, 3))

            # 축별 R2, RMSE, MAE를 수동으로 계산하고 로깅
            r2_fx = R2Score()(prediction[:, 0], ground_truth[:, 0])
            r2_fy = R2Score()(prediction[:, 1], ground_truth[:, 1])
            r2_fz = R2Score()(prediction[:, 2], ground_truth[:, 2])

            rmse_fx = MeanSquaredError(squared=False)(prediction[:, 0], ground_truth[:, 0])
            rmse_fy = MeanSquaredError(squared=False)(prediction[:, 1], ground_truth[:, 1])
            rmse_fz = MeanSquaredError(squared=False)(prediction[:, 2], ground_truth[:, 2])

            mae_fx = MeanAbsoluteError()(prediction[:, 0], ground_truth[:, 0])
            mae_fy = MeanAbsoluteError()(prediction[:, 1], ground_truth[:, 1])
            mae_fz = MeanAbsoluteError()(prediction[:, 2], ground_truth[:, 2])

            self.log("val/r2_fx", r2_fx, on_epoch=True, logger=True)
            self.log("val/r2_fy", r2_fy, on_epoch=True, logger=True)
            self.log("val/r2_fz", r2_fz, on_epoch=True, logger=True)
            self.log("val/rmse_fx", rmse_fx, on_epoch=True, logger=True)
            self.log("val/rmse_fy", rmse_fy, on_epoch=True, logger=True)
            self.log("val/rmse_fz", rmse_fz, on_epoch=True, logger=True)
            self.log("val/mae_fx", mae_fx, on_epoch=True, logger=True)
            self.log("val/mae_fy", mae_fy, on_epoch=True, logger=True)
            self.log("val/mae_fz", mae_fz, on_epoch=True, logger=True)


        # =================================================
        # Classification
        # =================================================
        elif self._task == "classification":
            with open(f'{self.directory}/train.txt', "a") as f:
                f.write(
                    f"[{self.current_epoch} Epoch] val_ce={metrics['val/cross_entropy']:.6f}, val_acc={metrics['val/accuracy']:.6f} ")

            prediction = torch.Tensor(self.val_results['prediction']) # (N, C)
            ground_truth = torch.Tensor(self.val_results['target']).to(torch.long) # (N,)

            loss = self.criterion(prediction, ground_truth)
            acc = Accuracy(task="multiclass", num_classes=len(self.classes))(prediction, ground_truth)
            ce = cross_entropy(prediction, ground_truth)

            self.log(f"val/accuracy", acc, on_epoch=True, logger=True)
            self.log(f"val/cross_entropy", ce, on_epoch=True, logger=True)
            self.log(f"loss_for_monitor", loss, on_epoch=True, logger=True)

            # 각 class별 Accuracy와 Cross_Entropy
            gt_class, pred_class = ground_truth, prediction.argmax(dim=1)  # (N,)
            for c in range(len(self.classes)):
                mask = (gt_class == c)
                if mask.sum() == 0: continue

                # Accuracy
                acc_c = (pred_class[mask] == c).float().mean()
                # Cross Entropy
                ce_c = F.cross_entropy(prediction[mask], gt_class[mask]) # (Nc, C), (Nc)

                self.log(f"val/acc_class_{c}", acc_c, on_epoch=True, logger=True)
                self.log(f"val/ce_class_{c}", ce_c, on_epoch=True, logger=True)

        self.val_metrics.reset()
        self.val_results = self._init_result_dict()



    def test_step(self, batch, batch_idx):

        # forward
        prediction = self(batch['img'])

        if self._task == "regression":
            # Calculate overall loss
            loss = self.criterion(prediction, batch['force'])

            # per-axis losses
            loss_fx = self.criterion(prediction[:, :, 0], batch['force'][:, :, 0])
            loss_fy = self.criterion(prediction[:, :, 1], batch['force'][:, :, 1])
            loss_fz = self.criterion(prediction[:, :, 2], batch['force'][:, :, 2])

            # logging
            self.log("test/loss", loss, batch_size=batch['img'].size(0), on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/loss_fx", loss_fx, batch_size=batch['img'].size(0), on_step=False, on_epoch=True, prog_bar=False)
            self.log("test/loss_fy", loss_fy, batch_size=batch['img'].size(0), on_step=False, on_epoch=True, prog_bar=False)
            self.log("test/loss_fz", loss_fz, batch_size=batch['img'].size(0), on_step=False, on_epoch=True, prog_bar=False)

            # (B, T, 3) -> (B*T, 3)
            preds_flat = prediction.detach().reshape((-1, 3))  # (B*T,3)
            targets_flat = batch['force'].detach().reshape((-1, 3))  # (B*T,3)

            self.test_metrics.update(preds_flat, targets_flat)
            self._save_result(self.test_results, prediction.detach(), batch['force'].detach(),
                              batch['sequence_id'].detach(), batch['subject_id'], batch['bodypart'])


        elif self._task == "classification":
            # [기존] ground_truth = batch['label'] # (N,)
            ground_truth = batch['label'] # (N, R)

            if self.hparams.classification_type == "OTO":  # MANY-TO-ONE
                # [기존] ground_truth = torch.mode(ground_truth, dim=1).values.to(torch.long)  # (N,)
                ground_truth = ground_truth[:, -1].to(torch.long)  # (N,) : 현재(마지막) 프레임 라벨
                prediction = prediction  # (N, 8)

            else:  # MANY-TO-MANY 에서도 하나의 값을 추론하도록
                N, T, C = prediction.shape

                ground_truth = torch.mode(ground_truth, dim=1).values.to(torch.long)  # (N,)
                prediction = torch.nn.functional.softmax(prediction, dim=2).mean(dim=1)  # (N, 5, 8) -> (N, 8)

            self.test_metrics.update(prediction, ground_truth)

            self.log("test/accuracy", self.test_metrics["accuracy"], on_epoch=True)
            self.log("test/cross_entropy", self.test_metrics["cross_entropy"], on_epoch=True)

            self._save_result(self.test_results, prediction.detach(), ground_truth.detach(),
                              batch['sequence_id'].detach(), batch['subject_id'], batch['bodypart'])



    def on_test_epoch_end(self):
        metrics = self.test_metrics.compute()
        self.log_dict(metrics)

        if self._task == "regression":
            prediction = torch.Tensor(self.test_results['prediction']).reshape((-1, 3)) # (B, T, 3) -> (B*T, 3)
            ground_truth = torch.Tensor(self.test_results['target']).reshape((-1, 3)) # (B, T, 3) -> (B*T, 3)

            # 축별 R2, RMSE, MAE를 수동으로 계산하고 로깅
            r2_all = R2Score()(prediction, ground_truth)
            r2_fx = R2Score()(prediction[:, 0], ground_truth[:, 0])
            r2_fy = R2Score()(prediction[:, 1], ground_truth[:, 1])
            r2_fz = R2Score()(prediction[:, 2], ground_truth[:, 2])

            rmse_all = MeanSquaredError(squared=False)(prediction, ground_truth)
            rmse_fx = MeanSquaredError(squared=False)(prediction[:, 0], ground_truth[:, 0])
            rmse_fy = MeanSquaredError(squared=False)(prediction[:, 1], ground_truth[:, 1])
            rmse_fz = MeanSquaredError(squared=False)(prediction[:, 2], ground_truth[:, 2])

            mae_all = MeanAbsoluteError()(prediction, ground_truth)
            mae_fx = MeanAbsoluteError()(prediction[:, 0], ground_truth[:, 0])
            mae_fy = MeanAbsoluteError()(prediction[:, 1], ground_truth[:, 1])
            mae_fz = MeanAbsoluteError()(prediction[:, 2], ground_truth[:, 2])

            self.log("test/r2_all", r2_all, on_epoch=True, logger=True)
            self.log("test/r2_fx", r2_fx, on_epoch=True, logger=True)
            self.log("test/r2_fy", r2_fy, on_epoch=True, logger=True)
            self.log("test/r2_fz", r2_fz, on_epoch=True, logger=True)

            self.log("test/rmse_all", rmse_all, on_epoch=True, logger=True)
            self.log("test/rmse_fx", rmse_fx, on_epoch=True, logger=True)
            self.log("test/rmse_fy", rmse_fy, on_epoch=True, logger=True)
            self.log("test/rmse_fz", rmse_fz, on_epoch=True, logger=True)

            self.log("test/mae_all", mae_all, on_epoch=True, logger=True)
            self.log("test/mae_fx", mae_fx, on_epoch=True, logger=True)
            self.log("test/mae_fy", mae_fy, on_epoch=True, logger=True)
            self.log("test/mae_fz", mae_fz, on_epoch=True, logger=True)

            # REPORT
            with open(f'{self.directory}/test_result.txt', "a") as f:
                f.write(f"[R2 SCORE] Overall: {r2_all:.6f} Fx: {r2_fx:.6f} Fy: {r2_fy:.6f} Fz: {r2_fz:.6f}\n")
                f.write(f"[RMSE] Overall: {rmse_all:.6f} Fx: {rmse_fx:.6f} Fy: {rmse_fy:.6f} Fz: {rmse_fz:.6f}\n")
                f.write(f"[MAE] Overall: {mae_all:.6f} Fx: {mae_fx:.6f} Fy: {mae_fy:.6f} Fz: {mae_fz:.6f}\n")


            # 시퀀스 별로 plot
            grouped_data, unique_seq = self.group_by_sequence(self.test_results)
            for sid in unique_seq:
                prediction = reconstruct_list_from_windows(grouped_data[sid]['prediction'])
                ground_truth = reconstruct_list_from_windows(grouped_data[sid]['target'])

                plot_sequence(ground_truth, prediction, f"{self.directory}/test", sid,
                              label=grouped_data[sid]['bodypart'][0],
                              name=grouped_data[sid]['subject_id'][0])


            # 피험자별로 Result Report
            grouped_data, unique_seq = self.group_by_name(self.test_results)
            with open(f'{self.directory}/test_result.txt', "a") as f:
                f.write(f"\n\n ========= REPORT FOR EACH SUBJECT ========= \n\n")
            for sid in unique_seq:
                prediction = torch.Tensor(reconstruct_list_from_windows(grouped_data[sid]['prediction']))
                ground_truth = torch.Tensor(reconstruct_list_from_windows(grouped_data[sid]['target']))

                r2_for_subject = R2Score()(prediction, ground_truth)
                rmse_for_subject = MeanSquaredError(squared=False)(prediction, ground_truth)
                mae_for_subject = MeanAbsoluteError()(prediction, ground_truth)

                r2_x = R2Score()(prediction[:, 0], ground_truth[:, 0])
                r2_y = R2Score()(prediction[:, 1], ground_truth[:, 1])
                r2_z = R2Score()(prediction[:, 2], ground_truth[:, 2])

                rmse_x = MeanSquaredError(squared=False)(prediction[:, 0], ground_truth[:, 0])
                rmse_y = MeanSquaredError(squared=False)(prediction[:, 1], ground_truth[:, 1])
                rmse_z = MeanSquaredError(squared=False)(prediction[:, 2], ground_truth[:, 2])

                mae_x = MeanAbsoluteError()(prediction[:, 0], ground_truth[:, 0])
                mae_y = MeanAbsoluteError()(prediction[:, 1], ground_truth[:, 1])
                mae_z = MeanAbsoluteError()(prediction[:, 2], ground_truth[:, 2])

                with open(f'{self.directory}/test_result.txt', "a") as f:
                    f.write(f"[{sid}] R2 SCORE: {r2_for_subject:.6f} RMSE: {rmse_for_subject:.6f} MAE: {mae_for_subject:.6f} (mean of force: {torch.mean(torch.norm(ground_truth, dim=1))})\n")
                    f.write(f"(R2 SCORE) X: {r2_x:.6f}, Y: {r2_y:.6f}, Z: {r2_z:.6f}\n")
                    f.write(f"(RMSE) X: {rmse_x:.6f}, Y: {rmse_y:.6f}, Z: {rmse_z:.6f}\n")
                    f.write(f"(MAE) X: {mae_x:.6f}, Y: {mae_y:.6f}, Z: {mae_z:.6f}\n")


            # bodypart 별로 Result Report
            grouped_data, unique_seq =  self.group_by_label(self.test_results)
            with open(f'{self.directory}/test_result.txt', "a") as f:
                f.write(f"\n\n ========= REPORT FOR EACH BODYPART ========= \n\n")
            for sid in unique_seq:
                prediction = torch.Tensor(reconstruct_list_from_windows(grouped_data[sid]['prediction']))
                ground_truth = torch.Tensor(reconstruct_list_from_windows(grouped_data[sid]['target']))

                r2_for_label = R2Score()(prediction, ground_truth)
                rmse_for_label = MeanSquaredError(squared=False)(prediction, ground_truth)
                mae_for_label = MeanAbsoluteError()(prediction, ground_truth)

                r2_x = R2Score()(prediction[:, 0], ground_truth[:, 0])
                r2_y = R2Score()(prediction[:, 1], ground_truth[:, 1])
                r2_z = R2Score()(prediction[:, 2], ground_truth[:, 2])

                rmse_x = MeanSquaredError(squared=False)(prediction[:, 0], ground_truth[:, 0])
                rmse_y = MeanSquaredError(squared=False)(prediction[:, 1], ground_truth[:, 1])
                rmse_z = MeanSquaredError(squared=False)(prediction[:, 2], ground_truth[:, 2])

                mae_x = MeanAbsoluteError()(prediction[:, 0], ground_truth[:, 0])
                mae_y = MeanAbsoluteError()(prediction[:, 1], ground_truth[:, 1])
                mae_z = MeanAbsoluteError()(prediction[:, 2], ground_truth[:, 2])

                with open(f'{self.directory}/test_result.txt', "a") as f:
                    bodypart_name = self._format_bodypart_name(sid)
                    f.write(f"[{bodypart_name}] R2 SCORE: {r2_for_label:.6f} RMSE: {rmse_for_label:.6f} MAE: {mae_for_label:.6f}\n")
                    f.write(f"(R2 SCORE) X: {r2_x:.6f}, Y: {r2_y:.6f}, Z: {r2_z:.6f}\n")
                    f.write(f"(RMSE) X: {rmse_x:.6f}, Y: {rmse_y:.6f}, Z: {rmse_z:.6f}\n")
                    f.write(f"(MAE) X: {mae_x:.6f}, Y: {mae_y:.6f}, Z: {mae_z:.6f}\n")


        elif self._task == "classification":
            # ==========================================================
            # 1. 결과 텐서 정리
            # ==========================================================
            logits = torch.tensor(self.test_results["prediction"])  # (N, C)
            gt = torch.tensor(self.test_results["target"]).long()  # (N,)
            pred = logits.argmax(dim=1)  # (N,)
            num_classes = len(self.classes)

            # numpy 변환 (sklearn용)
            gt_np = gt.cpu().numpy()
            pred_np = pred.cpu().numpy()

            # ==========================================================
            # 2. Overall Metrics
            # ==========================================================

            acc = Accuracy(task="multiclass", num_classes=num_classes)(logits, gt)
            ce = F.cross_entropy(logits, gt)
            precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
                gt_np, pred_np, average="macro", zero_division=0
            )

            precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
                gt_np, pred_np, average="weighted", zero_division=0
            )

            # Lightning log
            self.log("test/accuracy", acc, on_epoch=True)
            self.log("test/cross_entropy", ce, on_epoch=True)
            self.log("test/f1_macro", f1_macro, on_epoch=True)
            self.log("test/f1_weighted", f1_weighted, on_epoch=True)

            # ==========================================================
            # 3. Per-Class Metrics
            # ==========================================================

            precision, recall, f1, support = precision_recall_fscore_support(
                gt_np,
                pred_np,
                labels=np.arange(num_classes),
                average=None,
                zero_division=0
            )

            per_class_rows = []
            for c in range(num_classes):
                mask = (gt == c)
                n_c = int(mask.sum().item())

                if n_c == 0:
                    acc_c = float("nan")
                    ce_c = float("nan")
                else:
                    acc_c = (pred[mask] == c).float().mean().item()
                    ce_c = F.cross_entropy(logits[mask], gt[mask]).item()

                # Lightning log
                self.log(f"test/acc_class_{c}", acc_c, on_epoch=True)
                self.log(f"test/ce_class_{c}", ce_c, on_epoch=True)
                self.log(f"test/f1_class_{c}", f1[c], on_epoch=True)

                per_class_rows.append({
                    "class_index": c,
                    "class_name": bodypart_label[c],
                    "n": n_c,
                    "accuracy": acc_c,
                    "cross_entropy": ce_c,
                    "precision": precision[c],
                    "recall": recall[c],
                    "f1": f1[c],
                })

            # ==========================================================
            # 4. Confusion Matrix (raw + normalized)
            # ==========================================================

            confmat = torch.zeros((num_classes, num_classes), dtype=torch.long)

            for t, p in zip(gt, pred):
                confmat[t, p] += 1

            cm_normalized = confusion_matrix(
                gt_np,
                pred_np,
                labels=np.arange(num_classes),
                normalize="true"  # row normalize
            )

            # plot normalized confusion matrix

            try:
                class_names = list(self.classes.values())
            except Exception:
                class_names = [str(i) for i in range(num_classes)]

            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm_normalized,
                display_labels=class_names
            )

            fig, ax = plt.subplots(figsize=(8, 6))
            disp.plot(ax=ax, cmap="Blues", values_format=".2f", colorbar=True)
            ax.set_title("Normalized Confusion Matrix")
            plt.xticks(rotation=45)
            cm_path = f"{self.directory}/test_confusion_matrix.png"
            plt.tight_layout()
            plt.savefig(cm_path, dpi=300)
            plt.close(fig)

            # ==========================================================
            # 5. Write Report File
            # ==========================================================

            os.makedirs(self.directory, exist_ok=True)
            report_path = f"{self.directory}/test_result.txt"

            with open(report_path, "a", encoding="utf-8") as f:
                f.write("\n\n========= TEST REPORT (CLASSIFICATION) =========\n")
                f.write(f"Overall Accuracy: {acc.item():.6f}\n")
                f.write(f"Overall CrossEntropy: {ce.item():.6f}\n")
                f.write(f"F1 (macro): {f1_macro:.6f}\n")
                f.write(f"F1 (weighted): {f1_weighted:.6f}\n\n")
                f.write("[Per-class]\n")

                for r in per_class_rows:
                    f.write(
                        f"- class {r['class_index']} ({r['class_name']}): "
                        f"n={r['n']}, acc={r['accuracy']:.6f}, "
                        f"P={r['precision']:.6f}, "
                        f"R={r['recall']:.6f}, "
                        f"F1={r['f1']:.6f}, "
                        f"CE={r['cross_entropy']:.6f}\n"
                    )

                f.write("\n[Confusion Matrix] (rows=true, cols=pred)\n")

                for i in range(num_classes):
                    row = " ".join([f"{int(x):5d}" for x in confmat[i].tolist()])
                    f.write(row + "\n")

            # ==========================================================
            # 6. Save CSV
            # ==========================================================

            df = pd.DataFrame(per_class_rows)
            df.to_csv(f"{self.directory}/test_classification_summary.csv", index=False)

            raw_df = pd.DataFrame({
                "gt": gt_np,
                "pred": pred_np,
                "sequence_id": np.array(self.test_results["sequence_id"]),
                "subject_id": np.array(self.test_results["subject_id"]),
                "bodypart": np.array(self.test_results["bodypart"]),
            })

            raw_df.to_csv(f"{self.directory}/test_predictions.csv", index=False)

            # ==========================================================
            # 7. Subject-wise Report
            # ==========================================================

            subject_ids = np.array(self.test_results["subject_id"])
            unique_subjects = np.unique(subject_ids)

            with open(report_path, "a", encoding="utf-8") as f:
                f.write("\n\n========= REPORT FOR EACH SUBJECT =========\n\n")

            for sid in unique_subjects:
                idx = np.where(subject_ids == sid)[0]
                logits_sub = logits[idx]
                gt_sub = gt[idx]
                pred_sub = logits_sub.argmax(dim=1)

                if len(gt_sub) == 0:
                    continue

                acc_sub = Accuracy(task="multiclass", num_classes=num_classes)(logits_sub, gt_sub)
                ce_sub = F.cross_entropy(logits_sub, gt_sub)

                _, _, f1_sub, _ = precision_recall_fscore_support(
                    gt_sub.cpu().numpy(),
                    pred_sub.cpu().numpy(),
                    average="macro",
                    zero_division=0
                )

                self.log(f"test/accuracy_subject_{sid}", acc_sub, on_epoch=True)
                self.log(f"test/f1_subject_{sid}", f1_sub, on_epoch=True)

                with open(report_path, "a", encoding="utf-8") as f:
                    f.write(f"[Subject {sid}]\n")
                    f.write(f"Accuracy: {acc_sub.item():.6f}\n")
                    f.write(f"CrossEntropy: {ce_sub.item():.6f}\n")
                    f.write(f"F1(macro): {f1_sub:.6f}\n\n")

            # WandB logging

            if wandb.run is not None:
                wandb.log({"test/confusion_matrix": wandb.Image(cm_path)})

        self.test_metrics.reset()
        self.test_results = self._init_result_dict()


    # ======================================================
    # 그룹 & 필터 관련 Methods
    # ======================================================

    def group_by_sequence(self,
                          results):

        seq_ids = np.array(results["sequence_id"])
        unique_seq = np.unique(seq_ids)

        grouped = {}

        for sid in unique_seq:
            idx = np.where(seq_ids == sid)[0]

            preds = [results["prediction"][i] for i in idx]
            targets = [results["target"][i] for i in idx]
            subject_id = [results["subject_id"][i] for i in idx]
            sequence_label = [results["bodypart"][i] for i in idx]

            # (num_windows, T, 3) 로 stack
            grouped[sid] = {
                "subject_id": np.stack(subject_id),
                "bodypart": np.stack(sequence_label),
                "prediction": np.stack(preds),
                "target": np.stack(targets),
            }

        return grouped, unique_seq



    def group_by_name(self, results):

        seq_ids = np.array(results["subject_id"])
        unique_seq = np.unique(seq_ids)

        grouped = {}

        for sid in unique_seq:
            idx = np.where(seq_ids == sid)[0]

            preds = [results["prediction"][i] for i in idx]
            targets = [results["target"][i] for i in idx]
            sequence_id = [results["sequence_id"][i] for i in idx]
            bodypart = [results["bodypart"][i] for i in idx]

            # (num_windows, T, 3) 로 stack
            grouped[sid] = {
                "sequence_id": np.stack(sequence_id),
                "bodypart": np.stack(bodypart),
                "prediction": np.stack(preds),
                "target": np.stack(targets),
            }

        return grouped, unique_seq


    def group_by_label(self, results):

        seq_ids = np.array(results["bodypart"])
        unique_seq = np.unique(seq_ids)

        grouped = {}

        for sid in unique_seq:
            idx = np.where(seq_ids == sid)[0]

            preds = [results["prediction"][i] for i in idx]
            targets = [results["target"][i] for i in idx]
            sequence_id = [results["sequence_id"][i] for i in idx]
            subject_id = [results["subject_id"][i] for i in idx]

            # (num_windows, T, 3) 로 stack
            grouped[sid] = {
                "sequence_id": np.stack(sequence_id),
                "subject_id": np.stack(subject_id),
                "prediction": np.stack(preds),
                "target": np.stack(targets),
            }

        return grouped, unique_seq
    # ======================================================

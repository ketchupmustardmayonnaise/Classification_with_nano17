import json

import torch
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


class PredictionForce(pl.LightningModule):

    def __init__(self,
                 # === 모델 구조 관련 파라미터 ===
                 vision_feature_extractor,
                 temporal_feature_extractor,

                 # === 입력 관련 파라미터 ===
                 dim_of_regression : int = 3,
                 classes : dict = None,
                 task : str = 'regression',     # "regression" | "classification" | "finetune"

                 # === Hyperparameter ===
                 learning_rate : float = 1e-4,
                 weight_decay : float = 1e-3,

                 directory = None
                 ):

        super().__init__()

        self.save_hyperparameters(ignore=['vision_feature_extractor', 'temporal_feature_extractor'])

        self.vf_net = vision_feature_extractor
        self.tf_net = temporal_feature_extractor

        self.regression_head = nn.Linear(self.tf_net.size_of_output, dim_of_regression)
        self.classification_head = None

        self._task = task
        self.learning_rate = learning_rate

        self.directory = directory
        self.log_file = f'{self.directory}/train.txt'

        if classes is not None:
            self.classes = classes
        if self._task == "classification":
            self.classification_head = nn.Linear(self.tf_net.size_of_output, len(classes))


        self.criterion = torch.nn.MSELoss(reduction='mean')

        # ================= Metrics =================
        if self._task == "regression":
            self.criterion = torch.nn.MSELoss(reduction='mean')

            metrics = MetricCollection({
                "rmse": MeanSquaredError(squared=False),
                "mae": MeanAbsoluteError(),
                "r2": R2Score() })

            self.train_metrics = metrics.clone(prefix="train_")
            self.val_metrics = metrics.clone(prefix="val_")
            self.test_metrics = metrics.clone(prefix="test_")

        elif self._task == "classification":
            self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')

            metrics = MetricCollection({
                "acc": Accuracy(task="multiclass", num_classes=len(classes)),
                "ce": MeanMetric() # cross entropy 평균용
            })

            self.train_metrics = metrics.clone(prefix="train_")
            self.val_metrics = metrics.clone(prefix="val_")
            self.test_metrics = metrics.clone(prefix="test_")

        else:
            raise Exception("정확한 Task 기입")


        # Lists to store results for epoch end (for per-axis calculations)
        self.train_predictions = []
        self.train_targets = []
        self.val_predictions = []
        self.val_targets = []
        self.test_predictions = []  # 모든 테스트 샘플의 예측값
        self.test_targets = []  # 모든 테스트 샘플의 실제값



        # ================= Loss =================

        self.train_results = self._init_result_dict()
        self.val_results = self._init_result_dict()
        self.test_results = self._init_result_dict()



    def forward(self, x):
        batch_size, timestep, channels, height, width = x.shape

        x = self.vf_net(x)
        x = self.tf_net(x)

        if self._task == "classification":
            x = self.classification_head(x)
            return x

        elif self._task == "regression":
            x = self.regression_head(x)
            return x



    def step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]):
        prediction = self.forward(batch['img'])

        # Calculate overall loss
        loss = self.criterion(prediction, batch['force'])

        # per-axis losses
        loss_fx = self.criterion(prediction[:, :, 0], batch['force'][:, :, 0])
        loss_fy = self.criterion(prediction[:, :, 1], batch['force'][:, :, 1])
        loss_fz = self.criterion(prediction[:, :, 2], batch['force'][:, :, 2])

        # 이미지(images)와 함께 추가 메타데이터도 반환합니다.
        # 이 반환 값들은 training/validation/test_step에서 필요에 따라 사용됩니다.
        return (loss, loss_fx, loss_fy, loss_fz,
                prediction, batch['force'], batch['img'],
                batch['sequence_id'], batch['subject_id'])



    # =================== Methods ===================
    def _init_result_dict(self):
        return {
            "prediction": [], # shape: (num of overall sliding window, timestep, 3)
            "target": [], # shape: (num of overall sliding window, timestep, 3)
            "sequence_id": [], # shape: (num of overall sliding window, 1)
            "subject_id": []  # shape: (num of overall sliding window, 1)
        }

    def _save_result(self, storage, preds, target, sequence_index, subject_id):

        storage["prediction"].extend(preds.detach().cpu())
        storage["target"].extend(target.detach().cpu())
        storage["sequence_id"].extend(sequence_index)
        storage["subject_id"].extend(subject_id)


    # task 변경
    def set_task(self, value):

        if value == "classification":
            self.classification_head = nn.Linear(self.tf_net.size_of_output, len(self.classes))

            metrics = MetricCollection({
                "acc": Accuracy(task="multiclass", num_classes=len(self.classes)),
                "ce": MeanMetric()  # cross entropy 평균용
            }).clone(prefix="train_")

            self.train_metrics = metrics.clone(prefix="train_")
            self.val_metrics = metrics.clone(prefix="val_")
            self.test_metrics = metrics.clone(prefix="test_")

        elif value == "finetuning":
            self.learning_rate = self.hparams.learning_rate * 0.1

        else:
            raise Exception("정확한 Task 기입")

        self._task = value


    def configure_optimizers(self):
        # Initially only train the head - backbone will be added by callback
        if self._task == "regression":
            params = (
                    list(self.vf_net.parameters()) +
                    list(self.tf_net.parameters()) +
                    list(self.regression_head.parameters())
            )

        elif self._task == "classification":
            params = self.classification_head.parameters()  # head only

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
            preds_flat = prediction.detach().reshape(-1, 3)  # (B*T,3)
            targets_flat = batch['force'].detach().reshape(-1, 3)  # (B*T,3)

            self.train_metrics.update(preds_flat, targets_flat)
            self._save_result(self.train_results, prediction.detach(), batch['force'].detach(),
                              batch['sequence_id'], batch['subject_id'])


        elif self._task == "classification":
            # Calculate loss
            loss = self.criterion(prediction, batch['label'])

            pred_class = torch.argmax(prediction, dim=1)

            # 수정 예정
            # self.train_metrics["acc"](pred_class, batch['label'])
            # self.train_metrics(prediction, batch['label'])

        return {"loss": loss}



    def on_train_epoch_end(self):
        metrics = self.train_metrics.compute()
        self.log_dict(metrics)

        with open(f'{self.directory}/train.txt', "a") as f:
            f.write(f"train_rmse={metrics['train_rmse']:.6f}, train_mae={metrics['train_mae']:.6f}, train_r2={metrics['train_r2']:.6f}\n")

        if self._task == "regression":
            prediction = torch.cat(self.train_results['prediction'], dim=0)
            ground_truth = torch.cat(self.train_results['target'], dim=0)

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
            prediction = torch.cat(self.train_results['prediction'], dim=0)
            ground_truth = torch.cat(self.train_results['target'], dim=0)

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
            preds_flat = prediction.detach().reshape(-1, 3)  # (B*T,3)
            targets_flat = batch['force'].detach().reshape(-1, 3)  # (B*T,3)

            self.val_metrics.update(preds_flat, targets_flat)
            self._save_result(self.val_results, prediction.detach(), batch['force'].detach(),
                              batch['sequence_id'], batch['subject_id'])

        elif self._task == "classification":
            # Calculate loss
            loss = self.criterion(prediction.detach(), batch['label'].detach())
            pred_class = torch.argmax(prediction, dim=1)

        return {"loss": loss}



    def on_validation_epoch_end(self):
        metrics = self.val_metrics.compute()
        self.log_dict(metrics)

        with open(f'{self.directory}/train.txt', "a") as f:
            f.write(f"[{self.current_epoch} Epoch] val_rmse={metrics['val_rmse']:.6f}, val_mae={metrics['val_mae']:.6f}, val_r2={metrics['val_r2']:.6f} ")

        if self._task == "regression":
            prediction = torch.cat(self.val_results['prediction'], dim=0)
            ground_truth = torch.cat(self.val_results['target'], dim=0)

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
            preds_flat = prediction.detach().reshape(-1, 3)  # (B*T,3)
            targets_flat = batch['force'].detach().reshape(-1, 3)  # (B*T,3)

            self.test_metrics.update(preds_flat, targets_flat)
            self._save_result(self.test_results, prediction.detach(), batch['force'].detach(),
                              batch['sequence_id'], batch['subject_id'])

        elif self._task == "classification":
            loss = self.cls_loss(prediction, batch['label'])
            pred_class = torch.argmax(prediction, dim=1)

            self.test_metrics(pred_class, batch['label'])
            self._save_result(self.test_results, prediction.detach(), batch['label'].detach(),
                              batch['sequence_id'], batch['subject_id'])



    def on_test_epoch_end(self):
        metrics = self.test_results.compute()
        self.log_dict(metrics)

        if self._task == "regression":
            prediction = torch.cat(self.test_results['prediction'], dim=0)
            ground_truth = torch.cat(self.test_results['target'], dim=0)

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

            self.log("test/r2_fx", r2_fx, on_epoch=True, logger=True)
            self.log("test/r2_fy", r2_fy, on_epoch=True, logger=True)
            self.log("test/r2_fz", r2_fz, on_epoch=True, logger=True)
            self.log("test/rmse_fx", rmse_fx, on_epoch=True, logger=True)
            self.log("test/rmse_fy", rmse_fy, on_epoch=True, logger=True)
            self.log("test/rmse_fz", rmse_fz, on_epoch=True, logger=True)
            self.log("test/mae_fx", mae_fx, on_epoch=True, logger=True)
            self.log("test/mae_fy", mae_fy, on_epoch=True, logger=True)
            self.log("test/mae_fz", mae_fz, on_epoch=True, logger=True)

        self.test_metrics.reset()
        self.test_results = self._init_result_dict()

    def group_by_sequence(self,
                          results,
                          filter_type="sequence_id" # "sequence_id" or "subject_id"
    ):

        seq_ids = np.array(results[filter_type])
        unique_seq = np.unique(seq_ids)

        grouped = {}

        for sid in unique_seq:
            idx = np.where(seq_ids == sid)[0]

            preds = [results["prediction"][i] for i in idx]
            targets = [results["target"][i] for i in idx]

            # (num_windows, T, 3) 로 stack
            grouped[sid] = {
                "prediction": np.stack(preds),
                "target": np.stack(targets),
            }

        return grouped


    def metriccollection_to_float_dict(self, metric_collection):
        out = metric_collection.compute()
        return {
            k: v.item() if torch.is_tensor(v) else float(v)
            for k, v in out.items()
        }

    # def _calculate_and_log_epoch_end_metrics(self, outputs: List[Dict], stage: str):


import json

import torch
from torch.nn.functional import cross_entropy

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
from cross_entropy import CrossEntropyMetrics


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
                "accuracy": Accuracy(task="multiclass", num_classes=len(classes)),
                "cross_entropy": CrossEntropyMetrics() # cross entropy 평균용
            })

            self.train_metrics = metrics.clone(prefix="train_")
            self.val_metrics = metrics.clone(prefix="val_")
            self.test_metrics = metrics.clone(prefix="test_")

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
            x = x[:, -1] # 마지막 timestep (batch_size, channels, height, width)
            x = self.vf_net(x) # (batch_size, dim_hidden_features=96)
            x = self.classification_head(x) # (batch_size, 8)
            return x


    # =================== Methods ===================
    def _init_result_dict(self):
        return {
            "prediction": [], # shape: (num of overall sliding window, timestep, 3)
            "target": [], # shape: (num of overall sliding window, timestep, 3)
            "sequence_id": [], # shape: (num of overall sliding window, 1)
            "subject_id": [],  # shape: (num of overall sliding window, 1)
            "sequence_label": [] # shape: (num of overall sliding window, 1)
        }

    def _save_result(self, storage, preds, target, sequence_index, subject_id, sequence_label):

        storage["prediction"].extend(preds.detach().cpu().tolist())
        storage["target"].extend(target.detach().cpu().tolist())
        storage["sequence_id"].extend(sequence_index.detach().cpu().tolist())
        storage["subject_id"].extend(subject_id)
        storage["sequence_label"].extend(sequence_label.detach().cpu().tolist())


    # task 변경
    def set_task(self, value):

        if value == "classification":
            self.classification_head = nn.Linear(self.tf_net.size_of_output, len(self.classes))

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
                              batch['sequence_id'].detach(), batch['subject_id'], batch['sequence_label'].detach())


        elif self._task == "classification":
            # Calculate loss
            ground_truth = batch['label']
            loss = self.criterion(prediction, torch.argmax(ground_truth, dim=1))
            self.train_metrics.update(prediction, ground_truth)

            self.log("train/loss", loss, batch_size=batch['img'].size(0), on_step=False, on_epoch=True)
            self._save_result(self.train_results, prediction.detach(), batch['label'].detach(),
                              batch['sequence_id'].detach(), batch['subject_id'], batch['sequence_label'].detach())

        return {"loss": loss}



    def on_train_epoch_end(self):
        metrics = self.train_metrics.compute()
        self.log_dict(metrics)

        if self._task == "regression":
            with open(f'{self.directory}/train.txt', "a") as f:
                f.write(
                    f"train_rmse={metrics['train_rmse']:.6f}, train_mae={metrics['train_mae']:.6f}, train_r2={metrics['train_r2']:.6f}\n")

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

            prediction = torch.Tensor(self.train_results['prediction']) # (N, c)
            ground_truth = torch.Tensor(self.train_results['target']) # (N, c)

            acc = Accuracy(task="multiclass", num_classes=len(self.classes))(torch.argmax(prediction, dim=1), torch.argmax(ground_truth, dim=1))
            ce = cross_entropy(prediction, torch.argmax(ground_truth, dim=1).float())

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
            preds_flat = prediction.detach().reshape(-1, 3)  # (B*T,3)
            targets_flat = batch['force'].detach().reshape(-1, 3)  # (B*T,3)

            self.val_metrics.update(preds_flat, targets_flat)
            self._save_result(self.val_results, prediction.detach(), batch['force'].detach(),
                              batch['sequence_id'].detach(), batch['subject_id'], batch['sequence_label'].detach())

        elif self._task == "classification":

            # Calculate loss
            ground_truth = batch['label']  # (N, C)
            loss = self.criterion(prediction, torch.argmax(ground_truth, dim=1))
            self.val_metrics.update(prediction, ground_truth)

            # logging
            self._save_result(self.val_results, prediction.detach(), batch['label'].detach(),
                              batch['sequence_id'].detach(), batch['subject_id'], batch['sequence_label'].detach())

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
                    f"[{self.current_epoch} Epoch] val_rmse={metrics['val_rmse']:.6f}, val_mae={metrics['val_mae']:.6f}, val_r2={metrics['val_r2']:.6f} ")


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
                    f"[{self.current_epoch} Epoch] val_ce={metrics['val_ce']:.6f}, val_acc={metrics['val_acc']:.6f} ")

            prediction = torch.Tensor(self.val_results['prediction']) # (N, C)
            ground_truth = torch.Tensor(self.val_results['target']) # (N, C)

            loss = self.criterion(prediction, torch.argmax(ground_truth, dim=1).float())
            acc = Accuracy(task="multiclass", num_classes=len(self.classes))(torch.argmax(prediction, dim=1), torch.argmax(ground_truth, dim=1))
            ce = cross_entropy(prediction, torch.argmax(ground_truth, dim=1).float())

            self.log(f"val/accuracy", acc, on_epoch=True, logger=True)
            self.log(f"val/cross_entropy", ce, on_epoch=True, logger=True)
            self.log(f"loss_for_monitor", loss, on_epoch=True, logger=True)

            # 각 class별 Accuracy와 Cross_Entropy
            gt_cls = ground_truth.argmax(dim=1)  # (N,)
            pred_cls = prediction.argmax(dim=1)  # (N,)
            for c in range(len(self.classes)):
                mask = (gt_cls == c)
                if mask.sum() == 0:
                    continue

                # Accuracy
                acc_c = (pred_cls[mask] == c).float().mean()

                # Cross Entropy
                pred_c = prediction[mask]  # (Nc, C)
                gt_c = gt_cls[mask]  # (Nc,)
                ce_c = F.cross_entropy(pred_c, gt_c)

                self.log(f"train/acc_class_{c}", acc_c, on_epoch=True, logger=True)
                self.log(f"train/ce_class_{c}", ce_c, on_epoch=True, logger=True)

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
                              batch['sequence_id'].detach(), batch['subject_id'], batch['sequence_label'].detach())


        elif self._task == "classification":
            # Calculate loss
            loss = self.criterion(prediction, torch.argmax(batch['label'], dim=1).float())
            pred_class = torch.argmax(prediction, dim=1)

            self.test_metrics["acc"](pred_class, batch['label'])
            self.test_metrics["ce"](loss)  # ⚠ logits 아님, loss 넣기

            self.log("test/loss", loss, on_step=False, on_epoch=True)
            self.log("test/acc", self.train_metrics["acc"], on_epoch=True)
            self.log("test/ce", self.train_metrics["ce"], on_epoch=True)

            self._save_result(self.val_results, prediction.detach(), batch['label'].detach(),
                              batch['sequence_id'].detach(), batch['subject_id'], batch['sequence_label'].detach())



    def on_test_epoch_end(self):
        metrics = self.test_metrics.compute()
        self.log_dict(metrics)

        if self._task == "regression":
            prediction = torch.Tensor(self.test_results['prediction']).reshape(-1, 3) # (B, T, 3) -> (B*T, 3)
            ground_truth = torch.Tensor(self.test_results['target']).reshape(-1, 3) # (B, T, 3) -> (B*T, 3)

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
                              label=grouped_data[sid]['sequence_label'][0],
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
                    f.write(f"[{bodypart_label[sid]}] R2 SCORE: {r2_for_label:.6f} RMSE: {rmse_for_label:.6f} MAE: {mae_for_label:.6f}\n")
                    f.write(f"(R2 SCORE) X: {r2_x:.6f}, Y: {r2_y:.6f}, Z: {r2_z:.6f}\n")
                    f.write(f"(RMSE) X: {rmse_x:.6f}, Y: {rmse_y:.6f}, Z: {rmse_z:.6f}\n")
                    f.write(f"(MAE) X: {mae_x:.6f}, Y: {mae_y:.6f}, Z: {mae_z:.6f}\n")

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
            sequence_label = [results["sequence_label"][i] for i in idx]

            # (num_windows, T, 3) 로 stack
            grouped[sid] = {
                "subject_id": np.stack(subject_id),
                "sequence_label": np.stack(sequence_label),
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
            sequence_id = [results["target"][i] for i in idx]
            sequence_label = [results["sequence_label"][i] for i in idx]

            # (num_windows, T, 3) 로 stack
            grouped[sid] = {
                "sequence_id": np.stack(sequence_id),
                "sequence_label": np.stack(sequence_label),
                "prediction": np.stack(preds),
                "target": np.stack(targets),
            }

        return grouped, unique_seq


    def group_by_label(self, results):

        seq_ids = np.array(results["sequence_label"])
        unique_seq = np.unique(seq_ids)

        grouped = {}

        for sid in unique_seq:
            idx = np.where(seq_ids == sid)[0]

            preds = [results["prediction"][i] for i in idx]
            targets = [results["target"][i] for i in idx]
            sequence_id = [results["target"][i] for i in idx]
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

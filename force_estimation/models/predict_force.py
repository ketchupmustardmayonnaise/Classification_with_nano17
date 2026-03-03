import json

import torch
import matplotlib.pyplot as plt
from torch.nn.functional import cross_entropy
from wandb.integration.sklearn.calculate import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, precision_recall_fscore_support

from force_estimation.utils.plot_sequence import *
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
from force_estimation.utils.bodypart_label import bodypart_label


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

                 # === Hyperparameter ===
                 learning_rate : float = 1e-4,
                 weight_decay : float = 1e-3,

                 directory = None
                 ):

        super().__init__()

        self.save_hyperparameters(ignore=['vision_feature_extractor', 'temporal_feature_extractor'])

        self.vf_net = vision_feature_extractor
        self.tf_net = temporal_feature_extractor

        self.regression_head = nn.Linear(self.vf_net.size_of_output, dim_of_regression)
        self.classification_head = None

        self._task = task
        self.learning_rate = learning_rate

        self.directory = directory
        self.log_file = f'{self.directory}/train.txt'

        if classes is not None:
            self.classes = classes
        if self._task == "classification":
            if self.hparams.classification_type == "MTM":
                self.classification_head = nn.Linear(self.vf_net.size_of_output, len(classes))
            elif self.hparams.classification_type == "OTO":
                self.classification_head = nn.Linear(
                    self.vf_net.size_of_output * refer_recent_frames_for_classification, len(classes))


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
            x = x[:, -5:] # 마지막 5 프레임 timestep (batch_size, 5, channels, height, width)
            x = self.vf_net(x) # (batch_size, 5, dim_hidden_features=96)

            if self.hparams.classification_type == "MTM":
                x = self.classification_head(x) # (batch_size, 5, 8)
            elif self.hparams.classification_type == "OTO":
                x = torch.flatten(x, start_dim=1) # (batch_size, dim_hidden_features * 5)
                x = self.classification_head(x)  # (batch_size, 8)

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
        storage["sequence_id"].extend(sequence_index)
        storage["subject_id"].extend(subject_id)
        storage["bodypart"].extend(bodypart)


    # task 변경
    def set_task(self, value):

        if value == "classification":
            self.classification_head = nn.Linear(self.tf_net.size_of_output * self.hparams.refer_recent_frames_for_classification, len(self.classes))

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
            preds_flat = prediction.detach().reshape((-1, 3))  # (B*T,3)
            targets_flat = batch['force'].detach().reshape((-1, 3))  # (B*T,3)

            self.train_metrics.update(preds_flat, targets_flat)
            self._save_result(self.train_results, prediction.detach(), batch['force'].detach(),
                              batch['sequence_id'].detach(), batch['subject_id'], batch['bodypart'])


        elif self._task == "classification":
            # Calculate loss
            ground_truth = batch['label'] # (N, 5)

            if self.hparams.classification_type == "OTO": # ONE-TO-ONE
                ground_truth = torch.mode(ground_truth, dim=1).values.to(torch.long) # (N,)
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
            ground_truth = batch['label']  # (N, 5)

            if self.hparams.classification_type == "OTO":  # ONE-TO-ONE
                ground_truth = torch.mode(ground_truth, dim=1).values.to(torch.long)  # (N,)
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
            ground_truth = batch['label'] # (N,)

            if self.hparams.classification_type == "OTO":  # ONE-TO-ONE
                ground_truth = torch.mode(ground_truth, dim=1).values.to(torch.long)  # (N,)
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
                    f.write(f"[{bodypart_label[sid]}] R2 SCORE: {r2_for_label:.6f} RMSE: {rmse_for_label:.6f} MAE: {mae_for_label:.6f}\n")
                    f.write(f"(R2 SCORE) X: {r2_x:.6f}, Y: {r2_y:.6f}, Z: {r2_z:.6f}\n")
                    f.write(f"(RMSE) X: {rmse_x:.6f}, Y: {rmse_y:.6f}, Z: {rmse_z:.6f}\n")
                    f.write(f"(MAE) X: {mae_x:.6f}, Y: {mae_y:.6f}, Z: {mae_z:.6f}\n")

        elif self._task == "classification":
            # ---- collect ----
            logits = torch.tensor(self.test_results["prediction"])  # (N, C)
            gt = torch.tensor(self.test_results["target"]).long()  # (N,)
            pred = logits.argmax(dim=1)  # (N,)
            num_classes = len(self.classes)

            # ===============================
            # Precision / Recall / F1
            # ===============================
            gt_np = gt.cpu().numpy()
            pred_np = pred.cpu().numpy()

            precision, recall, f1, support = precision_recall_fscore_support(
                gt_np,
                pred_np,
                labels=np.arange(num_classes),
                average=None,  # per-class
                zero_division=0
            )

            # macro matrix
            precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
                gt_np, pred_np, average="macro", zero_division=0
            )

            # weighted matrix
            precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
                gt_np, pred_np, average="weighted", zero_division=0
            )

            # ---- log overall ----
            self.log("test/precision_macro", precision_macro, on_epoch=True)
            self.log("test/recall_macro", recall_macro, on_epoch=True)
            self.log("test/f1_macro", f1_macro, on_epoch=True)

            self.log("test/precision_weighted", precision_weighted, on_epoch=True)
            self.log("test/recall_weighted", recall_weighted, on_epoch=True)
            self.log("test/f1_weighted", f1_weighted, on_epoch=True)
            # =========================================

            # ---- overall metrics ----
            acc = Accuracy(task="multiclass", num_classes=num_classes)(logits, gt)
            ce = F.cross_entropy(logits, gt)

            self.log("test/accuracy", acc, on_epoch=True, logger=True)
            self.log("test/cross_entropy", ce, on_epoch=True, logger=True)

            # ---- confusion matrix ----
            confmat = torch.zeros((num_classes, num_classes), dtype=torch.long)
            for t, p in zip(gt, pred):
                confmat[t, p] += 1

            # per-class acc / ce
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

                self.log(f"test/acc_class_{c}", acc_c, on_epoch=True, logger=True)
                self.log(f"test/ce_class_{c}", ce_c, on_epoch=True, logger=True)

                self.log(f"test/precision_class_{c}", precision[c], on_epoch=True)
                self.log(f"test/recall_class_{c}", recall[c], on_epoch=True)
                self.log(f"test/f1_class_{c}", f1[c], on_epoch=True)

                # label name (classes가 dict일 수도 있어서 안전하게)
                try:
                    class_name = list(self.classes.keys())[c]  # classes={name:idx} 형태 가정
                except Exception:
                    class_name = str(c)

                per_class_rows.append({
                    "class_index": c,
                    "class_name": bodypart_label[c],
                    "n": n_c,
                    "accuracy": acc_c,
                    "cross_entropy": ce_c
                })

            # ---- write report file ----
            os.makedirs(self.directory, exist_ok=True)
            report_path = f"{self.directory}/test_result.txt"

            with open(report_path, "a", encoding="utf-8") as f:
                f.write("\n\n========= TEST REPORT (CLASSIFICATION) =========\n")
                f.write(f"Overall Accuracy: {acc.item():.6f}\n")
                f.write(f"Overall CrossEntropy: {ce.item():.6f}\n\n")

                f.write("[Per-class]\n")
                for r in per_class_rows:
                    f.write(
                        f"- class {r['class_index']} ({r['class_name']}): "
                        f"n={r['n']}, acc={r['accuracy']:.6f}, ce={r['cross_entropy']:.6f}\n"
                    )

                f.write("\n[Confusion Matrix] (rows=true, cols=pred)\n")
                # 보기 좋게 matrix 출력
                for i in range(num_classes):
                    row = " ".join([f"{int(x):5d}" for x in confmat[i].tolist()])
                    f.write(row + "\n")

                f.write("\n[Overall]\n")
                f.write(f"Precision (macro): {precision_macro:.6f}\n")
                f.write(f"Recall (macro): {recall_macro:.6f}\n")
                f.write(f"F1 (macro): {f1_macro:.6f}\n\n")

                f.write(f"Precision (weighted): {precision_weighted:.6f}\n")
                f.write(f"Recall (weighted): {recall_weighted:.6f}\n")
                f.write(f"F1 (weighted): {f1_weighted:.6f}\n")

                f.write("\n[Per-class Recall / Precision / F1]\n")
                for c in range(num_classes):
                    f.write(
                        f"class {bodypart_label[c]}: "
                        f"P={precision[c]:.6f}, "
                        f"R={recall[c]:.6f}, "
                        f"F1={f1[c]:.6f}, "
                        f"n={support[c]}\n"
                    )

            # ---- also save a CSV summary (optional but useful) ----
            df = pd.DataFrame(per_class_rows)
            df.to_csv(f"{self.directory}/test_classification_summary.csv", index=False)

            # ---- optional: save raw predictions for later ----
            raw_df = pd.DataFrame({
                "gt": gt.cpu().numpy(),
                "pred": pred.cpu().numpy(),
                "sequence_id": np.array(self.test_results["sequence_id"]),
                "subject_id": np.array(self.test_results["subject_id"]),
                "bodypart": np.array(self.test_results["bodypart"]),
            })
            raw_df.to_csv(f"{self.directory}/test_predictions.csv", index=False)

            # numpy 변환
            gt_np = gt.cpu().numpy()
            pred_np = pred.cpu().numpy()

            # ----------------------------
            # confusion matrix (normalized)
            # ----------------------------
            cm = confusion_matrix(
                gt_np,
                pred_np,
                labels=np.arange(num_classes),
                normalize='true' # ⭐ row normalize (가장 일반적)
            )

            # ----------------------------
            # plot
            # ----------------------------
            try:
                class_names = list(self.classes.values())
            except Exception:
                class_names = [str(i) for i in range(num_classes)]

            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

            fig, ax = plt.subplots(figsize=(8, 6))
            disp.plot(
                ax=ax,
                cmap="Blues",
                values_format=".2f",
                colorbar=True
            )

            ax.set_title("Normalized Confusion Matrix")
            plt.xticks(rotation=45)

            cm_path = f"{self.directory}/test_confusion_matrix.png"
            plt.tight_layout()
            plt.savefig(cm_path, dpi=300)
            plt.close(fig)

            print(f"Confusion matrix saved to {cm_path}")

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
            sequence_id = [results["target"][i] for i in idx]
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

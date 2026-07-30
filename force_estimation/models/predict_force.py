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


class PredictionForce(pl.LightningModule):

    def __init__(self,
                 # === 모델 구조 관련 파라미터 ===
                 vision_feature_extractor,
                 regression_feature_extractor=None,
                 classification_feature_extractor=None,

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

        self.save_hyperparameters(ignore=['vision_feature_extractor', 'regression_feature_extractor', 'classification_feature_extractor'])

        self._task = task
        self.learning_rate = learning_rate

        if classes is not None:
            self.classes = classes

        # 모델 정의
        # shared backbone
        self.vf_net = vision_feature_extractor

        # optional task branches
        self.rf_net = regression_feature_extractor
        self.cf_net = classification_feature_extractor

        # optional heads
        self.regression_head = None
        self.classification_head = None

        if self.rf_net is not None:
            self.regression_head = nn.Linear(
                self.vf_net.size_of_output,
                dim_of_regression
            )

        if self.cf_net is not None and self.classes is not None:
            self.classification_head = nn.Linear(
                self.cf_net.size_of_output,
                len(self.classes)
            )

        # directory 또는 log file
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
            self.train_metrics_fx = metrics.clone(prefix="train/", postfix="_fx")
            self.train_metrics_fy = metrics.clone(prefix="train/", postfix="_fy")
            self.train_metrics_fz = metrics.clone(prefix="train/", postfix="_fz")

            self.val_metrics = metrics.clone(prefix="val/")
            self.val_metrics_fx = metrics.clone(prefix="val/", postfix="_fx")
            self.val_metrics_fy = metrics.clone(prefix="val/", postfix="_fy")
            self.val_metrics_fz = metrics.clone(prefix="val/", postfix="_fz")

            # ================= Results =================
            self.test_metrics = metrics.clone(prefix="test/")
            self.test_results = self._init_result_dict()

        elif self._task == "classification":
            self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')

            metrics = MetricCollection({
                "accuracy": Accuracy(task="multiclass", num_classes=len(classes)),
                "cross_entropy": CrossEntropyMetrics() # cross entropy 평균용
            })

            self.train_metrics = metrics.clone(prefix="train/")
            self.val_metrics = metrics.clone(prefix="val/")
            self.test_metrics = metrics.clone(prefix="test/")

            self.train_results = self._init_result_dict()
            self.val_results = self._init_result_dict()
            self.test_results = self._init_result_dict()

        elif self._task == "multitask":
            if classes is None:
                raise Exception("Multitask task requires classes.")
            self.criterion_reg = torch.nn.MSELoss(reduction='mean')
            self.criterion_cls = torch.nn.CrossEntropyLoss(reduction='mean')

            reg_metrics = MetricCollection({
                "rmse": MeanSquaredError(squared=False),
                "mae": MeanAbsoluteError(),
                "r2": R2Score()
            })
            cls_metrics = MetricCollection({
                "accuracy": Accuracy(task="multiclass", num_classes=len(classes)),
                "cross_entropy": CrossEntropyMetrics()
            })

            self.train_metrics_reg = reg_metrics.clone(prefix="train/reg_")
            self.train_metrics_cls = cls_metrics.clone(prefix="train/cls_")

            self.val_metrics_reg = reg_metrics.clone(prefix="val/reg_")
            self.val_metrics_cls = cls_metrics.clone(prefix="val/cls_")

            self.test_metrics_reg = reg_metrics.clone(prefix="test/reg_")
            self.test_metrics_cls = cls_metrics.clone(prefix="test/cls_")

            self.test_results_reg = self._init_result_dict()
            self.test_results_cls = self._init_result_dict()

        else:
            raise Exception("정확한 Task 기입")



    def forward(self, x):
        batch_size, timestep, channels, height, width = x.shape

        x = self.vf_net(x) # (batch_size, timestep, dim_hidden_features=96)

        if self._task == "regression":
            if self.rf_net is None or self.regression_head is None:
                raise RuntimeError("Regression branch is not attached.")
            x = self.rf_net(x)
            x = self.regression_head(x)
            return x

        elif self._task == "classification":
            if self.cf_net is None or self.classification_head is None:
                raise RuntimeError("Classification branch is not attached.")
            x = self.cf_net(x) # (batch_size, dim_hidden_features=96) - 마지막 timestep의 feature
            x = self.classification_head(x)

            return x

        elif self._task == "multitask":
            if self.rf_net is None or self.regression_head is None:
                raise RuntimeError("Regression branch is not attached.")
            if self.cf_net is None or self.classification_head is None:
                raise RuntimeError("Classification branch is not attached.")
            reg_out = self.rf_net(x)
            reg_out = self.regression_head(reg_out)   # (B, T, dim_of_regression)
            cls_out = self.cf_net(x)
            cls_out = self.classification_head(cls_out)  # (B, num_classes)
            return {"regression": reg_out, "classification": cls_out}

        else:
            raise ValueError(f"Unknown task: {self._task}")


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
        storage["subject_id"].extend(subject_id)
        storage["bodypart"].extend(bodypart)


    # task 변경
    def set_task(self, value, network):

        if value == "classification":
            if self.classes is not None:
                self.cf_net = network
                self.classification_head = nn.Linear(self.vf_net.size_of_output, len(self.classes))
            else: raise Exception("classes가 확인되지 않습니다. Classification을 진행할 수 없습니다.")

            self._task = value
            metrics = MetricCollection({
                "accuracy": Accuracy(task="multiclass", num_classes=len(self.classes)),
                "cross_entropy": CrossEntropyMetrics()  # cross entropy 평균용
            })

            self.train_metrics = metrics.clone(prefix="train/")
            self.val_metrics = metrics.clone(prefix="val/")
            self.test_metrics = metrics.clone(prefix="test/")

            self.configure_optimizers()

        elif value == "regression":
            self.rf_net = network
            self.regression_head = nn.Linear(self.vf_net.size_of_output, self.hparams.dim_of_regression)

            self._task = value
            metrics = MetricCollection({
                "rmse": MeanSquaredError(squared=False),
                "mae": MeanAbsoluteError(),
                "r2": R2Score() })

            self.train_metrics = metrics.clone(prefix="train/")
            self.train_metrics_fx = metrics.clone(prefix="train/", postfix="_fx")
            self.train_metrics_fy = metrics.clone(prefix="train/", postfix="_fy")
            self.train_metrics_fz = metrics.clone(prefix="train/", postfix="_fz")

            self.val_metrics = metrics.clone(prefix="val/")
            self.val_metrics_fx = metrics.clone(prefix="val/", postfix="_fx")
            self.val_metrics_fy = metrics.clone(prefix="val/", postfix="_fy")
            self.val_metrics_fz = metrics.clone(prefix="val/", postfix="_fz")

            # ================= Results =================
            self.test_metrics = metrics.clone(prefix="test/")
            self.test_results = self._init_result_dict()

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
                    list(self.rf_net.parameters()) +
                    list(self.regression_head.parameters())
            )

        elif self._task == "classification":
            params = (
                    list(self.cf_net.parameters()) +
                    list(self.classification_head.parameters())
            )

        elif self._task == "finetune":
            params = self.parameters()  # 전체

        elif self._task == "multitask":
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
            self.log("train/loss", loss, batch_size=batch['img'].size(0), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("train/loss_fx", loss_fx, batch_size=batch['img'].size(0), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log("train/loss_fy", loss_fy, batch_size=batch['img'].size(0), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log("train/loss_fz", loss_fz, batch_size=batch['img'].size(0), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

            # (B, T, 3) -> (B*T, 3)
            preds_flat = prediction.detach().reshape((-1, 3))  # (B*T,3)
            targets_flat = batch['force'].detach().reshape((-1, 3))  # (B*T,3)

            self.train_metrics.update(preds_flat, targets_flat)
            self.train_metrics_fx.update(preds_flat[:, 0], targets_flat[:, 0])
            self.train_metrics_fy.update(preds_flat[:, 1], targets_flat[:, 1])
            self.train_metrics_fz.update(preds_flat[:, 2], targets_flat[:, 2])


        elif self._task == "classification":
            # Calculate loss
            ground_truth = batch['label'].to(torch.long) # (N,)
            prediction = prediction  # (N, 8)

            loss = self.criterion(prediction, ground_truth)
            self.train_metrics.update(prediction, ground_truth)

            self.log("train/loss", loss, batch_size=batch['img'].size(0), on_step=False, on_epoch=True, sync_dist=True)
            self._save_result(self.train_results, prediction.detach(), ground_truth.detach(),
                              batch['sequence_id'].detach(), batch['subject_id'], batch['bodypart'])

        elif self._task == "multitask":
            reg_out = prediction["regression"]   # (B, T, 3)
            cls_out = prediction["classification"]  # (B, num_classes)

            ground_truth_reg = batch['force']           # (B, T, 3)
            ground_truth_cls = batch['label'].to(torch.long)  # (B,)

            loss_reg = self.criterion_reg(reg_out, ground_truth_reg)
            loss_cls = self.criterion_cls(cls_out, ground_truth_cls)
            loss = loss_reg + loss_cls

            preds_flat = reg_out.detach().reshape((-1, 3))
            targets_flat = ground_truth_reg.detach().reshape((-1, 3))
            self.train_metrics_reg.update(preds_flat, targets_flat)
            self.train_metrics_cls.update(cls_out.detach(), ground_truth_cls)

            self.log("train/loss", loss, batch_size=batch['img'].size(0), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("train/loss_reg", loss_reg, batch_size=batch['img'].size(0), on_step=False, on_epoch=True, sync_dist=True)
            self.log("train/loss_cls", loss_cls, batch_size=batch['img'].size(0), on_step=False, on_epoch=True, sync_dist=True)

        return {"loss": loss}



    def on_train_epoch_end(self):
        if self._task == "multitask":
            reg_m = self.train_metrics_reg.compute()
            cls_m = self.train_metrics_cls.compute()
            self.log_dict(reg_m)
            self.log_dict(cls_m)
            self.train_metrics_reg.reset()
            self.train_metrics_cls.reset()
            return

        metrics = self.train_metrics.compute()
        self.log_dict(metrics)

        if self._task == "regression":
            with open(f'{self.directory}/train.txt', "a") as f:
                f.write(
                    f"train_rmse={metrics['train/rmse']:.6f}, train_mae={metrics['train/mae']:.6f}, train_r2={metrics['train/r2']:.6f}\n")

            self.log_dict(self.train_metrics.compute(), on_epoch=True, logger=True)
            self.log_dict(self.train_metrics_fx.compute(), on_epoch=True, logger=True)
            self.log_dict(self.train_metrics_fy.compute(), on_epoch=True, logger=True)
            self.log_dict(self.train_metrics_fz.compute(), on_epoch=True, logger=True)

            self.train_metrics_fx.reset()
            self.train_metrics_fy.reset()
            self.train_metrics_fz.reset()

        elif self._task == "classification":
            with open(f'{self.directory}/train.txt', "a") as f:
                f.write(
                    f"train_ce={metrics['train/cross_entropy']:.6f}, train_acc={metrics['train/accuracy']:.6f}\n")

            prediction = torch.Tensor(self.train_results['prediction']).to(self.device)  # (N, C)
            ground_truth = torch.Tensor(self.train_results['target']).to(self.device).to(torch.long)  # (N,)

            acc = Accuracy(task="multiclass", num_classes=len(self.classes)).to(self.device)(prediction, ground_truth)
            ce = cross_entropy(prediction, ground_truth)

            self.log(f"train/accuracy", acc, on_epoch=True, logger=True, sync_dist=False)
            self.log(f"train/cross_entropy", ce, on_epoch=True, logger=True, sync_dist=False)

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
            self.log("loss_for_monitor", loss, batch_size=batch['img'].size(0), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log("val/loss", loss, batch_size=batch['img'].size(0), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("val/loss_fx", loss_fx, batch_size=batch['img'].size(0), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log("val/loss_fy", loss_fy, batch_size=batch['img'].size(0), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log("val/loss_fz", loss_fz, batch_size=batch['img'].size(0), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

            # (B, T, 3) -> (B*T, 3)
            preds_flat = prediction.detach().reshape((-1, 3))  # (B*T,3)
            targets_flat = batch['force'].detach().reshape((-1, 3))  # (B*T,3)

            self.val_metrics.update(preds_flat, targets_flat)
            self.val_metrics_fx.update(preds_flat[:, 0], targets_flat[:, 0])
            self.val_metrics_fy.update(preds_flat[:, 1], targets_flat[:, 1])
            self.val_metrics_fz.update(preds_flat[:, 2], targets_flat[:, 2])

        elif self._task == "classification":
            # Calculate loss
            ground_truth = batch['label'].to(torch.long)  # (N,)
            prediction = prediction  # (N, 8)

            loss = self.criterion(prediction, ground_truth)
            self.val_metrics.update(prediction, ground_truth)

            # logging
            self.log("loss_for_monitor", loss, batch_size=batch['img'].size(0), on_step=False, on_epoch=True,
                     prog_bar=False, sync_dist=True)
            self._save_result(self.val_results, prediction.detach(), ground_truth.detach(),
                              batch['sequence_id'].detach(), batch['subject_id'], batch['bodypart'])

        elif self._task == "multitask":
            reg_out = prediction["regression"]   # (B, T, 3)
            cls_out = prediction["classification"]  # (B, num_classes)

            ground_truth_reg = batch['force']
            ground_truth_cls = batch['label'].to(torch.long)

            loss_reg = self.criterion_reg(reg_out, ground_truth_reg)
            loss_cls = self.criterion_cls(cls_out, ground_truth_cls)
            loss = loss_reg + loss_cls

            preds_flat = reg_out.detach().reshape((-1, 3))
            targets_flat = ground_truth_reg.detach().reshape((-1, 3))
            self.val_metrics_reg.update(preds_flat, targets_flat)
            self.val_metrics_cls.update(cls_out.detach(), ground_truth_cls)

            self.log("loss_for_monitor", loss, batch_size=batch['img'].size(0), on_step=False, on_epoch=True,
                     prog_bar=False, sync_dist=True)
            self.log("val/loss_reg", loss_reg, batch_size=batch['img'].size(0), on_step=False, on_epoch=True, sync_dist=True)
            self.log("val/loss_cls", loss_cls, batch_size=batch['img'].size(0), on_step=False, on_epoch=True, sync_dist=True)

        return {"loss": loss}



    def on_validation_epoch_end(self):
        if self._task == "multitask":
            reg_m = self.val_metrics_reg.compute()
            cls_m = self.val_metrics_cls.compute()
            self.log_dict(reg_m)
            self.log_dict(cls_m)
            self.val_metrics_reg.reset()
            self.val_metrics_cls.reset()
            return

        metrics = self.val_metrics.compute()
        self.log_dict(metrics)

        # =================================================
        # Regression
        # =================================================
        if self._task == "regression":
            with open(f'{self.directory}/train.txt', "a") as f:
                f.write(
                    f"[{self.current_epoch} Epoch] val_rmse={metrics['val/rmse']:.6f}, val_mae={metrics['val/mae']:.6f}, val_r2={metrics['val/r2']:.6f} ")

            self.log_dict(self.val_metrics.compute(), on_epoch=True, logger=True)
            self.log_dict(self.val_metrics_fx.compute(), on_epoch=True, logger=True)
            self.log_dict(self.val_metrics_fy.compute(), on_epoch=True, logger=True)
            self.log_dict(self.val_metrics_fz.compute(), on_epoch=True, logger=True)

            self.val_metrics_fx.reset()
            self.val_metrics_fy.reset()
            self.val_metrics_fz.reset()


        # =================================================
        # Classification
        # =================================================
        elif self._task == "classification":
            with open(f'{self.directory}/train.txt', "a") as f:
                f.write(
                    f"[{self.current_epoch} Epoch] val_ce={metrics['val/cross_entropy']:.6f}, val_acc={metrics['val/accuracy']:.6f} ")

            prediction = torch.Tensor(self.val_results['prediction']).to(self.device)  # (N, C)
            ground_truth = torch.Tensor(self.val_results['target']).to(self.device).to(torch.long)  # (N,)

            loss = self.criterion(prediction, ground_truth)
            acc = Accuracy(task="multiclass", num_classes=len(self.classes)).to(self.device)(prediction, ground_truth)
            ce = cross_entropy(prediction, ground_truth)

            self.log(f"val/accuracy", acc, on_epoch=True, logger=True, sync_dist=False)
            self.log(f"val/cross_entropy", ce, on_epoch=True, logger=True, sync_dist=False)

            # 각 class별 Accuracy와 Cross_Entropy
            # sync_dist=False: GPU마다 보유한 클래스가 달라 continue로 건너뛰면
            # 서로 다른 key로 all_reduce를 시도해 deadlock 발생
            gt_class, pred_class = ground_truth, prediction.argmax(dim=1)  # (N,)
            for c in range(len(self.classes)):
                mask = (gt_class == c)
                if mask.sum() == 0: continue

                # Accuracy
                acc_c = (pred_class[mask] == c).float().mean()
                # Cross Entropy
                ce_c = F.cross_entropy(prediction[mask], gt_class[mask]) # (Nc, C), (Nc)

                self.log(f"val/acc_class_{c}", acc_c, on_epoch=True, logger=True, sync_dist=False)
                self.log(f"val/ce_class_{c}", ce_c, on_epoch=True, logger=True, sync_dist=False)

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
            self.log("test/loss", loss, batch_size=batch['img'].size(0), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("test/loss_fx", loss_fx, batch_size=batch['img'].size(0), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log("test/loss_fy", loss_fy, batch_size=batch['img'].size(0), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log("test/loss_fz", loss_fz, batch_size=batch['img'].size(0), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

            # (B, T, 3) -> (B*T, 3)
            preds_flat = prediction.detach().reshape((-1, 3))  # (B*T,3)
            targets_flat = batch['force'].detach().reshape((-1, 3))  # (B*T,3)

            self.test_metrics.update(preds_flat, targets_flat)
            self._save_result(self.test_results, prediction.detach(), batch['force'].detach(),
                              batch['sequence_id'].detach(), batch['subject_id'], batch['bodypart'])


        elif self._task == "classification":
            # Calculate loss
            ground_truth = batch['label'].to(torch.long)  # (N,)
            prediction = prediction  # (N, 8)

            self.test_metrics.update(prediction, ground_truth)

            self.log("test/accuracy", self.test_metrics["accuracy"], on_epoch=True, sync_dist=True)
            self.log("test/cross_entropy", self.test_metrics["cross_entropy"], on_epoch=True, sync_dist=True)

            self._save_result(self.test_results, prediction.detach(), ground_truth.detach(),
                              batch['sequence_id'].detach(), batch['subject_id'], batch['bodypart'])

        elif self._task == "multitask":
            reg_out = prediction["regression"]   # (B, T, 3)
            cls_out = prediction["classification"]  # (B, num_classes)

            ground_truth_reg = batch['force']
            ground_truth_cls = batch['label'].to(torch.long)

            preds_flat = reg_out.detach().reshape((-1, 3))
            targets_flat = ground_truth_reg.detach().reshape((-1, 3))
            self.test_metrics_reg.update(preds_flat, targets_flat)
            self.test_metrics_cls.update(cls_out.detach(), ground_truth_cls)

            self._save_result(self.test_results_reg, reg_out.detach(), ground_truth_reg.detach(),
                              batch['sequence_id'].detach(), batch['subject_id'], batch['bodypart'])
            self._save_result(self.test_results_cls, cls_out.detach(), ground_truth_cls.detach(),
                              batch['sequence_id'].detach(), batch['subject_id'], batch['bodypart'])



    def on_test_epoch_end(self):
        if self._task == "multitask":
            reg_m = self.test_metrics_reg.compute()
            cls_m = self.test_metrics_cls.compute()
            self.log_dict(reg_m)
            self.log_dict(cls_m)
            self.test_metrics_reg.reset()
            self.test_metrics_cls.reset()
            self.test_results_reg = self._init_result_dict()
            self.test_results_cls = self._init_result_dict()
            return

        metrics = self.test_metrics.compute()
        self.log_dict(metrics)

        if self._task == "regression":
            # test 결과를 csv 파일로 저장함
            ground_truth = np.array(self.test_results["target"])  # (num_windows, window_size, 3)
            prediction = np.array(self.test_results["prediction"])  # (num_windows, window_size, 3)

            sequence_ids = np.array(self.test_results["sequence_id"])
            subject_ids = np.array(self.test_results["subject_id"])
            bodyparts = np.array(self.test_results["bodypart"])

            num_windows, window_size, _ = ground_truth.shape

            rows = []

            for w in range(num_windows):
                for f in range(window_size):
                    rows.append({
                        "window_idx": w,
                        "frame_idx": f,

                        "gt_x": ground_truth[w, f, 0],
                        "gt_y": ground_truth[w, f, 1],
                        "gt_z": ground_truth[w, f, 2],

                        "pred_x": prediction[w, f, 0],
                        "pred_y": prediction[w, f, 1],
                        "pred_z": prediction[w, f, 2],

                        "sequence_id": sequence_ids[w],
                        "subject_id": subject_ids[w],
                        "bodypart": bodyparts[w],
                    })

            raw_df = pd.DataFrame(rows)
            raw_df.to_csv(f"{self.directory}/test_predictions_regression.csv", index=False)

            # 축별 R2, RMSE, MAE를 수동으로 계산하고 로깅
            prediction = torch.Tensor(self.test_results['prediction'])[:, -1, :].to(self.device)  # (B, T, 3) -> (B, 3)
            ground_truth = torch.Tensor(self.test_results['target'])[:, -1, :].to(self.device)  # (B, T, 3) -> (B, 3)

            r2_all = R2Score().to(self.device)(prediction, ground_truth)
            r2_fx = R2Score().to(self.device)(prediction[:, 0], ground_truth[:, 0])
            r2_fy = R2Score().to(self.device)(prediction[:, 1], ground_truth[:, 1])
            r2_fz = R2Score().to(self.device)(prediction[:, 2], ground_truth[:, 2])

            rmse_all = MeanSquaredError(squared=False).to(self.device)(prediction, ground_truth)
            rmse_fx = MeanSquaredError(squared=False).to(self.device)(prediction[:, 0], ground_truth[:, 0])
            rmse_fy = MeanSquaredError(squared=False).to(self.device)(prediction[:, 1], ground_truth[:, 1])
            rmse_fz = MeanSquaredError(squared=False).to(self.device)(prediction[:, 2], ground_truth[:, 2])

            mae_all = MeanAbsoluteError().to(self.device)(prediction, ground_truth)
            mae_fx = MeanAbsoluteError().to(self.device)(prediction[:, 0], ground_truth[:, 0])
            mae_fy = MeanAbsoluteError().to(self.device)(prediction[:, 1], ground_truth[:, 1])
            mae_fz = MeanAbsoluteError().to(self.device)(prediction[:, 2], ground_truth[:, 2])

            self.log("test/r2_all", r2_all, on_epoch=True, logger=True, sync_dist=False)
            self.log("test/r2_fx", r2_fx, on_epoch=True, logger=True, sync_dist=False)
            self.log("test/r2_fy", r2_fy, on_epoch=True, logger=True, sync_dist=False)
            self.log("test/r2_fz", r2_fz, on_epoch=True, logger=True, sync_dist=False)

            self.log("test/rmse_all", rmse_all, on_epoch=True, logger=True, sync_dist=False)
            self.log("test/rmse_fx", rmse_fx, on_epoch=True, logger=True, sync_dist=False)
            self.log("test/rmse_fy", rmse_fy, on_epoch=True, logger=True, sync_dist=False)
            self.log("test/rmse_fz", rmse_fz, on_epoch=True, logger=True, sync_dist=False)

            self.log("test/mae_all", mae_all, on_epoch=True, logger=True, sync_dist=False)
            self.log("test/mae_fx", mae_fx, on_epoch=True, logger=True, sync_dist=False)
            self.log("test/mae_fy", mae_fy, on_epoch=True, logger=True, sync_dist=False)
            self.log("test/mae_fz", mae_fz, on_epoch=True, logger=True, sync_dist=False)

            # REPORT
            with open(f'{self.directory}/test_result.txt', "a") as f:
                f.write(f"[R2 SCORE] Overall: {r2_all:.6f} Fx: {r2_fx:.6f} Fy: {r2_fy:.6f} Fz: {r2_fz:.6f}\n")
                f.write(f"[RMSE] Overall: {rmse_all:.6f} Fx: {rmse_fx:.6f} Fy: {rmse_fy:.6f} Fz: {rmse_fz:.6f}\n")
                f.write(f"[MAE] Overall: {mae_all:.6f} Fx: {mae_fx:.6f} Fy: {mae_fy:.6f} Fz: {mae_fz:.6f}\n")


            # 시퀀스 별로 plot
            grouped_data, unique_seq = self.group_by_sequence(self.test_results)
            for sid in unique_seq:
                prediction = grouped_data[sid]['prediction'][:, -1, :]
                ground_truth = grouped_data[sid]['target'][:, -1, :]

                plot_sequence(ground_truth, prediction, f"{self.directory}/test", sid,
                              label=grouped_data[sid]['bodypart'][0],
                              name=grouped_data[sid]['subject_id'][0])


            # 피험자별로 Result Report
            grouped_data, unique_seq = self.group_by_name(self.test_results)
            with open(f'{self.directory}/test_result.txt', "a") as f:
                f.write(f"\n\n ========= REPORT FOR EACH SUBJECT ========= \n\n")
            for sid in unique_seq:
                prediction = torch.Tensor(grouped_data[sid]['prediction'])[:, -1, :].to(self.device) # (B, T, 3) -> (B, 3)
                ground_truth = torch.Tensor(grouped_data[sid]['target'])[:, -1, :].to(self.device) # (B, T, 3) -> (B, 3)

                r2_for_subject = R2Score().to(self.device)(prediction, ground_truth)
                rmse_for_subject = MeanSquaredError(squared=False).to(self.device)(prediction, ground_truth)
                mae_for_subject = MeanAbsoluteError().to(self.device)(prediction, ground_truth)

                r2_x = R2Score().to(self.device)(prediction[:, 0], ground_truth[:, 0])
                r2_y = R2Score().to(self.device)(prediction[:, 1], ground_truth[:, 1])
                r2_z = R2Score().to(self.device)(prediction[:, 2], ground_truth[:, 2])

                rmse_x = MeanSquaredError(squared=False).to(self.device)(prediction[:, 0], ground_truth[:, 0])
                rmse_y = MeanSquaredError(squared=False).to(self.device)(prediction[:, 1], ground_truth[:, 1])
                rmse_z = MeanSquaredError(squared=False).to(self.device)(prediction[:, 2], ground_truth[:, 2])

                mae_x = MeanAbsoluteError().to(self.device)(prediction[:, 0], ground_truth[:, 0])
                mae_y = MeanAbsoluteError().to(self.device)(prediction[:, 1], ground_truth[:, 1])
                mae_z = MeanAbsoluteError().to(self.device)(prediction[:, 2], ground_truth[:, 2])

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
                prediction = torch.Tensor(grouped_data[sid]['prediction'])[:, -1, :].to(self.device)  # (B, T, 3) -> (B, 3)
                ground_truth = torch.Tensor(grouped_data[sid]['target'])[:, -1, :].to(self.device)  # (B, T, 3) -> (B, 3)

                r2_for_label = R2Score().to(self.device)(prediction, ground_truth)
                rmse_for_label = MeanSquaredError(squared=False).to(self.device)(prediction, ground_truth)
                mae_for_label = MeanAbsoluteError().to(self.device)(prediction, ground_truth)

                r2_x = R2Score().to(self.device)(prediction[:, 0], ground_truth[:, 0])
                r2_y = R2Score().to(self.device)(prediction[:, 1], ground_truth[:, 1])
                r2_z = R2Score().to(self.device)(prediction[:, 2], ground_truth[:, 2])

                rmse_x = MeanSquaredError(squared=False).to(self.device)(prediction[:, 0], ground_truth[:, 0])
                rmse_y = MeanSquaredError(squared=False).to(self.device)(prediction[:, 1], ground_truth[:, 1])
                rmse_z = MeanSquaredError(squared=False).to(self.device)(prediction[:, 2], ground_truth[:, 2])

                mae_x = MeanAbsoluteError().to(self.device)(prediction[:, 0], ground_truth[:, 0])
                mae_y = MeanAbsoluteError().to(self.device)(prediction[:, 1], ground_truth[:, 1])
                mae_z = MeanAbsoluteError().to(self.device)(prediction[:, 2], ground_truth[:, 2])

                with open(f'{self.directory}/test_result.txt', "a") as f:
                    f.write(f"[{sid}] R2 SCORE: {r2_for_label:.6f} RMSE: {rmse_for_label:.6f} MAE: {mae_for_label:.6f}\n")
                    f.write(f"(R2 SCORE) X: {r2_x:.6f}, Y: {r2_y:.6f}, Z: {r2_z:.6f}\n")
                    f.write(f"(RMSE) X: {rmse_x:.6f}, Y: {rmse_y:.6f}, Z: {rmse_z:.6f}\n")
                    f.write(f"(MAE) X: {mae_x:.6f}, Y: {mae_y:.6f}, Z: {mae_z:.6f}\n")


        elif self._task == "classification":
            # ==========================================================
            # 1. 결과 텐서 정리
            # ==========================================================
            logits = torch.tensor(self.test_results["prediction"]).to(self.device)  # (N, C)
            gt = torch.tensor(self.test_results["target"]).to(self.device).long()  # (N,)
            pred = logits.argmax(dim=1)  # (N,)
            num_classes = len(self.classes)

            # numpy 변환 (sklearn용)
            gt_np = gt.cpu().numpy()
            pred_np = pred.cpu().numpy()

            # ==========================================================
            # 2. Overall Metrics
            # ==========================================================

            acc = Accuracy(task="multiclass", num_classes=num_classes).to(logits.device)(logits, gt)
            ce = F.cross_entropy(logits, gt)
            precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
                gt_np, pred_np, average="macro", zero_division=0
            )

            precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
                gt_np, pred_np, average="weighted", zero_division=0
            )

            # Lightning log
            self.log("test/accuracy", acc, on_epoch=True, sync_dist=False)
            self.log("test/cross_entropy", ce, on_epoch=True, sync_dist=False)

            self.log("test/precision_macro", precision_macro, on_epoch=True, sync_dist=False)
            self.log("test/precision_weighted", precision_weighted, on_epoch=True, sync_dist=False)

            self.log("test/recall_macro", recall_macro, on_epoch=True, sync_dist=False)
            self.log("test/recall_weighted", recall_weighted, on_epoch=True, sync_dist=False)

            self.log("test/f1_macro", f1_macro, on_epoch=True, sync_dist=False)
            self.log("test/f1_weighted", f1_weighted, on_epoch=True, sync_dist=False)

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

                # Lightning log (sync_dist=False: 데드락 방지 - n_c==0이면 continue 대신 nan으로 처리하므로 safe)
                self.log(f"test/acc_class_{c}", acc_c, on_epoch=True, sync_dist=False)
                self.log(f"test/ce_class_{c}", ce_c, on_epoch=True, sync_dist=False)
                self.log(f"test/f1_class_{c}", f1[c], on_epoch=True, sync_dist=False)

                per_class_rows.append({
                    "class_index": c,
                    "class_name": self.classes[c],
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

                f.write(f"Precision (macro): {precision_macro:.6f}\n")
                f.write(f"Precision (weighted): {precision_weighted:.6f}\n")

                f.write(f"Recall (macro): {recall_macro:.6f}\n")
                f.write(f"Recall (weighted): {recall_weighted:.6f}\n")

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

            raw_df.to_csv(f"{self.directory}/test_predictions_classification.csv", index=False)

            # ==========================================================
            # 7. Subject-wise Report
            # ==========================================================

            subject_ids = np.array(self.test_results["subject_id"])
            unique_subjects = np.unique(subject_ids)

            with open(report_path, "a", encoding="utf-8") as f:
                f.write("\n\n========= REPORT FOR EACH SUBJECT =========\n\n")

            for sid in unique_subjects:
                idx = np.where(subject_ids == sid)[0]

                idx_tensor = torch.from_numpy(idx).to(self.device)
                logits_sub = logits[idx_tensor]  # (Ns, C)
                gt_sub = gt[idx_tensor]  # (Ns,)
                pred_sub = logits_sub.argmax(dim=1)  # (Ns,)

                gt_sub_np = gt_sub.cpu().numpy()
                pred_sub_np = pred_sub.cpu().numpy()

                if len(gt_sub) == 0:
                    continue

                # ---------------------------
                # overall metrics for subject
                # ---------------------------
                acc_sub = Accuracy(task="multiclass", num_classes=num_classes).to(logits_sub.device)(logits_sub, gt_sub)
                ce_sub = F.cross_entropy(logits_sub, gt_sub)

                precision_macro_sub, recall_macro_sub, f1_macro_sub, _ = precision_recall_fscore_support(
                    gt_sub_np, pred_sub_np, average="macro", zero_division=0
                )
                precision_weighted_sub, recall_weighted_sub, f1_weighted_sub, _ = precision_recall_fscore_support(
                    gt_sub_np, pred_sub_np, average="weighted", zero_division=0
                )

                # ---------------------------
                # per-class metrics for subject
                # ---------------------------
                precision_sub, recall_sub, f1_sub, support_sub = precision_recall_fscore_support(
                    gt_sub_np,
                    pred_sub_np,
                    labels=np.arange(num_classes),
                    average=None,
                    zero_division=0
                )

                # ---------------------------
                # lightning log
                # ---------------------------
                # sync_dist=False: GPU마다 다른 subject 집합 → 다른 key로 all_reduce → deadlock
                self.log(f"test/accuracy_subject_{sid}", acc_sub, on_epoch=True, sync_dist=False)
                self.log(f"test/cross_entropy_subject_{sid}", ce_sub, on_epoch=True, sync_dist=False)

                self.log(f"test/precision_macro_subject_{sid}", precision_macro_sub, on_epoch=True, sync_dist=False)
                self.log(f"test/precision_weighted_subject_{sid}", precision_weighted_sub, on_epoch=True, sync_dist=False)

                self.log(f"test/recall_macro_subject_{sid}", recall_macro_sub, on_epoch=True, sync_dist=False)
                self.log(f"test/recall_weighted_subject_{sid}", recall_weighted_sub, on_epoch=True, sync_dist=False)

                self.log(f"test/f1_macro_subject_{sid}", f1_macro_sub, on_epoch=True, sync_dist=False)
                self.log(f"test/f1_weighted_subject_{sid}", f1_weighted_sub, on_epoch=True, sync_dist=False)

                with open(report_path, "a", encoding="utf-8") as f:
                    f.write(f"[Subject {sid}]\n")
                    f.write(f"Accuracy: {acc_sub.item():.6f}\n")
                    f.write(f"CrossEntropy: {ce_sub.item():.6f}\n")

                    f.write(f"Precision(macro): {precision_macro_sub:.6f}\n")
                    f.write(f"Precision(weighted): {precision_weighted_sub:.6f}\n")

                    f.write(f"Recall(macro): {recall_macro_sub:.6f}\n")
                    f.write(f"Recall(weighted): {recall_weighted_sub:.6f}\n")

                    f.write(f"F1(macro): {f1_macro_sub:.6f}\n")
                    f.write(f"F1(weighted): {f1_weighted_sub:.6f}\n\n")

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

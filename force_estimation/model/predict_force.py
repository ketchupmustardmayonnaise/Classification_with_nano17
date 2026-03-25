import json

import torch
import matplotlib.pyplot as plt
from torch.nn.functional import cross_entropy
from wandb.integration.sklearn.calculate import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, precision_recall_fscore_support

# from force_estimation.utils.plot_sequence import *
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
from force_estimation.model.resnet_small2 import resnet18_small


class SiamesePredictionForce(pl.LightningModule):
    """
    CFModel.Net을 pytorch_lightning으로 래핑한 Siamese regression 모델.
    입력: (B, 2, H, W) - siamese 이미지 쌍
    출력: (B, dim_of_regression) - 힘 예측
    """

    def __init__(self,
                 dim_of_regression: int = 3,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-3,
                 directory=None):
        super().__init__()
        self.save_hyperparameters()

        # Siamese shared backbone
        self.feature = resnet18_small(num_classes=128)
        self.feature.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)

        # Regression head (128 * 2 = 256)
        self.movement = nn.Sequential(
            nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, dim_of_regression),
        )

        self.criterion = nn.MSELoss(reduction='mean')
        self.directory = directory

        metrics = MetricCollection({
            "rmse": MeanSquaredError(squared=False),
            "mae": MeanAbsoluteError(),
            "r2": R2Score(),
        })
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

        self.train_results = self._init_result_dict()
        self.val_results = self._init_result_dict()
        self.test_results = self._init_result_dict()

    def forward(self, x):
        x = x.float()
        # x: (B, 2, H, W)
        x1 = self.feature(x[:, 0, :, :][:, np.newaxis, :, :])  # (B, 128)
        x2 = self.feature(x[:, 1, :, :][:, np.newaxis, :, :])  # (B, 128)

        x_out = torch.cat([x1, x2], -1)
        moved = self.movement(x_out)
        return moved  # (B, dim_of_regression)

    # =================== Helpers ===================

    def _init_result_dict(self):
        return {
            "prediction": [],
            "target": [],
            "sequence_id": [],
            "subject_id": [],
            "bodypart": [],
        }

    def _save_result(self, storage, preds, target, sequence_index, subject_id, bodypart):
        storage["prediction"].extend(preds.detach().cpu().tolist())
        storage["target"].extend(target.detach().cpu().tolist())
        storage["sequence_id"].extend(sequence_index.detach().cpu().tolist())
        storage["subject_id"].extend(subject_id)
        storage["bodypart"].extend(bodypart)

    def _log_per_axis(self, prefix, prediction, ground_truth):
        for i, axis in enumerate(['fx', 'fy', 'fz']):
            self.log(f"{prefix}/r2_{axis}", R2Score().to(self.device)(prediction[:, i], ground_truth[:, i]), on_epoch=True, logger=True, sync_dist=True)
            self.log(f"{prefix}/rmse_{axis}", MeanSquaredError(squared=False).to(self.device)(prediction[:, i], ground_truth[:, i]), on_epoch=True, logger=True, sync_dist=True)
            self.log(f"{prefix}/mae_{axis}", MeanAbsoluteError().to(self.device)(prediction[:, i], ground_truth[:, i]), on_epoch=True, logger=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.1, patience=10,
            ),
            'monitor': 'loss_for_monitor',
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    # =================== Train ===================

    def training_step(self, batch, batch_idx):
        prediction = self(batch['img'])   # (B, 3)
        target = batch['force']           # (B, 3)

        loss = self.criterion(prediction, target)
        loss_fx = self.criterion(prediction[:, 0], target[:, 0])
        loss_fy = self.criterion(prediction[:, 1], target[:, 1])
        loss_fz = self.criterion(prediction[:, 2], target[:, 2])

        bs = batch['img'].size(0)
        self.log("train/loss", loss, batch_size=bs, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/loss_fx", loss_fx, batch_size=bs, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/loss_fy", loss_fy, batch_size=bs, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/loss_fz", loss_fz, batch_size=bs, on_step=False, on_epoch=True, sync_dist=True)

        self.train_metrics.update(prediction.detach(), target.detach())
        self._save_result(self.train_results, prediction.detach(), target.detach(),
                          batch['sequence_id'].detach(), batch['subject_id'], batch['bodypart'])
        return {"loss": loss}

    def on_train_epoch_end(self):
        metrics = self.train_metrics.compute()
        self.log_dict(metrics)

        with open(f'{self.directory}/train.txt', "a") as f:
            f.write(f"train_rmse={metrics['train/rmse']:.6f}, train_mae={metrics['train/mae']:.6f}, train_r2={metrics['train/r2']:.6f}\n")

        prediction = torch.Tensor(self.train_results['prediction']).to(self.device)
        ground_truth = torch.Tensor(self.train_results['target']).to(self.device)
        self._log_per_axis("train", prediction, ground_truth)

        self.train_metrics.reset()
        self.train_results = self._init_result_dict()

    # =================== Validation ===================

    def validation_step(self, batch, batch_idx):
        prediction = self(batch['img'])   # (B, 3)
        target = batch['force']           # (B, 3)

        loss = self.criterion(prediction, target)
        loss_fx = self.criterion(prediction[:, 0], target[:, 0])
        loss_fy = self.criterion(prediction[:, 1], target[:, 1])
        loss_fz = self.criterion(prediction[:, 2], target[:, 2])

        bs = batch['img'].size(0)
        self.log("loss_for_monitor", loss, batch_size=bs, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/loss", loss, batch_size=bs, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/loss_fx", loss_fx, batch_size=bs, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/loss_fy", loss_fy, batch_size=bs, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/loss_fz", loss_fz, batch_size=bs, on_step=False, on_epoch=True, sync_dist=True)

        self.val_metrics.update(prediction.detach(), target.detach())
        self._save_result(self.val_results, prediction.detach(), target.detach(),
                          batch['sequence_id'].detach(), batch['subject_id'], batch['bodypart'])
        return {"loss": loss}

    def on_validation_epoch_end(self):
        metrics = self.val_metrics.compute()
        self.log_dict(metrics)

        with open(f'{self.directory}/train.txt', "a") as f:
            f.write(f"[{self.current_epoch} Epoch] val_rmse={metrics['val/rmse']:.6f}, val_mae={metrics['val/mae']:.6f}, val_r2={metrics['val/r2']:.6f} ")

        prediction = torch.Tensor(self.val_results['prediction']).to(self.device)
        ground_truth = torch.Tensor(self.val_results['target']).to(self.device)
        self._log_per_axis("val", prediction, ground_truth)

        self.val_metrics.reset()
        self.val_results = self._init_result_dict()

    # =================== Test ===================

    def test_step(self, batch, batch_idx):
        prediction = self(batch['img'])   # (B, 3)
        target = batch['force']           # (B, 3)

        loss = self.criterion(prediction, target)
        loss_fx = self.criterion(prediction[:, 0], target[:, 0])
        loss_fy = self.criterion(prediction[:, 1], target[:, 1])
        loss_fz = self.criterion(prediction[:, 2], target[:, 2])

        bs = batch['img'].size(0)
        self.log("test/loss", loss, batch_size=bs, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test/loss_fx", loss_fx, batch_size=bs, on_step=False, on_epoch=True, sync_dist=True)
        self.log("test/loss_fy", loss_fy, batch_size=bs, on_step=False, on_epoch=True, sync_dist=True)
        self.log("test/loss_fz", loss_fz, batch_size=bs, on_step=False, on_epoch=True, sync_dist=True)

        self.test_metrics.update(prediction.detach(), target.detach())
        self._save_result(self.test_results, prediction.detach(), target.detach(),
                          batch['sequence_id'].detach(), batch['subject_id'], batch['bodypart'])

    def on_test_epoch_end(self):
        metrics = self.test_metrics.compute()
        self.log_dict(metrics)

        ground_truth_np = np.array(self.test_results["target"])   # (N, 3)
        prediction_np = np.array(self.test_results["prediction"]) # (N, 3)
        sequence_ids = np.array(self.test_results["sequence_id"])
        subject_ids = np.array(self.test_results["subject_id"])
        bodyparts = np.array(self.test_results["bodypart"])

        # --- CSV ---
        rows = [
            {
                "gt_x": ground_truth_np[i, 0], "gt_y": ground_truth_np[i, 1], "gt_z": ground_truth_np[i, 2],
                "pred_x": prediction_np[i, 0], "pred_y": prediction_np[i, 1], "pred_z": prediction_np[i, 2],
                "sequence_id": sequence_ids[i], "subject_id": subject_ids[i], "bodypart": bodyparts[i],
            }
            for i in range(len(ground_truth_np))
        ]
        pd.DataFrame(rows).to_csv(f"{self.directory}/test_predictions_regression.csv", index=False)

        # --- Overall metrics ---
        pred_t = torch.Tensor(prediction_np).to(self.device)
        gt_t = torch.Tensor(ground_truth_np).to(self.device)

        r2_all = R2Score(num_outputs=self.hparams.dim_of_regression, multioutput='uniform_average').to(self.device)(pred_t, gt_t)
        rmse_all = MeanSquaredError(squared=False).to(self.device)(pred_t, gt_t)
        mae_all = MeanAbsoluteError().to(self.device)(pred_t, gt_t)

        self.log("test/r2_all", r2_all, on_epoch=True, logger=True, sync_dist=True)
        self.log("test/rmse_all", rmse_all, on_epoch=True, logger=True, sync_dist=True)
        self.log("test/mae_all", mae_all, on_epoch=True, logger=True, sync_dist=True)
        self._log_per_axis("test", pred_t, gt_t)

        axis_m = {}
        for i, axis in enumerate(['fx', 'fy', 'fz']):
            axis_m[axis] = (
                R2Score().to(self.device)(pred_t[:, i], gt_t[:, i]),
                MeanSquaredError(squared=False).to(self.device)(pred_t[:, i], gt_t[:, i]),
                MeanAbsoluteError().to(self.device)(pred_t[:, i], gt_t[:, i]),
            )

        with open(f'{self.directory}/test_result.txt', "a") as f:
            f.write(f"[R2 SCORE] Overall: {r2_all:.6f} Fx: {axis_m['fx'][0]:.6f} Fy: {axis_m['fy'][0]:.6f} Fz: {axis_m['fz'][0]:.6f}\n")
            f.write(f"[RMSE] Overall: {rmse_all:.6f} Fx: {axis_m['fx'][1]:.6f} Fy: {axis_m['fy'][1]:.6f} Fz: {axis_m['fz'][1]:.6f}\n")
            f.write(f"[MAE] Overall: {mae_all:.6f} Fx: {axis_m['fx'][2]:.6f} Fy: {axis_m['fy'][2]:.6f} Fz: {axis_m['fz'][2]:.6f}\n")

        # --- Sequence plots ---
        grouped_seq, unique_seq = self._group_by(self.test_results, "sequence_id")
        for sid in unique_seq:
            pred_seq = np.array(grouped_seq[sid]['prediction'])
            gt_seq = np.array(grouped_seq[sid]['target'])
            plot_sequence(gt_seq, pred_seq, f"{self.directory}/test", sid,
                          label=grouped_seq[sid]['bodypart'][0],
                          name=grouped_seq[sid]['subject_id'][0])

        # --- Per-subject report ---
        grouped_sub, unique_sub = self._group_by(self.test_results, "subject_id")
        with open(f'{self.directory}/test_result.txt', "a") as f:
            f.write(f"\n\n ========= REPORT FOR EACH SUBJECT ========= \n\n")
        for sid in unique_sub:
            self._write_group_report(sid, grouped_sub[sid])

        # --- Per-bodypart report ---
        grouped_bp, unique_bp = self._group_by(self.test_results, "bodypart")
        with open(f'{self.directory}/test_result.txt', "a") as f:
            f.write(f"\n\n ========= REPORT FOR EACH BODYPART ========= \n\n")
        for sid in unique_bp:
            self._write_group_report(sid, grouped_bp[sid])

        self.test_metrics.reset()
        self.test_results = self._init_result_dict()

    # =================== Grouping ===================

    def _group_by(self, results, key):
        ids = np.array(results[key])
        unique = np.unique(ids)
        grouped = {}
        for sid in unique:
            idx = np.where(ids == sid)[0]
            grouped[sid] = {k: [results[k][i] for i in idx] for k in results}
        return grouped, unique

    def _write_group_report(self, label, group_data):
        pred = torch.Tensor(np.array(group_data['prediction'])).to(self.device)
        gt = torch.Tensor(np.array(group_data['target'])).to(self.device)

        r2 = R2Score(num_outputs=self.hparams.dim_of_regression, multioutput='uniform_average').to(self.device)(pred, gt)
        rmse = MeanSquaredError(squared=False).to(self.device)(pred, gt)
        mae = MeanAbsoluteError().to(self.device)(pred, gt)

        r2_x = R2Score().to(self.device)(pred[:, 0], gt[:, 0])
        r2_y = R2Score().to(self.device)(pred[:, 1], gt[:, 1])
        r2_z = R2Score().to(self.device)(pred[:, 2], gt[:, 2])
        rmse_x = MeanSquaredError(squared=False).to(self.device)(pred[:, 0], gt[:, 0])
        rmse_y = MeanSquaredError(squared=False).to(self.device)(pred[:, 1], gt[:, 1])
        rmse_z = MeanSquaredError(squared=False).to(self.device)(pred[:, 2], gt[:, 2])
        mae_x = MeanAbsoluteError().to(self.device)(pred[:, 0], gt[:, 0])
        mae_y = MeanAbsoluteError().to(self.device)(pred[:, 1], gt[:, 1])
        mae_z = MeanAbsoluteError().to(self.device)(pred[:, 2], gt[:, 2])

        with open(f'{self.directory}/test_result.txt', "a") as f:
            f.write(f"[{label}] R2 SCORE: {r2:.6f} RMSE: {rmse:.6f} MAE: {mae:.6f}\n")
            f.write(f"(R2 SCORE) X: {r2_x:.6f}, Y: {r2_y:.6f}, Z: {r2_z:.6f}\n")
            f.write(f"(RMSE) X: {rmse_x:.6f}, Y: {rmse_y:.6f}, Z: {rmse_z:.6f}\n")
            f.write(f"(MAE) X: {mae_x:.6f}, Y: {mae_y:.6f}, Z: {mae_z:.6f}\n")

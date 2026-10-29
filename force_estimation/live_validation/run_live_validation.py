# -*- coding: utf-8 -*-
"""실기기에서 수집한 npz 폴더로 force estimation(regression) 과 bodypart classification 을 테스트함.

아래 '설정'만 고치고 실행하면 됨. 모델 구조, 학습 당시 window 길이, 전처리에 쓸 fold 데이터
경로는 모두 체크포인트 안에 들어 있어서 따로 적지 않아도 됨.

전처리는 그 fold 가 h5 를 만들 때 쓴 것을 그대로 따라감 (live_validation_dataset.py).
  - 정규화 mean/std    : <fold 데이터>/test.h5 의 attrs
  - idle/bodypart 경계 : <fold 데이터>/otsu_thresholds.json 의 thr_final

테스트는 PredictionForce.test_step / on_test_epoch_end 을 그대로 쓰므로 test_result.txt 와
시퀀스 plot 이 기존 테스트 결과와 같은 형식으로 나옴.
"""

import json
import os
import sys

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

FORCE_ESTIMATION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(FORCE_ESTIMATION_DIR)
for _path in (FORCE_ESTIMATION_DIR, PROJECT_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from models import PredictionForce, VisionFeatureExtractor, Transformer, LSTM
from utils.bodypart_label import bodypart_14_label
from utils.load_model import load_model_from_local
from live_validation.live_validation_dataset import LiveValidationDataset, read_h5_norm_stats


# ================================
# 설정  (경로는 force_estimation 폴더 기준)
# ================================
DATA_DIR = "./data/live_validation/2026-09-02"

CHECKPOINT = ("./saved_models/[FOLD1] 03-12_08-19-52_lightweight_convnexttiny_transformer"
              "/CLASSIFICATION_03-13_11-14-31/best_model_loss_for_monitor=0.2218.ckpt")

OUTPUT_DIR = "./saved_models/live_validation_runs/2026-09-02"

TASKS = ["regression", "classification"]    # 둘 중 하나만 돌리려면 하나만 남기면 됨

SUBDIRS = None      # 사용할 하위 폴더만. 예) ["subject10", "subject11"]
TAKES = None        # 같은 부위를 여러 번 수집했을 때 쓸 회차(1부터). 예) [1]

BATCH_SIZE = 64
# ================================


def resolve(path):
    """설정에 적은 상대 경로를 force_estimation 폴더 기준으로 품."""
    return path if os.path.isabs(path) else os.path.normpath(
        os.path.join(FORCE_ESTIMATION_DIR, path))


def read_checkpoint_config(checkpoint_path):
    """체크포인트에 저장된 학습 당시 설정과 sliding window 길이를 꺼냄."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    cfg = (checkpoint.get("datamodule_hyper_parameters") or {}).get("cfg")
    if cfg is None:
        raise RuntimeError(f"체크포인트에 학습 설정이 없습니다: {checkpoint_path}")

    # rf_net 의 positional encoding 길이 = 학습 당시 window 길이
    pe = checkpoint["state_dict"].get("rf_net.pos_encoder.pe")
    if pe is None:
        raise RuntimeError("체크포인트에 rf_net.pos_encoder.pe 가 없어 window 길이를 알 수 없습니다.")

    return cfg, int(pe.shape[1])


def build_model(cfg, window_length, checkpoint_path, output_dir, task):
    """학습 때와 같은 구조로 만들어서 체크포인트를 불러옴.

    regression 은 vf_net + rf_net, classification 은 vf_net + cf_net 을 씀.
    rf_net 은 classification 에서 쓰이지 않지만 state_dict 의 shape 을 맞추려고 같이 만듦.
    """

    model_cfg = cfg["model"]
    vf_net = VisionFeatureExtractor(model_cfg["encoder_type"], cfg["data"]["image_size"])

    nets = {
        "vision_feature_extractor": vf_net,
        "regression_feature_extractor": Transformer(
            size_of_input=vf_net.size_of_output,
            num_rnn_layer=model_cfg["transformer"]["num_layers"],
            num_of_heads=model_cfg["transformer"]["num_of_head"],
            dropout=model_cfg["dropout"],
            length_of_sequence=window_length,
            dim_feedforward=model_cfg["transformer"]["dim_feedforward"],
            export_multistep=model_cfg["transformer"]["export_multistep"]),
    }

    if task == "classification":
        nets["classification_feature_extractor"] = LSTM(
            size_of_input=vf_net.size_of_output,
            size_of_hidden_layer_features=model_cfg["lstm"]["size_of_hidden_layer_features"],
            num_layer=model_cfg["lstm"]["num_layers"],
            is_bidirectional=model_cfg["lstm"]["is_bidirectional"],
            export_multistep=model_cfg["lstm"]["export_multistep"])

    return load_model_from_local(PredictionForce, checkpoint_path,
                                 dim_of_regression=3,
                                 classes=bodypart_14_label,
                                 task=task,
                                 learning_rate=model_cfg["learning_rate"],
                                 weight_decay=model_cfg["weight_decay"],
                                 directory=output_dir,
                                 **nets)


def main():
    data_dir = resolve(DATA_DIR)
    checkpoint = resolve(CHECKPOINT)
    output_root = resolve(OUTPUT_DIR)

    cfg, window_length = read_checkpoint_config(checkpoint)

    # 전처리 기준이 되는 fold 데이터 폴더도 체크포인트에서 읽음
    fold_data = cfg["data"]["data_dir"]
    fold_data = fold_data if os.path.isabs(fold_data) else os.path.normpath(
        os.path.join(PROJECT_ROOT, fold_data.lstrip("./")))

    print("=" * 90)
    print(f"  데이터       : {data_dir}")
    print(f"  체크포인트   : {checkpoint}")
    print(f"  전처리 기준  : {fold_data}")
    print(f"  window       : {window_length} (체크포인트 기준)")
    print(f"  하위폴더/회차: {SUBDIRS or '전부'} / {TAKES or '전부'}")
    print(f"  결과         : {output_root}")
    print("=" * 90)

    otsu_info = json.loads(
        open(os.path.join(fold_data, "otsu_thresholds.json"), encoding="utf-8").read())
    img_mean, img_std = read_h5_norm_stats(os.path.join(fold_data, "test.h5"))

    dataset = LiveValidationDataset(root=data_dir,
                                    otsu_info=otsu_info,
                                    img_mean=img_mean,
                                    img_std=img_std,
                                    length_of_window=window_length,
                                    subdirs=SUBDIRS,
                                    takes=TAKES)

    for task in TASKS:
        output_dir = os.path.join(output_root, task)
        os.makedirs(output_dir, exist_ok=True)

        # on_test_epoch_end 가 test_result.txt 를 "a"(append)로 열기 때문에 먼저 지움
        report_path = os.path.join(output_dir, "test_result.txt")
        if os.path.exists(report_path):
            os.remove(report_path)

        print(f"\n{'#' * 90}\n#  {task}\n{'#' * 90}\n", flush=True)

        model = build_model(cfg, window_length, checkpoint, output_dir, task)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=torch.cuda.is_available())

        trainer = pl.Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu",
                             devices=1,
                             precision=16 if torch.cuda.is_available() else 32,
                             logger=False,
                             enable_checkpointing=False)
        trainer.test(model, dataloaders=loader)

        print(f"\n[{task}] -> {output_dir}\n")

        del trainer, model, loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

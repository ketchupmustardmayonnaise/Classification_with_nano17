import time
import os
import re
import hydra
import torch
import wandb
import pytorch_lightning as pl
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from models import *
from utils.bodypart_label import *
from utils.load_model import *
from omegaconf import DictConfig

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from data.capacitive_touchscreen_datamodule import CapacitiveImageDataModule

# ===================================
# wandb
# ===================================
from pytorch_lightning.loggers import WandbLogger


def _extract_fold_name(data_dir: str) -> str:
    match = re.search(r"(fold\d+)", os.path.normpath(data_dir))
    if match is None:
        raise ValueError(f"data.data_dir에서 fold 정보를 찾을 수 없습니다: {data_dir}")
    return match.group(1)


def _find_latest_run_dir(regression_fold_dir: str) -> str:
    if not os.path.isdir(regression_fold_dir):
        raise FileNotFoundError(f"Regression fold 디렉터리를 찾을 수 없습니다: {regression_fold_dir}")

    run_dirs = []
    for entry in os.scandir(regression_fold_dir):
        if not entry.is_dir():
            continue
        has_best_ckpt = any(
            filename.startswith("best_model_") and filename.endswith(".ckpt")
            for filename in os.listdir(entry.path)
        )
        if has_best_ckpt:
            run_dirs.append(entry.path)
    if not run_dirs:
        raise FileNotFoundError(f"best checkpoint가 있는 regression run 디렉터리가 없습니다: {regression_fold_dir}")

    return max(run_dirs, key=os.path.getmtime)


def _extract_monitor_score(ckpt_path: str) -> float:
    match = re.search(r"loss_for_monitor=([0-9]*\.?[0-9]+)", os.path.basename(ckpt_path))
    if match is None:
        return float("inf")
    return float(match.group(1))


def _find_best_checkpoint(run_dir: str) -> str:
    ckpt_paths = [
        os.path.join(run_dir, filename)
        for filename in os.listdir(run_dir)
        if filename.startswith("best_model_") and filename.endswith(".ckpt")
    ]
    if not ckpt_paths:
        raise FileNotFoundError(f"best checkpoint를 찾을 수 없습니다: {run_dir}")

    return min(ckpt_paths, key=_extract_monitor_score)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    OmegaConf.register_new_resolver("as_tuple", lambda *args: tuple(args))

    data_dir = to_absolute_path(cfg.data.data_dir)
    fold_name = _extract_fold_name(data_dir)
    regression_fold_dir = to_absolute_path(os.path.join("./saved_models/6fold_regression", fold_name))
    regression_run_dir = _find_latest_run_dir(regression_fold_dir)
    checkpoint_path = _find_best_checkpoint(regression_run_dir)

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(f"사용 장치: {device}")
    print(f"Using regression checkpoint: {checkpoint_path}")

    time_suffix = time.strftime('%m-%d_%H-%M-%S', time.localtime())

    # =========================================
    # callback 함수들
    # =========================================

    wandb_logger = WandbLogger(
        name=f'[CLASSIFICATION] {cfg.model.encoder_type}_{cfg.model.decoder_type}_{time_suffix}',
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        group="CLASSIFICATION",
        log_model= cfg.wandb.log_model
    )

    save_dir = os.path.join(regression_run_dir, f"CLASSIFICATION_{time_suffix}")
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    checkpoint_callback = ModelCheckpoint(
        monitor='loss_for_monitor',
        mode='min',
        save_last=True,
        save_top_k=3,
        dirpath=save_dir,
        filename='best_model_{loss_for_monitor:.4f}',
        verbose=True
    )

    # early stopping callback
    early_stop_callback = EarlyStopping(
        monitor='loss_for_monitor',
        patience=5,
        min_delta=1e-4,  # 0.001 미만 개선은 무시
        verbose=True)

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # 데이터셋
    data_module = CapacitiveImageDataModule(cfg)


    # =========================================
    # 학습 모델 정의
    # =========================================

    vf_net = VisionFeatureExtractor(cfg.model.encoder_type,
                                    cfg.data.image_size)

    tf_net = Transformer(size_of_input=vf_net.size_of_output,
                        num_rnn_layer=cfg.model.transformer.num_layers,
                        num_of_heads=cfg.model.transformer.num_of_head,

                        dropout=cfg.model.dropout,
                        length_of_sequence=cfg.data.length_of_window,

                        dim_feedforward=cfg.model.transformer.dim_feedforward,
                        export_multistep = cfg.model.transformer.export_multistep)

    model = load_model_from_local(
        PredictionForce, checkpoint_path,
        set_eval=False,  # 학습 시작 전 train mode 유지 (RNN backward 에러 방지)
        vision_feature_extractor=vf_net,
        temporal_feature_extractor=tf_net,

        dim_of_regression=3,
        # [기존] classes=bodypart_label,
        classes=bodypart_14_label,
        task="classification",

        # classification 관련 코드
        refer_recent_frames_for_classification=cfg.data.refer_recent_frames_for_classification,
        classification_type=cfg.model.classification_type,
        classification_head_type=cfg.model.classification_head_type,
        classification_head_hidden_size=cfg.model.classification_head_hidden_size,
        classification_head_num_layers=cfg.model.classification_head_num_layers,
        classification_head_dropout=cfg.model.classification_head_dropout,
        classification_tcn_kernel_size=cfg.model.classification_tcn_kernel_size,
        freeze_vf_net_for_classification=cfg.model.freeze_vf_net_for_classification,
        freeze_tf_net_for_classification=cfg.model.freeze_tf_net_for_classification,

        learning_rate=cfg.model.learning_rate,
        weight_decay=cfg.model.weight_decay,
        directory=save_dir,
    )

    trainer = pl.Trainer(
        accelerator='gpu' if use_cuda else 'cpu',
        devices=1,
        precision='16-mixed' if use_cuda else 32,
        logger=wandb_logger,
        max_epochs=200,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
    )

    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module, ckpt_path="best", weights_only=False)
    wandb.finish()


if __name__ == '__main__':
    main()

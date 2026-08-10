import os
import time
import torch
import wandb
import yaml
import pytorch_lightning as pl

from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from force_estimation.models import *
from force_estimation.utils.bodypart_label import bodypart_label
from force_estimation.data.capacitive_touchscreen_datamodule import CapacitiveImageDataModule, resolve_length_of_window


# ================================
# 사용자 설정
# ================================
BASE_CONFIG_PATH = "conf/config.yaml"
SWEEP_CONFIG_PATH = "conf/sweep_config.yaml"

SWEEP_RUN_COUNT = 5   # None이면 무한 실행
# ================================


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_cfg(base_config_path, sweep_params):
    """
    config.yaml을 읽은 뒤 sweep 파라미터만 override
    """

    OmegaConf.register_new_resolver("as_tuple", lambda *args: tuple(args), replace=True)

    cfg = OmegaConf.load(base_config_path)

    if "dropout" in sweep_params:
        cfg.model.dropout = float(sweep_params["dropout"])

    if "learning_rate" in sweep_params:
        cfg.model.learning_rate = float(sweep_params["learning_rate"])

    if "weight_decay" in sweep_params:
        cfg.model.weight_decay = float(sweep_params["weight_decay"])

    if "length_of_window" in sweep_params:
        cfg.data.length_of_window.regression = int(sweep_params["length_of_window"])

    return cfg


def train_one_run():
    """
    wandb agent가 호출하는 학습 함수
    """

    with wandb.init() as run:

        sweep_params = dict(wandb.config)

        cfg = build_cfg(BASE_CONFIG_PATH, sweep_params)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device:", device)

        time_suffix = time.strftime('%m-%d_%H-%M-%S', time.localtime())

        length_of_window = resolve_length_of_window(cfg, task="regression")

        run_name = (
            f"{cfg.model.encoder_type}_{cfg.model.decoder_type}"
            f"_{length_of_window}frames"
            f"_{time_suffix}"
        )

        wandb.run.name = run_name

        wandb_logger = WandbLogger(
            name=run_name,
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            group=cfg.wandb.group,
            log_model=cfg.wandb.log_model,
        )

        wandb.config.update(
            OmegaConf.to_container(cfg, resolve=True),
            allow_val_change=True
        )

        save_dir = f"{cfg.save_directory}/{time_suffix}_{cfg.model.encoder_type}_{cfg.model.decoder_type}"
        os.makedirs(save_dir, exist_ok=True)

        checkpoint_callback = ModelCheckpoint(
            monitor="loss_for_monitor",
            mode="min",
            save_top_k=3,
            dirpath=save_dir,
            filename="best_model_{loss_for_monitor:.4f}",
            verbose=True
        )

        early_stop_callback = EarlyStopping(
            monitor="loss_for_monitor",
            patience=5,
            min_delta=1e-4,
            verbose=True
        )

        lr_monitor = LearningRateMonitor(logging_interval="epoch")

        data_module = CapacitiveImageDataModule(cfg, task="regression")

        vf_net = VisionFeatureExtractor(
            cfg.model.encoder_type,
            cfg.data.image_size
        )

        rf_net = Transformer(
            size_of_input=vf_net.size_of_output,
            num_rnn_layer=cfg.model.transformer.num_layers,
            num_of_heads=cfg.model.transformer.num_of_head,
            dropout=cfg.model.dropout,
            length_of_sequence=length_of_window,
            dim_feedforward=cfg.model.transformer.dim_feedforward,
            export_multistep=cfg.model.transformer.export_multistep
        )

        model = PredictionForce(
            vf_net,
            regression_feature_extractor=rf_net,
            dim_of_regression=3,
            classes=bodypart_label,
            task="regression",
            learning_rate=cfg.model.learning_rate,
            weight_decay=cfg.model.weight_decay,
            directory=save_dir
        )

        trainer = pl.Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            precision=16,
            logger=wandb_logger,
            max_epochs=200,
            callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        )

        trainer.fit(model, datamodule=data_module)
        trainer.test(model, datamodule=data_module, ckpt_path="best")


def main():

    base_cfg = OmegaConf.load(BASE_CONFIG_PATH)
    sweep_cfg = load_yaml(SWEEP_CONFIG_PATH)

    if "project" not in sweep_cfg:
        sweep_cfg["project"] = base_cfg.wandb.project

    if "entity" not in sweep_cfg:
        sweep_cfg["entity"] = base_cfg.wandb.entity

    sweep_id = wandb.sweep(
        sweep=sweep_cfg,
        project=sweep_cfg["project"],
        entity=sweep_cfg.get("entity", None)
    )

    print("Sweep ID:", sweep_id)

    wandb.agent(
        sweep_id,
        function=train_one_run,
        count=SWEEP_RUN_COUNT
    )


if __name__ == "__main__":
    main()
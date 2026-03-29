import os
import sys
import time
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

PROJECT_PARENT = Path(__file__).resolve().parent.parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from force_estimation.data.capacitive_touchscreen_datamodule import CapacitiveImageDataModule
from force_estimation.models import PredictionForce, Transformer, VisionFeatureExtractor
from force_estimation.utils.bodypart_label import bodypart_14_label
from force_estimation.utils.load_model import load_model_from_local


def resolve_path(path_value, base_dir):
    path = Path(path_value)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def normalize_cfg_paths(cfg: DictConfig):
    script_dir = Path(__file__).resolve().parent
    cfg.save_directory = str(resolve_path(cfg.save_directory, script_dir))
    cfg.data.data_dir = str(resolve_path(cfg.data.data_dir, script_dir))
    return cfg


def validate_dataset_dir(dataset_dir):
    dataset_path = Path(dataset_dir)
    required_files = ["train.h5", "val.h5", "test.h5"]
    missing = [name for name in required_files if not (dataset_path / name).exists()]
    if missing:
        missing_text = ", ".join(missing)
        raise FileNotFoundError(f"Missing dataset files in {dataset_path}: {missing_text}")


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    OmegaConf.register_new_resolver("as_tuple", lambda *values: tuple(values), replace=True)
    cfg = normalize_cfg_paths(cfg)
    validate_dataset_dir(cfg.data.data_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 장치: {device}")
    print(f"데이터셋 경로: {cfg.data.data_dir}")

    save_root = Path(cfg.save_directory)
    save_root.mkdir(parents=True, exist_ok=True)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    save_dir_marker = save_root / ".current_save_dir"

    if local_rank == 0:
        time_suffix = time.strftime("%m-%d_%H-%M-%S", time.localtime())
        save_dir = save_root / f"{time_suffix}_{cfg.model.encoder_type}_{cfg.model.decoder_type}"
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir_marker, "w", encoding="utf-8") as marker_file:
            marker_file.write(str(save_dir))
    else:
        while not save_dir_marker.exists():
            time.sleep(0.2)
        with open(save_dir_marker, encoding="utf-8") as marker_file:
            save_dir = Path(marker_file.read().strip())

    time_suffix = time.strftime("_%m-%d_%H-%M-%S", time.localtime())
    wandb_group = f"{cfg.wandb.group}_fingerpad_fold{cfg.data.fold}"

    wandb_logger = WandbLogger(
        name=f"{cfg.model.encoder_type}_{cfg.model.decoder_type}_{time_suffix}",
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        group=wandb_group,
        log_model=cfg.wandb.log_model,
    )

    if local_rank == 0:
        wandb_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    checkpoint_callback = ModelCheckpoint(
        monitor="loss_for_monitor",
        mode="min",
        save_top_k=3,
        dirpath=str(save_dir),
        filename="best_model_{loss_for_monitor:.4f}",
        verbose=True,
    )

    early_stop_callback = EarlyStopping(
        monitor="loss_for_monitor",
        patience=5,
        min_delta=1e-4,
        verbose=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    data_module = CapacitiveImageDataModule(cfg)

    vf_net = VisionFeatureExtractor(cfg.model.encoder_type, cfg.data.image_size)

    rf_net = Transformer(
        size_of_input=vf_net.size_of_output,
        num_rnn_layer=cfg.model.transformer.num_layers,
        num_of_heads=cfg.model.transformer.num_of_head,
        dropout=cfg.model.dropout,
        length_of_sequence=cfg.data.length_of_window,
        dim_feedforward=cfg.model.transformer.dim_feedforward,
        export_multistep=cfg.model.transformer.export_multistep,
    )

    model = PredictionForce(
        vision_feature_extractor=vf_net,
        regression_feature_extractor=rf_net,
        dim_of_regression=3,
        classes=bodypart_14_label,
        task="regression",
        learning_rate=cfg.model.learning_rate,
        weight_decay=cfg.model.weight_decay,
        test_report_last_frame_only=cfg.test.get("report_last_frame_only", False),
        directory=str(save_dir),
    )

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else 32,
        logger=wandb_logger,
        max_epochs=200,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
    )

    trainer.fit(model, datamodule=data_module)

    best_ckpt_path = checkpoint_callback.best_model_path
    if best_ckpt_path:
        test_vf_net = VisionFeatureExtractor(cfg.model.encoder_type, cfg.data.image_size)
        test_rf_net = Transformer(
            size_of_input=test_vf_net.size_of_output,
            num_rnn_layer=cfg.model.transformer.num_layers,
            num_of_heads=cfg.model.transformer.num_of_head,
            dropout=cfg.model.dropout,
            length_of_sequence=cfg.data.length_of_window,
            dim_feedforward=cfg.model.transformer.dim_feedforward,
            export_multistep=cfg.model.transformer.export_multistep,
        )
        test_model = load_model_from_local(
            PredictionForce,
            best_ckpt_path,
            vision_feature_extractor=test_vf_net,
            regression_feature_extractor=test_rf_net,
            dim_of_regression=3,
            classes=bodypart_14_label,
            task="regression",
            learning_rate=cfg.model.learning_rate,
            weight_decay=cfg.model.weight_decay,
            test_report_last_frame_only=cfg.test.get("report_last_frame_only", False),
            directory=str(save_dir),
        )
    else:
        test_model = model

    trainer.test(test_model, datamodule=data_module)
    wandb.finish()


if __name__ == "__main__":
    main()

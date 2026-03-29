import sys
import re
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf

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

    if cfg.test.checkpoint_path is not None:
        cfg.test.checkpoint_path = str(resolve_path(cfg.test.checkpoint_path, script_dir))

    if cfg.test.output_dir is not None:
        cfg.test.output_dir = str(resolve_path(cfg.test.output_dir, script_dir))

    return cfg


def validate_dataset_dir(dataset_dir):
    dataset_path = Path(dataset_dir)
    required_files = ["test.h5"]
    missing = [name for name in required_files if not (dataset_path / name).exists()]
    if missing:
        missing_text = ", ".join(missing)
        raise FileNotFoundError(f"Missing dataset files in {dataset_path}: {missing_text}")


def extract_checkpoint_loss(checkpoint: Path):
    patterns = [
        r"best_model_loss_for_monitor=([0-9]+(?:\.[0-9]+)?)\.ckpt$",
        r"best_model_([0-9]+(?:\.[0-9]+)?)\.ckpt$",
    ]
    for pattern in patterns:
        match = re.search(pattern, checkpoint.name)
        if match:
            return float(match.group(1))
    return None


def resolve_checkpoint_info(cfg: DictConfig):
    checkpoint_path = cfg.test.checkpoint_path
    if checkpoint_path is None:
        raise ValueError(
            "cfg.test.checkpoint_path is not set. Pass a checkpoint file or run directory with Hydra, "
            "for example test.checkpoint_path=/abs/path/to/best_model_xxx.ckpt"
        )

    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"체크포인트 파일을 찾을 수 없습니다: {checkpoint}")

    if checkpoint.is_dir():
        best_candidates = sorted(checkpoint.glob("best_model_*.ckpt"))
        if best_candidates:
            parsed_candidates = []
            for candidate in best_candidates:
                loss = extract_checkpoint_loss(candidate)
                if loss is not None:
                    parsed_candidates.append((loss, candidate))

            if parsed_candidates:
                checkpoint = min(parsed_candidates, key=lambda item: item[0])[1]
            else:
                checkpoint = best_candidates[-1]
        else:
            ckpt_candidates = sorted(checkpoint.glob("*.ckpt"))
            if not ckpt_candidates:
                raise FileNotFoundError(f"체크포인트 디렉토리 안에 .ckpt 파일이 없습니다: {checkpoint}")
            checkpoint = ckpt_candidates[-1]

    output_dir = Path(cfg.test.output_dir) if cfg.test.output_dir is not None else checkpoint.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint, output_dir

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    OmegaConf.register_new_resolver("as_tuple", lambda *args: tuple(args), replace=True)
    cfg = normalize_cfg_paths(cfg)
    validate_dataset_dir(cfg.data.data_dir)
    checkpoint_path, output_dir = resolve_checkpoint_info(cfg)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 장치: {device}")
    print(f"데이터셋 경로: {cfg.data.data_dir}")
    print(f"체크포인트 경로: {checkpoint_path}")

    # 데이터셋
    data_module = CapacitiveImageDataModule(cfg)

    # =========================================
    # 학습 모델 정의
    # =========================================

    vf_net = VisionFeatureExtractor(cfg.model.encoder_type,
                                    cfg.data.image_size)

    rf_net = Transformer(size_of_input=vf_net.size_of_output,
                        num_rnn_layer=cfg.model.transformer.num_layers,
                        num_of_heads=cfg.model.transformer.num_of_head,

                        dropout=cfg.model.dropout,
                        length_of_sequence=cfg.data.length_of_window,

                        dim_feedforward=cfg.model.transformer.dim_feedforward,
                        export_multistep = cfg.model.transformer.export_multistep)

    '''
    cf_net = LSTM(size_of_input=vf_net.size_of_output,
                  size_of_hidden_layer_features=cfg.model.lstm.size_of_hidden_layer_features,
                  num_layer=cfg.model.lstm.num_layers,
                  is_bidirectional=cfg.model.lstm.is_bidirectional,
                  export_multistep=cfg.model.lstm.export_multistep)
    '''

    model = load_model_from_local(
        PredictionForce, str(checkpoint_path),
        vision_feature_extractor=vf_net,
        regression_feature_extractor=rf_net,
        # classification_feature_extractor=cf_net,

        dim_of_regression=3,
        classes=bodypart_14_label,
        task="regression",

        learning_rate=cfg.model.learning_rate,
        weight_decay=cfg.model.weight_decay,
        test_report_last_frame_only=cfg.test.get("report_last_frame_only", False),
        directory=str(output_dir),
    )

    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision='16-mixed' if torch.cuda.is_available() else 32,
        max_epochs=200
    )

    data_module.setup(stage='test')
    # test_mean = data_module.test_dataloader().get_mean_of_test_dataset()
    # PredictionForce.set_buffer("test_mean", test_mean)

    trainer.test(model, datamodule=data_module)


if __name__ == '__main__':
    main()

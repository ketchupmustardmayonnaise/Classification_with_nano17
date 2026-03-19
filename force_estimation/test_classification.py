import os
import time
import hydra
from omegaconf import OmegaConf
from models import *
from utils.bodypart_label import *
from utils.load_model import *
from omegaconf import DictConfig
import torch

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from data.capacitive_touchscreen_datamodule import CapacitiveImageDataModule

PATH = "./saved_models/02-17_03-32-34_lightweight_convnexttiny_transformer/CLASSIFICATION_03-05_09-13-14"
MODEL_NAME = "best_model_loss_for_monitor=0.2375.ckpt"


def _load_checkpoint_hparams(checkpoint_path: str):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"체크포인트를 찾을 수 없습니다: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return checkpoint.get("hyper_parameters", {})

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    OmegaConf.register_new_resolver("as_tuple", lambda *args: tuple(args))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 장치: {device}")

    checkpoint_path = f"{PATH}/{MODEL_NAME}"
    ckpt_hparams = _load_checkpoint_hparams(checkpoint_path)

    # 테스트에서 architecture mismatch를 막기 위해 ckpt 하이퍼파라미터를 우선 사용
    classification_type = ckpt_hparams.get("classification_type", cfg.model.classification_type)
    classification_head_type = ckpt_hparams.get("classification_head_type", cfg.model.classification_head_type)
    classification_head_hidden_size = ckpt_hparams.get("classification_head_hidden_size", cfg.model.classification_head_hidden_size)
    classification_head_num_layers = ckpt_hparams.get("classification_head_num_layers", cfg.model.classification_head_num_layers)
    classification_head_dropout = ckpt_hparams.get("classification_head_dropout", cfg.model.classification_head_dropout)
    classification_tcn_kernel_size = ckpt_hparams.get("classification_tcn_kernel_size", cfg.model.classification_tcn_kernel_size)
    refer_recent_frames = ckpt_hparams.get(
        "refer_recent_frames_for_classification",
        cfg.data.refer_recent_frames_for_classification,
    )

    if isinstance(classes, dict):
        classes = {int(k): v for k, v in classes.items()}

    print(
        "[Checkpoint hparams] "
        f"classification_type={classification_type}, "
        f"classification_head_type={classification_head_type}, "
        f"refer_recent_frames_for_classification={refer_recent_frames}"
    )

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
        strict=True,  # head 키 mismatch를 즉시 에러로 확인
        vision_feature_extractor=vf_net,
        temporal_feature_extractor=tf_net,

        dim_of_regression=3,
        # [기존] classes=bodypart_label,
        classes=bodypart_14_label,
        task="classification",
        refer_recent_frames_for_classification=refer_recent_frames,
        classification_type=classification_type,
        classification_head_type=classification_head_type,
        classification_head_hidden_size=classification_head_hidden_size,
        classification_head_num_layers=classification_head_num_layers,
        classification_head_dropout=classification_head_dropout,
        classification_tcn_kernel_size=classification_tcn_kernel_size,

        learning_rate=cfg.model.learning_rate,
        weight_decay=cfg.model.weight_decay,
        directory=PATH,
    )

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        precision=16,
        max_epochs=200
    )

    data_module.setup(stage='test')
    trainer.test(model, datamodule=data_module)


if __name__ == '__main__':
    main()

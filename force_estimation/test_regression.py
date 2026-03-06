import time
import hydra
from omegaconf import OmegaConf
from models import *
from utils.load_model import *
from utils.bodypart_label import bodypart_label
from omegaconf import DictConfig

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from data.capacitive_touchscreen_datamodule import CapacitiveImageDataModule

# ===================================
# wandb
# ===================================
from pytorch_lightning.loggers import WandbLogger

PATH = "./saved_models/02-17_03-32-34_lightweight_convnexttiny_transformer/03-05_08-15-46_lightweight_convnexttiny_transformer"
MODEL_NAME = "best_model_loss_for_monitor=0.2812.ckpt"

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    OmegaConf.register_new_resolver("as_tuple", lambda *args: tuple(args))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 장치: {device}")

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
        PredictionForce, f"{PATH}//{MODEL_NAME}",
        vision_feature_extractor=vf_net,
        temporal_feature_extractor=tf_net,

        dim_of_regression=3,
        classes=bodypart_label,
        task="regression",

        # [기존] use_force_axis_normalization 인자를 전달하지 않았음.
        use_force_axis_normalization=cfg.model.use_force_axis_normalization,
        # [기존] force_axis_norm_eps 인자를 전달하지 않았음.
        force_axis_norm_eps=cfg.model.force_axis_norm_eps,
        # [기존] data_dir 인자를 전달하지 않았음.
        data_dir=cfg.data.data_dir,

        learning_rate=cfg.model.learning_rate,
        weight_decay=cfg.model.weight_decay,
        # !!!!!!! 이거 수정 !!!!!!!
        directory=PATH,
    )

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        precision=16,
        max_epochs=200
    )

    data_module.setup(stage='test')
    # test_mean = data_module.test_dataloader().get_mean_of_test_dataset()
    # PredictionForce.set_buffer("test_mean", test_mean)

    # 이미 load_model_from_local로 체크포인트를 로드했으므로 ckpt_path 지정이 필요 없음
    trainer.test(model, datamodule=data_module)


if __name__ == '__main__':
    main()

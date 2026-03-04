import time
import hydra
from omegaconf import OmegaConf
from models import *
from utils.bodypart_label import bodypart_label
from utils.load_model import *
from omegaconf import DictConfig

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from data.capacitive_touchscreen_datamodule import CapacitiveImageDataModule

PATH = "./saved_models\\(offline augmentation) 02-17_03-32-34_lightweight_convnexttiny_transformer\\[ONE-TO-ONE] CLASSIFICATION_02-20_20-46-43"
MODEL_NAME = "model.ckpt"

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
        PredictionForce, f"{PATH}/{MODEL_NAME}",
        vision_feature_extractor=vf_net,
        temporal_feature_extractor=tf_net,

        dim_of_regression=3,
        classes=bodypart_label,
        task="classification",
        refer_recent_frames_for_classification=cfg.data.refer_recent_frames_for_classification,
        classification_type=cfg.model.classification_type,

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
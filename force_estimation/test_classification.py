import time
import hydra
from omegaconf import OmegaConf
from models import *
from utils.bodypart_label import *
from utils.load_model import *
from omegaconf import DictConfig

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from data.capacitive_touchscreen_datamodule import CapacitiveImageDataModule, resolve_length_of_window

PATH = "C:/Users/user/Documents/SynologyDrive/Research/[이정민] HandInterface+@/Experiment/Model/Ticwatch Pro 3/24 subject/[Train ALL BODY PART Data]/[FOLD1] 03-12_08-19-52_lightweight_convnexttiny_transformer"
MODEL_NAME = "best_model_loss_for_monitor=0.2877.ckpt"

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    OmegaConf.register_new_resolver("as_tuple", lambda *args: tuple(args))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 장치: {device}")

    # 데이터셋 - classification용 sliding window 길이를 사용함
    data_module = CapacitiveImageDataModule(cfg, task="classification")

    # =========================================
    # 학습 모델 정의
    # =========================================

    vf_net = VisionFeatureExtractor(cfg.model.encoder_type,
                                    cfg.data.image_size)

    # rf_net은 classification forward에서 쓰이지 않지만, 불러올 regression 체크포인트의
    # positional encoding(pe) 버퍼와 shape을 맞춰야 하므로 regression window 길이로 만듦
    rf_net = Transformer(size_of_input=vf_net.size_of_output,
                        num_rnn_layer=cfg.model.transformer.num_layers,
                        num_of_heads=cfg.model.transformer.num_of_head,

                        dropout=cfg.model.dropout,
                        length_of_sequence=resolve_length_of_window(cfg, task="regression"),

                        dim_feedforward=cfg.model.transformer.dim_feedforward,
                        export_multistep = cfg.model.transformer.export_multistep)

    cf_net = LSTM(size_of_input=vf_net.size_of_output,
                  size_of_hidden_layer_features=cfg.model.lstm.size_of_hidden_layer_features,
                  num_layer=cfg.model.lstm.num_layers,
                  is_bidirectional=cfg.model.lstm.is_bidirectional,
                  export_multistep=cfg.model.lstm.export_multistep)

    model = load_model_from_local(
        PredictionForce, f"{PATH}/{MODEL_NAME}",
        vision_feature_extractor=vf_net,
        regression_feature_extractor=rf_net,
        classification_feature_extractor=cf_net,

        dim_of_regression=3,
        classes=bodypart_14_label,
        task="classification",

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
import time
import hydra
from omegaconf import OmegaConf
from models import *
from utils.load_model import *
from utils.bodypart_label import *
from omegaconf import DictConfig

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from data.capacitive_touchscreen_datamodule import CapacitiveImageDataModule, resolve_length_of_window

# ===================================
# wandb
# ===================================
from pytorch_lightning.loggers import WandbLogger

PATH = "./saved_models\\[FOLD5] 03-19_04-57-03_lightweight_convnexttiny_transformer"
MODEL_NAME = "best_model_loss_for_monitor=0.3500.ckpt"

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    OmegaConf.register_new_resolver("as_tuple", lambda *args: tuple(args))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 장치: {device}")

    # 데이터셋
    length_of_window = resolve_length_of_window(cfg, task="regression")
    data_module = CapacitiveImageDataModule(cfg, task="regression")

    # =========================================
    # 학습 모델 정의
    # =========================================

    vf_net = VisionFeatureExtractor(cfg.model.encoder_type,
                                    cfg.data.image_size)

    rf_net = Transformer(size_of_input=vf_net.size_of_output,
                        num_rnn_layer=cfg.model.transformer.num_layers,
                        num_of_heads=cfg.model.transformer.num_of_head,

                        dropout=cfg.model.dropout,
                        length_of_sequence=length_of_window,

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
        PredictionForce, f"{PATH}//{MODEL_NAME}",
        vision_feature_extractor=vf_net,
        regression_feature_extractor=rf_net,
        # classification_feature_extractor=cf_net,

        dim_of_regression=3,
        classes=bodypart_14_label,
        task="regression",

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

    trainer.test(model, datamodule=data_module)


if __name__ == '__main__':
    main()
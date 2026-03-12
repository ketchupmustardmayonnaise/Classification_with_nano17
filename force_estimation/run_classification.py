import time
import hydra
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

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    OmegaConf.register_new_resolver("as_tuple", lambda *args: tuple(args))

    PATH = "./saved_models\\03-12_11-32-22_lightweight_convnexttiny_transformer"
    MODEL_NAME = "best_model_loss_for_monitor=0.3571.ckpt"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 장치: {device}")

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

    save_dir = f"{PATH}/CLASSIFICATION_{time_suffix}"
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

    rf_net = Transformer(size_of_input=vf_net.size_of_output,
                        num_rnn_layer=cfg.model.transformer.num_layers,
                        num_of_heads=cfg.model.transformer.num_of_head,

                        dropout=cfg.model.dropout,
                        length_of_sequence=cfg.data.length_of_window,

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
        directory=save_dir,
    )

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        precision=16,
        logger=wandb_logger,
        max_epochs=200,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
    )

    model.train()
    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module, ckpt_path="best")
    wandb.finish()


if __name__ == '__main__':
    main()
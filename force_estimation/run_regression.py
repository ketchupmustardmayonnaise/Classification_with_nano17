import os
import time
import torch
import hydra
import wandb
import pytorch_lightning as pl

from omegaconf import OmegaConf, DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from force_estimation.models import *
from force_estimation.utils.bodypart_label import *
from force_estimation.utils.resume import find_latest_run_dir, read_wandb_run_id, write_wandb_run_id
from force_estimation.data.capacitive_touchscreen_datamodule import CapacitiveImageDataModule

# 만약 리눅스에서 Background로 돌리고 싶다면
# config 파일에서 data.data_dir: ./data/current_dataset/output/bodypart/fold~ 이런 식으로 바꿔 주고
# 리눅스에서 path를 가장 최상위 폴더 (force_estimation의 상위 폴더)로 돌아긴 흐
# PYTHONPATH=$(pwd) nohup python force_estimation/run_regression.py > output.log 2>&1 &


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    OmegaConf.register_new_resolver("as_tuple", lambda *args: tuple(args))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 장치: {device}")

    # DDP subprocess에서는 LOCAL_RANK 환경변수가 설정됨
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    _save_dir_marker = os.path.join(cfg.save_directory, ".current_save_dir")
    _ckpt_path_marker = os.path.join(cfg.save_directory, ".current_ckpt_path")

    if local_rank == 0:
        resume_dir = None
        if cfg.resume.enabled:
            resume_dir = cfg.resume.run_dir if cfg.resume.run_dir else find_latest_run_dir(cfg.save_directory)
            if resume_dir is None or not os.path.isfile(os.path.join(resume_dir, "last.ckpt")):
                print(f"[resume] last.ckpt를 찾지 못해서 새로 학습을 시작합니다. (resume_dir={resume_dir})")
                resume_dir = None

        if resume_dir is not None:
            save_dir = resume_dir
            ckpt_path = os.path.join(resume_dir, "last.ckpt")
            print(f"[resume] 이어서 학습합니다: {save_dir}")
        else:
            time_suffix = time.strftime('%m-%d_%H-%M-%S', time.localtime())
            save_dir = f"{cfg.save_directory}/{time_suffix}_{cfg.model.encoder_type}_{cfg.model.decoder_type}"
            os.makedirs(save_dir, exist_ok=True)
            ckpt_path = None

        # 다른 rank들이 같은 경로를 쓸 수 있도록 기록
        with open(_save_dir_marker, "w") as f:
            f.write(save_dir)
        with open(_ckpt_path_marker, "w") as f:
            f.write(ckpt_path or "")
    else:
        # rank 0가 디렉토리를 생성할 때까지 대기
        while not (os.path.exists(_save_dir_marker) and os.path.exists(_ckpt_path_marker)):
            time.sleep(0.2)
        with open(_save_dir_marker) as f:
            save_dir = f.read().strip()
        with open(_ckpt_path_marker) as f:
            ckpt_path = f.read().strip() or None

    # 이어서 학습하는 경우, 같은 wandb run에 로그가 계속 쌓이도록 run id를 재사용
    wandb_run_id = read_wandb_run_id(save_dir)

    time_suffix = time.strftime('_%m-%d_%H-%M-%S', time.localtime())  # 로그 이름용

    # W&B logger (WandbLogger는 rank 0에서만 실제 run을 생성함)
    # wandb_run_id가 있으면(=이어서 학습) 같은 run에 이어서 기록하고, 없으면 새 run을 만듦
    wandb_logger = WandbLogger(
        name=f"{cfg.model.encoder_type}_{cfg.model.decoder_type}_{time_suffix}",
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        group=cfg.wandb.group,
        log_model=cfg.wandb.log_model,
        id=wandb_run_id,
        resume="allow" if wandb_run_id else None,
    )

    if local_rank == 0:
        wandb_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

        # run id를 저장해 둬서, 다음에 끊긴 학습을 이어갈 때 같은 wandb run을 재사용함
        write_wandb_run_id(save_dir, wandb_logger.experiment.id)

    # =========================================
    # callback 함수들
    # =========================================

    checkpoint_callback = ModelCheckpoint(
        monitor='loss_for_monitor',
        mode='min',
        save_top_k=3,
        save_last=True,  # 끊긴 학습을 이어갈 수 있도록 매 epoch last.ckpt도 함께 저장
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

    '''
    cf_net = LSTM(size_of_input=vf_net.size_of_output,
                  size_of_hidden_layer_features=cfg.model.lstm.size_of_hidden_layer_features,
                  num_layer=cfg.model.lstm.num_layers,
                  is_bidirectional=cfg.model.lstm.is_bidirectional,
                  export_multistep=cfg.model.lstm.export_multistep)
    '''

    model = PredictionForce(
                vision_feature_extractor=vf_net,
                regression_feature_extractor=rf_net,
                # classification_feature_extractor=cf_net,

                dim_of_regression=3, # axis
                classes=bodypart_14_label,
                task="regression",

                learning_rate=cfg.model.learning_rate,
                weight_decay=cfg.model.weight_decay,

                directory=save_dir
    )

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        # strategy="ddp",         # multi-GPU 전략
        precision=16,
        logger=wandb_logger,
        max_epochs=200,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
    )

    trainer.fit(model, datamodule=data_module, ckpt_path=ckpt_path)
    trainer.test(model, datamodule=data_module, ckpt_path="best")

    wandb.finish()


if __name__ == '__main__':
    main()
import os
import sys
import time
import yaml

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from force_estimation.dataset.rftdataset import RFTDataset
from model.predict_force import SiamesePredictionForce
from utils.util import copy_config


class StopAfterLRReductions(Callback):
    """
    LR이 max_reductions회 감소하면 학습을 종료하는 콜백.
    ReduceLROnPlateau와 함께 사용.
    """
    def __init__(self, max_reductions: int = 3):
        self.max_reductions = max_reductions
        self._prev_lr = None
        self._reduction_count = 0

    def on_train_epoch_end(self, trainer, pl_module):
        current_lr = trainer.optimizers[0].param_groups[0]['lr']
        if self._prev_lr is not None and current_lr < self._prev_lr - 1e-12:
            self._reduction_count += 1
            print(f'\n[StopAfterLRReductions] LR 감소 {self._reduction_count}/{self.max_reductions}: '
                  f'{self._prev_lr:.2e} → {current_lr:.2e}')
            if self._reduction_count >= self.max_reductions:
                print('[StopAfterLRReductions] 최대 LR 감소 횟수 도달. 학습 종료.')
                trainer.should_stop = True
        self._prev_lr = current_lr


class _FlatTestDataset(Dataset):
    """
    RFTDataset(test)는 시퀀스 단위로 반환하므로,
    SiamesePredictionForce의 test_step에 맞게 pair 단위로 펼쳐주는 래퍼.
    시퀀스는 처음 접근할 때 한 번만 로드해서 캐싱함.
    """
    def __init__(self, test_dataset: RFTDataset):
        self._ds = test_dataset

        data_list = test_dataset.data_list
        unique_seqs = sorted(data_list['sequence index'].unique())

        self._flat = []  # (seq_value, pair_idx_within_seq)
        for seq_val in unique_seqs:
            n_pairs = int((data_list['sequence index'] == seq_val).sum())
            for pair_idx in range(n_pairs):
                self._flat.append((int(seq_val), pair_idx))

        self._cache = {}

    def __len__(self):
        return len(self._flat)

    def __getitem__(self, index):
        seq_val, pair_idx = self._flat[index]
        if seq_val not in self._cache:
            self._cache[seq_val] = self._ds[seq_val]
        batch = self._cache[seq_val]
        return {
            'img':         batch['img'][pair_idx],          # (2, H, W)
            'force':       batch['force'][pair_idx],         # (3,)
            'sequence_id': batch['sequence_id'][pair_idx],  # scalar tensor
            'subject_id':  batch['subject_id'],             # str
            'bodypart':    batch['bodypart'],                # str
        }


def main(cfg: dict):
    train_mode    = bool(cfg.get('train_mode', True))
    use_wandb     = bool(cfg.get('use_wandb', False))
    batch_size    = int(cfg.get('batch_size', 64))
    num_epoch     = int(cfg.get('num_epoch', 200))
    learning_rate = float(cfg.get('learning_rate', 1e-3))
    weight_decay  = float(cfg.get('weight_decay', 1e-3))
    scheduler_factor = float(cfg.get('scheduler_factor', 0.1))
    scheduler_patience = int(cfg.get('scheduler_patience', 10))
    max_lr_reductions = int(cfg.get('max_lr_reductions', 3))
    num_workers = cfg.get('num_workers', 0)
    dataset_path  = cfg.get('dataset_path', './dataset/preprocessed_24subject')
    save_dir_base = cfg.get('save_dir', '../saved_models')

    time_suffix = time.strftime('%m-%d_%H-%M-%S', time.localtime())

    # ──────────────────────────────────────────────
    # 학습 모드
    # ──────────────────────────────────────────────
    if train_mode:
        save_dir = os.path.join(save_dir_base, time_suffix)
        os.makedirs(save_dir, exist_ok=True)
        copy_config(save_dir)

        train_dataset = RFTDataset(path=dataset_path, dataset_type='train')
        val_dataset   = RFTDataset(path=dataset_path, dataset_type='val')
        test_dataset  = _FlatTestDataset(RFTDataset(path=dataset_path, dataset_type='test'))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers, persistent_workers=True)
        val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=True)
        test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=True)

        model = SiamesePredictionForce(
            dim_of_regression=3,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            scheduler_factor=scheduler_factor,
            scheduler_patience=scheduler_patience,
            directory=save_dir,
        )

        checkpoint = ModelCheckpoint(
            dirpath=save_dir,
            filename='best_model',
            monitor='loss_for_monitor',
            mode='min',
            save_top_k=1,
        )
        early_stop = StopAfterLRReductions(max_reductions=max_lr_reductions)

        if use_wandb:
            logger = WandbLogger(
                project='HandInterfacesWatch',
                name=f'Siamese_{time_suffix}',
                config={
                    'batch_size': batch_size,
                    'learning_rate': learning_rate,
                    'weight_decay': weight_decay,
                    'scheduler_factor': scheduler_factor,
                    'scheduler_patience': scheduler_patience,
                    'num_epoch': num_epoch,
                    'max_lr_reductions': max_lr_reductions,
                }
            )
        else:
            logger = True  # 기본 CSVLogger

        trainer = pl.Trainer(
            max_epochs=num_epoch,
            callbacks=[checkpoint, early_stop],
            logger=logger,
            log_every_n_steps=1,
        )

        trainer.fit(model, train_loader, val_loader)
        trainer.test(model, test_loader, ckpt_path='best')

    # ──────────────────────────────────────────────
    # 테스트 모드
    # ──────────────────────────────────────────────
    else:
        model_dir = cfg['model_dir']
        save_dir  = os.path.dirname(model_dir)

        test_dataset = _FlatTestDataset(RFTDataset(path=dataset_path, dataset_type='test'))
        test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        model = SiamesePredictionForce.load_from_checkpoint(model_dir, directory=save_dir)

        trainer = pl.Trainer(logger=False)
        trainer.test(model, test_loader)


if __name__ == '__main__':
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    main(config)
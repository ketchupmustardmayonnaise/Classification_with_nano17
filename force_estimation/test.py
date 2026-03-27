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


def main(cfg: dict):
    train_mode    = bool(cfg.get('train_mode', True))
    use_wandb     = bool(cfg.get('use_wandb', False))
    batch_size    = int(cfg.get('batch_size', 64))
    num_epoch     = int(cfg.get('num_epoch', 200))
    learning_rate = float(cfg.get('learning_rate', 1e-3))
    weight_decay  = float(cfg.get('weight_decay', 1e-3))
    max_lr_reductions = int(cfg.get('max_lr_reductions', 3))
    dataset_path  = cfg.get('dataset_path', './dataset/preprocessed_24subject')
    save_dir_base = cfg.get('save_dir', '../saved_models')

    time_suffix = time.strftime('%m-%d_%H-%M-%S', time.localtime())


    # ──────────────────────────────────────────────
    # 테스트 모드
    # ──────────────────────────────────────────────
    model_dir = cfg['model_dir']
    save_dir = os.path.dirname(model_dir)

    test_dataset = _FlatTestDataset(RFTDataset(path=dataset_path, dataset_type='test'))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = SiamesePredictionForce.load_from_checkpoint(model_dir, directory=save_dir)

    trainer = pl.Trainer(logger=False)
    trainer.test(model, test_loader)



if __name__ == '__main__':
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    main(config)
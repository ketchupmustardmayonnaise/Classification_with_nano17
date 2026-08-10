from torchvision import transforms
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from .capacitive_touchscreen_dataset import CapTSDataset


def resolve_length_of_window(cfg, task: str = 'regression') -> int:
    """cfg.data.length_of_window에서 task에 해당하는 sliding window 길이를 꺼냄.

    config가 {regression: .., classification: ..} 형태가 아니라
    예전처럼 정수 하나로 적혀 있으면 그 값을 그대로 사용함.
    """
    length_of_window = cfg.data.length_of_window

    if isinstance(length_of_window, int):
        return length_of_window

    if task not in length_of_window:
        raise KeyError(f"config의 data.length_of_window에 '{task}' 항목이 없습니다. "
                       f"(현재: {list(length_of_window.keys())})")

    return int(length_of_window[task])


class CapacitiveImageDataModule(LightningDataModule):

    def __init__(self,
                 cfg,
                 task: str = 'regression',      # 'regression' | 'classification'
                 transform=transforms.ToTensor()):

        super().__init__()

        self.save_hyperparameters()

        self.length_of_window = resolve_length_of_window(cfg, task)
        print(f"[DataModule] task={task}, length_of_window={self.length_of_window}")


    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = CapTSDataset(length_of_window=self.length_of_window,
                                              num_classes=self.hparams.cfg.data.num_classes,
                                              path=f"{self.hparams.cfg.data.data_dir}/train.h5",
                                              dataset_type='train',

                                              xflip=self.hparams.cfg.data.xflip,
                                              yflip=self.hparams.cfg.data.yflip,
                                              translation=self.hparams.cfg.data.translation,
                                              rotation=self.hparams.cfg.data.rotation)

            self.val_dataset = CapTSDataset(length_of_window=self.length_of_window,
                                            num_classes=self.hparams.cfg.data.num_classes,
                                            path=f"{self.hparams.cfg.data.data_dir}/val.h5",
                                            dataset_type='val')

        if stage == 'test' or stage == 'predict':
            self.test_dataset = CapTSDataset(length_of_window=self.length_of_window,
                                             num_classes=self.hparams.cfg.data.num_classes,
                                             path=f"{self.hparams.cfg.data.data_dir}/test.h5",
                                             dataset_type='test')

    def train_dataloader(self):
        num_workers = self.hparams.cfg.data.get("num_workers", 0)
        return DataLoader(self.train_dataset, batch_size=self.hparams.cfg.data.batch_size,
                          num_workers=self.hparams.cfg.data.num_workers, shuffle=True, pin_memory=True, drop_last=True,
                          persistent_workers=(num_workers > 0))

    def val_dataloader(self):
        num_workers = self.hparams.cfg.data.get("num_workers", 0)
        return DataLoader(self.val_dataset, batch_size=self.hparams.cfg.data.batch_size,
                          num_workers=self.hparams.cfg.data.num_workers, shuffle=False, pin_memory=True, drop_last=True,
                          persistent_workers=(num_workers > 0))

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False, pin_memory=True)
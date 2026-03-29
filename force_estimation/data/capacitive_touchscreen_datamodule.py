from torchvision import transforms
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from .capacitive_touchscreen_dataset import CapTSDataset


class CapacitiveImageDataModule(LightningDataModule):

    def __init__(self,
                 cfg,
                 transform=transforms.ToTensor()):

        super().__init__()

        self.save_hyperparameters()


    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = CapTSDataset(length_of_window=self.hparams.cfg.data.length_of_window,
                                              num_classes=self.hparams.cfg.data.num_classes,
                                              path=f"{self.hparams.cfg.data.data_dir}/train.h5",
                                              dataset_type='train',

                                              xflip=self.hparams.cfg.data.xflip,
                                              yflip=self.hparams.cfg.data.yflip,
                                              translation=self.hparams.cfg.data.translation)

            self.val_dataset = CapTSDataset(length_of_window=self.hparams.cfg.data.length_of_window,
                                            num_classes=self.hparams.cfg.data.num_classes,
                                            path=f"{self.hparams.cfg.data.data_dir}/val.h5",
                                            dataset_type='val')

        if stage == 'test' or stage == 'predict':
            self.test_dataset = CapTSDataset(length_of_window=self.hparams.cfg.data.length_of_window,
                                             num_classes=self.hparams.cfg.data.num_classes,
                                             path=f"{self.hparams.cfg.data.data_dir}/test.h5",
                                             dataset_type='test',
                                             touch_only_last_frame=self.hparams.cfg.data.get("test_touch_only", False),
                                             touch_threshold_log=self.hparams.cfg.data.get("test_touch_log_threshold", None))

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

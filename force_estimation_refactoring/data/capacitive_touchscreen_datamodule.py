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
                                              num_classes=self.num_classes,
                                              path=f"{self.hparams.cfg.data.data_dir}/train_dataset.h5",
                                              dataset_type='train')

            self.val_dataset = CapTSDataset(length_of_window=self.hparams.cfg.data.length_of_window,
                                            num_classes=self.num_classes,
                                            path=f"{self.hparams.cfg.data.data_dir}/val_dataset.h5",
                                            dataset_type='val')

        if stage == 'test' or stage == 'predict':
            self.test_dataset = CapTSDataset(length_of_window=self.hparams.cfg.data.length_of_window,
                                             num_classes=self.num_classes,
                                             path=f"{self.hparams.cfg.data.data_dir}/test_dataset.h5",
                                             dataset_type='test')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.cfg.data.batch_size, num_workers=4, shuffle=True, pin_memory=True, drop_last=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.cfg.data.batch_size, num_workers=4, shuffle=False, pin_memory=True, drop_last=True, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False, pin_memory=True)
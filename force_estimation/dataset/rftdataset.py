import os
import zipfile
import torch
import torch.nn.functional as F
import h5py
import numpy as np
from torch.utils.data import Dataset
import bisect

from dataset.util import target_index


class RFTDataset(Dataset):
    def __init__(self, target_index, stride, path='./preprocessed_train_dataset', dataset_type='train'):
        self.path = path
        self.dataset_type = dataset_type
        self.target_index = target_index
        self.stride = stride

        self.img = []
        self.force = []
        self.label = []

        self.prefix = [0]
        self.frame = []

        with h5py.File(self.path, 'r') as f:
            sample_keys = list(f.keys())
            if 'frame' in sample_keys:
                self.frame = f[f"frame"][:]
                sample_keys.remove('frame')

            for seq in sample_keys:
                self.img.append(f[f"{seq}/img"][:])
                self.force.append(f[f"{seq}/filtered_force"][:])
                self.label.append(f[f"{seq}/label"][:])

        for i in range(len(self.frame)):
            self.frame[i] = int((self.frame[i] - target_index) / stride + 1)
            self.prefix.append(self.frame[i] + self.prefix[-1])
        self.len = sum(self.frame)

    def __getitem__(self, index):
        if self.dataset_type.lower() == 'test':

            img = self.img[index][:]  # (30, 3, 224, 224)
            force = self.force[index][:, 0:3]  # (3)
            label = self.label[index][:]  # (7)

            return (
                torch.from_numpy(img).float(),  # shape: (30, 3, 224, 224)
                torch.from_numpy(force).float(),  # shape: (6)
                torch.from_numpy(label).float()  # shape: (7)
            )
        else:
            sequence_idx = bisect.bisect_right(self.prefix, index) - 1
            frame_idx = index - self.prefix[sequence_idx]

            img = self.img[sequence_idx][frame_idx:frame_idx + self.target_index]
            force = self.force[sequence_idx][frame_idx:frame_idx + self.target_index,0:3]
            label = self.label[sequence_idx]

            return (
                torch.from_numpy(img).float(),  # shape: (30, 3, 224, 224)
                torch.from_numpy(force).float(),  # shape: (6)
                torch.from_numpy(label)  # shape: (7)
            )

    def __len__(self):
        if self.dataset_type.lower() == 'test':
            return len(self.frame)
        else: return self.len
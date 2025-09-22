import os
import zipfile
import torch
import torch.nn.functional as F
import h5py
import numpy as np
from torch.utils.data import Dataset
import bisect
import random
import math
import torchvision.transforms.functional as TF


class RFTDataset(Dataset):
    def __init__(self, path='./preprocessed_train_dataset', dataset_type='train'):
        self.path = path
        self.dataset_type = dataset_type
        self.augmentation = True
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
                self.force.append(f[f"{seq}/force"][:])
                self.label.append(f[f"{seq}/label"][:])

        for i in range(len(self.frame)):
            self.prefix.append(self.frame[i] + self.prefix[-1])

        self.len = sum(self.frame)

    def __getitem__(self, index):
        if self.dataset_type.lower() == 'test':
            img = self.img[index][:]
            force = self.force[index][:, 0:3]
            label = self.label[index][:]

            return (
                torch.from_numpy(img).float(),
                torch.from_numpy(force).float(),
                torch.from_numpy(label).float()
            )
        else:
            sequence_idx = bisect.bisect_right(self.prefix, index) - 1
            frame_idx = index - self.prefix[sequence_idx]

            img = torch.from_numpy(self.img[sequence_idx][frame_idx]).float()
            force = torch.from_numpy(self.force[sequence_idx][frame_idx, 0:3]).float()
            label = torch.from_numpy(self.label[sequence_idx]).float()


            ######## Data Augmentation ########
            if self.dataset_type.lower() == 'train' and self.augmentation == True :

                if random.random() < 0.5:
                    force[0] *= (-1) # x만 flip
                    
                    img = torch.stack([
                        TF.hflip(i)
                        for i in img
                    ])

            return (
                img,
                force,
                label
            )

    def __len__(self):
        if self.dataset_type.lower() == 'test':
            return len(self.frame)
        else: return self.len

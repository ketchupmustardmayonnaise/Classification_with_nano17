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
    def __init__(self, target_index, stride, path='./preprocessed_train_dataset', dataset_type='train', xflip=True):
        self.path = path
        self.dataset_type = dataset_type
        self.target_index = target_index
        self.stride = stride
        self.augmentation = True
        self.img = []
        self.force = []
        self.label = []
        self.xflip = xflip
        
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

            img = torch.from_numpy(self.img[sequence_idx][frame_idx:frame_idx + self.target_index]).float()
            force = torch.from_numpy(self.force[sequence_idx][frame_idx:frame_idx + self.target_index, 0:3]).float()
            label = torch.from_numpy(self.label[sequence_idx]).float()


            ######## Data Augmentation ########
            if self.dataset_type.lower() == 'train' and self.augmentation == True :

                if self.xflip and random.random() < 0.5:
                    force[:, 0] *= (-1)  # x만 flip

                    img = torch.stack([
                        TF.hflip(i)
                        for i in img
                    ])


                if random.random() < 0.5:
                    translate_x = random.randint(-3, 3)
                    translate_y = random.randint(-3, 3)
                    
                    img = torch.stack([
                        TF.affine(i, angle=0, translate=(translate_x, translate_y), scale=1, shear=0) 
                        for i in img
                    ])

                if random.random() < 0.5:
                    angle_deg = random.uniform(-15, 15)
                    angle_rad = math.radians(angle_deg)
                    
                    img = torch.stack([
                        TF.affine(i, angle=angle_deg, translate=(0, 0), scale=1, shear=0)
                        for i in img
                    ])

                    cos_a = math.cos(angle_rad)
                    sin_a = math.sin(angle_rad)
                    
                    rotation_matrix = torch.tensor([[cos_a, -sin_a, 0],
                                                    [sin_a,  cos_a, 0],
                                                    [0,      0,     1]], dtype=force.dtype)
                    
                    force = force @ rotation_matrix.T

            return (
                img,
                force,
                label
            )

    def __len__(self):
        if self.dataset_type.lower() == 'test':
            return len(self.frame)
        else: return self.len

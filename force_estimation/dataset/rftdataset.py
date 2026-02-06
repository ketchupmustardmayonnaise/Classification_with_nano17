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
    def __init__(self, target_index, path='./preprocessed_train_dataset', dataset_type='train', xflip=True):
        self.path = path
        self.dataset_type = dataset_type
        self.target_index = target_index
        self.stride = 1
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
                self.force.append(f[f"{seq}/force"][:])
                self.label.append(f[f"{seq}/label"][:])

        for i in range(len(self.frame)):
            self.frame[i] = int((self.frame[i] - target_index) / self.stride + 1)
            self.prefix.append(self.frame[i] + self.prefix[-1])
        self.len = sum(self.frame)


    def __getitem__(self, index):
        if self.dataset_type.lower() == 'test':
            img = self.img[index][:]
            force = self.force[index][:, 0:3]
            # force[:, 2] = np.maximum(force[:, 2], -4) # -4로 clipping
            label = self.label[index][:]

            return (
                torch.from_numpy(img).float(),
                torch.from_numpy(force).float(),
                torch.from_numpy(label).float()
            )
        else:
            sequence_idx = bisect.bisect_right(self.prefix, index) - 1
            frame_idx = index - self.prefix[sequence_idx]

            img = torch.from_numpy(self.img[sequence_idx][frame_idx:frame_idx + self.target_index])
            force = torch.from_numpy(self.force[sequence_idx][frame_idx:frame_idx + self.target_index, 0:3])

            # label
            window_labels = torch.from_numpy(self.label[sequence_idx][frame_idx:frame_idx + self.target_index])
            label = self.majority_last_onehot_torch(window_labels)


            ######## Data Augmentation ########
            if self.dataset_type.lower() == 'train' and self.augmentation == True :

                if self.xflip and random.random() < 0.5:
                    force[:, 0] *= (-1)  # x만 flip

                    img = torch.stack([
                        TF.hflip(i)
                        for i in img
                    ])

                '''
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
                '''

            return (
                img,
                force,
                label
            )

    def __len__(self):
        if self.dataset_type.lower() == 'test':
            return len(self.frame)
        else: return self.len

    def get_mean_of_test_dataset(self):
        assert self.dataset_type.lower() == 'test'

        force = np.concatenate(self.force, axis=0)[:, 0:3]
        return torch.from_numpy(np.mean(force, axis=0))

    @staticmethod
    def majority_last_onehot_torch(window_labels: torch.Tensor) -> torch.Tensor:
        """
        window_labels: (T, C) one-hot or soft labels
        return: scalar tensor (class index)
        """
        # 1. frame-wise class id (T,)
        frame_labels = window_labels.argmax(dim=1)

        # 2. class counts (C,)
        num_classes = window_labels.size(1)
        counts = torch.bincount(frame_labels, minlength=num_classes)

        # 3. majority classes
        max_count = counts.max()
        majority_mask = counts == max_count  # (C,)

        # 4. from last frame → first, pick first majority
        #    (vectorized: no Python loop)
        reversed_labels = frame_labels.flip(0)                  # (T,)
        is_majority = majority_mask[reversed_labels]            # (T,)
        idx = torch.argmax(is_majority.int())                   # first True
        label = reversed_labels[idx]

        return torch.nn.functional.one_hot(label, num_classes=num_classes).float()


    def scale_image(self, img, scale_factor=(2,2)):
        return F.interpolate(img, scale_factor=scale_factor, mode='bilinear')
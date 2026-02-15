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


class CapTSDataset(Dataset):
    def __init__(self, length_of_window, path='./preprocessed_train_dataset', dataset_type='train', xflip=True):
        self.path = path
        self.dataset_type = dataset_type
        self.length_of_window = length_of_window
        self.stride = 1
        self.augmentation = True
        self.img = []
        self.force = []
        self.label = []
        self.xflip = xflip

        self.prefix = [0]
        self.frame = []
        self.name = []

        with h5py.File(self.path, 'r') as f:
            sample_keys = list(f.keys())
            if 'frame' in sample_keys:
                self.frame = f[f"frame"][:]
                sample_keys.remove('frame')

            if 'name' in sample_keys:
                self.name = f[f"name"][:]
                sample_keys.remove('name')

            if 'label_each_sequence' in sample_keys:
                self.label_each_sequence = f[f"label_each_sequence"][:]
                sample_keys.remove('label_each_sequence')

            for seq in sample_keys:
                self.img.append(f[f"{seq}/img"][:])
                self.force.append(f[f"{seq}/force"][:])
                self.label.append(f[f"{seq}/label"][:])

        for i in range(len(self.frame)):
            self.frame[i] = int((self.frame[i] - length_of_window) / self.stride + 1)
            self.prefix.append(self.frame[i] + self.prefix[-1])
        self.len = sum(self.frame)



    def __getitem__(self, index):
        sequence_idx = bisect.bisect_right(self.prefix, index) - 1
        frame_idx = index - self.prefix[sequence_idx]

        img = torch.from_numpy(self.img[sequence_idx][frame_idx:frame_idx + self.length_of_window]).float()
        force = torch.from_numpy(self.force[sequence_idx][frame_idx:frame_idx + self.length_of_window, 0:3]).float()
        label = torch.from_numpy(self.label[sequence_idx][frame_idx + self.length_of_window - 1])

        ######## Data Augmentation ########
        if self.dataset_type.lower() == 'train' and self.augmentation == True:

            if self.xflip and random.random() < 0.5:
                force[:, 0] *= (-1)  # x만 flip

                img = torch.stack([
                    TF.hflip(i)
                    for i in img
                ])

        return {
            'img': img,
            'force': force,
            'label': label,
            'sequence_label' : self.label_each_sequence[sequence_idx],
            'sequence_id': sequence_idx,
            'subject_id': self.name[sequence_idx]
        }

    def __len__(self):
        return self.len

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
        reversed_labels = frame_labels.flip(0)  # (T,)
        is_majority = majority_mask[reversed_labels]  # (T,)
        idx = torch.argmax(is_majority.int())  # first True
        label = reversed_labels[idx]

        return torch.nn.functional.one_hot(label, num_classes=num_classes).float()

    def scale_image(self, img, scale_factor=(2, 2)):
        return F.interpolate(img, scale_factor=scale_factor, mode='bilinear')

    def return_unique_names(self):
        return np.unique(np.array(self.name))

    def return_sequence_labels(self):
        label_for_sequence = []
        for sequence_idx in range(len(self.label)):
            # label에서 0이 아닌 값만 뽑음 - 해당 시퀀스의 bodypart ground truth가 무엇인지
            labels = np.argmax(self.label[sequence_idx], axis=1)
            label_for_sequence.append(next((lab for lab in labels if lab != 0), None))

        return label_for_sequence
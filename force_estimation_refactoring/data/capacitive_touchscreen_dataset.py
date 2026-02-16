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
    def __init__(self, length_of_window, num_classes, path='./preprocessed_train_dataset', dataset_type='train', xflip=True):
        self.path = path
        self.dataset_type = dataset_type
        self.length_of_window = length_of_window
        self.num_classes = num_classes
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


        ######## Data Augmentation ########
        if self.dataset_type.lower() == 'train' and self.augmentation == True:
            if self.xflip:
                self.horizontal_flip_augmentaion()


        for i in range(len(self.frame)):
            self.frame[i] = int((self.frame[i] - length_of_window) / self.stride) + 1
            self.prefix.append(self.frame[i] + self.prefix[-1])
        self.len = sum(self.frame)



    def __getitem__(self, index):
        sequence_idx = bisect.bisect_right(self.prefix, index) - 1
        frame_idx = index - self.prefix[sequence_idx]

        img = torch.from_numpy(self.img[sequence_idx][frame_idx:frame_idx + self.length_of_window]).float()
        force = torch.from_numpy(self.force[sequence_idx][frame_idx:frame_idx + self.length_of_window, 0:3]).float()
        # label = torch.from_numpy(self.label[sequence_idx][frame_idx + self.length_of_window - 1])
        label = F.one_hot(self.label[sequence_idx], num_classes=self.num_classes).float()

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

    def scale_image(self, img, scale_factor=(2, 2)):
        return F.interpolate(img, scale_factor=scale_factor, mode='bilinear')


    def horizontal_flip_augmentaion(self):
        length = len(self.img)

        for i in range(length):
            img = torch.stack([
                TF.hflip(i)
                for i in self.img[length]
            ])
            self.img.append(img)
            self.force.append(self.force[length][0] * (-1))  # x만 flip

            self.label.append(self.label[length])
            self.label_each_sequence.append(self.label_each_sequence[length])
            self.frame.append(self.frame[length])
            self.name.append(self.name[length])



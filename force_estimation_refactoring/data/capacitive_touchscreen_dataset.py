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
import copy


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
                self.frame = f[f"frame"][:].tolist()
                sample_keys.remove('frame')

            if 'name' in sample_keys:
                self.name = f[f"name"][:].tolist()
                sample_keys.remove('name')

            if 'label_each_sequence' in sample_keys:
                self.label_each_sequence = f[f"label_each_sequence"][:].tolist()
                sample_keys.remove('label_each_sequence')

            for seq in sample_keys:
                self.img.append(f[f"{seq}/img"][:])
                self.force.append(f[f"{seq}/force"][:])
                self.label.append(f[f"{seq}/label"][:])


        ######## Data Augmentation ########
        if self.dataset_type.lower() == 'train' and self.augmentation == True:
            if self.xflip:
                self.horizontal_flip_augmentaion()

        # frame 개수와
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

        # label_idx = torch.tensor(self.label_each_sequence[sequence_idx], dtype=torch.long)
        # label = F.one_hot(label_idx, num_classes=self.num_classes).float()

        label = torch.tensor(self.label_each_sequence[sequence_idx], dtype=torch.long)

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


    def horizontal_flip_augmentaion(self):

        # 각 시퀀스마다 모두 시행
        length = len(self.img)
        for i in range(length):

            # 이미지는 Horizontal Flip을 시행함.
            img = torch.stack([
                TF.hflip(j)
                for j in torch.from_numpy(self.img[i])
            ]).numpy()
            self.img.append(img)

            # force는 x축 부호를 바꿈
            force = copy.deepcopy(self.force[i]) # (length of a sequence, 6)
            force[:, 0] *= (-1) # x만 flip
            self.force.append(force)

            # 다른 데이터들은 그대로 복사
            self.label.append(self.label[i])
            self.label_each_sequence.append(self.label_each_sequence[i])
            self.frame.append(self.frame[i])
            self.name.append(self.name[i])
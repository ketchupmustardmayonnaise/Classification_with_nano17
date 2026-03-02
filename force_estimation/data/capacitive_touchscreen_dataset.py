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
    def __init__(self,
                 length_of_window : int,
                 num_classes : int,
                 refer_recent_frames_for_classification : int,

                 path : str = './preprocessed_train_dataset',
                 dataset_type : str = 'train',

                 xflip : bool = True,
                 yflip : bool = True,
                 translation : bool = True):

        self.path = path
        self.dataset_type = dataset_type
        self.length_of_window = length_of_window
        self.num_classes = num_classes
        self.stride = 1

        # classification 할 때, 하나의 프레임만 참조하지 않고 여러 프레임을 동시에 참조하여 예측함.
        self.refer_recent_frames_for_classification = refer_recent_frames_for_classification

        self.frame = []
        self.name = []
        self.bodypart = []

        self.label = []
        self.img = []
        self.force = []

        self.xflip = xflip
        self.yflip = yflip
        self.translation = translation

        self.prefix = [0]

        with h5py.File(self.path, 'r') as f:
            self.img_mean = torch.tensor(f.attrs["img_mean"]).float()
            self.img_std = torch.tensor(f.attrs["img_std"]).float()

            # (C,) -> (1, C, 1, 1) 형태로 reshape (broadcast용)
            self.img_mean = self.img_mean.view(1, -1, 1, 1)
            self.img_std = self.img_std.view(1, -1, 1, 1)

            sample_keys = sorted([k for k in f.keys() if k.startswith("sequence_")])

            for seq in sample_keys:
                g = f[seq]

                self.img.append(g["img"][:])
                self.force.append(g["force"][:])
                self.label.append(g["label"][:])

                # metadata
                self.frame.append(g.attrs["frame"])
                self.bodypart.append(g.attrs["bodypart"])
                self.name.append(g.attrs["subject"])


        # Data Augmentation
        if self.dataset_type.lower() == 'train':
            if self.xflip:
                self.horizontal_flip_offline_augmentaion()
            if self.yflip:
                self.vertical_flip_offline_augmentaion()
            if self.translation:
                self.translation_offline_augmentation()

        # frame 개수와 length_of_window 값으로부터, 각 시퀀스에서 학습 가능한 데이터 개수 찾음
        for i in range(len(self.frame)):
            self.frame[i] = int((self.frame[i] - length_of_window) / self.stride) + 1
            self.prefix.append(self.frame[i] + self.prefix[-1])
        self.len = sum(self.frame)



    def __getitem__(self, index):
        sequence_idx = bisect.bisect_right(self.prefix, index) - 1
        frame_idx = index - self.prefix[sequence_idx]

        img = torch.from_numpy(self.img[sequence_idx][frame_idx:frame_idx + self.length_of_window]).float()
        # normalize image
        img = (img - self.img_mean) / (self.img_std + 1e-8)

        force = torch.from_numpy(self.force[sequence_idx][frame_idx:frame_idx + self.length_of_window, 0:3]).float()
        label = torch.from_numpy(self.label[sequence_idx][
                frame_idx + self.length_of_window - self.refer_recent_frames_for_classification:
                frame_idx + self.length_of_window]).long()

        return {
            'img': img,
            'force': force,
            'label': label,

            # metadata
            'bodypart' : self.bodypart[sequence_idx],
            'sequence_id': sequence_idx,
            'subject_id': self.name[sequence_idx]
        }

    def __len__(self):
        return self.len


    def horizontal_flip_offline_augmentaion(self):

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
            self.bodypart.append(self.bodypart[i])
            self.frame.append(self.frame[i])
            self.name.append(self.name[i])


    def vertical_flip_offline_augmentaion(self):

        # 각 시퀀스마다 모두 시행
        length = len(self.img)
        for i in range(length):
            # 이미지는 Horizontal Flip을 시행함.
            img = torch.stack([
                TF.vflip(j)
                for j in torch.from_numpy(self.img[i])
            ]).numpy()
            self.img.append(img)

            # force는 x축 부호를 바꿈
            force = copy.deepcopy(self.force[i])  # (length of a sequence, 6)
            force[:, 1] *= (-1)  # y만 flip
            self.force.append(force)

            # 다른 데이터들은 그대로 복사
            self.label.append(self.label[i])
            self.bodypart.append(self.bodypart[i])
            self.frame.append(self.frame[i])
            self.name.append(self.name[i])


    def translation_offline_augmentation(self, max_shift_px=1):
        """
        max_shift_px=1  -> x,y를 -1,0,1 픽셀 shift
        update_force=False -> 이미지 shift만 하고 force는 그대로 둠
        update_force=True  -> shift에 맞춰 force의 x,y도 같이 보정(원하면 사용)
        """

        original_length = len(self.img)

        # 8개 조합: (-1,0,1) x (-1,0,1) 에서 (0,0) 제외
        shifts = [(dx, dy)
                  for dx in range(-max_shift_px, max_shift_px + 1)
                  for dy in range(-max_shift_px, max_shift_px + 1)
                  if not (dx == 0 and dy == 0)]

        for i in range(original_length):
            base_img = torch.from_numpy(self.img[i])  # (T, C, H, W)
            base_force = self.force[i]

            # TF.affine을 위해 (C,H,W)로 맞춰야 함
            # 즉 base_img는 (T,C,H,W)라고 가정하고 진행.

            for dx, dy in shifts:
                # ---------- image shift ----------
                shifted = torch.stack([
                    TF.affine(
                        frame,
                        angle=0.0,
                        translate=[dx, dy],  # [x, y] 픽셀 이동
                        scale=1.0,
                        shear=[0.0, 0.0]
                    )
                    for frame in base_img
                ]).numpy()

                self.img.append(shifted)

                # ---------- force ----------
                force = copy.deepcopy(base_force)
                self.force.append(force)

                # ---------- meta 복사 ----------
                self.label.append(self.label[i])
                self.bodypart.append(self.bodypart[i])
                self.frame.append(self.frame[i])
                self.name.append(self.name[i])
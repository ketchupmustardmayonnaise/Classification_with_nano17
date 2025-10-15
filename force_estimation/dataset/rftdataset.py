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
import pandas as pd


class RFTDataset(Dataset):
    def __init__(self, path='./dataset/preprocessed_images', dataset_type='train'):
        self.data_list = pd.read_csv(f'{path}/{dataset_type}.csv', encoding='utf-8')
        self.dataset_type = dataset_type
        self.augmentation = True
        self.path = path

        if dataset_type == 'test':
            self.len = len(set(self.data_list.loc[:, 'sequence index'])) # 시퀀스 개수
        else:
            self.len = len(self.data_list)


    def __getitem__(self, index):
        if self.dataset_type.lower() == 'test':
            # test는 시퀀스 내 pair를 모두 반환함
            selected_sequence = self.data_list.loc[self.data_list['sequence index'] == index] # 해당 sequence 모두를 가져옴
            temporal_path = os.path.join(self.path, f'{self.dataset_type}/sequence_{index:03d}')

            img = []
            force = []
            for pair_index in range(len(selected_sequence)):
                a = selected_sequence.iloc[pair_index]['first image']
                first_image = np.load(os.path.join(temporal_path, selected_sequence.iloc[pair_index]['first image']))
                second_image = np.load(os.path.join(temporal_path, selected_sequence.iloc[pair_index]['second image']))

                temp_img = np.stack((first_image, second_image), axis=0)
                temp_force = [selected_sequence.iloc[pair_index]['difference of Fx'],
                              selected_sequence.iloc[pair_index]['difference of Fy'],
                              selected_sequence.iloc[pair_index]['difference of Fz']]

                max_of_image = selected_sequence.iloc[pair_index]['max capacitance value of the first image']
                temp_img = temp_img.astype(np.float64)
                temp_img /= max_of_image

                img.append(temp_img)
                force.append(temp_force)

            img = np.array(img)
            force = np.array(force)

            return (
                torch.from_numpy(img),
                torch.from_numpy(force),
            )

        else:
            # train, valiation은 pair별로 반환함.
            selected_data = self.data_list.iloc[index]
            temporal_path = os.path.join(self.path, f'{self.dataset_type}/sequence_{selected_data[0]:03d}')

            first_image = np.load(os.path.join(temporal_path, selected_data[1]))
            second_image = np.load(os.path.join(temporal_path, selected_data[2]))

            img = np.stack((first_image, second_image), axis=0)
            force = [selected_data[3], selected_data[4], selected_data[5]]

            max_of_image = selected_data[6]
            img = img.astype(np.float64)
            img /= max_of_image
            img = torch.from_numpy(img)

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
                torch.Tensor(force)
            )

    def __len__(self):
        return self.len

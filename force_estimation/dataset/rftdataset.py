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
    def __init__(self, path='./dataset/preprocessed_24subject', dataset_type='train'):
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
            selected_sequence = self.data_list.loc[self.data_list['sequence index'] == index]
            temporal_path = os.path.join(self.path, f'{self.dataset_type}/sequence_{index:03d}')

            img = []
            force = []
            for pair_index in range(len(selected_sequence)):
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
            n_pairs = len(selected_sequence)

            subject = selected_sequence.iloc[0]['subject']
            body_part = selected_sequence.iloc[0]['body_part']

            return {
                'img': torch.from_numpy(img),                          # (N_pairs, 2, H, W)
                'force': torch.from_numpy(force),                      # (N_pairs, 3)
                'sequence_id': torch.full((n_pairs,), index, dtype=torch.long),
                'subject_id': subject,
                'bodypart': body_part,
            }

        else:
            # train, validation은 pair별로 반환함.
            selected_data = self.data_list.iloc[index]
            sequence_index = int(selected_data['sequence index'])
            temporal_path = os.path.join(self.path, f'{self.dataset_type}/sequence_{sequence_index:03d}')

            first_image = np.load(os.path.join(temporal_path, selected_data['first image']))
            second_image = np.load(os.path.join(temporal_path, selected_data['second image']))

            img = np.stack((first_image, second_image), axis=0)
            force = [selected_data['difference of Fx'],
                     selected_data['difference of Fy'],
                     selected_data['difference of Fz']]

            max_of_image = selected_data['max capacitance value of the first image']
            img = img.astype(np.float64)
            img /= max_of_image
            img = torch.from_numpy(img)

            ######## Data Augmentation ########
            if self.dataset_type.lower() == 'train' and self.augmentation == True:

                if random.random() < 0.5:
                    force[0] *= (-1)  # x만 flip

                    img = torch.stack([
                        TF.hflip(i)
                        for i in img
                    ])

            return {
                'img': img,                                                          # (2, H, W)
                'force': torch.Tensor(force),                                        # (3,)
                'sequence_id': torch.tensor(sequence_index, dtype=torch.long),
                'subject_id': selected_data['subject'],
                'bodypart': selected_data['body_part'],
            }

    def __len__(self):
        return self.len

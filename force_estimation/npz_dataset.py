import os
import zipfile
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

num_classes = 7
IMG_MAX = 600
IMG_MAX_DIFF = 60
clipping_per_label = np.array([0.0, 0.0, -5.0, -6.0, 0.0, 0.0, 0.0])
origin_image_size = (10, 8)

class RFTDataset(Dataset):
    def __init__(self, image_size=(64, 64), image_type='RGB', path='./../dataset/rft64/train'):
        self.img = []
        self.force = []
        self.label = []

        self.img_ts = []
        self.force_ts = []
        self.frame = []

        self.image_type = image_type
        self.image_size = image_size

        if image_type.lower() == 'rgb':
            self.rgb_mean_and_max = [None, None, None, None, None, None]
            # origin_mean, origin_max, diff_from_initial_mean, diff_from_initial_max, diff_mean, diff_max

        # 불러오기
        for label in os.listdir(path): # 각 label에 대하여
            for file_name in os.listdir(path + '/' + label): # 각 파일들에 대하여
                with zipfile.ZipFile(path + '/' + label + '/' + file_name, 'r') as zf: # 각 zip파일들에 대하여
                    for npzfile in zf.namelist(): # 각 npz파일들에 대하여
                        with zf.open(npzfile) as nf:
                            data = np.load(nf)
                            if len(data['img_ts']) > len(data['force_ts']): continue

                            self.img.append(data['img'])
                            self.force.append(data['force'])
                            self.img_ts.append(data['img_ts'])
                            self.force_ts.append(data['force_ts'])
                            self.label.append(int(label))
                            self.frame.append(len(data['img']))

        print(np.sum(self.frame))

    def __getitem__(self, index):
        return self.img[index], self.label[index], self.force[index][:, 0:3]

    def __len__(self):
        return len(self.img)

    def preprocess(self):
        '''
        # self.img.shape = (# of sequence, # of chunk, frame, channel, height, width)
        # self.force.shape = (# of sequence, # of chunk, 6)  (6 features: (Fx, Fy, Fz, Tx, Ty, Tz))
        # self.label.shape = (# of sequence, # of chunk, num_classes)
        '''

        # 설정한 image_type에 맞게 이미지를 바꿈
        '''
        # origin: 그냥 원본 capacitance 이미지 그대로 씀
        # diff: 현재 이미지에서, 이전 t-1시점의 이미지를 뺀 이미지를 씀
        # rgb: 원본 이미지는 R 채널로, diff 이미지에서 양수는 G 채널로, diff 이미지에서 음수는 부호 바꿔서 B 채널로
        '''
        if self.image_type.lower() == 'rgb':
            self.divide_RGB_channels()
            for channel in range(0, 3):
                self.rgb_mean_and_max[channel * 2], self.rgb_mean_and_max[channel * 2 + 1] = self.extract_mean_and_max(channel)
        elif self.image_type.lower() == 'diff':
            self.image_to_diff()
            for idx in range(len(self.img)):
                img = []
                for frame in range(len(self.img[idx])):
                    img.append(np.expand_dims(self.img[idx][frame], axis=0))
                self.img[idx] = np.array(img, dtype=np.float32)
        elif self.image_type.lower() == 'origin':
            for idx in range(len(self.img)):
                img = []
                for frame in range(len(self.img[idx])):
                    img.append(np.expand_dims(self.img[idx][frame], axis=0))
                self.img[idx] = np.array(img, dtype=np.float32)
        else:
            raise Exception('Unknown image type')

        # capacitance 각각을 IMG_MAX로 나눔 (origin, rgb의 경우 512, diff의 경우 60)
        self.normalize_image(IMG_MAX, IMG_MAX_DIFF)
        # cap 이미지가 (10, 8)이기 때문에 너무 작아서 확대
        if self.image_size != origin_image_size: self.img = self.resize_image(self.image_size)

        # force data에 low-pass filter를 걺
        # b_f = self.force[5][:, 0]
        self.force = self.apply_butterworth_filter()
        # cap 이미지랑 force의 모으는 시간이 다르기 때문에, timestamp에 맞춰서 force data를 cap 시간에 interpolate
        self.force = self.interpolate_nano_to_capacitance(self.force, self.img_ts, self.force_ts)
        # b_a = self.force[5][:, 0]

        # force값을 clipping함. 0 이상인 것은 0으로, -6 이하인 것은 -6으로.
        self.outlier_filter(clipping_per_label)

        # label
        self.label = F.one_hot(torch.Tensor(self.label).to(torch.int64), num_classes=num_classes)

        # fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        #axes[0].plot(b_f)
        #axes[1].plot(b_a)
        #plt.show()

    def divide_RGB_channels(self): pass
    def extract_mean_and_max(self, channel): pass
    def interpolate_nano_to_capacitance(self, force_data, img_ts, force_ts): pass
    def apply_butterworth_filter(self): pass
    def measure_image_and_force_frequency(self, img_ts, force_ts): pass
    def resize_image(self, size): pass
    def normalize_image(self, img_max, img_max_diff): pass
    def outlier_filter(self, clipping_per_label): pass
    def print_information(self): pass
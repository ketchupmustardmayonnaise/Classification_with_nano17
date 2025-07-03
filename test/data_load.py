import os
import cv2
import torch
import ast
import numpy as np
import torch.nn.functional as F
from scipy.interpolate import interp1d
from torch.utils.data import Dataset
from scipy import signal, stats
from scipy.signal import filtfilt
from parameters import *

num_classes = 8
IMG_MAX = 512

class Image_and_force():
    def __init__(self, path='./data/two/test'):
        self.img = []
        self.force = []
        self.label = []

        self.img_timestamp = []
        self.force_timestamp = []

        self.index = -1

        for label in os.listdir(path):
            for file_name in os.listdir(path + '/' + label): # 각 파일들에 대하여
                if file_name.endswith('.csv'): # 파일 확장자가 csv인 경우만 처리
                    file_path = os.path.join(path + '/' + label, file_name)

                    file = open(file_path, 'r')
                    lines = file.readlines()
                    lines = [lst for lst in lines if lst != '\n' or lst != []]

                    line_index = 0
                    while line_index < len(lines):
                        lines[line_index] = lines[line_index].split('"')
                        lines[line_index] = [item for item in lines[line_index] if item.strip()]

                        temp_list = []
                        for item in lines[line_index]:
                            if item.startswith("[") and item.endswith("]"):
                                temp_list.append(ast.literal_eval(item))
                            else:
                                temp = item.split(",")
                                temp_list.extend([item for item in temp if item != ""])
                        lines[line_index] = temp_list

                        # 아무것도 기록되지 않은 줄이라면
                        if len(lines[line_index]) == 0:
                            del lines[line_index]
                            continue
                        elif lines[line_index][0] == '\n':
                            del lines[line_index]
                            continue

                        frame = int(lines[line_index][0])
                        self.img_timestamp.append(list(map(float, lines[line_index][1: 2 + 2 * (frame - 1): 2])))
                        img = list(lines[line_index][2: 2 + 2 * frame: 2])

                        # underflow 된 데이터들 0으로 처리
                        img = [
                            [0 if x >= 65000 else x for x in inner]
                            for inner in img
                        ]

                        try:
                            if any('fdx:' in item for item in lines[line_index]):
                                self.force_timestamp.append(list(map(float, lines[line_index][1 + 2 * frame: -1: 2])))
                                self.force.append(list(lines[line_index][2 + 2 * frame: -1: 2]))
                            else:
                                self.force_timestamp.append(list(map(float, lines[line_index][1 + 2 * frame:: 2])))
                                self.force.append(list(lines[line_index][2 + 2 * frame:: 2]))

                            self.img.append(np.array(img, dtype=np.float32).reshape((frame, 10, 8)))
                            self.label.append(int(label))
                        except:
                            print(file_name, line_index)

                        line_index += 1

        if FLAG_RESIZE:
            self.img = self.resize_image(SIZE)

        self.force = self.apply_butterworth_filter()
        self.force = self.interpolate_nano_to_capacitance(self.img, self.force, self.img_timestamp, self.force_timestamp)
        self.outlier_filter()
        self.preprocess()

        # label
        self.label = torch.Tensor(self.label)
        self.label = F.one_hot(self.label.to(torch.int64), num_classes=num_classes)

    def __len__(self):
        return len(self.img)

    def __getitem__(self):
        self.index += 1 if self.index < len(self.img) - 1 else 0
        return self.img[self.index], self.label[self.index], self.force[self.index][:, 2]

    def preprocess(self):
        for index in range(len(self.img)):
            self.img[index] = np.array(self.img[index]).astype(np.float32)
            self.img[index] = np.where(self.img[index] >= IMG_MAX, 1.0, self.img[index] / IMG_MAX)

    def interpolate_nano_to_capacitance(self, capacitance_data, force_data, capacitance_timestamp, force_timestamp):
        result = []
        for index in range(len(force_timestamp)):
            # 보간 함수 생성 (외삽은 금지 - 범위 밖이면 NaN 발생)
            interp_func = interp1d(force_timestamp[index], force_data[index], kind='linear', fill_value='extrapolate', axis=0)

            # sensor1의 타임스탬프에 대해 sensor2 값 보간
            r = interp_func(capacitance_timestamp[index])
            result.append(r)

        return result

    def apply_butterworth_filter(self):
        result = []

        nyq = 0.5 * 29
        cutoff = 10

        for index in range(len(self.force)):
            temp_force = np.array(self.force[index])
            b, a = signal.butter(2, cutoff / nyq, 'low')
            data = []
            for sig in range(6):
                r = filtfilt(b, a, temp_force[:, sig])
                data.append(r)

            result.append(np.array(data).T)
        return result

    def resize_image(self, size):
        result = []
        for data_index in range(len(self.img)):
            data = []
            for frame_index in range(len(self.img[data_index])):
                resized_img = cv2.resize(self.img[data_index][frame_index], size,
                                         interpolation=cv2.INTER_CUBIC)
                data.append(resized_img)
            result.append(data)

        return result

    def outlier_filter(self):
        for index in range(len(self.force)):
            img = np.array(self.img[index])
            force = np.array(self.force[index])

            indices = np.where(force[:, 2] < -6)[0]

            img = np.delete(img, indices, 0)
            force = np.delete(force, indices, 0)

            self.img[index] = img
            self.force[index] = force
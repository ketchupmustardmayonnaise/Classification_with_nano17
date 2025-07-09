import os
from sys import exception

import cv2
import math
import torch
import ast
import numpy as np
import torch.nn.functional as F
from scipy.interpolate import interp1d
from torch.utils.data import Dataset
from scipy.stats import pearsonr
from scipy import signal, stats
from scipy.signal import filtfilt
import matplotlib.pyplot as plt
from torchvision import transforms

num_classes = 8
IMG_MAX = 512

class CFDataset(Dataset):
    def __init__(self, target_indices, stride, image_resize='False', image_size=(28, 28), image_type='RGB', path='./data/origin'):
        self.target_indices = target_indices
        self.stride = stride

        self.img = []
        self.force = []
        self.label = []

        self.img_timestamp = []
        self.force_timestamp = []

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
                        img = list(lines[line_index][2: 2 + 2 * frame: 2])

                        # underflow 된 데이터들 0으로 처리
                        img = [
                            [0 if x >= 60000 else x for x in inner]
                            for inner in img
                        ]

                        try:
                            # force 저장
                            if any('fdx:' in item for item in lines[line_index]):
                                self.force_timestamp.append(list(map(float, lines[line_index][1 + 2 * frame: -1: 2])))
                                self.force.append(np.array(list(lines[line_index][2 + 2 * frame: -1: 2])))
                            else:
                                self.force_timestamp.append(list(map(float, lines[line_index][1 + 2 * frame:: 2])))
                                self.force.append(np.array(list(lines[line_index][2 + 2 * frame:: 2])))

                            # img 및 label 저장
                            self.img.append(np.array(img, dtype=np.float32).reshape((frame, 10, 8)))
                            self.img_timestamp.append(list(map(float, lines[line_index][1: 2 + 2 * (frame - 1): 2])))
                            self.label.append(int(label))
                        except:
                            print(file_name, line_index)

                        line_index += 1

        self.force_zero_calibration()
        if image_type == 'RGB': self.divide_3_channels()
        else:
            for idx in range(len(self.img)):
                for frame in range(len(self.img[idx])):
                    self.img[idx][frame] = np.expand_dims(self.img[idx][frame], axis=0)

        if image_resize:
            self.img = self.resize_image(image_size)

        self.force = self.apply_butterworth_filter()
        self.force = self.interpolate_nano_to_capacitance(self.img, self.force, self.img_timestamp, self.force_timestamp)
        self.img, self.force, self.label = self.break_images_smaller(self.img, self.force, self.label)

        self.outlier_filter()
        self.preprocess()

        # label
        self.label = torch.Tensor(self.label)
        self.label = F.one_hot(self.label.to(torch.int64), num_classes=num_classes)

    def __len__(self):
        return len(self.img)

    def __getitem__(self, index):
        return self.img[index], self.label[index], self.force[index][2]

    def preprocess(self):
        for index in range(len(self.img)):
            self.img[index] = np.array(self.img[index]).astype(np.float32)
            self.img[index] = np.where(self.img[index] >= IMG_MAX, 1.0, self.img[index] / IMG_MAX)
            self.img[index] = np.where(self.img[index] < 0, 0, self.img[index])

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
            temp_force = self.force[index]
            b, a = signal.butter(2, cutoff / nyq, 'low')
            data = []
            for sig in range(6):
                r = filtfilt(b, a, temp_force[:, sig])
                data.append(r)

            result.append(np.array(data).T)

            '''
            r = np.random.rand()
            if r < 0.01:
                b, a = signal.butter(2, 5 / nyq, 'low')
                c_5 = filtfilt(b, a, temp_force[:, 2])
                print(f"{index}, {self.label[index]}")

                plt.plot(temp_force[:, 2], color='r', label='before')
                plt.plot(data[2], color='blue', label='10 cutoff')
                plt.plot(c_5, color='c', label='5 cutoff')
                plt.xlabel('frame')
                plt.ylabel('force (N)')
                plt.legend()
                plt.show()
                plt.pause(5)
            '''
        return result

    def resize_image(self, size):
        result = []
        for data_index in range(len(self.img)):
            data = []
            for frame_index in range(len(self.img[data_index])):
                # 이미지를 (C, H, W)에서 (H, W, C)로 변경
                img = np.transpose(self.img[data_index][frame_index], (1, 2, 0))
                resized_img = cv2.resize(img, size,
                                         interpolation=cv2.INTER_CUBIC)
                # 이미지를 (H, W, C)에서 (C, H, W)로 변경
                resized_img = np.transpose(resized_img, (2, 0, 1))
                data.append(resized_img)
            result.append(data)

        return result

    def force_zero_calibration(self):
        for idx in range(len(self.force)):
            self.force[idx] -= self.force[idx][0]

    def RGB_image(self):
        multi_channel_img = []
        for data in range(len(self.img)):
            seq_img = []
            for frame in range(len(self.img[data])):
                img = np.array(self.img[data][frame])
                img_copy = img.copy() * 255
                img_copy = img_copy.astype(np.uint8)

                # Sobel edge
                sobelx = cv2.Sobel(img_copy, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(img_copy, cv2.CV_64F, 0, 1, ksize=3)
                sobel_mag = np.sqrt(sobelx ** 2 + sobely ** 2)

                blurred = cv2.GaussianBlur(img_copy, (5, 5), 0)
                sharp = cv2.addWeighted(img_copy, 1.5, blurred, -0.5, 0)

                img = np.stack([img, sobel_mag / 255, sharp / 255])
                seq_img.append(img)

            multi_channel_img.append(np.array(seq_img))
        multi_channel_img = np.array(multi_channel_img)

        self.img = multi_channel_img

    def divide_3_channels(self):
        ''' 맨처음 프레임은 버리되, 나머지 프레임은 3 Chanenl 이미지로 '''
        for sequence_idx in range(len(self.img)):
            new_seq = []
            for frame_idx in range(len(self.img[sequence_idx])):
                # 처음 프레임이라면 시행 X
                if frame_idx == 0:
                    continue

                current_frame = self.img[sequence_idx][frame_idx]
                previous_frame = self.img[sequence_idx][frame_idx - 1]

                diff = current_frame - previous_frame

                brighter = np.where(diff > 0, diff * current_frame, 0).reshape((1, 10, 8))
                darker = np.where(diff < 0, -diff * current_frame, 0).reshape((1, 10, 8))

                new_frame = np.vstack([current_frame.reshape((1, 10, 8)), brighter, darker])
                if np.isnan(new_frame).any():
                    continue
                new_seq.append(new_frame)


            self.img[sequence_idx] = new_seq
            self.img_timestamp[sequence_idx] = self.img_timestamp[sequence_idx][1:]


    def break_images_smaller(self, original_img, original_force, original_label):
        '''
        frame을 target_indices로 쪼갬 (image, shear 모두 적용됨)

        만약 원본 프레임이 30 frame이고, 이를 8 frame씩 쪼갠다면
        (30 - 8) + 1 만큼으로 쪼개짐

        (original_indices - target_indices) + 1
        '''
        img = [] # temporary image list
        force = []
        label = []

        for index in range(len(original_img)):
            frame = 0
            while frame + self.target_indices < len(original_img[index]):
                img.append(original_img[index][frame: frame + self.target_indices])
                temp_force = original_force[index][frame : frame + self.target_indices]

                force.append(temp_force[-1])
                label.append(original_label[index])
                frame += self.stride

        return img, np.array(force), label

    def outlier_filter(self):
        indices = np.where(self.force[:, 2] < -6)[0]
        # print(indices)

        self.img = np.delete(self.img, indices, 0)
        self.force = np.delete(self.force, indices, 0)
        self.label = np.delete(self.label, indices, 0)

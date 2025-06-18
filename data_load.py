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

TARGET_INDICES = 8
STRIDE = 1
SIZE = (28, 28)

FLAG_RESIZE = False
IMG_MAX = 512

class Image_and_force(Dataset):
    def __init__(self, path='./data/TWO'):
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
                        self.img_timestamp.append(list(map(float, lines[line_index][1: 2 + 2 * (frame - 1): 2])))
                        img = list(lines[line_index][2: 2 + 2 * frame: 2])
                        img = [
                            [0 if x >= 65000 else x for x in inner]
                            for inner in img
                        ]
                        self.img.append(np.array(img, dtype=np.float32).reshape((frame, 10, 8)))
                        self.label.append(int(label))

                        if any('fdx:' in item for item in lines[line_index]):
                            self.force_timestamp.append(list(map(float, lines[line_index][1 + 2 * frame: -1: 2])))
                            self.force.append(list(lines[line_index][2 + 2 * frame: -1: 2]))
                        else:
                            self.force_timestamp.append(list(map(float, lines[line_index][1 + 2 * frame:: 2])))
                            self.force.append(list(lines[line_index][2 + 2 * frame:: 2]))

                        line_index += 1

        self.force = self.apply_butterworth_filter()
        self.force = self.interpolate_nano_to_capacitance(self.img, self.force, self.img_timestamp, self.force_timestamp)
        self.img, self.force, self.label = self.break_images_smaller(self.img, self.force, self.label, TARGET_INDICES, STRIDE)
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
        self.img = self.img.astype(np.float32)
        self.img = np.where(self.img >= IMG_MAX, 1.0, self.img / IMG_MAX)
        self.img = torch.Tensor(self.img)

    def interpolate_nano_to_capacitance(self, capacitance_data, force_data, capacitance_timestamp, force_timestamp):
        result = []
        for index in range(len(force_timestamp)):
            # 보간 함수 생성 (외삽은 금지 - 범위 밖이면 NaN 발생)
            interp_func = interp1d(force_timestamp[index], force_data[index], kind='linear', fill_value='extrapolate', axis=0)

            # sensor1의 타임스탬프에 대해 sensor2 값 보간
            r = interp_func(capacitance_timestamp[index])
            result.append(r)

        return result

    def augmentation(self):
        self.flip()

        img_length = len(self.img)
        for angle in range(0, 360, 90):
           for index in range(0, img_length, 1):
               img, force = self.rotate_image_and_force(self.img[index], self.force[index], angle)
               self.img.append(np.array(img))
               self.force.append(force)
               self.label.append(self.label[index])

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

    def rotate_image_and_force(self, image, force, angle_deg):
        rotated_image = []

        fx = force[:, 0]
        fy = force[:, 1]
        fz = force[:, 2]
        tx = force[:, 3]
        ty = force[:, 4]
        tz = force[:, 5]

        new_force = []

        for index in range(len(image)):
            # 이미지 회전
            if FLAG_RESIZE:
                temp_img = np.float32(image[index]).reshape(SIZE)
            else:
                temp_img = np.float32(image[index]).reshape((10, 8))
            (h, w) = temp_img.shape
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
            rotated = cv2.warpAffine(temp_img, M, (w, h),
                                     borderMode=cv2.BORDER_REPLICATE)
            rotated_image.append(rotated)

        # 힘 벡터 회전
        theta = np.radians(angle_deg)
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        rotated_force = rotation_matrix @ np.array([fx, fy])
        rotated_torque = rotation_matrix @ np.array([tx, ty])

        new_force = np.vstack((rotated_force[0], rotated_force[1], fz,
                            rotated_torque[0], rotated_torque[1], tz)).T

        return rotated_image, new_force

    def flip(self):
        data_length = len(self.img)

        for data_index in range(data_length):
            augmentated_image = [[], [], []]
            augmentated_force = [[], [], []]
            for image_index in range(len(self.img[data_index])):
                augmentated_image[0].append([row[::-1] for row in self.img[data_index][image_index]]) # 좌우 flip
                augmentated_image[1].append(self.img[data_index][image_index][::-1])  # 상하 flip
                augmentated_image[2].append([row[::-1] for row in self.img[data_index][image_index][::-1]])  # 상하좌우 flip

            augmentated_image[0] = np.array(augmentated_image[0])
            augmentated_image[1] = np.array(augmentated_image[1])
            augmentated_image[2] = np.array(augmentated_image[2])

            temp = self.force[data_index]
            temp_x = np.copy(temp)
            temp_x[:, [0, 3]] *= (-1)
            temp_y = np.copy(temp)
            temp_y[:, [1, 4]] *= (-1)
            temp_xy = np.copy(temp)
            temp_xy[:, [0, 1, 3, 4]] *= (-1)

            augmentated_force[0] = temp_x
            augmentated_force[1] = temp_y
            augmentated_force[2] = temp_xy

            self.img.extend(augmentated_image)
            self.force.extend(augmentated_force)
            self.label.append(self.label[data_index])
            self.label.append(self.label[data_index])
            self.label.append(self.label[data_index])

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


    def break_images_smaller(self, original_img, original_force, original_label,
                             target_indices=TARGET_INDICES,
                             stride=STRIDE):
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
            while frame + target_indices < len(original_img[index]):
                img.append(original_img[index][frame: frame + target_indices])
                temp_force = original_force[index][frame : frame + target_indices]

                force.append(temp_force[-1])
                label.append(original_label[index])
                frame += stride

        return img, np.array(force), label

    def outlier_filter(self):
        indices = np.where((self.force[:, 2] <= -35) | (self.force[:, 2] > 0.1))[0]

        self.img = np.array(self.img)
        self.label = np.array(self.label)

        self.img = np.delete(self.img, indices, axis=0)
        self.label = np.delete(self.label, indices, axis=0)
        self.force = np.delete(self.force, indices, axis=0)

        # self.force[:, 2] *= -1

        # 몰려있는 데이터 삭제
        # z_scores = np.abs(stats.zscore(self.force[:, 2]))
        # 확률적으로 삭제할 데이터 범위 선택

        # np.random.seed(0)
        # too_many_datas_indices = np.where((self.force[:, 2] > -0.1) & (self.force[:, 2] < 0.1))[0]
        # drop_prob = 0.5

        # 확률적으로 삭제할 데이터들 선택
        # delete_flags = np.random.rand(len(too_many_datas_indices)) < drop_prob
        # indices_to_delete = too_many_datas_indices[delete_flags]

        # self.img = np.delete(self.img, indices_to_delete, axis=0)
        # self.label = np.delete(self.label, indices_to_delete, axis=0)
        # self.force = np.delete(self.force, indices_to_delete, axis=0)

        # self.img = self.img[self.force[:, 2] < 0]
        # self.label = self.label[self.force[:, 2] < 0]
        # self.force = self.force[self.force[:, 2] < 0]


    def visualization(self, count):
        import random, cv2

        for i in range(count): # count: 몇 번 보여 줄 것인가
            random_index = random.randint(0, len(self.img))
            for frame in range(len(self.img[random_index])):
                dst = cv2.resize(np.array(self.img[random_index][frame]), (224, 224), interpolation=cv2.INTER_AREA)
                dst = cv2.cvtColor(dst, cv2.COLOR_GRAY2RGB)
                cv2.putText(dst, str(self.label[random_index]), (15, 15), 1, 1.3, (255, 0, 0), 2)
                cv2.line(dst, (200, 100),
                         (200, 100 + int(self.force[random_index][2].item() * 50)),
                         (0, 255, 0), 3)

                cv2.imshow("image", dst)
                cv2.waitKey(500)
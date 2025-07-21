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
import matplotlib.pyplot as plt

num_classes = 8
IMG_MAX = 512
IMG_MAX_DIFF = 60

class CFDataset(Dataset):
    def __init__(self, target_index, stride, image_resize=False, image_size=(28, 28), image_type='RGB', path='./data/origin'):
        self.target_index = target_index
        self.stride = stride
        self.image_type = image_type

        self.img = []
        self.force = []
        self.label = []

        self.frame = []

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
                        self.frame.append(frame)
                        img = list(lines[line_index][2: 2 + 2 * frame: 2])

                        # capacitive image에서 underflow 된 데이터들 0으로 처리
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

        print(sum(self.frame))
        # img_timestamp와 force_timestamp 기반으로, 동시에 수집한 데이터들인지 확인함
        self.check_overlap_percentage()
        # force data에서, 첫 번째 force로 나머지를 뺌 (초기값 보정)
        self.force_zero_calibration()

        # 설정한 image_type에 맞게 이미지를 바꿈

        # origin: 그냥 원본 capacitance 이미지 그대로 씀
        # diff: 현재 이미지에서, 이전 t-1시점의 이미지를 뺀 이미지를 씀
        # rgb: 원본 이미지는 R 채널로, diff 이미지에서 양수는 G 채널로, diff 이미지에서 음수는 부호 바꿔서 B 채널로
        if image_type.lower() == 'rgb':
            self.divide_3_channels()
        elif image_type.lower() == 'diff':
            self.image_to_diff()
            for idx in range(len(self.img)):
                img = []
                for frame in range(len(self.img[idx])):
                    img.append(np.expand_dims(self.img[idx][frame], axis=0))
                self.img[idx] = np.array(img, dtype=np.float32)
        elif image_type.lower() == 'origin':
            for idx in range(len(self.img)):
                img = []
                for frame in range(len(self.img[idx])):
                    img.append(np.expand_dims(self.img[idx][frame], axis=0))
                self.img[idx] = np.array(img, dtype=np.float32)
        else: raise Exception('Unknown image type')

        # cap 이미지가 (10, 8)이기 때문에 너무 작아서 확대
        if image_resize:
            self.img = self.resize_image(image_size)

        # force data에 low-pass filter를 걺
        self.force = self.apply_butterworth_filter()
        # cap 이미지랑 force의 모으는 시간이 다르기 때문에, timestamp에 맞춰서 force data를 cap 시간에 interpolate
        self.force = self.interpolate_nano_to_capacitance(self.img, self.force, self.img_timestamp, self.force_timestamp)
        # 이미지를 target_index 씩 자름
        self.img, self.force, self.label = self.break_images_smaller(self.img, self.force, self.label)

        # force값을 clipping함. 0 이상인 것은 0으로, -6 이하인 것은 -6으로.
        self.outlier_filter()
        # capacitance 각각을 IMG_MAX로 나눔 (origin, rgb의 경우 512, diff의 경우 60)
        self.preprocess()

        # label
        self.label = F.one_hot(torch.Tensor(self.label).to(torch.int64), num_classes=num_classes)

    def __len__(self):
        return len(self.img)

    def __getitem__(self, index):
        return np.array(self.img[index]), self.label[index], self.force[index][2]

    # capacitance 이미지 입장에서, force와 timestamp가 얼마나 겹치는지 확인
    def check_overlap_percentage(self):
        index_for_delete = []
        for sequence_idx in range(len(self.img)):
            overlap_start = max(self.img_timestamp[sequence_idx][0], self.force_timestamp[sequence_idx][0])
            overlap_end = min(self.img_timestamp[sequence_idx][-1], self.force_timestamp[sequence_idx][-1])

            overlap_duration = max(0, (overlap_end - overlap_start))
            total_duration_of_cap = self.img_timestamp[sequence_idx][-1] - self.img_timestamp[sequence_idx][0]

            if (overlap_duration / total_duration_of_cap) < 0.5: index_for_delete.append(sequence_idx)

        for idx in range(len(index_for_delete)-1, -1, -1): # 맨 끝에서부터 삭제 (맨 앞에서부터 삭제하면 다른 데이터가 삭제될 수 있음)
            del self.img[index_for_delete[idx]]
            del self.force[index_for_delete[idx]]
            del self.img_timestamp[index_for_delete[idx]]
            del self.force_timestamp[index_for_delete[idx]]
            del self.label[index_for_delete[idx]]
            del self.frame[index_for_delete[idx]]


    def preprocess(self):
        if self.image_type.lower() == 'rgb' or self.image_type.lower() == 'origin':
            for index in range(len(self.img)):
                self.img[index] = np.array(self.img[index]).astype(np.float32)
                self.img[index] = np.where(self.img[index] >= IMG_MAX, 1.0, self.img[index] / IMG_MAX)
                self.img[index] = np.where(self.img[index] < 0, 0, self.img[index])

        elif self.image_type.lower() == 'diff':
            for index in range(len(self.img)):
                self.img[index] = np.array(self.img[index]).astype(np.float32)
                self.img[index] = np.where(self.img[index] >= IMG_MAX_DIFF, 1.0, self.img[index] / IMG_MAX_DIFF)

        else:
            raise Exception('Unknown image type')


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

    def exponential_filter(self, alpha):
        result = []
        for data_idx in range(len(self.img)):
            one_data_result = []
            img = alpha * self.img[data_idx][0]
            one_data_result.append(img)

            for frame in range(1, len(self.img[data_idx])):
                img = alpha * self.img[data_idx][frame] + (1 - alpha) * img
                one_data_result.append(img)

            result.append(np.expand_dims(np.array(one_data_result), axis=1))

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
                if len(resized_img.shape) == 2: resized_img = np.expand_dims(resized_img, axis=2)
                # 이미지를 (H, W, C)에서 (C, H, W)로 변경
                resized_img = np.transpose(resized_img, (2, 0, 1))
                data.append(resized_img)
            result.append(data)

        return result

    def force_zero_calibration(self):
        for idx in range(len(self.force)):
            self.force[idx] -= self.force[idx][0]

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

                brighter = np.where(diff > 0, diff, 0).reshape((1, 10, 8))
                darker = np.where(diff < 0, -diff, 0).reshape((1, 10, 8))

                new_frame = np.vstack([current_frame.reshape((1, 10, 8)), brighter, darker])
                if np.isnan(new_frame).any():
                    continue
                new_seq.append(new_frame)


            self.img[sequence_idx] = new_seq
            self.img_timestamp[sequence_idx] = self.img_timestamp[sequence_idx][1:]
            self.frame[sequence_idx] -= 1

    def image_to_diff(self):
        ''' 맨처음 프레임은 버리되, 나머지 프레임은 diff 이미지로 '''
        for sequence_idx in range(len(self.img)):
            new_seq = []
            for frame_idx in range(len(self.img[sequence_idx])):
                # 처음 프레임이라면 시행 X
                if frame_idx == 0:
                    continue

                current_frame = self.img[sequence_idx][frame_idx]
                previous_frame = self.img[sequence_idx][frame_idx - 1]

                diff = current_frame - previous_frame
                new_seq.append(diff)

            self.img[sequence_idx] = new_seq
            self.img_timestamp[sequence_idx] = self.img_timestamp[sequence_idx][1:]
            self.frame[sequence_idx] -= 1


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

        previous_frame = 0
        for index in range(len(original_img)):
            frame = 0

            while frame + self.target_index < len(original_img[index]):
                img.append(original_img[index][frame: frame + self.target_index])
                temp_force = original_force[index][frame : frame + self.target_index]

                force.append(temp_force[-1])
                label.append(original_label[index])
                frame += self.stride

            self.frame[index] = len(img) - previous_frame
            previous_frame = len(img)

        return np.array(img), force, label

    def outlier_filter(self):
        force = np.array(self.force)

        indices = np.where(force[:, 2] < -6)[0]
        force[indices, 2] = -6

        indices = np.where(force[:, 2] > 0)[0]
        force[indices, 2] = 0

        self.force = force
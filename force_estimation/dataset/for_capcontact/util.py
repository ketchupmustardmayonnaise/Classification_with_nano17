import numpy as np
import cv2
from scipy import signal
from scipy.interpolate import interp1d
import os
import torch
import torch.nn.functional as F
import zipfile
import yaml

def load_data_from_npz(path):
    img = []
    force = []

    label = []
    frame = []

    img_ts = []
    force_ts = []

    for lab in os.listdir(path):  # 각 label에 대하여
        for file_name in os.listdir(path + '/' + lab):  # 각 파일들에 대하여
            with zipfile.ZipFile(path + '/' + lab + '/' + file_name, 'r') as zf:  # 각 zip파일들에 대하여
                for npzfile in zf.namelist():  # 각 npz파일들에 대하여
                    with zf.open(npzfile) as nf:
                        data = np.load(nf)
                        if len(data['img_ts']) > len(data['force_ts']): continue

                        img.append(data['img'])
                        force.append(data['force'])

                        img_ts.append(data['img_ts'])
                        force_ts.append(data['force_ts'])

                        label.append(int(lab))
                        frame.append(len(data['img']))

    print(f"{path}: {sum(frame)} image data")

    return img, force, label, img_ts, force_ts

def z_score_of_image(img: np.array, mean: list, std: list):

    # 일단 image는 RGB 채널로 나누어져 있으며
    # break_images_smaller() 함수 실행함으로써 쪼개진 각 chunk라고 가정

    # mean과 std는 3차원 리스트로, 각각 R, G, B의 평균과 표준편차

    mean = np.array(mean)
    std = np.array(std)

    mean_reshaped = mean.reshape(1, -1, 1, 1)
    std_reshaped = std.reshape(1, -1, 1, 1)
    
    img = (img - mean_reshaped) / (std_reshaped + 1e-7)
    return img


def z_score_of_force(force_list:list, mean:list, std:list):
    
    
    mean = np.array(mean)
    std = np.array(std)
    
    std = std + 1e-8
    
    normalized_force = [ (np.array(force) - mean) / std for force in force_list ]
    
    return normalized_force



def divide_RGB_channels(img):
    ''' 첫 번째와 두 번째 프레임은 버리되, 나머지 프레임은 3 Chanenl 이미지로 '''
    result = []
    for sequence_idx in range(len(img)):
        new_seq = []
        for frame_idx in range(len(img[sequence_idx])):
            # 처음 프레임이라면 시행 X
            if frame_idx == 0 or frame_idx == 1:
                continue

            current_frame = img[sequence_idx][frame_idx]
            previous_frame = img[sequence_idx][frame_idx - 1]
            p_previous_frame = img[sequence_idx][frame_idx - 2]

            diff = current_frame - previous_frame
            diff_ = previous_frame - p_previous_frame
            p_diff = diff - diff_

            new_frame = np.vstack([current_frame.reshape((1, 80, 64)), diff.reshape((1, 80, 64)), p_diff.reshape((1, 80, 64))])
            if np.isnan(new_frame).any():
                continue
            new_seq.append(new_frame)
        result.append(np.array(new_seq))

    return result


def interpolate_nano_to_capacitance(force, img_ts, force_ts):
    result = []
    for index in range(len(force_ts)):
        # 보간 함수 생성 (외삽은 금지 - 범위 밖이면 NaN 발생)
        interp_func = interp1d(force_ts[index], force[index], kind='linear', fill_value='extrapolate',
                               axis=0)

        # sensor1의 타임스탬프에 대해 sensor2 값 보간
        r = interp_func(img_ts[index])
        result.append(r)

    return result


def measure_image_and_force_frequency(img_ts, force_ts):
    def measure_frequency(timestamps):
        sum_of_intervals = 0.0
        sum_of_frame_force = 0

        for seq in timestamps:
            interval = np.mean(np.diff(seq))
            sum_of_frame_force += len(seq) - 1
            sum_of_intervals += interval * (len(seq) - 1)

        mean_interval = sum_of_intervals / sum_of_frame_force
        frequency = 1 / mean_interval
        return frequency

    # force frequency 측정
    force_hz = measure_frequency(force_ts)

    # image frequency 측정
    image_hz = measure_frequency(img_ts)

    return image_hz, force_hz


def apply_butterworth_filter(force, img_ts, force_ts):
    result = []

    image_hz, force_hz = measure_image_and_force_frequency(img_ts, force_ts)

    nyq = 0.5 * force_hz
    cutoff = 6

    for index in range(len(force)):
        temp_force = force[index]
        b, a = signal.butter(4, cutoff / nyq, 'low')
        data = []
        for sig in range(6):
            r = signal.filtfilt(b, a, temp_force[:, sig])
            data.append(r)

        result.append(np.array(data).T)

    return result

def resize_image(input_image, size):
    # print(size)
    if size == (10, 8) or size == [10, 8]:
        pass
    else :
        # src (원본 이미지)는 (h, w, c) 순서로 입력해 줘야 함.
        # size는, (h, w) 순서가 아니라 (w, h) 순서로 입력해 줘야 함.
        input_image = np.stack([
            cv2.resize(img.astype(np.float32).transpose(1, 2, 0), (size[1], size[0])).transpose(2, 0, 1)
            for img in input_image
        ])

    return input_image.astype(np.float32)


def normalize_image(img: np.array, max: list, mean: list):
    '''
    입력:
     - img: 단일 시퀀스의 영상 이미지 (shape: (# of frame of one sequence, channel, height, width)
     - max: 각 채널별 max값 list (shape: (3) (max of R channel, max of G channel, max of B channel)
     - mean 각 채널별 mean값 list (shape: (3) (mean of R channel, mean of G channel, mean of B channel)
    '''
    with open("../config.yaml", 'r+', encoding='utf-8') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        
    img = img.astype('float32')

    # Gray Scale 0 - 1로 정규화
    if cfg['image_type'].lower() == 'rgb' :
        img[:, 0] = ((img[ :, 0] * 255 / max[0]))
        img[:, 0] -= (mean[0] * 255 / max[0])

        img[:, 1] = ((img[ :, 0] * 255 / max[0]))
        img[:, 1] -= (mean[0] * 255 / max[0])

        img[:, 2] = ((img[ :, 0] * 255 / max[0]))
        img[:, 2] -= (mean[0] * 255 / max[0])
    elif cfg['image_type'].lower() == 'diff' :
        img[:, 0] = ((img[ :, 0] * 255 / max[0]))
        img[:, 0] -= (mean[0] * 255 / max[0])

        img[:, 1] -= mean[1]

        img[:, 2] -= mean[2]
    else: raise Exception('Unknown rgb type')
        

    return img


def preprocess(img, force, label, img_ts, force_ts,
               image_size, num_classes):
    '''
    모든 전처리 여기에서 진행
    '''
    # force data에 low-pass filter를 걺
    force = apply_butterworth_filter(force, img_ts, force_ts)

    # cap 이미지랑 force의 모으는 시간이 다르기 때문에, timestamp에 맞춰서 force data를 cap 시간에 interpolate
    force = interpolate_nano_to_capacitance(force, img_ts, force_ts)

    # force = z_score_of_force(force, f_mean, f_std)
    
    frame = []

    for sequence_index in range(len(img)):
        frame.append(int(len(img[sequence_index])))
    
    

    # label
    label = F.one_hot(torch.Tensor(label).to(torch.int64), num_classes=num_classes)

    return img, force, label, frame
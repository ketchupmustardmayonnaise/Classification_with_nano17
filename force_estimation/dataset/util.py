import numpy as np
import cv2
from scipy import signal
from scipy.interpolate import interp1d
import os
import torch
import torch.nn.functional as F
import zipfile
import yaml

def load_data_from_npz(path, target_index):
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
                        if len(data['img']) < target_index: continue

                        img.append(data['img'])
                        force.append(data['force'])

                        img_ts.append(data['img_ts'])
                        force_ts.append(data['force_ts'])

                        label.append(int(lab))
                        frame.append(len(data['img']))

    print(f"{path}: {sum(frame)} image data")

    return img, force, label, img_ts, force_ts


def divide_image_and_force(img:np.array, force:np.array, threshold=130, data_type='train'):
    # 이때 입력으로 들어오는 image는 각 시퀀스
    # img shape: (frame 개수, 10, 8)

    # # ============================================
    # (첫 번째 이미지, 두 번째 이미지)로 나누는 과정
    # return으로 줘야 하는 image의 shape: (쌍 개수, 2, 10, 8)
    # return으로 줘야 하는 force의 shape: (쌍 개수, 6)

    result_image = []
    result_force = []

    start = None
    end = None
    start_and_end = []

    # 이미지의 끝까지 가면서, (start, end) 쌍을 찾음
    for frame_index in range(len(img)):
        if start is None:
            if np.max(img[frame_index]) >= threshold:
                start = frame_index

        elif end is None:
            if np.max(img[frame_index]) < threshold:
                end = frame_index

        # 'start' variable and 'end' variable is determined
        if start is not None and end is not None:
            start_and_end.append((start, end))
            start = None
            end = None

    if data_type in ['train', 'val', 'validation']:

        # 이제 start부터 end까지 쌍을 냄
        for (start_frame, end_frame) in start_and_end:
            for s_index in range(start_frame, end_frame):
                for e_index in range(s_index+2, end_frame, 2): # 홀수는 홀수대로, 짝수는 짝수대로
                    result_image.append([img[s_index], img[e_index]])
                    result_force.append(force[e_index] - force[s_index])

        result_image = np.array(result_image)
        result_force = np.array(result_force)

        return result_image, result_force

    elif data_type == 'test':

        # 이제 start부터 end까지 쌍을 냄
        for (start_frame, end_frame) in start_and_end:
            for e_index in range(start_frame, end_frame):
                result_image.append([img[start_frame], img[e_index]])
                result_force.append(force[e_index] - force[start_frame])

        result_image = np.array(result_image)
        result_force = np.array(result_force)

        return result_image, result_force



def normalize_image(img: np.array):
    # 이때 입력으로 들어오는 image는 각 시퀀스
    # img shape: (쌍 개수, 2, 10, 8)

    img = img.astype(np.float32)

    # ============================================
    # 이미지를 normalize하는 과정
    # normalize 방법: 첫 번째 이미지의 max값으로, 첫 번째 이미지와 두 번째 이미지를 나눔.

    for pair_index in range(len(img)):
        pair_max = np.max(img[pair_index][0]) # 한 쌍의 첫 번째 이미지의 최댓값
        img[pair_index] /= pair_max
    # ============================================

    return img


def normalize_force(force: np.array, mean, std):
    # 이때의 force는 sequence별로
    force = (force - np.array(mean)) / np.array(std)
    return force



def translate_image(img: np.array):
    # 이때 입력으로 들어오는 image는 각 시퀀스
    # img shape: (frame 개수, 10, 8)

    for frame_index in range(len(img)):
        sum_of_image = np.sum(img[frame_index][img[frame_index] > 0]) # 양수의 값만 더함

        sum_of_x = 0
        sum_of_y = 0

        for iy in range(img.shape[1]):
            for ix in range(img.shape[2]):
                # 양수라면
                if img[frame_index, iy, ix] >= 0:
                    sum_of_x += (ix * img[frame_index, iy, ix])
                    sum_of_y += (iy * img[frame_index, iy, ix])

        centroid_x = sum_of_x / sum_of_image
        centroid_y = sum_of_y / sum_of_image

        # ==============================
        # 이미지 translation
        dx = -(centroid_x - 3.5)
        dy = -(centroid_y - 4.5)
        mtrx = np.float32([[1, 0, dx],
                           [0, 1, dy]])

        dst_image = cv2.warpAffine(img[frame_index], mtrx,
                                   (8, 10), # (width, height)
                                   cv2.INTER_LINEAR,
                                   cv2.BORDER_CONSTANT,
                                   borderValue=0)

        img[frame_index] = dst_image

    return img


def pad_image(img: np.array, target_shape):
    result_image = []
    for frame_index in range(len(img)):
        h, w = img[frame_index].shape

        pad_h = target_shape[0] - h  # 22
        pad_w = target_shape[1] - w  # 24

        # 위-아래, 좌-우 패딩 분할
        pad_top = pad_h // 2  # 11
        pad_bottom = pad_h - pad_top  # 11
        pad_left = pad_w // 2  # 12
        pad_right = pad_w - pad_left

        # 중앙 패딩
        padded_img = np.pad(
            img[frame_index],
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode='constant',  # 0으로 채움
        )
        result_image.append(padded_img)
    return result_image


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


def preprocess(img, force, label, img_ts, force_ts,
               img_max, img_mean, img_std,
               force_mean, force_std, num_classes,
               data_type):
    '''
    모든 전처리 여기에서 진행
    '''

    # force data에 low-pass filter를 걺
    force = apply_butterworth_filter(force, img_ts, force_ts)

    # cap 이미지랑 force의 모으는 시간이 다르기 때문에, timestamp에 맞춰서 force data를 cap 시간에 interpolate
    force = interpolate_nano_to_capacitance(force, img_ts, force_ts)

    # image 크기: (시퀀스 개수, 각 시퀀스의 프레임 개수, 2, 32, 23)
    frame = []
    for sequence_index in range(len(img)):
        # !! 주의 !!
        # 이미지를 translate한 후, divide_and_normalize_image를 실행하여야 함.
        img[sequence_index] = translate_image(img[sequence_index])
        img[sequence_index] = pad_image(img[sequence_index], (32, 32))
        img[sequence_index], force[sequence_index] = divide_image_and_force(img[sequence_index], force[sequence_index], data_type=data_type)
        img[sequence_index] = normalize_image(img[sequence_index])
        # force[sequence_index] = normalize_force(force[sequence_index], force_mean, force_std)
        frame.append(int(len(img[sequence_index])))

    # label
    label = F.one_hot(torch.Tensor(label).to(torch.int64), num_classes=num_classes)

    return img, force, label, frame
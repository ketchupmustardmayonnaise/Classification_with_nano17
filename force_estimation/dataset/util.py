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



def divide_RGB_channels(img, img_ts):
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

            new_frame = np.vstack([current_frame.reshape((1, 10, 8)), diff.reshape((1, 10, 8)), p_diff.reshape((1, 10, 8))])
            if np.isnan(new_frame).any():
                continue
            new_seq.append(new_frame)

        img_ts[sequence_idx] = img_ts[sequence_idx][2:]
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


def extract_frame_segments(
        original_img,
        original_force,
        threshold=100):
    """
    비디오 시퀀스에서 지정된 (start, end) 범위의 프레임 세그먼트들을 추출합니다.

    Args:
        video_sequence (FrameSequence):
            전체 프레임 시퀀스. Shape은 [총 프레임 수, height, width, ...]여야 합니다.
        frame_ranges (FrameRanges):
            추출할 프레임 범위의 (start, end) 튜플 리스트.
            'start'는 포함되고 'end'는 제외됩니다 (Python 슬라이싱 관례: [start:end]).

    Returns:
        Tuple[FrameSequence, ...]:
            각 범위에 해당하는 프레임 시퀀스 세그먼트들을 담고 있는 튜플.
    """

    result_img = []
    result_force = []

    for sequence_index in range(len(original_img)):

        # ===============================
        # 1. start와 end 쌍 찾기
        # ===============================
        start = None
        end = None
        start_and_end = []

        # 이미지의 끝까지 가면서, (start, end) 쌍을 찾음
        for frame_index in range(len(original_img[sequence_index])):
            if start is None:
                if np.max(original_img[sequence_index][frame_index][0]) >= threshold:
                    start = frame_index

            elif end is None:
                if np.max(original_img[sequence_index][frame_index][0]) < threshold:
                    end = frame_index

            # 'start' variable and 'end' variable is determined
            if start is not None and end is not None:
                if end - start >= 120:
                    start_and_end.append((start, end))
                start = None
                end = None

        # 만약 시퀀스의 끝이 threshold 이상으로 끝난 경우 처리
        if start is not None and end is None:
            start_and_end.append((start, len(original_img[sequence_index])))

        # ===============================
        # 2. segment들로 나누기
        # ===============================

        # 각 (start, end) 쌍에 대해 반복
        for s_index, e_index in start_and_end:

            img_segment = original_img[sequence_index][s_index:e_index]
            force_segment = original_force[sequence_index][s_index:e_index]

            result_img.append(img_segment)
            result_force.append(force_segment)

    return result_img, result_force


def preprocess(img, force, label, img_ts, force_ts,
               img_max, img_mean, img_std, num_classes):
    '''
    모든 전처리 여기에서 진행
    '''

    img = divide_RGB_channels(img, img_ts)

    # force data에 low-pass filter를 걺
    force = apply_butterworth_filter(force, img_ts, force_ts)

    # cap 이미지랑 force의 모으는 시간이 다르기 때문에, timestamp에 맞춰서 force data를 cap 시간에 interpolate
    force = interpolate_nano_to_capacitance(force, img_ts, force_ts)

    img, force = extract_frame_segments(img, force)
    
    frame = []
    for sequence_index in range(len(img)):
        # 이미지를 채널별로 normalize
        img[sequence_index] = z_score_of_image(img[sequence_index], img_mean, img_std)
        frame.append(int(len(img[sequence_index])))

    # label
    label = F.one_hot(torch.Tensor(label).to(torch.int64), num_classes=num_classes)

    return img, force, label, frame
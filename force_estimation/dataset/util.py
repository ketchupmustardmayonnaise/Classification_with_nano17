import shutil

import numpy as np
import cv2
from scipy import signal
from scipy.interpolate import interp1d
import os
import torch
import torch.nn.functional as F
import zipfile
import yaml
import pandas as pd

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


def save_frame_sequences(sequence, seq_index, base_dir):
    """
    영상 시퀀스 데이터를 시퀀스별 폴더와 프레임별 .npz 파일로 저장합니다.

    Args:
        sequences (list of numpy.ndarray): (frame_count, height, width) 형태의 시퀀스 리스트.
        base_dir (str): 모든 출력을 저장할 기본 디렉토리 이름.
    """
    # 최상위 출력 폴더 생성
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        print(f"✅ 최상위 출력 폴더 생성: '{base_dir}'")

    sequence_dir_name = f'sequence_{seq_index:03d}'
    sequence_path = os.path.join(base_dir, sequence_dir_name)

    # 시퀀스별 폴더 생성
    if not os.path.exists(sequence_path):
        os.makedirs(sequence_path)

    # 시퀀스 내의 각 프레임 반복
    for frame_index in range(len(sequence)):
        # (32, 32) 크기의 프레임 데이터 추출
        frame_data = sequence[frame_index]

        # 파일 이름 설정 (예: 0000.npz)
        file_name = f'{frame_index:04d}.npy'
        file_path = os.path.join(sequence_path, file_name)

        # .npz 파일로 저장
        # np.savez는 키-값 쌍으로 저장하며, 불러올 때 키로 접근합니다.
        # 일반적으로 'frame'이라는 키를 사용합니다.
        np.save(file_path, frame_data)


def make_pairs(img:np.array, force:np.array, pair_list:list, sequence_index, data_type='train'):
    # 이때 입력으로 들어오는 image는 각 시퀀스
    # img shape: (frame 개수, 10, 8)

    # # ============================================
    # (첫 번째 이미지, 두 번째 이미지)로 나누는 과정
    # return으로 줘야 하는 image의 shape: (쌍 개수, 2, 10, 8)
    # return으로 줘야 하는 force의 shape: (쌍 개수, 6)

    frame_count = 0

    # pair_list 기록
    if data_type in ['train', 'val', 'validation']:
        for start_frame in range(len(img)):
            for end_frame in range(start_frame + 2, min(len(img), start_frame + 50), 2):
                first_filename = f'{start_frame:04d}.npy'
                second_filename = f'{end_frame:04d}.npy'
                diff_of_force = force[end_frame] - force[start_frame]
                max_of_first_image = np.max(img[start_frame])

                pair_list.append([sequence_index, first_filename, second_filename,
                                  diff_of_force[0], diff_of_force[1], diff_of_force[2],
                                  max_of_first_image])
                frame_count += 1

    elif data_type == 'test':
        start_frame = 0
        for end_frame in range(start_frame + 1, len(img)):
            first_filename = f'{start_frame:04d}.npy'
            second_filename = f'{end_frame:04d}.npy'
            diff_of_force = force[end_frame] - force[start_frame]
            max_of_first_image = np.max(img[start_frame])

            pair_list.append([sequence_index, first_filename, second_filename,
                              diff_of_force[0], diff_of_force[1], diff_of_force[2],
                              max_of_first_image])
            frame_count += 1

    return frame_count


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


def extract_frame_segments(
        original_img,
        original_force,
        frame=120,
        threshold=130):
    '''
    이미지 전체를 보면서, 이미지의 최대 capacitance value가 threshold를 넘지 못하는 경우
    터치가 안 된 순간으로 간주하고, 이를 제외함.

    즉 터치를 시작한 순간부터 뗀 순간까지로 모두 분할함

    또한 frame 이하인 것 또한 지움 (우리 모델의 LSTM은 sliding window 미만인 프레임은 학습하지 못하기 때문에, 동등한 비교를 위하여)
    '''

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
                if np.max(original_img[sequence_index][frame_index]) >= threshold:
                    start = frame_index

            elif end is None:
                if np.max(original_img[sequence_index][frame_index]) < threshold:
                    end = frame_index

            # 'start' variable and 'end' variable is determined
            if start is not None and end is not None:
                if end - start >= frame:
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


def filter_force(force):
    force[force[:, 2] > 0, 2] = 0
    return force



def preprocess(img, force, label, img_ts, force_ts, data_type, num_classes=7):
    '''
    모든 전처리 여기에서 진행
    '''
    path = f'./preprocessed_images/{data_type}'

    # 먼저 이전 데이터와 지금 데이터와 섞이지 않기 위해, npy 파일들을 저장할 디렉토리를 삭제함.
    if os.path.exists(path):
        shutil.rmtree(path)

    # force data에 low-pass filter를 걺
    force = apply_butterworth_filter(force, img_ts, force_ts)

    # cap 이미지랑 force의 모으는 시간이 다르기 때문에, timestamp에 맞춰서 force data를 cap 시간에 interpolate
    force = interpolate_nano_to_capacitance(force, img_ts, force_ts)

    img, force = extract_frame_segments(img, force, frame=120)

    # image 크기: (시퀀스 개수, 각 시퀀스의 프레임 개수, 2, 32, 23)
    frame = []
    pair_list = []
    for sequence_index in range(len(img)):
        force[sequence_index] = filter_force(force[sequence_index]) # z force가 0 이상은 것은 0으로
        # !! 주의 !!
        # 이미지를 translate한 후, divide_and_normalize_image를 실행하여야 함.
        img[sequence_index] = translate_image(img[sequence_index]) # 이미지를 중심 정렬
        img[sequence_index] = pad_image(img[sequence_index], (32, 32)) # 이미지를 32 * 32로 padding
        save_frame_sequences(img[sequence_index], sequence_index, path)
        frame_count = make_pairs(img[sequence_index], force[sequence_index],
                                             pair_list, sequence_index, data_type=data_type)
        frame.append(frame_count)

    header = ['sequence index', 'first image', 'second image',
              'difference of Fx', 'difference of Fy', 'difference of Fz',
              'max capacitance value of the first image']
    pair_list = pd.DataFrame(data=pair_list,
                             columns=header)
    pair_list.to_csv(f'preprocessed_images/{data_type}.csv', index=False)

    # label
    label = F.one_hot(torch.Tensor(label).to(torch.int64), num_classes=num_classes)

    return img, force, label, frame
import os
import zipfile
import numpy as np
import shutil
from PIL import Image

path = './data_to_save_images_of_sequence'
img_max = [281.0000, 251.0000, 251.0000]
img_mean = [9.2207, 0.0346, -0.0139]
img_std = [43.1450, 4.8452, 6.2904]


def divide_RGB_channels(img, img_ts):
    ''' 첫 번째와 두 번째 프레임은 버리되, 나머지 프레임은 3 Chanenl 이미지로 '''
    new_sequence = []
    for frame_idx in range(len(img)):
        # 처음 프레임이라면 시행 X
        if frame_idx == 0 or frame_idx == 1:
            continue

        current_frame = img[frame_idx]
        previous_frame = img[frame_idx - 1]
        p_previous_frame = img[frame_idx - 2]

        diff = current_frame - previous_frame
        diff_ = previous_frame - p_previous_frame
        p_diff = diff - diff_

        new_frame = np.vstack([current_frame.reshape((1, 10, 8)), diff.reshape((1, 10, 8)), p_diff.reshape((1, 10, 8))])
        if np.isnan(new_frame).any():
            continue
        new_sequence.append(new_frame)

    img_ts = img_ts[2:]
    result = np.array(new_sequence)

    return result, img_ts


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

def extract_frame_segments(
        original_img,
        threshold=130):
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

    # ===============================
    # 1. start와 end 쌍 찾기
    # ===============================
    start = None
    end = None
    start_and_end = []

    # 이미지의 끝까지 가면서, (start, end) 쌍을 찾음
    for frame_index in range(len(original_img)):
        if start is None:
            if np.max(original_img[frame_index][0]) >= threshold:
                start = frame_index

        elif end is None:
            if np.max(original_img[frame_index][0]) < threshold:
                end = frame_index

        # 'start' variable and 'end' variable is determined
        if start is not None and end is not None:
            if end - start >= 120:
                start_and_end.append((start, end))
            start = None
            end = None

    # 만약 시퀀스의 끝이 threshold 이상으로 끝난 경우 처리
    if start is not None and end is None:
        start_and_end.append((start, len(original_img)))

    # ===============================
    # 2. segment들로 나누기
    # ===============================

    # 각 (start, end) 쌍에 대해 반복
    for s_index, e_index in start_and_end:
        img_segment = original_img[s_index:e_index]
        result_img.append(img_segment)

    return result_img


sequence_index = 0 # 시퀀스별로 폴더가 생성됨.
for file_name in os.listdir(path + '/'):  # 각 zip 파일들에 대하여
    with zipfile.ZipFile(path + '/' + file_name, 'r') as zf:
        for npzfile in zf.namelist():  # 각 npz파일(=sequence)들에 대하여
            with zf.open(npzfile) as nf:
                max_val = np.array([200, 200, 200]).reshape(1, -1, 1, 1)
                min_val = np.array([0, 0, 0]).reshape(1, -1, 1, 1)

                data = np.load(nf)

                img = data['img']
                img_ts = data['img_ts']

                img, img_ts = divide_RGB_channels(img, img_ts)
                img = extract_frame_segments(img)

                for index in range(len(img)):

                    img[index] = img[index].astype('float64')
                    img[index][1, :, :] *= 10
                    img[index][2, :, :] *= 20
                    # img[index] /= np.array(img_std).reshape(1, -1, 1, 1)

                    # img[index] = z_score_of_image(img[index], img_mean, img_std)

                    # min~max 사이로 클리핑
                    img[index] = np.clip(np.array(img[index]), min_val, max_val)

                    # img[index] -= np.array(img_mean).reshape(1, -1, 1, 1)

                    # 정규화 (0~1)
                    # img[index] = (img[index] - min_val) / (max_val - min_val)

                    # img[index] /= np.array(img_max).reshape(1, -1, 1, 1)

                    img[index] += np.array([50, 50, 50]).reshape(1, -1, 1, 1)

                    # 0~255로 스케일링 후 uint8 변환
                    # img[index] = (img[index] * 255).astype(np.uint8)
                    img[index] = img[index].astype(np.uint8)

                    # 각 픽셀을 20x20 블록으로 확대
                    scale = 20
                    img_big = np.kron(img[index], np.ones((scale, scale), dtype=np.uint8))

                    img[index] = np.transpose(img[index], (0, 2, 3, 1))


                    if os.path.exists(f'./result/{sequence_index}'):
                        shutil.rmtree(f'./result/{sequence_index}')
                    os.makedirs(f'./result/{sequence_index}')

                    for frame_index in range(len(img[index])):
                        Image.fromarray(img[index][frame_index]).save(f'./result/{sequence_index}/{frame_index + 1}.pdf')
                        Image.fromarray(img[index][frame_index][:, :, 0]).save(f'./result/{sequence_index}/{frame_index+1}_0.pdf')
                        Image.fromarray(img[index][frame_index][:, :, 1]).save(f'./result/{sequence_index}/{frame_index + 1}_1.pdf')
                        Image.fromarray(img[index][frame_index][:, :, 2]).save(f'./result/{sequence_index}/{frame_index + 1}_2.pdf')

                    sequence_index += 1
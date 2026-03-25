"""
24 Subject Dataset Preprocessing Script
- FINGERPAD- body parts만 처리
- 기존 util.py의 preprocess 파이프라인을 6x6 이미지에 맞게 적용
- is_valid_frame을 추가로 활용하여 유효하지 않은 프레임 제거
"""

import os
import shutil
import numpy as np
import cv2
import pandas as pd
from scipy import signal
from scipy.interpolate import interp1d
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from util import (
    apply_butterworth_filter,
    interpolate_nano_to_capacitance,
    filter_force,
    save_frame_sequences,
    make_pairs,
    pad_image,
)

# ========== 경로 설정 ==========
DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '24_subject_dataset')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'preprocessed_24subject')

# ========== Subject 분할 ==========
TRAIN_SUBJECTS = [f'subject{i}' for i in [2, 3, 5, 6,
                                          9, 10, 11, 12,
                                          14, 15, 16, 17,
                                          19, 20, 21, 22]]   # subject1 ~ subject18
VAL_SUBJECTS   = [f'subject{i}' for i in [1, 4, 18, 24]]  # subject19 ~ subject21
TEST_SUBJECTS  = [f'subject{i}' for i in [7, 8, 13, 23]]  # subject22 ~ subject24


# ========== 6x6 이미지용 함수 ==========

def translate_image_6x6(img: np.ndarray) -> np.ndarray:
    """
    6x6 capacitance 이미지의 centroid를 중심(2.5, 2.5)으로 정렬.
    기존 translate_image와 동일한 로직, 이미지 크기만 6x6으로 변경.
    """
    CENTER_X = 2.5  # 6 pixel width  → 중심 = (6-1)/2 = 2.5
    CENTER_Y = 2.5  # 6 pixel height → 중심 = (6-1)/2 = 2.5

    img = img.astype(np.float32)

    for frame_index in range(len(img)):
        frame = img[frame_index]

        sum_of_image = np.sum(frame[frame > 0])  # 양수 값의 합
        if sum_of_image == 0:
            continue  # 터치 없는 프레임은 건너뜀

        sum_of_x = 0.0
        sum_of_y = 0.0
        for iy in range(img.shape[1]):
            for ix in range(img.shape[2]):
                if frame[iy, ix] >= 0:
                    sum_of_x += ix * frame[iy, ix]
                    sum_of_y += iy * frame[iy, ix]

        centroid_x = sum_of_x / sum_of_image
        centroid_y = sum_of_y / sum_of_image

        dx = -(centroid_x - CENTER_X)
        dy = -(centroid_y - CENTER_Y)
        mtrx = np.float32([[1, 0, dx],
                            [0, 1, dy]])

        img[frame_index] = cv2.warpAffine(
            frame, mtrx,
            (6, 6),  # (width, height)
            cv2.INTER_LINEAR,
            cv2.BORDER_CONSTANT,
            borderValue=0
        )

    return img


def otsu_threshold_1d(values: np.ndarray) -> float:
    """
    1D 값 배열에 Otsu's threshold 알고리즘 적용.
    클래스 간 분산을 최대화하는 threshold 반환.
    """
    hist, bin_edges = np.histogram(values, bins=256)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    total    = hist.sum()
    sum_total = np.dot(bin_centers, hist)

    max_variance = 0.0
    threshold = bin_centers[0]
    weight_bg = 0
    sum_bg    = 0.0

    for i in range(len(hist)):
        weight_bg += hist[i]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break

        sum_bg   += bin_centers[i] * hist[i]
        mean_bg   = sum_bg / weight_bg
        mean_fg   = (sum_total - sum_bg) / weight_fg
        variance  = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2

        if variance > max_variance:
            max_variance = variance
            threshold    = bin_centers[i]

    return float(threshold)


def compute_otsu_touch_threshold(subjects: list) -> float:
    """
    Train subjects의 모든 프레임에 대해 log(1 + sum(frame))을 계산하고
    Otsu's threshold를 적용한 뒤, exp(t) - 1로 되돌려
    터치 구간 분할에 사용할 sum threshold를 반환.

    Args:
        subjects: threshold 계산에 사용할 subject 목록 (보통 TRAIN_SUBJECTS)

    Returns:
        sum_threshold: np.sum(frame) >= sum_threshold 이면 터치로 간주
    """
    imgs, _, _, _, is_valid_frames, _ = load_fingerpad_data(subjects)
    imgs = replace_bad_frames(imgs, is_valid_frames)
    imgs = [np.clip(img, 0, None) for img in imgs]

    log_sums = np.array([
        np.log1p(np.sum(frame))
        for img_seq in imgs
        for frame in img_seq
    ])

    t_log         = otsu_threshold_1d(log_sums)
    sum_threshold = np.expm1(t_log)  # exp(t) - 1

    print(f"Otsu threshold (log1p-sum space): {t_log:.4f}")
    print(f"Touch threshold (sum space):      {sum_threshold:.2f}")

    return sum_threshold


def extract_frame_segments_with_validity(
        original_img,
        original_force,
        meta: list,
        frame: int = 120,
        threshold: float = 200.0,
):
    """
    터치 구간을 분할.
    각 프레임의 np.sum(frame) >= threshold 이면 활성(터치) 프레임으로 판단.

    Args:
        original_img:   list of (T, 6, 6) ndarray
        original_force: list of (T, 6) ndarray  (보간 완료 후)
        meta:           list of (subject, body_part, filename) 튜플
        frame:          최소 segment 길이 (미만 제거)
        threshold:      터치 판정 기준 (sum of frame)

    Returns:
        result_img, result_force, result_meta
    """
    result_img = []
    result_force = []
    result_meta = []

    for seq_idx in range(len(original_img)):
        img_seq = original_img[seq_idx]
        force_seq = original_force[seq_idx]

        start = None
        end = None
        start_and_end = []

        for frame_idx in range(len(img_seq)):
            is_active = np.sum(img_seq[frame_idx]) >= threshold

            if start is None:
                if is_active:
                    start = frame_idx
            elif end is None:
                if not is_active:
                    end = frame_idx

            if start is not None and end is not None:
                if end - start >= frame:
                    start_and_end.append((start, end))
                start = None
                end = None

        # 시퀀스 끝까지 active한 경우
        if start is not None and end is None:
            end = len(img_seq)
            if end - start >= frame:
                start_and_end.append((start, end))

        for s, e in start_and_end:
            result_img.append(img_seq[s:e])
            result_force.append(force_seq[s:e])
            result_meta.append(meta[seq_idx])

    return result_img, result_force, result_meta


# ========== 불량 프레임 교체 ==========

def replace_bad_frames(imgs: list, is_valid_frames: list, pixel_threshold: int = 1000) -> list:
    """
    로드 직후, 아래 두 조건 중 하나라도 해당하는 프레임을 처리.
      1. is_valid_frame이 False인 프레임
      2. 어느 한 픽셀이라도 pixel_threshold 이상인 프레임

    - 맨 앞 연속 불량 구간: drop (제거)
    - 이후 불량 프레임: 이전 프레임으로 덮어씌움
    - 전체 시퀀스가 불량이면 빈 배열 반환

    Args:
        imgs:             list of (T, 6, 6) ndarray
        is_valid_frames:  list of (T,) bool ndarray
        pixel_threshold:  픽셀 값 이상이면 불량으로 판정

    Returns:
        수정된 imgs (맨 앞 불량 프레임 drop 후 반환)
    """
    result = []
    for seq_idx in range(len(imgs)):
        img_seq = imgs[seq_idx].copy()   # (T, 6, 6)
        valid   = is_valid_frames[seq_idx]

        def is_bad(frame, valid_flag):
            return (not valid_flag) or (np.max(frame) >= pixel_threshold)

        # 1. 맨 앞 연속 불량 구간의 끝 인덱스 찾기 → drop
        first_good = 0
        while first_good < len(img_seq) and is_bad(img_seq[first_good], valid[first_good]):
            first_good += 1

        img_seq = img_seq[first_good:]
        valid   = valid[first_good:]

        # 2. 나머지 불량 프레임 → 이전 프레임으로 교체
        for i in range(1, len(img_seq)):
            if is_bad(img_seq[i], valid[i]):
                img_seq[i] = img_seq[i - 1].copy()

        result.append(img_seq)

    return result


# ========== 데이터 로드 ==========

def load_fingerpad_data(subject_list: list):
    """
    주어진 subject 목록에서 FINGERPAD-* body part 데이터를 모두 로드.

    Returns:
        imgs, forces, img_tss, force_tss, is_valid_frames, meta
        meta: list of (subject, body_part, filename) 튜플
    """
    imgs, forces, img_tss, force_tss, is_valid_frames = [], [], [], [], []
    meta = []

    for subject in subject_list:
        subject_dir = os.path.join(DATASET_DIR, subject)
        if not os.path.isdir(subject_dir):
            print(f"[경고] 폴더 없음: {subject_dir}")
            continue

        for body_part in sorted(os.listdir(subject_dir)):
            if not body_part.startswith('FINGERPAD-'):
                continue

            part_dir = os.path.join(subject_dir, body_part)
            for fname in sorted(os.listdir(part_dir)):
                if not fname.endswith('.npz'):
                    continue

                fpath = os.path.join(part_dir, fname)
                data = np.load(fpath)

                # img_ts가 force_ts보다 길면 skip (기존 코드와 동일)
                if len(data['img_ts']) > len(data['force_ts']):
                    print(f"[skip] img_ts > force_ts: {fpath}")
                    continue

                imgs.append(data['img'])                       # (T, 6, 6)
                forces.append(data['force'])                   # (F, 6)
                img_tss.append(data['img_ts'])                 # (T,)
                force_tss.append(data['force_ts'])             # (F,)
                is_valid_frames.append(data['is_valid_frame']) # (T,)
                meta.append((subject, body_part, fname))

    total_frames = sum(len(im) for im in imgs)
    print(f"로드 완료: {len(imgs)}개 시퀀스, {total_frames}개 프레임 "
          f"({len(subject_list)}명의 subject, FINGERPAD- only)")
    return imgs, forces, img_tss, force_tss, is_valid_frames, meta


# ========== 전처리 메인 함수 ==========

def preprocess_24subject(
        subjects: list,
        data_type: str,
        threshold: float = 200.0,
        min_frames: int = 150,
        pad_size: tuple = (32, 32),
):
    """
    24 subject 데이터셋에서 FINGERPAD- 데이터를 전처리하고 저장.

    Pipeline:
      1. 불량 프레임 교체 (is_valid_frame=False 또는 픽셀>=1000 → 이전 프레임)
      2. Butterworth low-pass filter (force)
      3. force → img timestamp로 보간
      4. 터치 구간 분할 (threshold 기준)
      5. 각 시퀀스:
         - z force clipping (> 0 → 0)
         - centroid 정렬 (translate_image_6x6)
         - 32x32 padding
         - 프레임 저장 (.npy)
         - pair 생성 → CSV 저장

    Args:
        subjects:    처리할 subject 이름 리스트
        data_type:   'train' | 'val' | 'test'
        threshold:   터치 판정 기준 (np.sum(frame) >= threshold). compute_otsu_touch_threshold()로 계산된 값을 권장
        min_frames:  최소 segment 길이
        pad_size:    padding 목표 크기
    """
    output_path = os.path.join(OUTPUT_DIR, data_type)

    # 이전 결과 삭제
    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    # 1. 데이터 로드
    imgs, forces, img_tss, force_tss, is_valid_frames, meta = load_fingerpad_data(subjects)

    if len(imgs) == 0:
        print(f"[{data_type}] 데이터 없음. 건너뜀.")
        return [], [], []

    # 2. 불량 프레임 교체 (is_valid_frame=False 또는 픽셀 값 >= 1000 → 이전 프레임으로 덮어씌움)
    imgs = replace_bad_frames(imgs, is_valid_frames)

    # 2-1. 음수 픽셀 → 0으로 클리핑
    imgs = [np.clip(img, 0, None) for img in imgs]

    # 3. Butterworth low-pass filter
    forces = apply_butterworth_filter(forces, img_tss, force_tss)

    # 4. force를 img timestamp에 맞게 보간
    forces = interpolate_nano_to_capacitance(forces, img_tss, force_tss)

    # 5. 터치 구간 분할 (불량 프레임은 이미 교체됐으므로 threshold만 사용)
    imgs, forces, meta = extract_frame_segments_with_validity(
        imgs, forces, meta,
        frame=min_frames, threshold=threshold,
    )
    print(f"[{data_type}] 터치 구간 분할 후: {len(imgs)}개 segment")

    # 5. 시퀀스별 전처리 및 저장
    pair_list = []
    frame_counts = []

    for seq_idx in range(len(imgs)):
        subject, body_part, _ = meta[seq_idx]
        forces[seq_idx] = filter_force(forces[seq_idx])          # z force clipping
        imgs[seq_idx]   = translate_image_6x6(imgs[seq_idx])     # centroid 정렬
        imgs[seq_idx]   = pad_image(imgs[seq_idx], pad_size)      # 32x32 padding
        save_frame_sequences(imgs[seq_idx], seq_idx, output_path) # .npy 저장
        count = make_pairs(
            imgs[seq_idx], forces[seq_idx],
            pair_list, seq_idx, data_type=data_type,
            subject=subject, body_part=body_part,
        )
        frame_counts.append(count)

    total_pairs = sum(frame_counts)
    print(f"[{data_type}] 총 {total_pairs}개 pair 생성")

    # 6. CSV 저장
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    header = [
        'sequence index', 'first image', 'second image',
        'difference of Fx', 'difference of Fy', 'difference of Fz',
        'max capacitance value of the first image',
        'subject', 'body_part',
    ]
    df = pd.DataFrame(data=pair_list, columns=header)
    csv_path = os.path.join(OUTPUT_DIR, f'{data_type}.csv')
    df.to_csv(csv_path, index=False)
    print(f"[{data_type}] CSV 저장: {csv_path}\n")

    return imgs, forces, frame_counts


# ========== 실행 ==========

if __name__ == '__main__':
    print("=" * 60)
    print("24 Subject Dataset Preprocessing (FINGERPAD- only)")
    print("=" * 60)
    print(f"Train: {TRAIN_SUBJECTS}")
    print(f"Val  : {VAL_SUBJECTS}")
    print(f"Test : {TEST_SUBJECTS}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60 + "\n")

    # Train 데이터 기준으로 Otsu threshold 계산 (log1p-sum space → sum space)
    touch_threshold = compute_otsu_touch_threshold(TRAIN_SUBJECTS)
    print(f"\nTouch threshold: {touch_threshold:.2f}\n")

    preprocess_24subject(TRAIN_SUBJECTS, 'train', threshold=touch_threshold)
    preprocess_24subject(VAL_SUBJECTS,   'val',   threshold=touch_threshold)
    preprocess_24subject(TEST_SUBJECTS,  'test',  threshold=touch_threshold)

    print("전처리 완료.")

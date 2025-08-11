import numpy as np
import cv2  # 리사이즈를 위해 OpenCV 사용 (예시)
from util import *

# --- 설정 (Configuration) ---
path = 'train'
target_index = 30
stride = 1
image_size = (224, 224) # (너비, 높이) 순서

# --- 데이터 로딩 (Data Loading) ---
img, _, _, img_ts, _ = load_data_from_npz(path, target_index)
img = divide_RGB_channels(img, img_ts) # img는 이제 NumPy 배열의 리스트

# --- 개선된 통계 계산 로직 ---

# 리사이즈된 모든 프레임을 저장할 리스트
resized_sequences = []

print("Resizing all frames...")
# 1. 모든 시퀀스와 프레임을 순회하며 리사이즈 후 리스트에 수집
for sequence in img:
    # sequence의 shape: (frames, 3, H, W)
    resized_seq = resize_image(sequence, image_size)
    resized_sequences.append(resized_seq)

# 2. 수집된 프레임 리스트를 하나의 거대한 NumPy 배열로 통합
#    이 과정이 메모리를 가장 많이 사용하지만, 기존 코드보다 훨씬 효율적입니다.
print("Concatenating frames into a single array...")
all_frames_np = np.concatenate(resized_sequences, axis=0)
# all_frames_np shape: (총 프레임 수, 3, 224, 224)

# 더 이상 필요 없는 리스트는 메모리에서 해제
del resized_sequences 
del img

# 3. 통합된 배열에서 벡터화 연산으로 통계치를 한 번에 계산
print("Calculating statistics...")
# 채널별로 데이터 분리
channel_1 = all_frames_np[:, 0, :, :]
channel_2 = all_frames_np[:, 1, :, :]
channel_3 = all_frames_np[:, 2, :, :]

# 평균 및 표준편차 계산
mean_ch1, std_ch1 = channel_1.mean(), channel_1.std()
mean_ch2, std_ch2 = channel_2.mean(), channel_2.std()
mean_ch3, std_ch3 = channel_3.mean(), channel_3.std()

# 최대값 계산
max_ch1, max_ch2, max_ch3 = channel_1.max(), channel_2.max(), channel_3.max()

# 4. 결과 출력
print("\n--- Channel 1 ---")
print(f"Mean: {mean_ch1:.4f}, Std: {std_ch1:.4f}, Max: {max_ch1:.4f}")
print("\n--- Channel 2 ---")
print(f"Mean: {mean_ch2:.4f}, Std: {std_ch2:.4f}, Max: {max_ch2:.4f}")
print("\n--- Channel 3 ---")
print(f"Mean: {mean_ch3:.4f}, Std: {std_ch3:.4f}, Max: {max_ch3:.4f}")
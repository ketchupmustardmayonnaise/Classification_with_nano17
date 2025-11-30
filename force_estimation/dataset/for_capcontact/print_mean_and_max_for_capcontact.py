import numpy as np
import cv2  # 리사이즈를 위해 OpenCV 사용 (예시)
from util import *
import matplotlib.pyplot as plt
from capcontact_generator import Generator

# --- 설정 (Configuration) ---
path = '../train'
batch_size = 32

# --- 데이터 로딩 (Data Loading) ---
img, force, _, img_ts, force_ts = load_data_from_npz(path)
flatten_image = []
generator_result = []

# 하나의 numpy로 만듦
for sequence in img:
    flatten_image.extend(sequence)
flatten_image = np.array(flatten_image)
flatten_image[flatten_image < 0] = 0 # 0 미만인 것은 0으로
flatten_image = np.expand_dims(flatten_image, 1)

# capcontact generator에 통과 - 배치별로 처리
batch_start = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator()
generator.to(device)

while True:
    batch_img = torch.Tensor(flatten_image[batch_start : batch_start + batch_size]).to(device)
    generator_result.extend(generator(batch_img).detach().cpu().numpy().tolist())

    batch_start += batch_size
    print(f"{batch_start} / {len(flatten_image)}")

    if batch_start >= len(flatten_image): break

generator_result = np.squeeze(np.array(generator_result))
img = divide_RGB_channels(generator_result, img_ts) # img는 이제 NumPy 배열의 리스트


print("Resizing all frames...")
# 1. 모든 시퀀스와 프레임을 순회하며 리사이즈 후 리스트에 수집
for sequence in img:
    # sequence의 shape: (frames, 3, H, W)
    resized_sequences.append(resize_image(sequence, image_size))

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
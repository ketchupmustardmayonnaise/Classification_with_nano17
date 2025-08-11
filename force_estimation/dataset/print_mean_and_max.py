'''
path에 있는 데이터들의 채널별 mean, max, std를 확인

!!! 주의 !!!
break_images_smaller 함수를 실행하기 전과 후의 결과가 다를 것.
그리고 break_images_smaller 함수를 실행하기 전에는, interpolate_nano_to_capacitance 함수를 꼭 해 주어야 함.
'''

from util import *

# configuration
path = 'train'  # train dataset 기준으로
target_index = 30
stride = 1
image_size = (224, 224)

# 불러오기
img, force, label, img_ts, force_ts = load_data_from_npz(path, target_index)
img = divide_RGB_channels(img, img_ts)
# force data에 low-pass filter를 걺
force = apply_butterworth_filter(force, img_ts, force_ts)
# cap 이미지랑 force의 모으는 시간이 다르기 때문에, timestamp에 맞춰서 force data를 cap 시간에 interpolate
force = interpolate_nano_to_capacitance(force, img_ts, force_ts)
# target_index씩 자름
img, force, label = break_images_smaller(img, force, label, target_index, stride)

# image shape = (seq, frame, c, h, w)
r_value = []
g_value = []
b_value = []

r_max = -1e10
g_max = -1e10
b_max = -1e10

for seq in range(len(img)):
    r_value.extend(img[seq][:, 0].flatten().tolist())
    g_value.extend(img[seq][:, 1].flatten().tolist())
    b_value.extend(img[seq][:, 2].flatten().tolist())

    if np.max(img[seq][:, 0]) > r_max: r_max = np.max(img[seq][:, 0])
    if np.max(img[seq][:, 1]) > g_max: g_max = np.max(img[seq][:, 1])
    if np.max(img[seq][:, 2]) > b_max: b_max = np.max(img[seq][:, 2])

print(f'r_max: {r_max}, g_max: {g_max}, b_max: {b_max}')
print(f'r_mean: {np.mean(r_value)}, g_mean: {np.mean(g_value)}, b_mean: {np.mean(b_value)}')
print(f'r_std: {np.std(r_value)}, g_std: {np.std(g_value)}, b_std: {np.std(b_value)}')
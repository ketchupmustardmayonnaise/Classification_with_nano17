import numpy as np
from force_estimation.dataset.preprocess import PreprocessDataset
import matplotlib.pyplot as plt

def check_condition(img_frame, force_frame):
    """
    특정 조건을 확인하는 빈 함수
    img_frame: [height, width]
    force_frame: [3] (x, y, z축 힘)
    """
    # 여기에 조건을 작성하세요.
    # 예: np.mean(img_frame) > 100 이거나 force_frame[2] > 5.0 등

    # if sum(img_frame) < 25 and np.linalg.norm(force_frame, axis=1) < 0.02:
    if np.sum(img_frame) < 35:
        return True  # 혹은 False
    else:
        return False


def plot_selected_images(image_list, force_list, threshold=0.2):
    """
    force의 절대값이 threshold 이상인 프레임의 이미지를 plot합니다.
    """
    selected_samples = []

    # 1. 데이터 순회하며 조건에 맞는 프레임 수집
    for frame_idx in range(image_list.shape[0]):
        # force의 3축 중 어느 하나라도 절댓값이 10 이상인지 확인
        if np.any(np.abs(force_list[frame_idx]) >= threshold):
            selected_samples.append({
                'image': image_list[frame_idx],
                'force': force_list[frame_idx],
            })

    # 2. 시각화 (최대 16개까지만 예시로 출력)
    num_to_show = min(len(selected_samples), 16)
    cols = 4
    rows = (num_to_show + cols - 1) // cols

    plt.figure(figsize=(12, 3 * rows))

    for i in range(num_to_show):
        sample = selected_samples[i]
        plt.subplot(rows, cols, i + 1)

        # interpolation='nearest'를 사용해야 6x6 픽셀이 뚜렷하게 보입니다.
        plt.imshow(sample['image'], cmap='gray', interpolation='nearest')

        # 제목에 해당 프레임의 force 값 표시
        force_str = f"F: {np.round(sample['force'], 2)}"
        # plt.title(f"{sample['info']}\n{force_str}", fontsize=9)
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def collect_selected_forces(image_list, force_list):
    selected_forces = []
    selected_images = []

    # 1. 이미지 시퀀스와 포스 시퀀스 쌍을 하나씩 꺼냄
    for img_seq, force_seq in zip(image_list, force_list):
        # 각 시퀀스의 프레임 수 확인 (두 리스트의 프레임 수는 동일하다고 가정)
        num_frames = img_seq.shape[0]

        # 2. 해당 시퀀스 내부의 프레임을 하나씩 순회
        for i in range(num_frames):
            img_frame = img_seq[i]
            force_frame = force_seq[i]

            # 3. 빈 함수(조건)를 통과하는지 확인
            if check_condition(img_frame, force_frame):
                # 조건에 맞으면 해당 시퀀스의 i번째 force 저장
                selected_forces.append(force_frame)
                selected_images.append(img_frame)

    # 최종적으로 넘파이 배열로 변환 [선택된_개수, 3]
    return np.array(selected_images), np.array(selected_forces)

# --- 사용 예시 ---
pd = PreprocessDataset()
DATASET_PATH = "../bodypart/for_analyze"

img, force, label, img_ts, force_ts = pd.load_data_from_npz(DATASET_PATH)

for seq in range(len(img)):
    img[seq], img_ts[seq] = pd.replace_bad_frame_with_previous(img[seq], img_ts[seq])
    img[seq] = pd.clip_zero(img[seq])
    img[seq] = pd.fix_corrupted_frames(img[seq])
force = pd.apply_butterworth_filter(force, img_ts, force_ts)
force = pd.interpolate_nano_to_capacitance(force, img_ts, force_ts)

result_image, result_force = collect_selected_forces(img, force)
result_force = result_force[:, 0:3]

f = np.linalg.norm(result_force, axis=(1,))
# plot_selected_images(result_image, result_force)
print(np.max(f))
print(np.mean(f))
print(np.std(f))
print(np.percentile(f, 95))
# print(f"선택된 힘 데이터의 형태: {result.shape}")
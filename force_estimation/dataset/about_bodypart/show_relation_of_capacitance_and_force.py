import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from force_estimation.dataset.preprocess import PreprocessDataset

DATASET_PATH = "../bodypart/for_analyze"

pd = PreprocessDataset()

img, force, label, img_ts, force_ts = pd.load_data_from_npz(DATASET_PATH)

for seq in range(len(img)):
    img[seq], img_ts[seq] = pd.replace_bad_frame_with_previous(img[seq], img_ts[seq])
    img[seq] = pd.clip_zero(img[seq])
    img[seq] = pd.fix_corrupted_frames(img[seq])
force = pd.apply_butterworth_filter(force, img_ts, force_ts)
force = pd.interpolate_nano_to_capacitance(force, img_ts, force_ts)

# 모든 시퀀스의 이미지를 하나의 numpy로 모음
all_frames_capacitance = np.concatenate(img)

# 모든 시퀀스의 mean과 max를 측정
mean_of_each_frame = np.mean(all_frames_capacitance, axis=(1, 2))
max_of_each_frame = np.max(all_frames_capacitance, axis=(1, 2))
sum_of_each_frame = np.sum(all_frames_capacitance, axis=(1, 2))

# force는 z값만 남겨 둠.
xyz_force = np.concatenate(force)[:, 0:3]
scale_of_force_of_each_frame = np.linalg.norm(xyz_force, axis=1)

thresholds = np.linspace(5, 200, 40)  # 40개 구간으로 나누기

force_thresholds = 0.02


# ==========================================
percentages = []

for th in thresholds:
    mask = max_of_each_frame <= th
    try:
        ratio = np.sum(scale_of_force_of_each_frame[mask] > force_thresholds) / mask.sum() * 100  # %
        percentages.append(ratio)
    except:
        percentages.append(np.nan)

plt.plot(thresholds, percentages, marker='o')
plt.xlabel("Max Capacitance Threshold")
plt.ylabel(f"Percentage of Force > {force_thresholds}N (%)")
plt.title(f"Proportion of Force > {force_thresholds}N")
plt.grid(True)
plt.show()
# ==========================================


# ==========================================
sum_of_force = []
for th in thresholds:
    mask = max_of_each_frame <= th
    try:
        sum = np.sum(scale_of_force_of_each_frame[mask]) / mask.sum()
        sum_of_force.append(sum)
    except:
        percentages.append(np.nan)

plt.plot(thresholds, sum_of_force, marker='o')
plt.xlabel("Max Capacitance Threshold")
plt.ylabel("Mean of Force")
plt.title("Mean of Force")
plt.axhline(force_thresholds, 0, 200, color='red', linestyle='--', label=f'Magnitude of force = {force_thresholds}N')
plt.text(5, force_thresholds+0.01, f'Magnitude of force = {force_thresholds}N', color='red')
plt.grid(True)
plt.show()
# ==========================================



# ==========================================
# 각 프레임 capacitance의 mean값과, z force 상관관계 재기
percentages = []

for th in thresholds:
    mask = mean_of_each_frame < th
    try:
        ratio = np.sum(scale_of_force_of_each_frame[mask] > force_thresholds) / mask.sum() * 100  # %
        percentages.append(ratio)
    except:
        percentages.append(np.nan)

plt.plot(thresholds, percentages, marker='o')
plt.xlabel("Capacitance Threshold")
plt.ylabel(f"Percentage of Force > {force_thresholds}N (%)")
plt.title(f"Proportion of Force > {force_thresholds}N")
plt.grid(True)
plt.show()
# ==========================================


# ==========================================
# 각 프레임 capacitance의 mean값과, z force 상관관계 재기
sum_of_force = []
for th in thresholds:
    mask = mean_of_each_frame < th
    try:
        sum = np.sum(scale_of_force_of_each_frame[mask]) / mask.sum()
        sum_of_force.append(sum)
    except:
        percentages.append(np.nan)

plt.plot(thresholds, sum_of_force, marker='o')
plt.xlabel("Mean Capacitance Threshold")
plt.ylabel("Mean of Force")
plt.title("Mean of Force")
plt.grid(True)
plt.show()
# ==========================================




# ==========================================
# 각 프레임 capacitance의 sum값과, z force 상관관계 재기
percentages = []

for th in thresholds:
    mask = sum_of_each_frame < th
    try:
        ratio = np.sum(scale_of_force_of_each_frame[mask] > force_thresholds) / mask.sum() * 100  # %
        percentages.append(ratio)
    except:
        percentages.append(np.nan)

plt.plot(thresholds, percentages, marker='o')
plt.xlabel("Capacitance Threshold")
plt.ylabel(f"Percentage of Force > {force_thresholds}N (%)")
plt.title(f"Proportion of Force > {force_thresholds}N")
plt.grid(True)
plt.show()
# ==========================================


# ==========================================
# 각 프레임 capacitance의 sum값과, z force 상관관계 재기
sum_of_force = []
for th in thresholds:
    mask = sum_of_each_frame < th
    try:
        sum = np.sum(scale_of_force_of_each_frame[mask]) / mask.sum()
        sum_of_force.append(sum)
    except:
        percentages.append(np.nan)

plt.plot(thresholds, sum_of_force, marker='o')
plt.xlabel("Mean Capacitance Threshold")
plt.ylabel("Mean of Force")
plt.title("Mean of Force")
plt.grid(True)
plt.show()
# ==========================================


# ==========================================
# 각 프레임 capacitance의 sum값과, z force 상관관계 재기
sum_of_force = []
for th in thresholds:
    mask = sum_of_each_frame < th
    try:
        p95 = np.percentile(scale_of_force_of_each_frame[mask], 95)
        sum_of_force.append(p95)
    except:
        percentages.append(np.nan)

plt.plot(thresholds, sum_of_force, marker='o')
plt.xlabel("Sum Capacitance Threshold")
plt.ylabel("95% confidence interval of Force")
plt.title("95% confidence interval of Force")
plt.grid(True)
plt.show()
# ==========================================
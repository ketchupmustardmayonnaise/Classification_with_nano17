import numpy as np
import matplotlib.pyplot as plt

NPZ_PATH = (
    "24_subject_dataset/"
    "subject1/FINGERPAD-INDEX/0_subject1_FINGERPAD-INDEX_PRESS_20260224115059.npz"
)

data = np.load(NPZ_PATH, allow_pickle=True)
img = data["img"].astype(np.float32)          # (N, H, W)
is_valid = data["is_valid_frame"]              # (N,)  bool

# --- 프레임 보정 ---
# is_valid_frame=False 또는 max(frame) >= 1000 이면 이전 프레임으로 교체
cleaned = img.copy()
for i in range(len(cleaned)):
    if not is_valid[i] or cleaned[i].max() >= 1000:
        if i == 0:
            cleaned[i] = np.zeros_like(cleaned[i])   # 첫 프레임은 0으로
        else:
            cleaned[i] = cleaned[i - 1]

# 음수 → 0 clipping
cleaned = np.clip(cleaned, 0, None)

# 프레임별 sum(cap)
sum_cap = cleaned.reshape(len(cleaned), -1).sum(axis=1)

# --- 보정된/제거된 프레임 인덱스 표시용 ---
invalid_mask = (~is_valid) | (img.reshape(len(img), -1).max(axis=1) >= 1000)
invalid_indices = np.where(invalid_mask)[0]

# --- Plot ---
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(sum_cap, linewidth=1.2, label="sum(cap)")

if len(invalid_indices) > 0:
    ax.vlines(
        invalid_indices, ymin=sum_cap.min(), ymax=sum_cap.max(),
        colors="red", linewidth=0.8, alpha=0.5, label=f"replaced frames ({len(invalid_indices)})"
    )

ax.set_xlabel("Frame")
ax.set_ylabel("sum(cap)")
ax.set_title(f"sum(cap) per frame\n{NPZ_PATH.split('/')[-1]}")
ax.legend()
plt.tight_layout()
plt.savefig("plot_sum_cap.svg", dpi=150)
plt.savefig("plot_sum_cap.png", dpi=150)
print(f"Saved: plot_sum_cap.png")
print(f"Total frames: {len(sum_cap)}, Replaced frames: {len(invalid_indices)}")
plt.show()
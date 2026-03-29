"""
FINGERPAD 데이터의 각 프레임에 대해 log(1 + sum(img))를 계산하고
연속적인 히스토그램(KDE)으로 시각화하는 스크립트.

전처리 순서 (build_h5_cross_subject.py 동일):
  1. 음수 픽셀 → 0 (clip_zero)
  2. is_valid_frame == False 프레임 → 이전 프레임으로 덮어씌움
  3. 어느 픽셀이라도 >= 1000 인 프레임 → 이전 프레임으로 덮어씌움
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# 모듈 경로 추가
DATASET_ROOT = Path(__file__).parent / "24_subject_dataset"
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))  # project root
BODYPART = "MOUTH"

from force_estimation.data.current_dataset.module.image_preprocess import (
    clip_zero,
    replace_invalid_frame_with_previous,
    replace_bad_frame_with_previous,
)

IMG_BAD_THRESHOLD = 1000.0


def preprocess_img(img: np.ndarray, is_valid_frame: np.ndarray) -> np.ndarray:
    img = clip_zero(img)
    img = replace_invalid_frame_with_previous(img, is_valid_frame)
    img = replace_bad_frame_with_previous(img, threshold=IMG_BAD_THRESHOLD)
    return img


def collect_fingerpad_log_values(dataset_root: Path) -> np.ndarray:
    """모든 subject에서 FINGERPAD*로 시작하는 파일을 수집하고 log(1+sum(img))를 반환."""
    log_values = []

    for subject_dir in sorted(dataset_root.iterdir()):
        if not subject_dir.is_dir():
            continue
        for bodypart_dir in sorted(subject_dir.iterdir()):
            if not bodypart_dir.name.startswith(BODYPART):
                continue
            for npz_file in sorted(bodypart_dir.glob("*.npz")):
                data = np.load(npz_file)
                img = data["img"]                      # (T, 6, 6), int32
                is_valid = data["is_valid_frame"]      # (T,), bool

                img = preprocess_img(img, is_valid)

                # 각 프레임의 픽셀 합산 후 log(1 + sum)
                frame_sums = img.reshape(img.shape[0], -1).sum(axis=1).astype(float)
                frame_log = np.log1p(frame_sums)
                log_values.append(frame_log)

                print(f"  {subject_dir.name}/{bodypart_dir.name}/{npz_file.name}: "
                      f"{img.shape[0]} frames")

    return np.concatenate(log_values)


def plot_histogram(log_values: np.ndarray):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    n_bins = 80

    # --- 왼쪽: 히스토그램 + KDE (density) ---
    ax = axes[0]
    ax.hist(
        log_values,
        bins=n_bins,
        density=True,
        alpha=0.45,
        color="steelblue",
        edgecolor="white",
        linewidth=0.4,
        label="Histogram (density)",
    )

    kde = gaussian_kde(log_values, bw_method="scott")
    x_grid = np.linspace(log_values.min(), log_values.max(), 1000)
    ax.plot(x_grid, kde(x_grid), color="crimson", linewidth=2, label="KDE")

    ax.set_xlabel("log(1 + sum(img))", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"{BODYPART} frames — log(1+sum(img)) distribution", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- 오른쪽: 히스토그램 (count) ---
    ax2 = axes[1]
    ax2.hist(
        log_values,
        bins=n_bins,
        density=False,
        alpha=0.55,
        color="darkorange",
        edgecolor="white",
        linewidth=0.4,
    )
    ax2.set_xlabel("log(1 + sum(img))", fontsize=12)
    ax2.set_ylabel("Count (frames)", fontsize=12)
    ax2.set_title(f"{BODYPART} frames — count histogram", fontsize=13)
    ax2.grid(True, alpha=0.3)

    stats_text = (
        f"N = {len(log_values):,} frames\n"
        f"mean = {log_values.mean():.3f}\n"
        f"std  = {log_values.std():.3f}\n"
        f"min  = {log_values.min():.3f}\n"
        f"max  = {log_values.max():.3f}"
    )
    ax2.text(
        0.97, 0.97, stats_text,
        transform=ax2.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8),
    )

    plt.tight_layout()
    out_path = Path(__file__).parent / f"{BODYPART}_log_sum_histogram.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved: {out_path}")
    plt.show()


if __name__ == "__main__":
    print(f"Collecting {BODYPART} data...")
    log_values = collect_fingerpad_log_values(DATASET_ROOT)
    print(f"\nTotal frames: {len(log_values):,}")
    plot_histogram(log_values)
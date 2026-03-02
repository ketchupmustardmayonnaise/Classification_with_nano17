import os
import numpy as np
from collections import defaultdict

from force_estimation.data.previous_dataset.preprocess import PreprocessDataset


# =========================
# ===== USER SETTINGS =====
# =========================

ROOT = "../dataset/bodypart_not_cross_person/all"   # dataset root (train/val/test 있는 폴더)
SCORE_MODE = "relu_sum"    # "relu_sum", "abs_sum", "sum"
K_STD = 0.0                # threshold = otsu + K_STD * std


# =========================
# ===== OTSU FUNCTIONS =====
# =========================

def otsu_threshold_1d(x, nbins=1024):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    hist, edges = np.histogram(x, bins=nbins)
    p = hist.astype(float) / (hist.sum() + 1e-12)
    centers = (edges[:-1] + edges[1:]) / 2

    w0 = np.cumsum(p)
    w1 = 1.0 - w0
    mu = np.cumsum(p * centers)
    mu_t = mu[-1]

    denom = w0 * w1
    denom[denom == 0] = np.nan
    sigma_b2 = (mu_t * w0 - mu) ** 2 / denom

    idx = np.nanargmax(sigma_b2)
    return float(centers[idx])


def log_otsu_threshold(values, nbins=1024):
    v = np.asarray(values, dtype=float)
    v = np.maximum(v, 0.0)

    L = np.log1p(v)
    t_log = otsu_threshold_1d(L, nbins=nbins)
    T_original = float(np.expm1(t_log))
    return T_original


# =========================
# ===== SCORE FUNCTION =====
# =========================

def seq_to_S(img_seq, score_mode="relu_sum"):
    """
    img_seq: (T, H, W)
    return: (T,)
    """
    X = np.asarray(img_seq)

    if score_mode == "relu_sum":
        return np.maximum(X, 0).sum(axis=(1, 2))
    elif score_mode == "abs_sum":
        return np.abs(X).sum(axis=(1, 2))
    elif score_mode == "sum":
        return X.sum(axis=(1, 2))
    else:
        raise ValueError(score_mode)


# =========================
# ===== MAIN LOGIC =====
# =========================

def main():

    train_dir = os.path.join(ROOT, "train")
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Train directory not found: {train_dir}")

    pd = PreprocessDataset()

    print("Loading train data...")
    img_list, _, labels, img_ts, _, _ = pd.load_data_from_npz(train_dir)

    S_by_label = defaultdict(list)

    print("Collecting scores per label...")

    for img_seq, y, ts in zip(img_list, labels, img_ts):

        # ===== 최소 전처리 =====
        img_seq, _ = pd.replace_bad_frame_with_previous(img_seq, ts)
        img_seq = pd.clip_zero(img_seq)
        img_seq = pd.fix_corrupted_frames(img_seq)

        S = seq_to_S(img_seq, score_mode=SCORE_MODE)
        S_by_label[int(y)].append(S)

    # concat
    S_by_label = {
        y: np.concatenate(v)
        for y, v in S_by_label.items()
        if len(v) > 0
    }

    print("\n===== TRAIN LABEL-WISE OTSU THRESHOLDS =====\n")

    for y in sorted(S_by_label.keys()):
        S_all = S_by_label[y]

        T_otsu = log_otsu_threshold(S_all)
        std_S = float(np.std(S_all))
        T_final = T_otsu + K_STD * std_S

        print(f"Label {y}")
        print(f"  #frames       : {len(S_all)}")
        print(f"  Otsu threshold: {T_otsu:.6f}")
        print(f"  std(S)        : {std_S:.6f}")
        print(f"  Final T       : {T_final:.6f}")
        print()


if __name__ == "__main__":
    main()
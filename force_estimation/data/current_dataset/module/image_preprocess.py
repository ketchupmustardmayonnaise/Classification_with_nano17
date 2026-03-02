import argparse
from pathlib import Path
import json
import numpy as np
import h5py
from scipy import signal
from scipy.interpolate import interp1d

# -------------------------
# Frame cleaning (timestamps)
# -------------------------
def keep_strictly_increasing(ts: np.ndarray):
    """
    ts에서 strictly increasing이 되도록 프레임 인덱스를 남김.
    - ts[i+1] - ts[i] > 0 인 구간만 유지
    """
    ts = np.asarray(ts)
    if ts.ndim != 1 or ts.size == 0:
        return np.arange(ts.size, dtype=np.int64)
    keep = np.ones(ts.size, dtype=bool)
    d = np.diff(ts.astype(np.float64))
    keep[1:] = d > 0
    return np.where(keep)[0]


# -------------------------
# Image preprocessing
# -------------------------
def replace_bad_frame_with_previous(img, threshold=1000, return_mask=False):
    img = np.asarray(img).copy()

    # 프레임 단위로 "어느 픽셀이라도 threshold 이상이면 bad"
    bad = np.any(img >= threshold, axis=(1, 2))  # shape (A,)

    for t in range(len(bad)):
        if bad[t] and t > 0:
            img[t] = img[t - 1].copy()

    if return_mask:
        return img, bad
    return img


def replace_invalid_frame_with_previous(img, is_valid_frame):
    """
    is_valid_frame==False 이면 이전 프레임으로 덮어씀.
    (t=0에서 invalid면 그대로 둠)
    """
    img = np.asarray(img).copy()
    valid = np.asarray(is_valid_frame).astype(bool)
    if valid.ndim != 1 or valid.shape[0] != img.shape[0]:
        return img
    for t in range(len(valid)):
        if (not valid[t]) and t > 0:
            img[t] = img[t - 1].copy()
    return img


def clip_zero(img):
    return np.clip(img, a_min=0, a_max=None)


def to_3ch_rgb_like(img, img_ts):
    """
    (A,6,6) -> (A-2, 3, 6, 6)
    채널:
      ch0 = current
      ch1 = current - prev
      ch2 = (current - prev) - (prev - prev2)
    그리고 img_ts도 처음 2프레임 버린 것과 동일하게 2: 로 맞춤.
    """
    img = np.asarray(img)
    A = img.shape[0]
    if A < 3:
        return None, None

    out = []
    for t in range(2, A):
        cur = img[t]
        prev = img[t - 1]
        prev2 = img[t - 2]
        diff = cur - prev
        diff_prev = prev - prev2
        p_diff = diff - diff_prev
        frame = np.stack([cur, diff, p_diff], axis=0)  # (3,6,6)
        out.append(frame)

    out = np.stack(out, axis=0).astype(np.float32)     # (A-2,3,6,6)
    out_ts = np.asarray(img_ts)[2:].astype(np.float64) # (A-2,)
    return out, out_ts


def compute_channel_mean_std(seqs, max_frames=3000000):
    """
    train set에서 채널별 mean/std 계산 (streaming 비슷하게 단순 샘플링).
    seqs: list[np.ndarray], each (T,3,6,6)
    """
    # 프레임을 너무 많이 합치지 않도록 subsample
    all_frames = []
    total = 0
    for x in seqs:
        if x is None:
            continue
        T = x.shape[0]
        if total + T <= max_frames:
            all_frames.append(x)
            total += T
        else:
            remain = max_frames - total
            if remain <= 0:
                break
            all_frames.append(x[:remain])
            total += remain
            break

    if total == 0:
        raise RuntimeError("No frames to compute mean/std from train set")

    X = np.concatenate(all_frames, axis=0)  # (N,3,6,6)
    mean = X.mean(axis=(0, 2, 3))
    std = X.std(axis=(0, 2, 3))
    std = np.maximum(std, 1e-6)
    return mean.astype(np.float32), std.astype(np.float32)


def normalize_zscore(x, mean, std):
    # x: (T,3,6,6)
    mean = mean.reshape(1, 3, 1, 1)
    std = std.reshape(1, 3, 1, 1)
    return (x - mean) / std
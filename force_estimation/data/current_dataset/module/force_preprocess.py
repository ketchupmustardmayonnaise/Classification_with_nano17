import argparse
from pathlib import Path
import json
import numpy as np
import h5py
from scipy import signal
from scipy.interpolate import interp1d

# -------------------------
# Force preprocessing
# -------------------------

def butter_lowpass_filtfilt(force, force_ts, cutoff_hz=6.0, order=4):
    """
    force: (B,6)
    force_ts: (B,)
    샘플레이트는 300 Hz
    """
    force = np.asarray(force, dtype=np.float64)
    fs_dt = 300
    if not np.isfinite(fs_dt) or fs_dt <= 0:
        return force.astype(np.float32)

    fs = 1.0 / fs_dt
    nyq = 0.5 * fs
    wn = cutoff_hz / nyq
    wn = min(max(wn, 1e-6), 0.999999)

    b, a = signal.butter(order, wn, btype="low")
    out = np.zeros_like(force)
    for k in range(force.shape[1]):
        out[:, k] = signal.filtfilt(b, a, force[:, k])
    return out.astype(np.float32)


def interp_force_to_img_ts(force, force_ts, img_ts):
    """
    외삽은 하지 않도록, force_ts 범위 밖의 img_ts 프레임은 drop.
    return: force_interp (T,6), img_keep_idx (T,)
    """
    force_ts = np.asarray(force_ts, dtype=np.float64)
    img_ts = np.asarray(img_ts, dtype=np.float64)
    force = np.asarray(force, dtype=np.float32)

    if force_ts.size < 2 or img_ts.size < 1:
        return None, None

    t0, t1 = force_ts[0], force_ts[-1]
    keep = (img_ts >= t0) & (img_ts <= t1) & np.isfinite(img_ts)
    keep_idx = np.where(keep)[0]
    if keep_idx.size == 0:
        return None, None

    f = interp1d(force_ts, force, axis=0, kind="linear", bounds_error=False, fill_value=np.nan)
    y = f(img_ts[keep_idx])  # (T,6)
    # 혹시 NaN이 생기면 제거
    ok = np.all(np.isfinite(y), axis=1)
    keep_idx = keep_idx[ok]
    y = y[ok]
    if y.shape[0] == 0:
        return None, None
    return y.astype(np.float32), keep_idx.astype(np.int64)

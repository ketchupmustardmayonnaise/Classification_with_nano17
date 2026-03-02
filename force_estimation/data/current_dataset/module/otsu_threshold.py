import numpy as np

from .image_preprocess import *
from .utils import *
from .bodypart_tag import *

def otsu_threshold_1d(x: np.ndarray) -> float:
    """
    1D Otsu threshold (numpy only).
    x: 1D array
    """
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 10:
        return float(np.nan)

    # 히스토그램 기반 Otsu
    hist, bin_edges = np.histogram(x, bins=256)
    hist = hist.astype(np.float64)
    p = hist / np.sum(hist)

    omega = np.cumsum(p)  # class probabilities
    mu = np.cumsum(p * (bin_edges[:-1] + bin_edges[1:]) / 2.0)
    mu_t = mu[-1]

    # between-class variance
    denom = omega * (1.0 - omega)
    denom[denom == 0] = np.nan
    sigma_b2 = (mu_t * omega - mu) ** 2 / denom

    idx = int(np.nanargmax(sigma_b2))
    thr = (bin_edges[idx] + bin_edges[idx + 1]) / 2.0
    return float(thr)


def compute_otsu_kstd_by_bodypart(
    root: Path,
    subjects_in_train: list,
    img_bad_threshold: float,
    k: float,
):
    """
    bodypart별로 log1p(sum(img)) 분포를 모아서
    otsu + k*std_idle 계산.
    return: dict {bodypart_name: {"otsu":..., "std_idle":..., "thr":...}}
    """
    dist = {name: [] for name in TAG_INV.keys()}

    for subject, bodypart, npz_path in iter_npz(root):
        if subject not in subjects_in_train:
            continue
        if bodypart not in TAG_INV:
            continue

        img, img_ts, is_valid_frame, force, force_ts = load_one_npz(npz_path)

        # timestamp 역행 제거
        keep_idx = keep_strictly_increasing(img_ts)
        img = img[keep_idx]
        img_ts = img_ts[keep_idx]
        if is_valid_frame is not None:
            is_valid_frame = np.asarray(is_valid_frame)[keep_idx]

        # 네 현재 순서대로 image preprocess (3채널 변환 전, 1채널 기준)
        img = clip_zero(img)
        if is_valid_frame is not None:
            img = replace_invalid_frame_with_previous(img, is_valid_frame)
        img = replace_bad_frame_with_previous(img, threshold=img_bad_threshold)

        # 프레임별 sum -> log1p
        s = img.reshape(img.shape[0], -1).sum(axis=1).astype(np.float64)
        x = np.log1p(np.clip(s, 0, None))

        dist[bodypart].append(x)

    out = {}
    for bodypart, chunks in dist.items():
        if len(chunks) == 0:
            out[bodypart] = {"otsu": None, "std_idle": None, "thr": None}
            continue
        x = np.concatenate(chunks, axis=0)
        thr_otsu = otsu_threshold_1d(x)
        if not np.isfinite(thr_otsu):
            out[bodypart] = {"otsu": None, "std_idle": None, "thr": None}
            continue

        idle = x[x <= thr_otsu]
        std_idle = float(np.std(idle)) if idle.size else 0.0
        thr_final = float(thr_otsu + k * std_idle)

        out[bodypart] = {"otsu": np.expm1(float(thr_otsu)), "std_idle": np.expm1(std_idle), "thr": np.expm1(thr_final)}

    return out
import numpy as np
from collections import defaultdict
from preprocess import PreprocessDataset

def seq_to_S(X, score_mode="relu_sum"):
    if score_mode == "relu_sum":
        return np.maximum(X, 0).sum(axis=(1, 2))
    elif score_mode == "abs_sum":
        return np.abs(X).sum(axis=(1, 2))
    elif score_mode == "sum":
        return X.sum(axis=(1, 2))
    else:
        raise ValueError


def collect_S_by_bodypart(img_list, labels, score_mode="relu_sum"):
    S_by = defaultdict(list)

    for X, y in zip(img_list, labels):
        S = seq_to_S(X, score_mode=score_mode)
        S_by[y].append(S)

    # label별 concat
    S_by = {y: np.concatenate(v) for y, v in S_by.items()}
    return S_by


def otsu_threshold_1d(x, nbins=512):
    """
    x: 1D array-like (float)
    returns: otsu threshold in the same domain as x
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("Input has no finite values.")

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


def log_otsu_threshold(values, nbins=512, clip_negative=True):
    """
    values: python list / np.array of S values (>=0 권장)
    returns:
      T_original: threshold in original domain
      t_log: threshold in log1p(domain)
    """
    v = np.asarray(values, dtype=float)

    if clip_negative:
        v = np.maximum(v, 0.0)

    L = np.log1p(v)
    t_log = otsu_threshold_1d(L, nbins=nbins)
    T_original = float(np.expm1(t_log))
    return T_original, t_log


pd = PreprocessDataset()

path = 'dataset/bodypart_not_cross_person/all/train'
img, force, label, img_ts, force_ts, name = pd.load_data_from_npz(path)

for sequence_index in range(len(img)):
    img[sequence_index], img_ts[sequence_index] = pd.replace_bad_frame_with_previous(img[sequence_index],
                                                                                     img_ts[sequence_index])
    img[sequence_index] = pd.clip_zero(img[sequence_index])
    img[sequence_index] = pd.fix_corrupted_frames(img[sequence_index])

S_by = collect_S_by_bodypart(img, label, score_mode="relu_sum")

for index in range(len(S_by)):
    T, t_log = log_otsu_threshold(S_by[index+1], nbins=1024)
    print(f"{index+1}: log-domain otsu: {t_log}, original-domain threshold: {T}")
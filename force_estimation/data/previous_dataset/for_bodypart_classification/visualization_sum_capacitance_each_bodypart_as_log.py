import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from force_estimation.data.previous_dataset.preprocess import PreprocessDataset

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


def plot_each_bodypart_hist(S_by, bins=200, density=True, log=True):
    for y, S in sorted(S_by.items(), key=lambda kv: str(kv[0])):

        hist, edges = np.histogram(S, bins=bins, density=density)
        centers = (edges[:-1] + edges[1:]) / 2

        if log:
            Slog = np.log1p(S)  # log(1+S)
            plt.figure(figsize=(6, 4))
            plt.hist(Slog, bins=300, density=True)
            plt.xlabel("log1p(Sum of Capacitance)")
            plt.ylabel("Density")
            plt.title("Histogram of log1p(Sum of Capacitance)")
            plt.show()
        else:
            plt.figure(figsize=(6, 4))
            plt.hist(S, bins=300, density=True)
            plt.xlabel("Sum of Capacitance")
            plt.ylabel("Density")
            plt.title("Histogram of Sum of Capacitance (S)")
            plt.show()

pd = PreprocessDataset()

path = 'dataset/bodypart_cross_person/for_analyze'
img, force, label, img_ts, force_ts, name = pd.load_data_from_npz(path)

for sequence_index in range(len(img)):
    img[sequence_index], img_ts[sequence_index] = pd.replace_bad_frame_with_previous(img[sequence_index],
                                                                                     img_ts[sequence_index])
    img[sequence_index] = pd.clip_zero(img[sequence_index])
    img[sequence_index] = pd.fix_corrupted_frames(img[sequence_index])

S_by = collect_S_by_bodypart(img, label, score_mode="relu_sum")
print({k: v.shape for k, v in S_by.items()})  # label별 프레임 수 확인

plot_each_bodypart_hist(S_by, bins=200, density=True, log=True)
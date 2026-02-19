from preprocess import PreprocessDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import gaussian_kde
from matplotlib.animation import FuncAnimation


def compute_global_S_D_stats(img_list, score_mode="relu_sum"):
    all_S = []
    all_D = []

    for X in img_list:
        if score_mode == "relu_sum":
            S = np.maximum(X, 0).sum(axis=(1,2))
        elif score_mode == "abs_sum":
            S = np.abs(X).sum(axis=(1,2))
        elif score_mode == "sum":
            S = X.sum(axis=(1,2))
        else:
            raise ValueError

        D = np.zeros_like(S)
        if len(S) > 1:
            D[1:] = np.abs(S[1:] - S[:-1])

        all_S.append(S)
        all_D.append(D)

    all_S = np.concatenate(all_S)
    all_D = np.concatenate(all_D)

    stats = {
        "S_mean": all_S.mean(),
        "S_std": all_S.std(),
        "S_p05": np.percentile(all_S, 5),
        "S_p10": np.percentile(all_S, 10),
        "S_p20": np.percentile(all_S, 20),
        "S_p30": np.percentile(all_S, 30),
        "S_p40": np.percentile(all_S, 40),
        "S_p50": np.percentile(all_S, 50),
        "S_p90": np.percentile(all_S, 90),
        "D_mean": all_D.mean(),
        "D_std": all_D.std(),
        "D_p05": np.percentile(all_D, 5),
        "D_p10": np.percentile(all_D, 10),
        "D_p20": np.percentile(all_D, 20),
        "D_p50": np.percentile(all_D, 50),
        "D_p90": np.percentile(all_D, 90),
    }

    return stats


def make_idle_mask_simple(
    X,                 # (T,H,W)
    S_th,
    score_mode="relu_sum"
):
    if score_mode == "relu_sum":
        S = np.maximum(X, 0).sum(axis=(1,2))
    elif score_mode == "abs_sum":
        S = np.abs(X).sum(axis=(1,2))
    elif score_mode == "sum":
        S = X.sum(axis=(1,2))
    else:
        raise ValueError

    D = np.zeros_like(S)
    if len(S) > 1:
        D[1:] = np.abs(S[1:] - S[:-1])

    idle = (S <= S_th)

    return idle, S, D


def evaluate_idle_ratio(img_list, S_th, D_th):
    results = []

    for i, X in enumerate(img_list):
        idle, S, D = make_idle_mask_simple(X, S_th)

        T = len(S)
        results.append({
            "seq_idx": i,
            "T": T,
            "idle_frac": idle.sum() / T,
        })

    return pd.DataFrame(results)


import matplotlib.pyplot as plt

def plot_sequence_idle(
    X,
    S_th,
    seq_idx=None,
    score_mode="relu_sum"
):
    idle, S, D = make_idle_mask_simple(X, S_th, score_mode)

    T = len(S)
    t_axis = np.arange(T)

    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=False)

    # ---- S(t) time plot ----
    axs[0].plot(t_axis, S)
    axs[0].axhline(S_th, linestyle="--", label="S_th")
    axs[0].fill_between(t_axis, S.min(), S.max(), where=idle, alpha=0.2, label="idle")
    axs[0].set_ylabel("S(t)")
    axs[0].set_title(f"Sequence {seq_idx}" if seq_idx is not None else "Sequence")
    axs[0].legend()

    # ---- S histogram (bimodal check) ----
    x = S.astype(float)
    kde = gaussian_kde(x)  # bandwidth는 자동
    xs = np.linspace(x.min(), x.max(), 500)

    axs[1].plot(xs, kde(xs))
    axs[1].set_xlabel("S value")
    axs[1].set_ylabel("Density (KDE)")

    plt.tight_layout()
    plt.show()


def plot_multiple_sequences(
    img_list,
    S_th,
    D_th,
    n_examples=3,
    score_mode="relu_sum"
):
    for i in range(min(n_examples, len(img_list))):
        plot_sequence_idle(
            img_list[i],
            S_th,
            seq_idx=i,
            score_mode=score_mode
        )



pd = PreprocessDataset()

path = 'dataset/bodypart_cross_person/for_analyze'
img, force, label, img_ts, force_ts, name = pd.load_data_from_npz(path)

for sequence_index in range(len(img)):
    img[sequence_index], img_ts[sequence_index] = pd.replace_bad_frame_with_previous(img[sequence_index],
                                                                                     img_ts[sequence_index])
    img[sequence_index] = pd.clip_zero(img[sequence_index])
    img[sequence_index] = pd.fix_corrupted_frames(img[sequence_index])

stats = compute_global_S_D_stats(img)

S_th = stats["S_p40"]
D_th = stats["D_p90"]
print(S_th)

plot_multiple_sequences(img, S_th, D_th, n_examples=300)
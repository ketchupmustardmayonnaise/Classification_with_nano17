import argparse
from pathlib import Path
import json
import numpy as np
import h5py
from scipy import signal
from scipy.interpolate import interp1d

# -------------------------
# Utils: split by subject
# -------------------------
def split_subjects(subjects, seed=0, train_ratio=0.8, val_ratio=0.1):
    """
    subjects: list[str]
    return: train_subjects, val_subjects, test_subjects
    """
    rng = np.random.default_rng(seed)
    subjects = sorted(subjects)
    perm = rng.permutation(subjects)

    n = len(perm)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_train = min(max(n_train, 1), n - 2) if n >= 3 else max(n_train, 1)
    n_val = min(max(n_val, 1), n - n_train - 1) if n - n_train >= 2 else max(n_val, 0)

    train = perm[:n_train].tolist()
    val = perm[n_train:n_train + n_val].tolist()
    test = perm[n_train + n_val:].tolist()
    return train, val, test


def leave_out_split(subjects, test_subjects):
    """
    Multiple-subject leave-out:
    - test_subjects: list[str]
    return: remain_subjects, test_subjects
    """
    subjects = sorted(subjects)
    test_subjects = sorted(set(test_subjects))

    missing = [s for s in test_subjects if s not in subjects]
    if missing:
        raise ValueError(f"test_subjects not found: {missing} (available: {subjects})")

    remain = [s for s in subjects if s not in test_subjects]
    if len(remain) == 0:
        raise ValueError("All subjects are assigned to test. Need at least 1 subject for train/val.")

    return remain, test_subjects


def parse_subject_list(s: str):
    """
    "subject3,subject7 subject9" 같은 입력을 list[str]로 파싱
    """
    if s is None:
        return []
    # 콤마를 공백으로 바꾸고 split
    parts = s.replace(",", " ").split()
    return [p.strip() for p in parts if p.strip()]


# -------------------------
# IO: read dataset structure
# -------------------------
def iter_npz(root: Path):
    # Dataset/subject/bodypart/*.npz
    for subject_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        for body_dir in sorted([p for p in subject_dir.iterdir() if p.is_dir()]):
            for npz_path in sorted(body_dir.glob("*.npz")):
                yield subject_dir.name, body_dir.name, npz_path


def load_one_npz(npz_path: Path):
    d = np.load(npz_path, allow_pickle=False)
    # key 이름은 유연하게:
    img = d["img"]
    img_ts = d["img_ts"]
    force = d["force"]
    force_ts = d["force_ts"]
    is_valid_frame = d["is_valid_frame"] if "is_valid_frame" in d else (d["is_valid"] if "is_valid" in d else None)
    return img, img_ts, is_valid_frame, force, force_ts
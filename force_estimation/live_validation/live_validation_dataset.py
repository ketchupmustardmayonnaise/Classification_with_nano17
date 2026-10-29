# -*- coding: utf-8 -*-
"""실기기에서 수집한 npz 를 h5 데이터셋과 '완전히 같은 방식'으로 전처리해서
CapTSDataset 과 같은 모양의 sliding window 샘플을 내보내는 Dataset.

전처리는 build_h5_cross_subject.py 의 build_one_split() 을 그대로 따라감.
(timestamp 정리 -> clip_zero -> invalid/bad frame 교체 -> 3채널 변환 -> force filter/interp)
그래서 전처리 로직을 새로 쓰지 않고 current_dataset.module 의 함수들을 그대로 호출함.

정규화 mean/std 와 otsu threshold 는 fold 마다 다르므로 '그 fold 가 쓴 값'을 인자로 받음.
"""

import bisect
import re
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from force_estimation.data.current_dataset.module import (
    TAG_INV,
    load_one_npz,
    keep_strictly_increasing,
    clip_zero,
    replace_invalid_frame_with_previous,
    replace_bad_frame_with_previous,
    to_3ch_rgb_like,
    butter_lowpass_filtfilt,
    interp_force_to_img_ts,
)


# 실기기 앱이 저장한 파일 이름의 bodypart 표기 -> 학습 데이터셋의 bodypart 이름.
# npz 안에 같이 저장된 pred_bodypart(앱이 실시간으로 예측한 14-class id)로 검증했음.
LIVE_BODYPART_MAP = {
    "PAD(THUMB)":    "FINGERPAD-THUMB",          # 1
    "PAD(INDEX)":    "FINGERPAD-INDEX",          # 2
    "PAD(MIDDLE)":   "FINGERPAD-MIDDLE",         # 3
    "PALMER(ONE)":   "PALMARSIDEFINGERS-ONE",    # 4
    "PALMER(TWO)":   "PALMARSIDEFINGERS-TWO",    # 5
    "PALMER(THREE)": "PALMARSIDEFINGERS-THREE",  # 6
    "PALM":          "PALM",                     # 7
    "FIST(WRIST)":   "ROCKDOWN-WRISTSIDE",       # 8
    "FIST(PINKY)":   "ROCKDOWN-FINGERSIDE",      # 9
    "FIST(THUMB)":   "ROCKUP",                   # 10
    "MOUTH":         "MOUTH",                    # 11
    "EAR(PINNA)":    "EAR-PINNA",                # 12
    "EAR(HOLE)":     "EAR-HOLE",                 # 13
}

# 아래 두 형식을 모두 받음. 마지막 숫자는 수집 시각(timestamp).
#   0_subject1_EAR(HOLE)_step1_20260824105244.npz
#   0_subject15_EAR(HOLE)_20260831203218.npz
FILENAME_PATTERN = re.compile(
    r"^\d+_(?P<subject>subject\d+)_(?P<bodypart>.+?)(?:_step\d+)?_(?P<timestamp>\d{8,})$")


def parse_filename(npz_path):
    """npz 경로에서 (subject, bodypart 이름, 수집 시각) 을 뽑음."""

    matched = FILENAME_PATTERN.match(Path(npz_path).stem)
    if matched is None:
        raise ValueError(f"파일 이름 형식을 알 수 없습니다: {npz_path}")

    raw_bodypart = matched.group("bodypart")
    if raw_bodypart not in LIVE_BODYPART_MAP:
        raise KeyError(f"매핑되지 않은 bodypart 표기입니다: '{raw_bodypart}' ({npz_path})")

    return (matched.group("subject"), LIVE_BODYPART_MAP[raw_bodypart],
            int(matched.group("timestamp")))


def iter_npz(root, subdirs=None, takes=None):
    """root 아래의 npz 를 (subject, bodypart, path) 로 훑음.

    폴더 구조는 <root>/*.npz 와 <root>/<하위폴더>/*.npz 둘 다 지원함.

    subdirs : 사용할 하위 폴더 이름 목록. None 이면 전부.
    takes   : 사용할 회차 번호 목록(1부터). None 이면 전부.
              같은 (subject, bodypart) 를 여러 번 수집했을 때 수집 시각 순으로 회차를 매김.
    """

    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"폴더를 찾을 수 없습니다: {root}")

    if subdirs:
        paths = sorted(p for name in subdirs for p in (root / name).glob("*.npz"))
    else:
        paths = sorted(root.glob("*.npz")) or sorted(root.glob("*/*.npz"))

    if takes:
        wanted = set(takes)
        grouped = {}
        for path in paths:
            subject, bodypart, timestamp = parse_filename(path)
            grouped.setdefault((subject, bodypart), []).append((timestamp, path))

        keep = {path for items in grouped.values()
                for number, (_, path) in enumerate(sorted(items), start=1) if number in wanted}
        paths = [p for p in paths if p in keep]

    if not paths:
        raise FileNotFoundError(f"npz 파일이 없습니다: {root} (subdirs={subdirs}, takes={takes})")

    for path in paths:
        subject, bodypart, _ = parse_filename(path)
        yield subject, bodypart, path


def read_h5_norm_stats(h5_path):
    """CapTSDataset 이 읽는 것과 같은 위치(h5 attrs)에서 img_mean / img_std 를 읽음."""

    with h5py.File(h5_path, "r") as f:
        return (np.asarray(f.attrs["img_mean"], dtype=np.float32),
                np.asarray(f.attrs["img_std"], dtype=np.float32))


def preprocess_npz(npz_path, bodypart, otsu_info,
                   img_bad_threshold=1000.0, cutoff_hz=6.0, filter_order=4):
    """npz 하나를 h5 에 저장되던 것과 똑같은 상태(img/force/label)까지 전처리함.

    못 쓰는 시퀀스면 None 을 돌려줌.
    """

    img, img_ts, is_valid_frame, force, force_ts = load_one_npz(Path(npz_path))

    # (0) image timestamp 역행/중복 프레임 제거
    keep_idx = keep_strictly_increasing(img_ts)
    img, img_ts = img[keep_idx], img_ts[keep_idx]
    if is_valid_frame is not None:
        is_valid_frame = np.asarray(is_valid_frame)[keep_idx]

    # (0b) force timestamp 역행/중복 샘플 제거
    keep_f = keep_strictly_increasing(force_ts)
    force, force_ts = force[keep_f], force_ts[keep_f]

    # image preprocess
    img = clip_zero(img)
    if is_valid_frame is not None:
        img = replace_invalid_frame_with_previous(img, is_valid_frame)
    img = replace_bad_frame_with_previous(img, threshold=img_bad_threshold)

    img3, img3_ts = to_3ch_rgb_like(img, img_ts)
    if img3 is None or img3.shape[0] < 1:
        return None

    # force preprocess: filter -> interpolate -> z-clip
    force = butter_lowpass_filtfilt(force, force_ts, cutoff_hz=cutoff_hz, order=filter_order)
    force_interpolated, keep_img_idx = interp_force_to_img_ts(force, force_ts, img3_ts)
    if force_interpolated is None:
        return None

    # interpolate 에서 drop 된 프레임에 맞춰 img 도 동기화
    img3 = img3[keep_img_idx]
    force_interpolated[:, 2] = np.minimum(force_interpolated[:, 2], 0.0)

    # 프레임마다 idle(0) / bodypart(1~13) 레이블.
    # img3 의 첫 번째 채널 frame-wise sum 이 그 bodypart 의 thr_final 을 넘으면 bodypart.
    threshold = otsu_info.get(bodypart, {}).get("thr_final")
    if threshold is None:
        raise Exception(f"No threshold found for bodypart: {bodypart}")

    img_sum = img3[:, 0].reshape(img3.shape[0], -1).sum(axis=1)
    label = np.where(img_sum > float(threshold), int(TAG_INV[bodypart]), 0).astype(np.int32)

    return {"img": img3, "force": force_interpolated, "label": label,
            "frame": int(img3.shape[0]), "bodypart": bodypart, "source_npz": str(npz_path)}


class LiveValidationDataset(Dataset):
    """수집한 npz 전체를 sliding window 로 훑는 test 용 Dataset.

    CapTSDataset 과 동일하게
      - window 는 length_of_window 로 뜨고 stride = 1
      - label 은 window 의 '마지막' 프레임 label
      - 정규화는 (img - img_mean) / (img_std + 1e-8)
    """

    def __init__(self, root, otsu_info, img_mean, img_std, length_of_window,
                 subdirs=None, takes=None):

        self.length_of_window = int(length_of_window)
        self.img_mean = torch.tensor(img_mean).float().view(1, -1, 1, 1)
        self.img_std = torch.tensor(img_std).float().view(1, -1, 1, 1)

        self.sequences, self.subjects = [], []
        skipped = []

        for subject, bodypart, npz_path in iter_npz(root, subdirs=subdirs, takes=takes):
            sequence = preprocess_npz(npz_path, bodypart, otsu_info)

            if sequence is None or sequence["frame"] < self.length_of_window:
                skipped.append(str(npz_path))
                continue

            sequence["subject"] = subject
            self.sequences.append(sequence)
            self.subjects.append(subject)

        if not self.sequences:
            raise RuntimeError(f"사용할 수 있는 시퀀스가 없습니다: {root}")

        # 각 시퀀스에서 뜰 수 있는 window 개수 -> 누적합으로 index 매핑
        self.prefix = [0]
        for sequence in self.sequences:
            self.prefix.append(self.prefix[-1] + sequence["frame"] - self.length_of_window + 1)

        print(f"[LiveValidationDataset] sequences={len(self.sequences)}, "
              f"windows={self.prefix[-1]}, window={self.length_of_window}")
        for path in skipped:
            print(f"  [skip] {path}")

    def __len__(self):
        return self.prefix[-1]

    def __getitem__(self, index):
        sequence_idx = bisect.bisect_right(self.prefix, index) - 1
        start = index - self.prefix[sequence_idx]
        end = start + self.length_of_window
        sequence = self.sequences[sequence_idx]

        img = torch.from_numpy(sequence["img"][start:end]).float()
        img = (img - self.img_mean) / (self.img_std + 1e-8)

        return {
            "img": img,                                                      # (T,3,6,6)
            "force": torch.from_numpy(sequence["force"][start:end, 0:3]).float(),   # (T,3)
            "label": sequence["label"][end - 1],        # window 마지막 프레임의 레이블
            "bodypart": sequence["bodypart"],
            "sequence_id": sequence_idx,
            "subject_id": sequence["subject"],
        }

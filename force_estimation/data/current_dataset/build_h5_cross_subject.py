from force_estimation.data.current_dataset.module import *


# 랜덤 split
# python build_h5_cross_subject.py --root C:\Users\Jeongmin\PycharmProjects\Classification_with_nano17\force_estimation\data\dataset\12_dataset
# --out_dir C:\Users\Jeongmin\PycharmProjects\Classification_with_nano17\force_estimation\data\dataset\12_dataset\h5 --split random --seed 0 --train_ratio 0.8 --val_ratio 0.1

# leave-out
# python build_h5_cross_subject.py
# --root C:\Users\Jeongmin\PycharmProjects\Classification_with_nano17\force_estimation\data\current_dataset\12_dataset --out_dir C:\Users\Jeongmin\PycharmProjects\Classification_with_nano17\force_estimation\data\current_dataset\dataset\12_dataset_h5
# --split loso --test_subjects "subject10 subject11 subject12" --val_subjects "subject5" --otsu_k 1.0


# -------------------------
# Main build
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Dataset root folder")
    ap.add_argument("--out_dir", type=str, default="out_h5", help="output folder")
    ap.add_argument("--split", type=str, default="random", choices=["random", "loso"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_subjects", type=str, default=None,
                    help="for loso: comma/space separated. e.g., 'subject3,subject7 subject9'")
    ap.add_argument("--val_subjects", type=str, default=None,
                    help="optional: comma/space separated. e.g., 'subject1 subject2'")
    ap.add_argument("--img_bad_threshold", type=float, default=1000.0)
    ap.add_argument("--cutoff_hz", type=float, default=6.0)
    ap.add_argument("--filter_order", type=int, default=4)
    ap.add_argument("--otsu_k", type=float, default=0.0, help="final_thr = otsu + k*std_idle (log1p(sum(img)))")
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


    # Dataset 폴더 내에서 각 subject별로 정리가 되어 있어야 함.
    subjects = sorted([p.name for p in root.iterdir() if p.is_dir()])
    if len(subjects) == 0:
        raise RuntimeError(f"No subject folders found in {root}")


    # ===========================================
    # ---- split 결정 ----
    val_manual = parse_subject_list(args.val_subjects)

    if args.split == "random":
        # random은 기존처럼 train/val/test를 한 번에 뽑되,
        # val_subjects가 있으면 val을 고정하고 나머지를 재계산한다.
        train_subj, val_subj, test_subj = split_subjects(
            subjects, seed=args.seed, train_ratio=args.train_ratio, val_ratio=args.val_ratio
        )

    else:
        # loso: test는 사용자가 지정
        test_list = parse_subject_list(args.test_subjects)
        if len(test_list) == 0:
            raise ValueError("--test_subjects is required for --split loso (can be multiple subjects)")
        remain, test_subj = leave_out_split(subjects, test_subjects=test_list)

        # remain 중에서, val_subjects 지정되면 그걸 val로 고정
        if len(val_manual) > 0:
            val_set = set(val_manual)

            # 존재하지 않는 subject가 있는가?
            missing = sorted([s for s in val_set if s not in subjects])
            if missing:
                raise ValueError(f"--val_subjects contains unknown subjects: {missing}")

            # test dataset과 겹치는 subject가 있는가?
            overlap = sorted(val_set & set(test_subj))
            if overlap:
                raise ValueError(f"--val_subjects overlaps with test subjects: {overlap}")

            # remain 중에서 val subject와 train subject를 나눔.
            val_subj = sorted(val_set)
            train_subj = sorted([s for s in remain if s not in val_set])
            if len(train_subj) == 0:
                raise ValueError("Train subjects became empty after applying --val_subjects.")

        else:
            # val_subjects 미지정이면: remain에서 val_ratio로 랜덤 선택
            rng = np.random.default_rng(args.seed)
            remain = sorted(remain)
            perm = rng.permutation(remain)

            n_remain = len(perm)

            # 만약 remain으로 남은 1명이라면
            if n_remain == 1:
                # train만 1명, val은 비움(어쩔 수 없음)
                train_subj, val_subj = perm.tolist(), []
            else:
                n_val = max(1, int(round(n_remain * args.val_ratio)))
                n_val = min(n_val, n_remain - 1)  # train 최소 1명 남기기
                val_subj = perm[:n_val].tolist()
                train_subj = perm[n_val:].tolist()

    # ---- final sanity check ----
    assert len(set(train_subj) & set(val_subj)) == 0
    assert len(set(train_subj) & set(test_subj)) == 0
    assert len(set(val_subj) & set(test_subj)) == 0

    split_info = {"train": train_subj, "val": val_subj, "test": test_subj}
    (out_dir / "split.json").write_text(json.dumps(split_info, indent=2), encoding="utf-8")
    print("[Split]", split_info)
    # ===========================================


    # ---- Otsu + k*std thresholds (by bodypart, using TRAIN subjects only) ----
    otsu_info = compute_otsu_kstd_by_bodypart(
        root=root,
        subjects_in_train=train_subj,
        img_bad_threshold=args.img_bad_threshold,
        k=args.otsu_k,
    )
    (out_dir / "otsu_thresholds.json").write_text(json.dumps(otsu_info, indent=2), encoding="utf-8")
    print("[Otsu] wrote:", out_dir / "otsu_thresholds.json")
    # -----------------------------------------------------------------------



    # 1) 1차 패스: train만 모아서 3채널 변환까지 한 후 mean/std 계산
    train_seqs_for_stats = []
    train_meta = []

    for subject, bodypart, npz_path in iter_npz(root):
        if subject not in train_subj:
            continue

        img, img_ts, is_valid_frame, force, force_ts = load_one_npz(npz_path)

        # (0) timestamp 역행/중복 프레임 제거
        keep_idx = keep_strictly_increasing(img_ts)
        img = img[keep_idx]
        img_ts = img_ts[keep_idx]
        if is_valid_frame is not None:
            is_valid_frame = np.asarray(is_valid_frame)[keep_idx]

        # (1) 음수는 0으로
        img = clip_zero(img)

        # (2) invalid 프레임(is_valid_frame False) -> 이전 프레임 복사
        if is_valid_frame is not None:
            img = replace_invalid_frame_with_previous(img, is_valid_frame)

        # (3) 큰 값 프레임 -> 이전 프레임 복사 (+ 카운트)
        img, bad_mask = replace_bad_frame_with_previous(img, threshold=args.img_bad_threshold, return_mask=True)

        # (4) 3채널 변환 (앞 2프레임 drop)
        img3, img3_ts = to_3ch_rgb_like(img, img_ts)
        if img3 is None:
            continue

        train_seqs_for_stats.append(img3)
        train_meta.append((subject, bodypart, str(npz_path)))

    mean, std = compute_channel_mean_std(train_seqs_for_stats)
    stats = {"img_mean": mean.tolist(), "img_std": std.tolist()}
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print("[Stats] mean:", mean, "std:", std)



    # 2) 2차 패스: train/val/test 각각 H5 생성
    def build_one_split(split_name, subjects_in_split):
        h5_path = out_dir / f"{split_name}.h5"
        n_saved = 0
        str_dt = h5py.string_dtype(encoding="utf-8")

        with h5py.File(h5_path, "w") as h5f:
            h5f.attrs["split_name"] = split_name
            h5f.attrs["subjects"] = json.dumps(subjects_in_split)
            h5f.attrs["img_mean"] = mean
            h5f.attrs["img_std"] = std
            h5f.attrs["cutoff_hz"] = float(args.cutoff_hz)
            h5f.attrs["filter_order"] = int(args.filter_order)

            seq_idx = 0
            for subject, bodypart, npz_path in iter_npz(root):
                if subject not in subjects_in_split:
                    continue

                img, img_ts, is_valid_frame, force, force_ts = load_one_npz(npz_path)

                # (0) timestamp 역행/중복 프레임 제거
                keep_idx = keep_strictly_increasing(img_ts)
                img = img[keep_idx]
                img_ts = img_ts[keep_idx]
                if is_valid_frame is not None:
                    is_valid_frame = np.asarray(is_valid_frame)[keep_idx]

                # image preprocess
                img = clip_zero(img)
                if is_valid_frame is not None:
                    img = replace_invalid_frame_with_previous(img, is_valid_frame)
                img = replace_bad_frame_with_previous(img, threshold=args.img_bad_threshold)

                img3, img3_ts = to_3ch_rgb_like(img, img_ts)
                if img3 is None or img3.shape[0] < 1:
                    continue

                # force preprocess: filter -> interpolate -> z-clip
                force = butter_lowpass_filtfilt(force, force_ts, cutoff_hz=args.cutoff_hz, order=args.filter_order)
                force_interpolated, keep_img_idx = interp_force_to_img_ts(force, force_ts, img3_ts)
                if force_interpolated is None: continue

                # interpolate에서 drop된 프레임에 맞춰 img도 동기화
                img3, img3_ts = img3[keep_img_idx], img3_ts[keep_img_idx]

                # z force clipping (>=0 -> 0)
                force_interpolated[:, 2] = np.minimum(force_interpolated[:, 2], 0.0)


                # ==============================================
                # NEW: per-frame label (idle vs bodypart)
                # 조건: img3의 첫 번째 채널 sum이 thr(bodypart)를 넘으면 bodypart, 아니면 idle
                # ==============================================
                idle_label_id = 0

                # bodypart별 thr 가져오기 (compute_otsu_kstd_by_bodypart 결과의 "thr" 사용)
                bodypart_info = otsu_info.get(bodypart, {})
                threshold = bodypart_info.get("thr_final", None)

                # 첫 번째 채널 기준 frame-wise sum
                s_sum = img3[:, 0].reshape(img3.shape[0], -1).sum(axis=1).astype(np.float64)

                if threshold is None: raise Exception(f"No threshold found for bodypart: {bodypart}")
                else:
                    threshold = float(threshold)
                    label_frame = np.where(
                        s_sum > threshold,
                        MERGED_MAP[int(TAG_INV[bodypart])],
                        idle_label_id
                    ).astype(np.int32)
                # ==============================================

                # save
                g = h5f.create_group(f"sequence_{seq_idx:04d}")
                g.create_dataset("img", data=img3, compression="gzip", compression_opts=9)      # (T,3,6,6)
                g.create_dataset("force", data=force_interpolated, compression="gzip", compression_opts=9) # (T,6)
                g.create_dataset("label", data=label_frame, compression="gzip", compression_opts=9)  # (T,)

                g.attrs["subject"] = subject
                g.attrs["bodypart"] = bodypart
                g.attrs["source_npz"] = str(npz_path)
                g.attrs["frame"] = img3.shape[0]
                # --------------------------------

                seq_idx += 1
                n_saved += 1

        print(f"[{split_name}] saved sequences: {n_saved} -> {h5_path}")
        return h5_path

    build_one_split("train", train_subj)
    build_one_split("val", val_subj)
    build_one_split("test", test_subj)

    print("[DONE] wrote split.json, stats.json, train/val/test h5")


if __name__ == "__main__":
    main()
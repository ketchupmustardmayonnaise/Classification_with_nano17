import zipfile
import numpy as np
import os
import io
import random
from pathlib import Path

# ======================================
# configuration
input_folder = Path("original_data/bodypart_except_jmh")   # zip 파일들이 들어 있는 폴더
output_folder = Path("output") # train/test/val zip 저장 폴더
output_folder.mkdir(exist_ok=True)

bodypart = ['idle', 'finger', 'palmar_side_finger', 'palm', 'rock down', 'rock up', 'mouth', 'ear']
# ======================================

for lab in os.listdir(input_folder):
    # ----- 1. 모든 zip 내부 npz 파일 경로 수집 -----
    all_data = []  # [(zip_path, inner_name_bytes), ...]

    for zip_path in input_folder.glob(f"{lab}/*.zip"):
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for name in zf.namelist():
                if name.endswith(".npz"):
                    all_data.append((zip_path, name))

    print(f"총 {len(all_data)}개의 npz 파일을 찾았습니다.")

    # ----- 2. 섞기 -----
    random.shuffle(all_data)

    # ----- 3. 8:2으로 분할 -----
    n = len(all_data)
    n_train = int(n * 0.8)
    n_val = n - n_train

    splits = {
        "train": all_data[:n_train],
        "val": all_data[n_train:],
    }

    # ----- 4. 각각 zip으로 저장 -----
    for split_name, data_list in splits.items():
        out_zip_path = output_folder / f"{bodypart[int(lab)]}_{split_name}.zip"
        with zipfile.ZipFile(out_zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as out_zip:
            for zip_path, inner_name in data_list:
                # 원본 zip 이름 (확장자 제거)
                base_name = zip_path.stem
                # 원본 zip 내 파일 index
                idx = Path(inner_name).stem
                # 새 이름
                new_name = f"{base_name}_{idx}.npz"

                # 원본 zip에서 npz 데이터를 읽어서 새 zip에 추가
                with zipfile.ZipFile(zip_path, 'r') as src_zip:
                    with src_zip.open(inner_name) as f:
                        file_data = f.read()
                        out_zip.writestr(new_name, file_data)

        print(f"{split_name}.zip 생성 완료 ({len(data_list)}개 파일 포함)")

print("✅ 모든 zip 파일이 성공적으로 생성되었습니다.")
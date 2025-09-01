import zipfile
import random
import os
current_path = os.getcwd()
print(current_path)

# 원본 ZIP 경로
src_zip_path = f'{current_path}\\split'  # 여기에 원래 zip 경로 넣기

for file_name in os.listdir(src_zip_path):  # 각 파일들에 대하여
    if file_name.endswith('.zip'):
        with zipfile.ZipFile(src_zip_path + '/' + file_name, 'r') as zf:  # 각 zip파일들에 대하여
            print("Exists:", os.path.exists(src_zip_path))
            print("Readable:", os.access(src_zip_path, os.R_OK))
            print("Writable:", os.access(src_zip_path, os.W_OK))

            # 출력 ZIP 파일 이름들
            output_zips = {
                'train': f'TRAIN_{file_name}.zip',
                'val': f'VAL_{file_name}.zip',
                'test': f'TEST_{file_name}.zip'
            }

            # Step 1: 원본 ZIP에서 npz 파일 리스트 가져오기
            npz_files = [name for name in zf.namelist() if name.endswith('.npz')]

            # Step 2: 섞고 나누기
            random.shuffle(npz_files)  # 랜덤 셔플
            n_total = len(npz_files)
            n_train = int(n_total * 0.7)
            n_val = int(n_total * 0.2)
            n_test = n_total - n_train - n_val  # 나머지

            split_dict = {
                'train': npz_files[:n_train],
                'val': npz_files[n_train:n_train + n_val],
                'test': npz_files[n_train + n_val:]
            }

            # Step 3: 각각을 새 zip 파일로 저장
            for split_name, file_list in split_dict.items():
                with zipfile.ZipFile(output_zips[split_name], 'a') as out_zip:
                    for fn in file_list:
                        # ZIP 내 파일을 읽고 그대로 새 ZIP에 쓰기
                        with zf.open(fn) as source_file:
                            data = source_file.read()
                            out_zip.writestr(fn, data)

            print(f"✅ 분할 완료: {file_name} train.zip, val.zip, test.zip 생성됨.")
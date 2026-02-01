import os
import numpy as np
import zipfile

target_dir = 'original_data/bodypart_adjusted'

# ! ================================ !
# 변수 생성
# ! ================================ !
img = []
force = []

img_ts = []
force_ts = []

# ! ================================ !
# frequency 측정
# ! ================================ !

def measure_frequency(timestamps):
    sum_of_intervals = 0.0
    sum_of_frame_force = 0

    diff = []
    diff_seq = []

    for seq in timestamps:
        interval = np.mean(np.diff(seq))
        diff_seq.append(np.diff(seq))
        diff.extend(np.diff(seq).tolist())
        sum_of_frame_force += len(seq) - 1
        sum_of_intervals += interval * (len(seq) - 1)

    diff = np.array(diff)
    count20 = np.sum(diff == 20)
    count20m = np.sum(diff == -20)
    count10 = np.sum(diff >= 10)
    count10m = np.sum(diff <= -0)

    mean_interval = sum_of_intervals / sum_of_frame_force
    frequency = 1 / mean_interval
    return frequency

# ! ================================ !
# 데이터 불러오기
# ! ================================ !

# 폴더 존재 여부 확인
if not os.path.exists(target_dir):
    print(f"오류: '{target_dir}' 경로를 찾을 수 없습니다.")

print(f"--- '{target_dir}' 내의 ZIP 파일 탐색 시작 ---")

# os.walk는 하위 폴더를 모두 순회할 때까지 내부적으로 반복
for root, dirs, files in os.walk(target_dir):
    for file in files:
        # 확장자가 .zip인 파일만 필터링
        if file.lower().endswith('.zip'):
            full_path = os.path.join(root, file)

            with zipfile.ZipFile(full_path, 'r') as zf:  # 각 zip파일들에 대하여
                for npzfile in zipfile.ZipFile(full_path, 'r').namelist(): # 각 npz파일들에 대하여
                    with zf.open(npzfile) as nf:
                        data = np.load(nf)
                        img.append(data['img'])
                        force.append(data['force'])
                        img_ts.append(data['img_ts'])
                        force_ts.append(data['force_ts'])


# ! ================================ !
# 데이터 불러오기
# ! ================================ !

print(measure_frequency(img_ts))
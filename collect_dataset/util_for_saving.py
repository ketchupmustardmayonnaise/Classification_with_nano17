import os
import io
import config
import zipfile
import numpy as np


# capacitance 이미지 입장에서, force와 timestamp가 얼마나 겹치는지 확인
def check_overlap_percentage(img_ts, force_ts):
    overlap_start = max(img_ts[0], force_ts[0])
    overlap_end = min(img_ts[-1], force_ts[-1])

    overlap_duration = max(0, (overlap_end - overlap_start))
    total_duration_of_cap = img_ts[-1] - img_ts[0]

    if (overlap_duration / total_duration_of_cap) < 0.99:
        return False
    else:
        return True


# force timestamp가 image timestamp과 완전히 동일 시간에 수집된 데이터인지 확인함
# 이때, force timestamp는 한번 필터링된 timestamp여야 함.
def is_valid_timestamp(img_timestamp, force_timestamp):

    # force가 image보다 시작이 늦다면
    if force_timestamp[0] > img_timestamp[0]:
        return False

    # force가 image보다 끝이 빠르다면
    if force_timestamp[-1] < img_timestamp[-1]:
        return False

    # 만약 이미지 시퀀스와 force 시퀀스 조금이라도 안 맞는다면
    if not check_overlap_percentage(img_timestamp, force_timestamp):
        return False

    return True


# force_timestamp에서, time에 근접한 index를 가져옴.
def find_closest_datetime(force_timestamp, time, is_search_of_first=None):
    # 맨 처음 프레임의 인덱스를 뽑고 싶은데, force가 이미지 저장 이전에 모두 기록됐을 때
    if is_search_of_first and force_timestamp[-1] < time : return None
    # 마지막 프레임의 인덱스를 뽑고 싶은데, force가 이미지 다 저장한 이후부터로 기록됐을 때
    if (not is_search_of_first) and force_timestamp[0] > time : return None

    idx = np.searchsorted(force_timestamp, time)

    if idx == 0: return 0
    if idx == len(force_timestamp): return len(force_timestamp)
    else:
        if is_search_of_first: return idx - 1
        elif not is_search_of_first: return idx


# image sequence와 force sequence를 받으면 이를 저장함.
# image_of_sequence와 force_of_sequence 모두, (timestamp, data) 가 함께 저장되어 있음.
def save_image_and_force(image_data, image_timestamp, force_data, force_timestamp, dir, name, index, bodypart, date):

    # 이미지 시작 & 끝 time 가져오기
    image_start_time = image_timestamp[0]
    image_end_time = image_timestamp[-1]

    # 만약 길이가 맞지 않는다면
    if len(force_timestamp) != len(force_data) \
            or len(image_timestamp) != len(image_data):
        print("try again(1)")
        return False

    # 이미지 시작과 끝을 force timestamp에서 index로 찾음
    force_start_index = find_closest_datetime(force_timestamp, image_start_time, is_search_of_first=True)
    force_end_index = find_closest_datetime(force_timestamp, image_end_time, is_search_of_first=False)


    # !! ================ 1차검증 ================ !!
    # 만약 이미지 시작 & 끝을 force timestamp에서 못 찾는다면
    if force_start_index == None or force_end_index == None:
        print("try again(2)")
        return False
    # !! ===================================== !!


    # 안전한 수집을 위하여, 파악한 데이터보다 더 길게 저장하도록 함
    if force_start_index > 0: force_start_index -= 1
    if force_end_index < len(force_timestamp) - 1: force_end_index += 1

    # force를 numpy로 저장
    temp_ft = np.copy(force_timestamp)
    temp_f = np.copy(force_data)
    force_timestamp = temp_ft[force_start_index: force_end_index]
    force = temp_f[force_start_index: force_end_index]


    # !! ================ 2차검증 ================ !!
    # force timestamp가 제대로 모아진 데이터인지 확인함
    if not is_valid_timestamp(image_timestamp, force_timestamp):
        print("try again(3)")
        print(image_timestamp[0], force_timestamp[0])
        print(image_timestamp[-1], force_timestamp[-1])
        return False
    # !! ===================================== !!

    # !! ================ LABELING ================ !!
    print("레이블을 선택하세요")
    while True:
        try:
            label = config.LABEL[
                int(input("((1) PRESS (2) SHEAR_1LV (3) SHEAR_2LV (4) SHEAR_3LV (5) FREE (0) DISCARD): "))]
            break
        except:
            print()

    # !! ===================================== !!

    # 폴더 생성 후 저장
    if not os.path.isdir(dir): os.mkdir(dir)
    zipfiledir = f"{dir}/{name}_{bodypart}_{date}.zip"

    mem = io.BytesIO()
    np.savez(mem, img=image_data, img_ts=image_timestamp, force=force, force_ts=force_timestamp)

    with zipfile.ZipFile(zipfiledir, 'a') as zf:
        zf.writestr(f'{index}_{label}.npz', mem.getvalue())

    print("🔴 {}번째 데이터가 저장되었습니다 (이미지 프레임: {}, 힘 프레임: {})\n"
          .format(index, image_data.shape[0], len(force)))

    return True
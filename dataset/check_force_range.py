import os
import ast
import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
from scipy import signal

img = [ [] for i in range(8) ]
force = [[] for i in range(8)]
z_force = [ [] for i in range(8) ]

img_timestamp = [[] for i in range(8)]
force_timestamp = [[] for i in range(8)]

path = './data/f'

def break_images_smaller(original_img, original_force,
                         target_indices=1,
                         stride=1):
    '''
    frame을 target_indices로 쪼갬 (image, shear 모두 적용됨)

    만약 원본 프레임이 30 frame이고, 이를 8 frame씩 쪼갠다면
    (30 - 8) + 1 만큼으로 쪼개짐

    (original_indices - target_indices) + 1
    '''
    img = [] # temporary image list
    force = []

    for index in range(len(original_img)):
        frame = 0
        while frame + target_indices < len(original_img[index]):
            img.append(original_img[index][frame: frame + target_indices])
            temp_force = original_force[index][frame : frame + target_indices]

            force.append(temp_force[-1])
            frame += stride

    return img, np.array(force)

def interpolate_nano_to_capacitance(force_data, capacitance_timestamp, force_timestamp):
    result = []
    for index in range(len(force_timestamp)):
        # 보간 함수 생성 (외삽은 금지 - 범위 밖이면 NaN 발생)
        interp_func = interp1d(force_timestamp[index], force_data[index], kind='linear', fill_value='extrapolate', axis=0)

        # sensor1의 타임스탬프에 대해 sensor2 값 보간
        r = interp_func(capacitance_timestamp[index])
        result.append(r)

    return result

def apply_butterworth_filter(force):
    result = []

    nyq = 0.5 * 29
    cutoff = 10

    for index in range(len(force)):
        temp_force = np.array(force[index])
        b, a = signal.butter(2, cutoff / nyq, 'low')
        data = []
        for sig in range(6):
            r = signal.filtfilt(b, a, temp_force[:, sig])
            data.append(r)

        result.append(np.array(data).T)

    return result

for label in os.listdir(path):
    for file_name in os.listdir(path + '/' + label):  # 각 파일들에 대하여
        if file_name.endswith('.csv'):  # 파일 확장자가 csv인 경우만 처리
            file_path = os.path.join(path + '/' + label, file_name)

            file = open(file_path, 'r')
            lines = file.readlines()
            lines = [lst for lst in lines if lst != '\n' or lst != []]

            line_index = 0
            while line_index < len(lines):
                lines[line_index] = lines[line_index].split('"')
                lines[line_index] = [item for item in lines[line_index] if item.strip()]

                temp_list = []
                for item in lines[line_index]:
                    if item.startswith("[") and item.endswith("]"):
                        temp_list.append(ast.literal_eval(item))
                    else:
                        temp = item.split(",")
                        temp_list.extend([item for item in temp if item != ""])
                lines[line_index] = temp_list

                # 아무것도 기록되지 않은 줄이라면
                if len(lines[line_index]) == 0:
                    del lines[line_index]
                    continue
                elif lines[line_index][0] == '\n':
                    del lines[line_index]
                    continue

                frame = int(lines[line_index][0])
                img_timestamp[int(label)].append(list(map(float, lines[line_index][1: 2 + 2 * (frame - 1): 2])))
                img_temp = list(lines[line_index][2: 2 + 2 * frame: 2])
                img_temp = [
                    [0 if x >= 65000 else x for x in inner]
                    for inner in img_temp
                ]
                img[int(label)].append(np.array(img_temp, dtype=np.float32).reshape((frame, 10, 8)))

                if any('fdx:' in item for item in lines[line_index]):
                    force_timestamp[int(label)].append(list(map(float, lines[line_index][1 + 2 * frame: -1: 2])))
                    force_temp = list(lines[line_index][2 + 2 * frame: -1: 2])
                    force[int(label)].append(force_temp)
                else:
                    force_timestamp[int(label)].append(list(map(float, lines[line_index][1 + 2 * frame:: 2])))
                    force_temp = list(lines[line_index][2 + 2 * frame:: 2])
                    force[int(label)].append(force_temp)

                z_force[int(label)].extend(np.array(force_temp)[:, 2])
                line_index += 1


norm = [[] for i in range(8)]
std = [[] for i in range(8)]
max = [[] for i in range(8)]

df = []

for lab in range(8):
    force[lab] = apply_butterworth_filter(force[lab])
    force[lab] = interpolate_nano_to_capacitance(force[lab], img_timestamp[lab], force_timestamp[lab])
    img[lab], force[lab] = break_images_smaller(img[lab], force[lab])
    force[lab] = force[lab][:, 2]

    norm[lab].extend(np.mean(np.array(img[lab]).reshape((len(img[lab]), -1)), axis=1))
    std[lab].extend(np.std(np.array(img[lab]).reshape((len(img[lab]), -1)), axis=1))
    max[lab].extend(np.max(np.abs(np.array(img[lab]).reshape((len(img[lab]), -1))), axis=1))

    df.append(pd.DataFrame({
        'norm':norm[lab],
        'std':std[lab],
        'max':max[lab],
        'force':force[lab]
    }))

    print(df[lab].corr())
    print()

for label in range(8):
    print(f"{label}번째 데이터\t평균:{np.mean(z_force[label]):2f}\tSTD:{np.std(z_force[label]):2f}\tABS MAX:{np.max(np.abs(z_force[label])):2f}\tMAX:{np.max(z_force[label]):2f}")
'''
여러 개의 npz 시퀀스 파일을 다 불러온 후
시퀀스들을 섞어서 저장하는 코드

현재 train dataset의 모든 npz file을 불러오면 메모리가 없어서 안 됨
다른 건 다 괜찮은데, 특히 이미지를 (224, 224)로 확대하면서 메모리가 부족함

때문에 모든 npz 시퀀스 파일들을 불러온 후
하나의 .h5 파일에 데이터들을 저장함
'''

'''
dataset.h5
├── sample_0/
│   ├── img     (shape=(30, 3, 224, 224), float16)
│   ├── force   (shape=(6), float16)
│   └── label   (shape=(7), int)
├── sample_1/
│   ├── ...
'''
import h5py
import sys
import os
sys.path.append(os.path.dirname(__file__))
from util import *
import numpy as np
import yaml
import traceback


# ========================================================
# configuration
with open("../config.yaml", 'r+', encoding='utf-8') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

dataset_types = ['train', 'val', 'test']
# ========================================================


for d_type in dataset_types :
    dataset_type = d_type

    original_image_size = (10, 8)

    img, force, label, img_ts, force_ts = load_data_from_npz(dataset_type)
    img, force, label, frame = preprocess(img, force, label, img_ts, force_ts, d_type)

    print(f'{d_type} complete')


print("Complete to resize lots of images and save")
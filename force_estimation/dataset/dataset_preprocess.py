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

image_size = config.get('image_size')
target_index = config.get('target_index')
stride = config.get('stride')
num_classes = 7

dataset_types = ['train', 'val', 'test']

img_max = [713.0000, 642.0000, 642.0000]
img_mean = [17.8756, 0.0001, 0.0000]
img_std = [47.0908, 4.2171, 5.8668]

f_mean = [ 3.38455280e-07, -4.57362519e-06, -1.18898044e-05, 2.87566208e-07, 8.71646891e-08, 3.76575034e-07]
f_std = [0.03641479, 0.05464274, 0.0668863, 0.00312084, 0.00155052, 0.00047453]

z_threshold = -7
# ========================================================


for d_type in dataset_types :
    dataset_type = d_type

    original_image_size = (10, 8)

    img, force, label, img_ts, force_ts = load_data_from_npz(dataset_type, target_index)
    img, force, label, frame = preprocess(img, force, label, img_ts, force_ts, img_max, img_mean, img_std,
                                          f_mean, f_std, num_classes, d_type)
    
    file_name = f'{dataset_type}_dataset.h5'
    
    with h5py.File(file_name, 'w') as h5f:
        # 각 sequence 저장
        for index in range(0, len(img)):
            try:
                sample_name = f'sequence_{index:04d}'
                group = h5f.create_group(sample_name)

                # 저장
                group.create_dataset('img', data=img[index], compression='gzip', compression_opts=9) # shape: (# of sequence, 3, 224, 224), float32
                group.create_dataset('force', data=force[index], compression='gzip', compression_opts=9) # shape: (# of sequence, 6), float
                group.create_dataset('label', data=label[index], compression='gzip', compression_opts=9) # shape: (# of sequence, 7), int

                if index % 10 == 0:
                    print(f"save {index} / {len(img)} data...")

            except Exception as e:
                print(e)
                traceback.print_exc()
                sys.exit(1)

        h5f.create_dataset('frame', data=frame, compression='gzip', compression_opts=9)

    '''
    with h5py.File(file_name, 'a') as f:
        sample_keys = list(f.keys())
        if 'frame' in sample_keys: sample_keys.remove('frame')

        for key in sample_keys:
            base_group = f[f'{key}']
            force = f[f'{key}/force']
            filtered_force = np.copy(force)

            for i in range(filtered_force.shape[0]):
                if filtered_force[i, 2] > 0: filtered_force[i, 2] = 0
                if filtered_force[i, 2] < z_threshold: filtered_force[i, 2] = z_threshold

            base_group.create_dataset('filtered_force', data=filtered_force, compression='gzip', compression_opts=9)
    '''
    print(f'{d_type} complete')


print("Complete to resize lots of images and save")
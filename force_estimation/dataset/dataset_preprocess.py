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
import wandb


# ========================================================
# configuration
with open("../config.yaml", 'r+', encoding='utf-8') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

image_size = (24, 12)
num_classes = 7

dataset_types = ['train', 'val', 'test']

img_max = [281.0000, 264.0000, 264.0000]
img_mean = [12.6907, -0.0000, -0.0000]
img_std = [36.3523, 5.6774, 7.6739]

f_mean = [1.51290077e-01,4.33448139e-02,-1.19329080e+00, 7.06875598e-03, -1.86735749e-02, 2.39680678e-05]
f_std = [0.56331648, 0.66661553, 1.36792603, 0.02959986, 0.02169826, 0.00322406]

z_threshold = -7
# ========================================================

# wandb에 artifact로 저장
wandb.login()
run = wandb.init(project="compare_model", job_type="upload-dataset", name="dataset")
artifact = wandb.Artifact(name=f"dataset", type="dataset",
                          metadata={"image_size": image_size,
                                    "z_threshold": z_threshold,
                                    "img_max": img_max,
                                    "img_mean": img_mean,
                                    "img_std": img_std})

for d_type in dataset_types :
    dataset_type = d_type

    original_image_size = (10, 8)

    img, force, label, img_ts, force_ts = load_data_from_npz(dataset_type)
    img, force, label, frame = preprocess(img, force, label, img_ts, force_ts, img_max, img_mean, img_std, f_mean, f_std, image_size, num_classes)
    
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

    with h5py.File(file_name, 'a') as f:
        sample_keys = list(f.keys())
        if 'frame' in sample_keys: sample_keys.remove('frame')

        for key in sample_keys:
            base_group = f[f'{key}']
            force = f[f'{key}/force']
            filtered_force = np.copy(force)

            if len(filtered_force.shape) == 1:
                if filtered_force[2] > 0: filtered_force[2] = 0
                if filtered_force[2] < z_threshold: filtered_force[2] = z_threshold
            else: # test dataset
                for i in range(filtered_force.shape[0]):
                    if filtered_force[i, 2] > 0: filtered_force[i, 2] = 0
                    if filtered_force[i, 2] < z_threshold: filtered_force[i, 2] = z_threshold

            base_group.create_dataset('filtered_force', data=filtered_force, compression='gzip', compression_opts=9)
            
    print(f'{d_type} complete')
    artifact.add_file(local_path=file_name)

run.log_artifact(artifact)
print("Complete to resize lots of images and save")
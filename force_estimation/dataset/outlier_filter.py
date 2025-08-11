import h5py
import numpy as np
path = 'val_dataset.h5'

z_filter = -5

with h5py.File(path, 'a') as f:
    sample_keys = list(f.keys())
    if 'frame' in sample_keys: sample_keys.remove('frame')

    for key in sample_keys:
        base_group = f[f'{key}']
        force = f[f'{key}/force']
        filtered_force = np.copy(force)

        if len(filtered_force.shape) == 1:
            if filtered_force[2] > 0: filtered_force[2] = 0
            if filtered_force[2] < -5: filtered_force[2] = -5
        else: # test dataset
            for i in range(filtered_force.shape[0]):
                if filtered_force[i, 2] > 0: filtered_force[i, 2] = 0
                if filtered_force[i, 2] < -5: filtered_force[i, 2] = -5

        base_group.create_dataset('filtered_force', data=filtered_force, compression='gzip', compression_opts=9)
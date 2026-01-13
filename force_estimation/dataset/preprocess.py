import numpy as np
import cv2
from scipy import signal
from scipy.interpolate import interp1d
import os
import torch
import torch.nn.functional as F
import zipfile
import traceback
import h5py

dataset_types = ['train', 'val', 'test']

class PreprocessDataset:

    def __init__(self,
                 original_image_size=(6, 6), resized_image_size=(6, 6),
                 type_of_image="rgb", type_of_normalize="zscore",
                 cutoff_filter=6, order_of_filter=4,
                 num_classes=7):

        self.original_image_size = original_image_size
        self.resized_image_size = resized_image_size
        self.type_of_image = type_of_image
        self.type_of_normalize = type_of_normalize
        self.cutoff_filter = cutoff_filter
        self.order_of_filter = order_of_filter
        self.num_classes = num_classes

        self.img_max = None
        self.img_std = None
        self.img_mean = None


    def load_data_from_npz(self, path):
        img = []
        force = []

        label = []
        frame = []

        img_ts = []
        force_ts = []

        for lab in os.listdir(path):  # 각 label에 대하여
            for file_name in os.listdir(path + '/' + lab):  # 각 파일들에 대하여
                with zipfile.ZipFile(path + '/' + lab + '/' + file_name, 'r') as zf:  # 각 zip파일들에 대하여
                    for npzfile in zf.namelist():  # 각 npz파일들에 대하여
                        with zf.open(npzfile) as nf:
                            data = np.load(nf)
                            if len(data['img_ts']) > len(data['force_ts']): continue

                            img.append(data['img'])
                            force.append(data['force'])

                            img_ts.append(data['img_ts'])
                            force_ts.append(data['force_ts'])

                            label.append(int(lab))
                            frame.append(len(data['img']))

        print(f"{path}: {sum(frame)} image data")
        return img, force, label, img_ts, force_ts



    # ==================================
    # image
    # ==================================

    def delete_bad_frame(self, input_image, threshold=1000) -> np.array:
        '''
        threshold를 넘어가는 비정상적으로 큰 값들을 포함한 프레임들 제거
        입력: 각 시퀀스
        '''
        input_image = np.array(input_image)
        bad_frames = np.any(input_image >= threshold, axis=(1, 2))
        return input_image[~bad_frames]


    def clip_zero(self, input_image) -> np.array:
        '''
        capacitive image에서 0 미만인 것들을 0으로 올림
        입력: 각 시퀀스
        '''
        return np.clip(input_image, a_min=0, a_max=None)


    def normalize_image(self, img: np.array, mean: list, std: list) -> list:
        mean = np.array(mean)
        std = np.array(std)

        mean_reshaped = mean.reshape(1, -1, 1, 1)
        std_reshaped = std.reshape(1, -1, 1, 1)

        if self.type_of_normalize.lower() == "zscore":
            result_img = (img - mean_reshaped) / (std_reshaped + 1e-7)
        elif self.type_of_normalize.lower() == "mean":
            result_img = img - mean_reshaped
        else:
            raise Exception("Unknown normalize type")

        return result_img


    def convert_3channel_image(self, input_image, input_ts) -> (list, list):
        '''
        첫 번째와 두 번째 프레임은 버리되, 나머지 프레임은 3 Chanenl 이미지로
        입력: 모든 시퀀스 전체
        '''
        result_image = []
        result_ts = []

        for sequence_idx in range(len(input_image)):
            new_seq = []
            for frame_idx in range(len(input_image[sequence_idx])):
                # 처음 프레임이라면 시행 X
                if frame_idx == 0 or frame_idx == 1:
                    continue

                current_frame = input_image[sequence_idx][frame_idx]
                previous_frame = input_image[sequence_idx][frame_idx - 1]
                p_previous_frame = input_image[sequence_idx][frame_idx - 2]

                diff = current_frame - previous_frame
                diff_ = previous_frame - p_previous_frame
                p_diff = diff - diff_

                if self.type_of_image.lower() == "rgb":
                    new_frame = np.vstack([current_frame.reshape((1, self.original_image_size[0], self.original_image_size[1])),
                                           diff.reshape((1, self.original_image_size[0], self.original_image_size[1])),
                                           p_diff.reshape((1, self.original_image_size[0], self.original_image_size[1]))])

                elif self.type_of_image.lower() == "grayscale" or self.type_of_image.lower() == "gray":
                    new_frame = np.vstack([current_frame.reshape((1, self.original_image_size[0], self.original_image_size[1])),
                                           current_frame.reshape((1, self.original_image_size[0], self.original_image_size[1])),
                                           current_frame.reshape((1, self.original_image_size[0], self.original_image_size[1]))])

                new_seq.append(new_frame)

            result_ts.append(input_ts[sequence_idx][2:])
            result_image.append(np.array(new_seq))

        return result_image, result_ts


    def resize_image(self, input_image, size) -> np.array:
        if size == self.original_image_size:
            result_image = np.array(input_image)

        else:
            # src (원본 이미지)는 (h, w, c) 순서로 입력해 줘야 함.
            # size는, (h, w) 순서가 아니라 (w, h) 순서로 입력해 줘야 함.
            result_image = np.stack([
                cv2.resize(img.astype(np.float32).transpose(1, 2, 0), (size[1], size[0])).transpose(2, 0, 1)
                for img in input_image
            ])

        return result_image.astype(np.float32)


    # ==================================
    # force
    # ==================================

    def interpolate_nano_to_capacitance(self, force, img_ts, force_ts) -> list:
        result = []
        for index in range(len(force_ts)):
            # 보간 함수 생성 (외삽은 금지 - 범위 밖이면 NaN 발생)
            interp_func = interp1d(force_ts[index], force[index], kind='linear', fill_value='extrapolate',
                                   axis=0)

            # sensor1의 타임스탬프에 대해 sensor2 값 보간
            r = interp_func(img_ts[index])
            result.append(r)

        return result


    def measure_image_and_force_frequency(self, img_ts, force_ts) -> (float, float):
        def measure_frequency(timestamps):
            sum_of_intervals = 0.0
            sum_of_frame_force = 0

            for seq in timestamps:
                interval = np.mean(np.diff(seq))
                sum_of_frame_force += len(seq) - 1
                sum_of_intervals += interval * (len(seq) - 1)

            mean_interval = sum_of_intervals / sum_of_frame_force
            frequency = 1 / mean_interval
            return frequency

        # force frequency 측정
        force_hz = measure_frequency(force_ts)

        # image frequency 측정
        image_hz = measure_frequency(img_ts)

        return image_hz, force_hz


    def apply_butterworth_filter(self, force, img_ts, force_ts,
                                 cutoff=6, order=4):
        result = []

        image_hz, force_hz = self.measure_image_and_force_frequency(img_ts, force_ts)

        nyq = 0.5 * force_hz

        for index in range(len(force)):
            temp_force = force[index]
            b, a = signal.butter(order, cutoff / nyq, 'low')
            data = []
            for sig in range(6):
                r = signal.filtfilt(b, a, temp_force[:, sig])
                data.append(r)

            result.append(np.array(data).T)

        return result


    # ==================================
    # About Preprocess
    # ==================================


    def preprocess(self, path, is_save=True):
        img, force, label, img_ts, force_ts = self.load_data_from_npz(path)

        for sequence_index in range(len(img)):
            img[sequence_index] = self.delete_bad_frame(img[sequence_index])
            img[sequence_index] = self.clip_zero(img[sequence_index])

        img, img_ts = self.convert_3channel_image(img, img_ts)
        force = self.apply_butterworth_filter(force, img_ts, force_ts,
                                              cutoff=self.cutoff_filter,
                                              order=self.order_of_filter) # force data에 low-pass filter를 걺

        # cap 이미지랑 force의 모으는 시간이 다르기 때문에, timestamp에 맞춰서 force data를 cap 시간에 interpolate
        force = self.interpolate_nano_to_capacitance(force, img_ts, force_ts)
        frame = []

        for sequence_index in range(len(img)):
            # 이미지를 채널별로 normalize
            img[sequence_index] = self.resize_image(img[sequence_index], self.resized_image_size)
            img[sequence_index] = self.normalize_image(img[sequence_index], self.img_mean, self.img_std)
            # z force를 0으로 clipping
            force[sequence_index][:, 2] = np.clip(force[sequence_index][:, 2], a_min=None, a_max=0)

            frame.append(int(len(img[sequence_index])))

        # label
        label = F.one_hot(torch.Tensor(label).to(torch.int64), num_classes=self.num_classes)
        if is_save: self.save_dataset(path, img, force, label, frame)

        return img, force, label, frame



    def save_dataset(self, path, img, force, label, frame):
        file_name = f'{path}_dataset.h5'

        with h5py.File(file_name, 'w') as h5f:
            # 각 sequence 저장
            for index in range(0, len(img)):
                try:
                    sample_name = f'sequence_{index:04d}'
                    group = h5f.create_group(sample_name)

                    # 저장
                    group.create_dataset('img', data=img[index], compression='gzip',
                                         compression_opts=9)  # shape: (# of sequence, 3, 224, 224), float32
                    group.create_dataset('force', data=force[index], compression='gzip',
                                         compression_opts=9)  # shape: (# of sequence, 6), float
                    group.create_dataset('label', data=label[index], compression='gzip',
                                         compression_opts=9)  # shape: (# of sequence, 7), int

                    if index % 10 == 0:
                        print(f"save {index} / {len(img)} data...")

                except Exception as e:
                    print(e)
                    traceback.print_exc()
                    sys.exit(1)

            h5f.create_dataset('frame', data=frame, compression='gzip', compression_opts=9)



    def extract_informations(self, path_of_train_dataset) -> (np.array, np.array, np.array):
        '''
        Train Dataset으로부터 mean, max, std 추출
        '''

        img, force, _, img_ts, force_ts = self.load_data_from_npz(path_of_train_dataset)
        for sequence_index in range(len(img)):
            img[sequence_index] = self.delete_bad_frame(img[sequence_index])
            img[sequence_index] = self.clip_zero(img[sequence_index])
        img_frequency, force_frequency = self.measure_image_and_force_frequency(img_ts, force_ts)

        print(f"이미지 수집 주기: {img_frequency}")
        print(f"force 수집 주기: {force_frequency}")

        img, img_ts = self.convert_3channel_image(img, img_ts)  # img는 이제 NumPy 배열의 리스트
        resized_sequences = []
        print("Resizing all frames...")

        # 1. 모든 시퀀스와 프레임을 순회하며 리사이즈 후 리스트에 수집
        for sequence in img:
            # sequence의 shape: (frames, 3, H, W)
            resized_sequences.append(self.resize_image(sequence, self.resized_image_size))

        # 2. 수집된 프레임 리스트를 하나의 거대한 NumPy 배열로 통합
        #    이 과정이 메모리를 가장 많이 사용하지만, 기존 코드보다 훨씬 효율적입니다.
        print("Concatenating frames into a single array...")
        all_frames_np = np.concatenate(resized_sequences, axis=0)

        # 더 이상 필요 없는 리스트는 메모리에서 해제
        del resized_sequences
        del img

        img_max = np.max(all_frames_np, axis=(0, 2, 3))
        img_std = np.std(all_frames_np, axis=(0, 2, 3))
        img_mean = np.mean(all_frames_np, axis=(0, 2, 3))

        print(f"img_max: {img_max}")
        print(f"img_std: {img_std}")
        print(f"img_mean: {img_mean}\n")

        force = np.concatenate(force, axis=0)
        force[:, 2] = np.clip(force[:, 2], a_min=None, a_max=0)

        print(f"force max: {np.max(force, axis=0)}")
        print(f"force min: {np.min(force, axis=0)}\n")

        return img_max, img_std, img_mean



pd = PreprocessDataset()
pd.img_max, pd.img_std, pd.img_mean = pd.extract_informations('train')

for d_type in dataset_types:
    pd.preprocess(d_type, is_save=True)
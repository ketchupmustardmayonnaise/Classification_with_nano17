import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import signal
from npz_dataset import RFTDataset

def _divide_RGB_channels(self):
    ''' 맨처음 프레임은 버리되, 나머지 프레임은 3 Chanenl 이미지로 '''

    for sequence_idx in range(len(self.img)):
        new_seq = []
        initial_img = None
        for frame_idx in range(len(self.img[sequence_idx])):
            # 처음 or 두 번째 프레임이라면 시행 X
            if frame_idx == 0:
                initial_img = self.img[sequence_idx][0]
            elif frame_idx == 1: continue

            current_frame = self.img[sequence_idx][frame_idx]
            previous_frame = self.img[sequence_idx][frame_idx - 1]
            p_previous_frame = self.img[sequence_idx][frame_idx - 2]

            diff_from_initial = current_frame - initial_img
            diff = current_frame - previous_frame
            diff_ = previous_frame - p_previous_frame
            p_diff = diff - diff_

            new_frame = np.vstack([current_frame.reshape((1, 10, 8)), diff.reshape((1, 10, 8)), p_diff.reshape((1, 10, 8))])
            if np.isnan(new_frame).any():
                continue
            new_seq.append(new_frame)

        self.img[sequence_idx] = np.array(new_seq)
        self.img_ts[sequence_idx] = self.img_ts[sequence_idx][1:]
        self.frame[sequence_idx] -= 2

def _extract_mean_and_max(self, channel):
    cap = []
    max = 0
    for sequence_idx in range(len(self.img)):
        if max < np.max(self.img[sequence_idx][:, channel]): max = np.max(self.img[sequence_idx][:, channel])
        cap.extend(self.img[sequence_idx][:, channel].flatten().tolist())

    mean = np.mean(np.array(cap))
    return mean, max

def _interpolate_nano_to_capacitance(self, force_data, img_ts, force_ts):
    result = []
    for index in range(len(force_ts)):
        # 보간 함수 생성 (외삽은 금지 - 범위 밖이면 NaN 발생)
        interp_func = interp1d(force_ts[index], force_data[index], kind='linear', fill_value='extrapolate',
                               axis=0)

        # sensor1의 타임스탬프에 대해 sensor2 값 보간
        r = interp_func(img_ts[index])
        result.append(r)

    return result


def _apply_butterworth_filter(self):
    result = []

    image_hz, force_hz = _measure_image_and_force_frequency(self.img_ts, self.force_ts)

    nyq = 0.5 * force_hz
    cutoff = 5

    for index in range(len(self.force)):
        temp_force = self.force[index]
        b, a = signal.butter(2, cutoff / nyq, 'low')
        data = []
        for sig in range(6):
            r = signal.filtfilt(b, a, temp_force[:, sig])
            data.append(r)

        result.append(np.array(data).T)

    return result

def _measure_image_and_force_frequency(img_ts, force_ts):
    def _measure_frequency(timestamps):
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
    force_hz = _measure_frequency(force_ts)

    # image frequency 측정
    image_hz = _measure_frequency(img_ts)

    return image_hz, force_hz

def _resize_image(self, size):
    result = []
    for data_index in range(len(self.img)):
        data = []
        for frame_index in range(len(self.img[data_index])):
            # 이미지를 (C, H, W)에서 (H, W, C)로 변경
            img = np.transpose(self.img[data_index][frame_index], (1, 2, 0))
            resized_img = cv2.resize(img.astype('float32'), size,
                                     interpolation=cv2.INTER_CUBIC)
            if len(resized_img.shape) == 2: resized_img = np.expand_dims(resized_img, axis=2)
            # 이미지를 (H, W, C)에서 (C, H, W)로 변경
            resized_img = np.transpose(resized_img, (2, 0, 1))
            data.append(resized_img)
        result.append(np.array(data))

    return result


def _normalize_image(self, img_max, img_max_diff):
    # 0 - 1로 정규화
    if self.image_type.lower() == 'rgb':
        for sequence_idx in range(len(self.img)):
            self.img[sequence_idx] = self.img[sequence_idx].astype('float32')

            self.img[sequence_idx][:, 0] = (self.img[sequence_idx][:, 0] * 255 / self.rgb_mean_and_max[1]) # [0, 255]
            self.img[sequence_idx][:, 0] -= (self.rgb_mean_and_max[0] * 255 / self.rgb_mean_and_max[1])

            self.img[sequence_idx][:, 1] -= self.rgb_mean_and_max[2]

            self.img[sequence_idx][:, 2] -= self.rgb_mean_and_max[4]

    elif self.image_type.lower() == 'origin':
        for index in range(len(self.img)):
            self.img[index] = np.array(self.img[index]).astype(np.float32)
            self.img[index] = np.where(self.img[index] >= img_max, 1.0, self.img[index] / img_max)
            self.img[index] = np.where(self.img[index] < 0, 0, self.img[index])

    elif self.image_type.lower() == 'diff':
        for index in range(len(self.img)):
            self.img[index] = np.array(self.img[index]).astype(np.float32)
            self.img[index] = np.where(self.img[index] >= img_max_diff, 1.0, self.img[index] / img_max_diff)

    else:
        raise Exception('Unknown image type')


def _outlier_filter(self, clipping_per_label):
    for sequence_idx in range(len(self.force)):
        force = np.array(self.force[sequence_idx])
        label = np.array(self.label[sequence_idx])

        threshold = clipping_per_label[label]

        indices = np.where(force[:, 2] < threshold)[0]
        force[indices, 2] = threshold

        indices = np.where(force[:, 2] > 0)[0]
        force[indices, 2] = 0

        self.force[sequence_idx] = force


def _print_information(self):
    # hz
    img_hz, force_hz = _measure_image_and_force_frequency(self.img_ts, self.force_ts)
    print(f"img_hz: {img_hz}, force_hz: {force_hz}\n")

    # cap
    cap = []
    for seq in self.img:
        cap.extend(seq.flatten())
    cap = np.array(cap)

    t = np.where((cap >= 100) & (cap < 199))

    print(f"ALL: {len(cap)}\n"
          f"0-99: {(len(np.where((cap >= 0) & (cap < 100))[0]) / len(cap) * 100):2f}\n"
          f"100-199: {(len(np.where((cap >= 100) & (cap < 199))[0]) / len(cap)) * 100:2f}\n"
          f"200-299: {(len(np.where((cap >= 200) & (cap < 299))[0]) / len(cap)) * 100:2f}\n"
          f"300-399: {(len(np.where((cap >= 300) & (cap < 399))[0]) / len(cap)) * 100:2f}\n"
          f"400-499: {(len(np.where((cap >= 400) & (cap < 499))[0]) / len(cap)) * 100:2f}\n"
          f"500-599: {(len(np.where((cap >= 500) & (cap < 599))[0]) / len(cap)) * 100:2f}\n"
          f"600-699: {(len(np.where((cap >= 600) & (cap < 699))[0]) / len(cap)) * 100:2f}\n"
          f"700- : {(len(np.where(cap >= 700)) / len(cap)):2f}\n")

    # filter
    nyq = 0.5 * force_hz

    for index in range(len(self.force)):
        temp_force = self.force[index]
        data = []
        sig = 2 # z축만 보여줌

        b, a = signal.butter(2, img_hz / nyq, 'low')
        r_img_hz = signal.filtfilt(b, a, temp_force[:, sig])

        b,a = signal.butter(2, 20 / nyq, 'low')
        r_20 = signal.filtfilt(b, a, temp_force[:, sig])

        b, a = signal.butter(2, 10 / nyq, 'low')
        r_10 = signal.filtfilt(b, a, temp_force[:, sig])

        plt.plot(temp_force[:, sig], color='b', label='before_apply', alpha=0.5)
        plt.plot(r_img_hz, color='r', label='image')
        plt.plot(r_20, color='y', label='20 Hz')
        plt.plot(r_10, color='g', label='10 Hz')
        plt.ylabel('Force [N]')
        plt.xlabel('Frame')
        plt.legend()
        plt.show()

    return None

RFTDataset.divide_RGB_channels = _divide_RGB_channels
RFTDataset.extract_mean_and_max = _extract_mean_and_max
RFTDataset.interpolate_nano_to_capacitance = _interpolate_nano_to_capacitance
RFTDataset.apply_butterworth_filter = _apply_butterworth_filter
RFTDataset.measure_image_and_force_frequency = _measure_image_and_force_frequency
RFTDataset.resize_image = _resize_image
RFTDataset.normalize_image = _normalize_image
RFTDataset.outlier_filter = _outlier_filter
RFTDataset.print_information = _print_information
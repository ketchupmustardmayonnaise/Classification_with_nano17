from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np

mark_file = '20N_mark_data.csv'
nano_file = '20N_nano_data.txt'

nano_f = open(nano_file, 'r')
nano_lines = nano_f.readlines()
nano_before_data = []
nano_after_data = []
nano_timestamp = []

for line in nano_lines:
    line = line.strip()
    line = line.split(', ')
    nano_timestamp.append(float(line[0]))

    after_data = line[1].split(' ')
    after_data = [x for x in after_data if x != '']
    before_data = line[2].split(' ')
    before_data = [x for x in before_data if x != '']

    nano_before_data.append(float(before_data[3]) * (-1))
    nano_after_data.append(float(after_data[3]) * (-1))

mark_f = open(mark_file, 'r')
mark_lines = mark_f.readlines()
mark_data = []
mark_timestamp = []

for line in mark_lines:
    line = line.strip()
    line = line.split('\t')
    # mark_timestamp.append(float(line[3]) + nano_timestamp[0])
    mark_data.append(float(line[1]))

# 보간을 위한 x축 생성
x_before = np.linspace(0, 1, len(nano_before_data))
x_after = np.linspace(0, 1, len(nano_after_data))
x_new = np.linspace(0, 1, len(mark_data))

# 보간 함수 생성 및 적용
interp_func_before = interp1d(x_before, nano_before_data, kind='linear', axis=0, fill_value='extrapolate')
interp_nano_data_before = interp_func_before(x_new)

interp_func_after = interp1d(x_after, nano_after_data, kind='linear', axis=0, fill_value='extrapolate')
interp_nano_data_after = interp_func_after(x_new)

plt.plot(mark_data, label='mark', color='red')
# plt.plot(interp_nano_data_before, label='nano', color='blue')
plt.plot(interp_nano_data_after, label='nano', color='green')
plt.show()
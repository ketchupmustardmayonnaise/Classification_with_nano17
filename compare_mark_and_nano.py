from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

mark_file = '10N_mark_data.csv'
nano_file = '10N_nano_data.txt'

nano_f = open(nano_file, 'r')
nano_lines = nano_f.readlines()
nano_data = []
nano_timestamp = []

for line in nano_lines:
    line = line.strip()
    line = line.split(', ')
    nano_timestamp.append(float(line[0]))
    nano_data.append(float(line[1]) * (-1))

mark_f = open(mark_file, 'r')
mark_lines = mark_f.readlines()
mark_data = []
mark_timestamp = []

for line in mark_lines:
    line = line.strip()
    line = line.split('\t')
    mark_timestamp.append(float(line[3]) + nano_timestamp[0])
    mark_data.append(float(line[1]))

interp_func = interp1d(nano_timestamp, nano_data, kind='linear', fill_value='extrapolate', axis=0)
interp_nano_data = interp_func(mark_timestamp)

plt.plot(mark_data, label='mark')
plt.plot(interp_nano_data, label='nano')
plt.show()
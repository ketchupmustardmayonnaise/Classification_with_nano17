import time
import numpy as np
import nidaqmx
import ctypes

device_name = "Dev3"  # DAQ 장치 이름

# ATI .cal 파일 경로 설정
CAL_FILE_PATH = "./read_sensor/FT13162.cal"
DLL_PATH = "./read_sensor/atidaq.dll"

# 전역 변수로 캘리브레이션 인스턴스 및 관련 함수 포인터를 저장
ftconfig_lib = None
_cal_instance_global = None
_convert_to_ft_func = None # ConvertToFT 함수 포인터

try:
    # DLL_PATH는 atidaq.dll이지만, 실제 calibration 함수는 ftconfig.c에서 컴파일된 것이므로
    # atidaq.dll이 ftconfig의 기능을 포함하고 있다고 가정합니다.
    ftconfig_lib = ctypes.CDLL(DLL_PATH)
    print(f"Successfully attempted to load library from: {DLL_PATH}")
except OSError as e:
    print(f"Error loading library: {e}")
    print("Please ensure 'atidaq.dll' is in the same directory or accessible via PATH.")
    print("Continuing with hardcoded calibration matrix.")
    ftconfig_lib = None # 라이브러리 로드 실패 시 None으로 설정


# Calibration 구조체 (ctypes 포인터 처리를 위해 간소화)
class Calibration(ctypes.Structure):
    _fields_ = [] # 실제 구조체 정의는 C DLL 내부에서 처리되므로 Python에서는 빈 구조체로 선언


# C 함수들의 ctypes 래퍼 정의
if ftconfig_lib:
    try:
        ftconfig_lib.createCalibration.restype = ctypes.POINTER(Calibration)
        # 중요: createCalibration의 argtypes를 설정합니다.
        ftconfig_lib.createCalibration.argtypes = [ctypes.c_char_p, ctypes.c_ushort]

        ftconfig_lib.destroyCalibration.argtypes = [ctypes.POINTER(Calibration)]
        ftconfig_lib.destroyCalibration.restype = None

        ftconfig_lib.GetMatrix.argtypes = [ctypes.POINTER(Calibration),
                                           np.ctypeslib.ndpointer(dtype=np.float32, shape=(6, 6), flags='C_CONTIGUOUS')]
        ftconfig_lib.GetMatrix.restype = ctypes.c_short

        ftconfig_lib.ConvertToFT.argtypes = [
            ctypes.POINTER(Calibration),
            np.ctypeslib.ndpointer(dtype=np.float32, shape=(6,), flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.float32, shape=(6,), flags='C_CONTIGUOUS')
        ]
        ftconfig_lib.ConvertToFT.restype = None
        _convert_to_ft_func = ftconfig_lib.ConvertToFT

    except AttributeError as e:
        print(f"Warning: Missing expected functions in ftconfig library: {e}. Some DLL features might be unavailable.")
        # 필요한 함수가 없으면 해당 함수 포인터를 None으로 설정
        if not hasattr(ftconfig_lib, 'createCalibration'): ftconfig_lib.createCalibration = None
        if not hasattr(ftconfig_lib, 'destroyCalibration'): ftconfig_lib.destroyCalibration = None
        if not hasattr(ftconfig_lib, 'GetMatrix'): ftconfig_lib.GetMatrix = None
        _convert_to_ft_func = None # ConvertToFT도 사용 불가

# --- .cal 파일로부터 캘리브레이션 매트릭스 로드 시도 ---
user_axis = None # 초기값은 None

if ftconfig_lib and ftconfig_lib.createCalibration and ftconfig_lib.GetMatrix:
    try:
        cal_file_bytes = CAL_FILE_PATH.encode('utf-8') # 경로를 bytes로 인코딩

        # 올바른 인자를 전달하여 createCalibration 호출
        _cal_instance_global = ftconfig_lib.createCalibration(cal_file_bytes, ctypes.c_ushort(1))
        if not _cal_instance_global:
            raise RuntimeError("Failed to create Calibration instance via C function (createCalibration returned NULL).")

        temp_matrix = np.zeros((6, 6), dtype=np.float32)
        result = ftconfig_lib.GetMatrix(_cal_instance_global, temp_matrix)
        if result != 0:
            raise RuntimeError(f"Failed to get matrix from calibration, error code: {result}")

        user_axis = temp_matrix
        print(f"Successfully loaded calibration matrix from {CAL_FILE_PATH}")
        # print("Calibration Matrix:\n", user_axis)

    except Exception as e:
        print(f"Error during .cal file loading: {e}")
        print("Falling back to hardcoded calibration matrix.")
        # 에러 발생 시 _cal_instance_global이 유효하다면 파괴
        if _cal_instance_global and ftconfig_lib and ftconfig_lib.destroyCalibration:
            ftconfig_lib.destroyCalibration(_cal_instance_global)
            _cal_instance_global = None
        _convert_to_ft_func = None # DLL 기반 변환도 사용 불가

        user_axis = np.array([
            [0.01690, 0.08292, -0.08283, -3.24987, 0.17391, 3.30169],
            [0.07981, 3.90319, -0.03074, -1.84989, -0.16747, -1.96248],
            [3.69925, -0.06051, 3.91911, -0.06182, 3.86466, -0.00285],
            [0.23745, 23.51132, 21.95623, -11.52137, -22.45015, -11.75438],
            [-23.67481, -0.07585, 13.51575, 19.37138, 11.34589, -19.95318],
            [0.28997, 14.41725, 0.47660, 13.91273, 0.93382, 14.76661]
        ])
else:
    print("ftconfig library or its required functions for calibration loading (createCalibration, GetMatrix) not loaded. Using hardcoded calibration matrix.")
    _convert_to_ft_func = None # DLL 기반 변환 사용 불가
    user_axis = np.array([
        [0.01690, 0.08292, -0.08283, -3.24987, 0.17391, 3.30169],
        [0.07981, 3.90319, -0.03074, -1.84989, -0.16747, -1.96248],
        [3.69925, -0.06051, 3.91911, -0.06182, 3.86466, -0.00285],
        [0.23745, 23.51132, 21.95623, -11.52137, -22.45015, -11.75438],
        [-23.67481, -0.07585, 13.51575, 19.37138, 11.34589, -19.95318],
        [0.28997, 14.41725, 0.47660, 13.91273, 0.93382, 14.76661]
    ])

timestamp_list = []
force_shm = None
ts_shm = None

data_for_saved = []
sample_rate_data = []

data_per_sec = 0
WINDOW_SIZE = 100

ft_bias = [0, 0, 0, 0, 0, 0]
daq_task = None # NI-DAQmx 태스크 인스턴스

def setup_daq_task(device_name):
    global daq_task
    if daq_task is None:
        daq_task = nidaqmx.Task()
        for i in range(6):
            daq_task.ai_channels.add_ai_voltage_chan(f"{device_name}/ai{i}", min_val=-10.0, max_val=10.0)
        daq_task.timing.cfg_samp_clk_timing(rate=300.0, sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS)
        daq_task.start() # 태스크 시작
    return daq_task

def read_sensor_value():
    global daq_task
    if daq_task is None:
        setup_daq_task(device_name)

    # 1개 샘플을 읽습니다.
    read_data = daq_task.read(number_of_samples_per_channel=1)

    # NIDAQmx read 결과가 리스트 안에 단일 값으로 올 수 있으므로 평탄화
    if isinstance(read_data, list) and all(isinstance(d, list) and len(d) == 1 for d in read_data):
        read_data = [d[0] for d in read_data]
    return read_data


def calculation(raw_data, bias):
    global _cal_instance_global, _convert_to_ft_func  # 전역 변수 사용 명시

    voltages_minus_bias = np.array(raw_data, dtype=np.float32) - np.array(bias, dtype=np.float32)
    result_ft = np.zeros(6, dtype=np.float32)

    # DLL의 ConvertToFT 함수를 사용할 수 있는 경우
    if _convert_to_ft_func and _cal_instance_global:
        try:
            _convert_to_ft_func(_cal_instance_global, voltages_minus_bias, result_ft)
            return result_ft
        except Exception as e:
            print(f"Error calling ConvertToFT from DLL: {e}. Falling back to NumPy calculation.")
            # 에러 발생 시 NumPy 계산으로 대체
            return user_axis @ voltages_minus_bias
    else:
        # DLL 함수를 사용할 수 없는 경우 (초기 로드 실패 또는 ConvertToFT 없음)
        return user_axis @ voltages_minus_bias


def read_nano17():
    tick_data = read_sensor_value()
    ft = calculation(tick_data, ft_bias)
    return ft


def calibration():
    global ft_bias
    # 초기 바이어스 읽기 (평균 캘리브레이션)
    print("Performing initial bias calibration (reading 100 samples)...")
    bias_samples = []
    for _ in range(100):  # 100개 샘플로 평균 바이어스 계산
        bias_samples.append(read_sensor_value())
        time.sleep(0.001)  # 짧은 대기 (샘플링 레이트에 맞춰 조정)

    ft_bias = np.mean(bias_samples, axis=0)


def close_nano17():
    global daq_task
    if daq_task:
        daq_task.stop()
        daq_task.close()
        daq_task = None
        print("DAQmx Task closed.")
import socket
import time
import collect_dataset.config as cfg
import multiprocessing
import subprocess
import re
import numpy as np
from ppadb.client import Client as AdbClient
from multiprocessing import shared_memory

MAX_CAP_VALUE = 600
MIN_CAP_VALUE = 0

USE_PREVIOUS_FRAME = True # 여러 개의 데이터가 동시에 들어오면 이를 모두 저장할 것인지?
CONNECT_METHOD = "socket"
CURRENT_DIFFERENCE_MS = 0.020 # 50 Hz

ERROR_PACKET = [-255] * cfg.capacitive_screen_shape[0] * cfg.capacitive_screen_shape[1]

class ReadCapacitance(multiprocessing.Process):

    def __init__(self, queue, shm_name, record_event):
        super(ReadCapacitance, self).__init__()

        # 연결 관련 변수들
        self.host = "127.0.0.1"
        self.port = 8888
        self.buffer_size = 8192
        self.socket = None
        self.is_connected = False
        self.connect_method = CONNECT_METHOD

        self.client = AdbClient(host=self.host, port=5037)
        try: self.device = self.client.devices()[0]
        except: self.device = None

        self.running = True
        self.data_queue = queue
        self.sleep_time = 0.005

        self.record_event = record_event
        # ❗ 여기서는 "이름만" 저장
        self.shm_name = shm_name
        self.shm = None
        self.shm_view = None

        self.previous_timestamp = None

    def _attach_shared_memory(self):
        # 실제 프로세스(run이 돈 프로세스) 안에서만 호출
        if self.shm is None:
            self.shm = shared_memory.SharedMemory(name=self.shm_name)
            self.shm_view = np.ndarray(
                shape=cfg.capacitive_screen_shape,
                dtype=np.int32,
                buffer=self.shm.buf
            )


    def connect(self):
        # 소켓 생성 및 연결
        try:
            if self.connect_method == "socket":
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.connect((self.host, self.port))
            elif self.connect_method == "adb":
                client = AdbClient(host="127.0.0.1", port=5037)
                devices = client.devices()
                self.device = devices[0]
                self.process = subprocess.run(['adb','shell','cat','/proc/diff_log'], text=True)
                subprocess.call('su', shell=True)

        except ConnectionRefusedError:
            print("❌ 연결 거부. 안드로이드 서버나 포워딩 설정 확인.")
        except TimeoutError:
            print("❌ 연결 시간 초과.")
        except Exception as e:
            print(f"❌ 소켓 통신 오류: {e}.")


    def read(self):
        if self.is_connected:
            try:
                if self.connect_method == "socket":
                    data = self.socket.recv(self.buffer_size).decode('utf-8', errors='ignore')
                    data = re.split(r'[,\n]', data)
                    data = [t.strip() for t in data if t.strip()]
                    timestamp = float(data[0])
                    data = [int(x) for x in data[1:] if x.lstrip('-').isdigit()]
                elif self.connect_method == "adb":
                    stdout = self.device.shell("cat /proc/diff_log")
                    data = re.split(r'[,\n]', stdout)
                    data = [t.strip() for t in data if t.strip()]
                    timestamp = time.time()

                data_to_read = []
                if not USE_PREVIOUS_FRAME:
                    if len(data) != cfg.capacitive_screen_shape[0] * cfg.capacitive_screen_shape[1]:
                        # print("Error, the number of data is invalid: %d.\n" % len(data))
                        return None
                    else:
                        data_to_read.append([timestamp, data])
                else:
                    # 정상 패킷이 들어온다면
                    if len(data) != 0 and len(data) % (cfg.capacitive_screen_shape[0] * cfg.capacitive_screen_shape[1]) == 0:
                        num_of_frame = int((cfg.capacitive_screen_shape[0] * cfg.capacitive_screen_shape[1]) / len(data))
                        for i in range(0, num_of_frame):
                            ts = timestamp - (num_of_frame-1 - i) * CURRENT_DIFFERENCE_MS
                            data_to_read.append([ts, data[i * cfg.capacitive_screen_shape[0] * cfg.capacitive_screen_shape[1]
                                                            : cfg.capacitive_screen_shape[0] * cfg.capacitive_screen_shape[1]]])
                            self.previous_timestamp = ts
                    # 오류 패킷이 들어온다면
                    else:
                        raise Exception("Error, the number of data is invalid: %d.\n" % len(data))

            except Exception as e:
                data_to_read = []
                if self.previous_timestamp is not None:
                    # 데이터 저장
                    data_to_read.append([self.previous_timestamp + CURRENT_DIFFERENCE_MS,
                                         ERROR_PACKET])
                    # timestamp를 새로 변경
                    self.previous_timestamp = self.previous_timestamp + CURRENT_DIFFERENCE_MS
                    return data_to_read
                else:
                    # print("Error, the number of data is invalid: %d.\n" % len(data))
                    return None

            return data_to_read

        else: return None


    def stop(self):
        self.running = False
        if self.socket is not None:
            self.socket.close()
        if self.shm is not None:
            self.shm.close()  # unlink는 main에서 이미 해주고 있으니 여기선 close만


    def run(self):
        # ✅ 자식 프로세스에서 shared memory 연결
        self._attach_shared_memory()

        # PC의 8888 포트를 장치(device)의 8888 포트로 포워딩 설정
        self.device.forward("tcp:8888", "tcp:8888")
        print("✅ Port Forwarding 설정 완료: PC:8888 -> Android:8888")

        while True:
            try:
                self.connect()
                if self.connect_method == "socket":
                    current_time = str(time.time()) + "\n"  # 예: "1709123456.78\n"
                    self.socket.sendall(current_time.encode('utf-8'))
                    print(f"[PC] Time sent to Phone: {current_time.strip()}")
                self.is_connected = True
                break
            except:
                print("스마트워치 연결 실패, 2초 후 재연결 시도")
                time.sleep(2)


        # ==================================================
        # 연결 완료
        # ==================================================

        while self.running:
            datas = self.read()

            if datas is not None:
                try:
                    for timestamp, data in datas:
                        # 공유 메모리에 데이터 전송
                        arr = np.asarray(data, dtype=np.int32).reshape(cfg.capacitive_screen_shape)
                        self.shm_view[:] = arr

                        # 녹화 중이라면 큐에 저장
                        if self.record_event.is_set():
                            self.data_queue.put_nowait((timestamp, data))
                except:
                    continue

            time.sleep(self.sleep_time)
import multiprocessing.shared_memory
import keyboard

import collect_dataset.config
from show_data_window import *
from collect_dataset.read_sensor.read_ft_sensor import *
from collect_dataset.read_sensor.read_capacitance import *
from util_for_saving import *

# 변수 생성
is_recording = False
ft_data_queue_for_save = multiprocessing.Queue() # for save
ft_data_queue_for_plot = multiprocessing.Queue() # for plot
capacitance_data_queue = multiprocessing.Queue() # for save
shm_for_capacitive_image = multiprocessing.shared_memory.SharedMemory(create=True, size=np.zeros(config.capacitive_screen_shape, dtype=np.int32).nbytes) # for plot

record_event = multiprocessing.Event()
calibration_request_event = multiprocessing.Event()
calibration_done_event = multiprocessing.Event()

capacitance_process = ReadCapacitance(capacitance_data_queue, shm_for_capacitive_image.name, record_event)
ftsensor_process = ReadFTSensor(ft_data_queue_for_save, ft_data_queue_for_plot,
                                record_event, calibration_request_event, calibration_done_event)


def set_recording():
    global capacitance_process, ftsensor_process, record_event, calibration_request_event, calibration_done_event

    if record_event.is_set():
        record_event.clear()

        try:
            sequence_of_capacitive_image = []
            sequence_of_force = []

            while not capacitance_data_queue.empty():
                sequence_of_capacitive_image.append(capacitance_data_queue.get_nowait())

            while not ft_data_queue_for_save.empty():
                sequence_of_force.append(ft_data_queue_for_save.get_nowait())

            # image와 timestamp 분리하기
            img_timestamp = np.array(list(zip(*sequence_of_capacitive_image))[0], dtype=float)
            img = np.array(list(zip(*sequence_of_capacitive_image))[1]).reshape(
                (len(sequence_of_capacitive_image), config.capacitive_screen_shape[0], config.capacitive_screen_shape[1]))

            # 앞뒤 조금 자름
            img_timestamp = img_timestamp[4:-4]
            img = img[4:-4]

            force_timestamp = np.array(list(zip(*sequence_of_force))[0], dtype=float)
            force_data = np.array(list(zip(*sequence_of_force))[1])

            if force_timestamp[-1] < img_timestamp[-1]:
                print(f"🚫 저장 실패, 다시 하십시오 (force time[-1]: {force_timestamp[-1]} img time[-1]: {img_timestamp[-1]})\n")
            else:
                # 너무 짧으면 저장하지 않음
                if len(sequence_of_capacitive_image) < 4:
                    print("🚫 저장 실패, 다시 하십시오 (capacitance image가 너무 짧음)\n")
                else:
                    # 저장에 성공하면 index를 1 올림
                    if save_image_and_force(img, img_timestamp,
                                            force_data, force_timestamp,
                                            config.DIR, config.NAME, config.FILE_INDEX, config.BODYPART, config.DATE):
                        config.FILE_INDEX = config.FILE_INDEX + 1
                    else:
                        print("🚫 저장 실패, 다시 하십시오\n")
        except Exception as e:
            print(f"🚫 Recording 중 에러 발생: {e}")

    else:
        # 녹화 시작 요청 → calibrate 먼저 필요
        calibration_done_event.clear()
        calibration_request_event.set()

        # calibration 완료될 때까지 기다림
        calibration_done_event.wait()
        print(">> Calibration done. Recording START!")

        record_event.set()
        calibration_request_event.clear()


if __name__ == '__main__':
    print("Enter your name: ")
    config.NAME = str(input())

    print("Enter current BODYPART: ")
    print("( (1) FINGER PAD (2) PALMAR SIDE OF FINGERS (3) PALM (4) HAND_ROCK_DOWN )")
    print("( (5) HAND_ROCK_UP (6) MOUTH (7) EAR )")
    config.BODYPART = config.TAG[int(input())]

    keyboard.add_hotkey("space", lambda: set_recording())


    # 변수 생성

    capacitance_process.start()
    ftsensor_process.start()

    try:
        plotter = Plotter(ft_data_queue_for_plot, shm_for_capacitive_image.name, record_event)
        plotter.start()

    finally:
        # 6. 종료 처리
        capacitance_process.stop()  # stop 메서드가 구현되어 있다고 가정
        ftsensor_process.stop()

        # 프로세스 종료 대기
        capacitance_process.terminate()  # 혹은 join
        ftsensor_process.terminate()
        capacitance_process.join()
        ftsensor_process.join()

        shm_for_capacitive_image.close()
        shm_for_capacitive_image.unlink()
        print("프로그램 종료")
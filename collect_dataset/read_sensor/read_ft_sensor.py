import multiprocessing
import threading
import time

import collect_dataset.read_sensor._test_Nano17 as ftsensor

class ReadFTSensor(multiprocessing.Process):

    def __init__(self, queue_for_save, queue_for_plot,
                 record_event, calibration_request_event, calibration_done_event):
        super(ReadFTSensor, self).__init__()

        self.running = True
        self.sleep_time = 0.002
        self.queue_for_save = queue_for_save
        self.record_event = record_event
        self.calibration_request_event = calibration_request_event
        self.calibration_done_event = calibration_done_event

        # plot하기 위하여 ft 데이터 전송
        self.queue_for_plot = queue_for_plot


    def calibrate(self):
        try:
            ftsensor.calibration() # 1초 정도 소요
            return True
        except:
            return False


    def read(self):
        data = ftsensor.read_nano17()
        return data


    def run(self):
        self.calibrate()

        while self.running:

            # 1. calibration 요청 체크
            if self.calibration_request_event.is_set():
                self.calibrate()

                self.calibration_done_event.set()
                self.calibration_request_event.clear()

            data = self.read()
            timestamp = time.time()
            self.queue_for_plot.put_nowait((timestamp, data))

            if self.record_event.is_set():
                self.queue_for_save.put_nowait((timestamp, data))

            time.sleep(self.sleep_time)


    def stop(self):
        ftsensor.close_nano17()
        self.running = False
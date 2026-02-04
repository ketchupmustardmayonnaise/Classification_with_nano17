from matplotlib.animation import FuncAnimation

import config as cfg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Ellipse
from multiprocessing import shared_memory

class Plotter:
    def __init__(self, ft_data_queue, shm_name_for_capacitive_image, record_event):
        self.fig = None
        self.ft_data_lines = None
        self.ft_data_list = []
        self.ft_data_timestamp = []

        self.axes = None
        self.qdMesh = None
        self.record_text = None
        self.capacitance_text_obj = []

        self.ft_data_queue = ft_data_queue

        self.shm = shared_memory.SharedMemory(name=shm_name_for_capacitive_image)
        self.shm_array_view = np.ndarray(cfg.capacitive_screen_shape, dtype=np.int32, buffer=self.shm.buf)

        self.record_event = record_event


    def prepare_plot(self):
        # prepare plot
        self.axes = []
        self.fig = plt.figure(figsize=(24, 12))

        self.axes.append(plt.subplot2grid((3, 4), (0, 2), colspan=2, rowspan=1))
        self.axes.append(plt.subplot2grid((3, 4), (1, 2), colspan=2, rowspan=1))
        self.axes.append(plt.subplot2grid((3, 4), (2, 2), colspan=2, rowspan=1))

        self.axes.append(plt.subplot2grid((3, 4), (0, 0), colspan=2, rowspan=3))
        self.ft_data_lines = []

        labels = ['Fx', 'Fy', 'Fz']
        colors = ['b', 'g', 'r']

        for i in range(3):
            ft_data_line, = self.axes[i].plot([], [], color=colors[i], label=labels[i])
            self.axes[i].set_ylabel(labels[i])
            self.axes[i].legend(loc='upper right')
            self.axes[i].set_ylim(-4, 4)

            self.ft_data_lines.append(ft_data_line)

            if i == 0 or i == 1:
                tick = [-2, -1, 0, 1, 2]
                self.axes[i].set_yticks(tick)
                self.axes[i].yaxis.grid(True)
            else:
                tick = list(range(cfg.z_threshold, 1))
                self.axes[i].set_yticks(tick)
                self.axes[i].yaxis.grid(True)

        # z축 범위는 -5 ~ 1
        self.axes[2].set_ylim(cfg.z_threshold, 1)
        self.axes[2].set_xlabel('Time (s)')
        plt.tight_layout()

        # Heatmap
        self.axes[3].set_xlim(0, cfg.capacitive_screen_shape[1])
        self.axes[3].set_ylim(0, cfg.capacitive_screen_shape[0])

        self.arrData = np.zeros(cfg.capacitive_screen_shape, dtype=int)
        self.qdMesh = self.axes[3].pcolormesh(self.arrData,
                                    vmin=0, vmax=400,
                                    cmap=cm.gray)
        self.annotate_heatmap(self.axes[3], self.arrData)

        # 텍스트
        self.fig_text = self.fig.text(
            0.25, 0.96,  # 가운데, 위쪽
            "RECORDING",
            ha="center", va="center",
            fontsize=20,
            color="red",
            weight="bold",
            visible=False
        )
        w, h = self.fig.get_size_inches()
        ratio = w / h  # 가로/세로 비
        radius = 0.008
        self.fig_circle = Ellipse(
            (0.20, 0.96),
            width=2 * radius,
            height=2 * radius * ratio,
            transform=self.fig.transFigure,
            facecolor="red",
            edgecolor="none",
            visible=False
        )

        self.fig.add_artist(self.fig_circle)

    def show_status(self, text, color="red"):
        self.fig_text.set_text(text)
        self.fig_text.set_visible(True)

        self.fig_circle.set_facecolor(color)
        self.fig_circle.set_visible(True)

        self.fig.canvas.draw_idle()

    def hide_status(self):
        self.fig_text.set_visible(False)
        self.fig_circle.set_visible(False)
        self.fig.canvas.draw_idle()

    def update(self, frame):
        # !! ======================== recording ======================== !!
        if self.record_event.is_set():
            self.qdMesh.set_cmap(cm.Blues_r)
            self.show_status("RECORDING")
        else:
            self.qdMesh.set_cmap(cm.gray)
            self.hide_status()

        # !! ======================== force data ======================== !!
        try:
            while not self.ft_data_queue.empty():
                timestamp, data = self.ft_data_queue.get_nowait()
                self.ft_data_list.append(data)
                self.ft_data_timestamp.append(timestamp)

            # 최근 1000개의 데이터만 기록
            if len(self.ft_data_list) > 2000:
                self.ft_data_list = self.ft_data_list[-2000:]
                self.ft_data_timestamp = self.ft_data_timestamp[-2000:]

            for i in range(3):
                self.ft_data_lines[i].set_data(self.ft_data_timestamp, np.array(self.ft_data_list)[:, i])
                self.axes[i].set_xlim(max(0, self.ft_data_timestamp[-1] - cfg.SLIDING_WINDOW_SEC), self.ft_data_timestamp[-1])
        except:
            pass

        # !! ======================== capacitive screen ======================== !!
        try:
            # 공유 메모리 접근
            cap_data = self.shm_array_view.copy()
            cap_data = np.flipud(cap_data)  # plot으로 보여줄 때는 거꾸로 뒤집어서 보여줘야 함

            self.qdMesh.set_array(cap_data.ravel())
            self.annotate_heatmap(self.axes[3], cap_data)
        except Exception as e:
            # [수정] pass 대신 에러를 출력해야 디버깅이 가능합니다.
            print(f"Plot update error: {e}")


    def annotate_heatmap(self, ax, data):
        # [수정] 리스트가 비어있는지 확인
        if not self.capacitance_text_obj:
            # create text (처음 한 번만 실행됨)
            for i in range(data.shape[1]):
                for j in range(data.shape[0]):
                    # [핵심 수정] 변수에 대입(=)하지 말고 리스트에 추가(append)해야 함
                    text_obj = ax.text(i + 0.5, j + 0.5, '%d' % data[j, i],
                                       horizontalalignment='center',
                                       verticalalignment='center')
                    self.capacitance_text_obj.append(text_obj)
        else:
            # set text (이미 객체가 있으면 텍스트만 변경)
            nNum = 0
            for i in range(data.shape[1]):
                for j in range(data.shape[0]):
                    try:
                        self.capacitance_text_obj[nNum].set_text('%d' % data[j, i])
                        nNum += 1
                    except IndexError:
                        # 데이터 크기가 안 맞을 경우 예외 처리
                        print("Index Error in heatmap annotation")


    def start(self):
        self.prepare_plot()
        self.ani = FuncAnimation(self.fig, self.update, interval=10)
        plt.show()
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import npz_dataset_method
from npz_dataset import RFTDataset

dataset = RFTDataset(path='./../dataset/rft64/origin')
dataset.preprocess()

def play_sequence(images, forces):
    fig = plt.figure(figsize=(10, 5))

    ax_img = fig.add_subplot(1, 2, 1)
    img_display = ax_img.imshow(np.transpose(images[0], (1, 2, 0)))
    ax_img.set_title("Image Sequence")
    ax_img.axis("off")

    ax_force = fig.add_subplot(1, 2, 2)
    lines = [ax_force.plot([], [], label=label)[0] for label in ['Fx', 'Fy', 'Fz']]
    ax_force.set_xlim(0, len(forces))
    ax_force.set_ylim(-5, 5)
    ax_force.set_title("Force over Time")
    ax_force.set_xlabel("Frame")
    ax_force.set_ylabel("Force (N)")
    ax_force.legend()

    x_data = []
    y_data = [[], [], []]  # Fx, Fy, Fz

    def init():
        img_display.set_data(np.transpose(images[0], (1, 2, 0)))
        for i in range(3):
            lines[i].set_data([], [])
        return [img_display] + lines

    def update(frame):
        img = np.transpose(images[frame], (1, 2, 0))
        img_display.set_data(img)

        x_data.append(frame)
        for i in range(3):
            y_data[i].append(forces[frame, i])
            lines[i].set_data(x_data, y_data[i])

        ax_force.set_xlim(max(0, frame - 50), frame + 10)
        return [img_display] + lines

    # 애니메이션 실행 및 blocking
    ani = FuncAnimation(
        fig,
        update,
        frames=len(images),
        init_func=init,
        interval=50,
        blit=False
    )

    plt.tight_layout()
    plt.show(block=False) # plt.show()는 blocking이므로, 다음 시퀀스로 자연스럽게 넘어감
    plt.pause((50 * len(images)) / 1000)
    plt.close()

# ---------- 전체 시퀀스 반복 재생 ----------
for index in range(dataset.__len__()):
    images, _, forces = dataset.__getitem__(index)
    print(f"Playing sequence {index+1}/{len(dataset)}...")
    play_sequence(images, forces)
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation
from force_estimation import PreprocessDataset

# =======================================
# configuration

dataset_types = ['train', 'val', 'test']

original_image_size = (6, 6)
resized_image_size = (6, 6)

type_of_image = "rgb"
type_of_normalize = "zscore"

cutoff_filter = 6
order_of_filter = 4

num_classes = 7
# =======================================


# cap 값 표시하기 위한 텍스트 오브젝트
text_object = []

def annotateHeatMap(ax, arrData):
    if np.any(arrData > 500):
        print(arrData)

    if (len(text_object) != 0):
        # set text
        nNum = 0
        for i in range(arrData.shape[1]):
            for j in range(arrData.shape[0]):
                text_object[nNum].set_text(arrData[j, i])
                nNum += 1
    else:
        # create text
        for i in range(arrData.shape[1]):
            for j in range(arrData.shape[0]):
                txObj = ax.text(i + 0.5, j + 0.5, '%d' % arrData[j, i],
                                horizontalalignment='center',
                                verticalalignment='center')
                text_object.append(txObj)


def play_sequence(images, forces, length_of_image):
    fig = plt.figure(figsize=(20, 5))

    img = np.transpose(images, (0, 2, 3, 1))

    ax_img_red = fig.add_subplot(1, 4, 1)
    img_display_red = ax_img_red.imshow(img[0, :, :, 0], cm.gray,  vmin=-127, vmax=255)
    annotateHeatMap(ax_img_red, np.zeros((10, 8), dtype=int))

    ax_img_green = fig.add_subplot(1, 4, 2)
    img_display_green = ax_img_green.imshow(img[0, :, :, 1], cm.gray,  vmin=-127, vmax=255)

    ax_img_blue = fig.add_subplot(1, 4, 3)
    img_display_blue = ax_img_blue.imshow(img[0, :, :, 2], cm.gray,  vmin=-127, vmax=255)

    ax_img_red.set_title("origin")
    ax_img_green.set_title("diff")
    ax_img_blue.set_title("diff1 - diff2")
    ax_img_red.axis("off")

    ax_force = fig.add_subplot(1, 4, 4)
    lines = [ax_force.plot([], [], label=label)[0] for label in ['Fx', 'Fy', 'Fz']]
    ax_force.set_xlim(0, len(forces))
    ax_force.set_ylim(-2, 2)
    tick = [-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3]
    plt.yticks(tick)
    ax_force.grid(True)
    ax_force.set_title("Force over Time")
    ax_force.set_xlabel("Frame")
    ax_force.set_ylabel("Force (N)")
    ax_force.legend()

    x_data = []
    y_data = [[], [], []]  # Fx, Fy, Fz

    def init():
        img = np.transpose(images[0], (1, 2, 0))
        img_display_red.set_data(img[:, :, 0])
        annotateHeatMap(ax_img_red, np.zeros((10, 8), dtype=int))
        img_display_green.set_data(img[:, :, 1])
        img_display_blue.set_data(img[:, :, 2])
        for i in range(3):
            lines[i].set_data([], [])
        return [img_display_red, img_display_green, img_display_blue] + lines

    def update(frame):
        img = np.transpose(images[frame], (1, 2, 0))
        img_display_red.set_data(img[:, :, 0])
        img_display_green.set_data(img[:, :, 1])
        img_display_blue.set_data(img[:, :, 2])

        x_data.append(frame)
        for i in range(3):
            y_data[i].append(forces[frame, i])
            lines[i].set_data(x_data, y_data[i])

        ax_force.set_xlim([0, length_of_image])

        # ===========================================
        annotateHeatMap(ax_img_red, img[:, :, 0])
        # ===========================================

        return [img_display_red, img_display_green, img_display_blue] + lines

    # 애니메이션 실행 및 blocking
    ani = FuncAnimation(
        fig,
        update,
        frames=[i for i in range(length_of_image)],
        init_func=init,
        interval=10,
        blit=False
    )

    plt.tight_layout()
    plt.show(block=False) # plt.show()는 blocking이므로, 다음 시퀀스로 자연스럽게 넘어감
    plt.pause(75 * length_of_image / 1000)
    plt.close()

pd = PreprocessDataset(original_image_size=original_image_size,
                       resized_image_size=resized_image_size,
                       type_of_image=type_of_image,
                       type_of_normalize=type_of_normalize,
                       cutoff_filter=cutoff_filter,
                       order_of_filter=order_of_filter,
                       num_classes=num_classes)
pd.img_max, pd.img_std, pd.img_mean = pd.extract_informations('train')
img, force, _, frame = pd.preprocess('test', is_save=False)

for index in range(0, len(img)):
    print(f"Playing sequence {index}/{len(img)}...")
    play_sequence(img[index], force[index], img[index].shape[0])
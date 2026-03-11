import os
import torch
import numpy as np
import wandb

import seaborn as sns
import matplotlib.pyplot as plt


def reconstruct_list_from_windows(raw_windows : list,
                                  index : int = None) -> list:
    if index is None:
        index = len(raw_windows[0])

    reconstructed_list = []
    reconstructed_list.extend(raw_windows[0][:index].tolist())  # 첫 번째 데이터는 모두 저장
    for i in range(1, len(raw_windows)):
        reconstructed_list.append(raw_windows[i][index - 1].tolist())  # 그 이후부턴 마지막 프레임만 저장

    return reconstructed_list


def plot_sequence(f, pf, save_dir, index,
                  label=None,
                  use_wandb=False, name=None):
    """
    재정렬된 force와 predicted force가 주어지면, 이를 plot하여 저장함.
    때문에 force 리스트와 predicted force 리스트의 shape은 완전히 같아야 함. (frames of each sequence, 3)
    """

    # 하나의 sequence에 대하여 수행

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))  # plot 만듦

    if label is not None:
        plt.suptitle(f"{index}th test sequence data ({label})")
    else:
        plt.suptitle(f"{index}th test sequence data")

    # 해당 plot에 대해서만 MSE, MAE
    mse = torch.nn.functional.mse_loss(torch.Tensor(f).float(), torch.Tensor(pf).float(), reduction='none')
    mae = torch.nn.functional.l1_loss(torch.Tensor(f).float(), torch.Tensor(pf).float(), reduction='none')

    for num_data in range(3):  # 3개 Fx, Fy, Fz에 대하여
        axes[num_data].plot(np.array(f)[:, num_data], color='r', label='force')
        axes[num_data].plot(np.array(pf)[:, num_data], color='b', label='predicted force')
        axes[num_data].set_xlabel('Frame')
        axes[num_data].set_ylabel('Force (N)')
        if num_data == 0 or num_data == 1:
            axes[num_data].set_ylim([-3, 3])
        elif num_data == 2:
            axes[num_data].set_ylim([-10, 0.5])

        axes[num_data].text(x=0.05, y=0.05,
                            s=f'RMSE: {torch.sqrt_(torch.mean(mse, axis=0)[num_data]):4f},'
                              f'MSE: {torch.mean(mse, axis=0)[num_data]:.4f},'
                              f'MAE: {torch.mean(mae, axis=0)[num_data]:.4f}',
                            transform=axes[num_data].transAxes,
                            horizontalalignment='left',
                            verticalalignment='top')

        if num_data == 0:
            axes[num_data].set_title('FX')
        elif num_data == 1:
            axes[num_data].set_title('FY')
        elif num_data == 2:
            axes[num_data].set_title('FZ')

    plt.legend()
    if not os.path.exists(f"{save_dir}"):
        os.makedirs(f"{save_dir}")
        os.makedirs(f"{save_dir}/png")
        os.makedirs(f"{save_dir}/svg")

    if label is None:
        if name is None:
            filename = f'test_{index}th_data'
        else: filename = f'[{str(name)}] test_{index}th_data'
    else:
        if name is None:
            filename = f'[{label}] test_{index}th_data'
        else: filename = f'[{str(name)}] [{label}] test_{index}th_data'

    plt.savefig(f'{save_dir}/png/{filename}.png',
                dpi=300, facecolor='white', edgecolor='black',
                format='png', pad_inches=0.1, transparent=False)
    plt.savefig(f'{save_dir}/svg/{filename}.svg',
                dpi=300,
                facecolor='white', edgecolor='black',
                format='svg', pad_inches=0.1, transparent=False)

    if use_wandb:
        img = wandb.Image(plt.imread(f'{save_dir}/test_{index}th_data.png'))
        wandb.log({f"test_{index}th_data": img})

    plt.cla()  # clear the current axes
    plt.clf()  # clear the current figure
    plt.close()  # closes the current figure
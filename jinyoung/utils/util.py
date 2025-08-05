import argparse
import os
import shutil
import ast
import matplotlib.pyplot as plt
import torch
import numpy as np

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
    

def copy_config(dir):
    # 현재 파일 위치 기준으로 config.yaml 경로 설정
    current_dir = os.getcwd()  # 또는 원하는 기준 경로
    src_path = os.path.join(current_dir, "config.yaml")

    # 복사할 대상 경로
    dst_dir = dir
    dst_path = os.path.join(dst_dir, "config.yaml")

    # 복사 실행
    shutil.copy(src_path, dst_path)
    

def write_basic_information(save_dir,use_wandb, use_all_sequence, target_indices, stride,
                                image_type, image_resize, image_size, augmentation_angle,
                                encoder, encoder_output,
                                fc_num_layer, fc_hidden_dim,
                                decoder, rnn_layer, decoder_output,
                                optimizer, scheduler, dropout, batch_size, loss, learning_rate, weight_decay):
    basic_txt_path = save_dir + "/basic.txt"
    basic_txt_file = open(basic_txt_path, 'a')
    basic_txt_file.write(f"use_wandb: {use_wandb}\n\n") # wandb를 사용할 것인지?
    basic_txt_file.write(f"encoder: {encoder}\n")
    basic_txt_file.write(f"encoder_output: {encoder_output}\n\n")

    basic_txt_file.write(f"use_all_sequence: {use_all_sequence}\n")
    basic_txt_file.write(f"target_indices: {target_indices}\n")
    basic_txt_file.write(f"stride: {stride}\n\n")

    basic_txt_file.write(f"image_type: {image_type}\n")
    basic_txt_file.write(f"image_resize: {image_resize}\n")
    basic_txt_file.write(f"image_size: {image_size}\n")
    basic_txt_file.write(f"augmentation_angle: {augmentation_angle}\n\n")

    basic_txt_file.write(f"decoder: {decoder}\n")
    basic_txt_file.write(f"rnn_layer: {rnn_layer}\n")
    basic_txt_file.write(f"decoder_output: {decoder_output}\n\n")

    basic_txt_file.write(f"fc_num_layer: {fc_num_layer}\n")
    basic_txt_file.write(f"fc_hidden_dim: {fc_hidden_dim}\n\n")

    basic_txt_file.write(f"optimizer: {optimizer}\n")
    basic_txt_file.write(f"scheduler: {scheduler}\n")
    basic_txt_file.write(f"dropout: {dropout}\n\n")

    basic_txt_file.write(f"loss: {loss}\n")
    basic_txt_file.write(f"batch_size: {batch_size}\n")
    basic_txt_file.write(f"learning_rate: {learning_rate}\n")
    basic_txt_file.write(f"weight_decay: {weight_decay}\n\n")
    basic_txt_file.close()


def parse_args():
    """커맨드라인 인자를 파싱하고 반환합니다."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_mode', default=None, type=str)
    parser.add_argument('--use_wandb', default=None, type=str2bool)
    parser.add_argument('--encoder', default=None, type=str)
    parser.add_argument('--encoder_output', default=None, type=str)
    parser.add_argument('--decoder', default=None, type=str)
    parser.add_argument('--rnn_layer', default=None, type=int)
    parser.add_argument('--decoder_output', default=None, type=int)
    parser.add_argument('--fc_num_layer', default=None, type=int)
    parser.add_argument('--fc_hidden_dim', default=None, type=int)
    parser.add_argument('--target_index', default=None, type=int)
    parser.add_argument('--stride', default=None, type=int)
    parser.add_argument('--augmentation_angle', default=None, type=int)
    parser.add_argument('--image_type', default=None, type=str)
    parser.add_argument('--image_resize', default=None, type=str2bool)
    parser.add_argument('--image_size', default=None, type=ast.literal_eval)
    parser.add_argument('--save_dir', default=None, type=str)
    parser.add_argument('--test_data_dir', default=None, type=str)
    parser.add_argument('--train_data_dir', default=None, type=str)
    parser.add_argument('--val_data_dir', default=None, type=str)
    parser.add_argument('--optimizer_type', default=None, type=str)
    parser.add_argument('--scheduler_type', default=None, type=str)
    parser.add_argument('--batch_size', default=None, type=int)
    parser.add_argument("--num_epoch", default=None, type=int)
    parser.add_argument("--learning_rate", default=None, type=float)
    parser.add_argument("--weight_decay", default=None, type=float)
    parser.add_argument('--dropout', default=None, type=float)
    parser.add_argument('--loss', default=None, type=str)
    return parser.parse_args()


def lists_for_plot(f, pf, target_index, plot_index):
    # plot을 위해, 시퀀스 별로 index 뽑음
    f_result = f[:len(f) - (target_index - plot_index)]

    predicted_force_sequence = []
    predicted_force_sequence.extend(pf[0][:plot_index])  # 첫 번째 데이터는 반만 저장
    for i in range(1, len(pf)):
        predicted_force_sequence.append(pf[i][plot_index - 1])

    return f_result, predicted_force_sequence

def lists_for_plot_mean(f, pf):
    # plot을 위해, 시퀀스 별로 index 뽑음
    predicted_force_sequence = [[] for _ in range(len(f))]

    # predicted force의 모든 값을 리스트에 append함.
    for i in range(0, len(pf)):
        for j in range(0, len(pf[i])):
            predicted_force_sequence[i+j].append(pf[i][j])

    for i in range(0, len(predicted_force_sequence)):
        predicted_force_sequence[i] = np.mean(np.array(predicted_force_sequence[i]), axis=0)

    return f, predicted_force_sequence

def plot(f, pf, plot_index, save_dir):
    dir = f'{save_dir}/{plot_index}th_frame_of_sliding_window'
    if not os.path.exists(dir):
        os.makedirs(dir)

    for index in range(len(f)):  # 테스트셋 35번째 이미지만 그림
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))  # plot 만듦

        _f = f[index]
        _pf = pf[index]

        if plot_index <= 0:
            plt.suptitle(f"mean frame")
        else:
            plt.suptitle(f"{index}th frame of sliding window")

        # 해당 plot에 대해서만 MSE, MAE
        mse = torch.mean(torch.nn.functional.mse_loss(torch.Tensor(_f).float(), torch.Tensor(_pf).float(),
                                                      reduction='none'), axis=0)
        mae = torch.mean(torch.nn.functional.l1_loss(torch.Tensor(_f).float(), torch.Tensor(_pf).float(),
                                                     reduction='none'), axis=0)

        for num_data in range(3):  # 3개 Fx, Fy, Fz에 대하여
            axes[num_data].plot(np.array(_f)[:, num_data], color='r', label='force')
            axes[num_data].plot(np.array(_pf)[:, num_data], color='b', label='predicted force')
            axes[num_data].set_xlabel('Frame')
            axes[num_data].set_ylabel('Force (N)')
            if num_data == 0 or num_data == 1: axes[num_data].set_ylim([-1.5, 1.5])

            axes[num_data].text(x=0.05, y=0.05,
                                s=f'MSE: {mse[num_data]:.4f}, MAE: {mae[num_data]:.4f}',
                                transform=axes[num_data].transAxes,
                                horizontalalignment='left',
                                verticalalignment='top')

            if num_data == 0:
                axes[num_data].set_title('X')
            elif num_data == 1:
                axes[num_data].set_title('Y')
            elif num_data == 2:
                axes[num_data].set_title('Z')

        plt.legend()
        plt.savefig(dir + f'/test_{index}th_data.png', dpi=300, facecolor='white', edgecolor='black',
                    format='png', bbox_inches='tight', pad_inches=0.1, transparent=False)
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import wandb
import torch
import os
from sklearn.metrics import confusion_matrix
from NOT_USE.with_classification.bodypart_label import bodypart_label


class Visualization:

    @staticmethod
    def save_accuracy_metrics(path, ground_truth, prediction):

        ground_truth = np.array(ground_truth)
        prediction = np.array(prediction)

        # classification 결과의 confusion matrix
        cm = confusion_matrix(ground_truth, prediction, normalize='true')
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=bodypart_label.values(),
                    yticklabels=bodypart_label.values())
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()  # ← 여백 자동 조정

        plt.savefig(f'{path}/confusion_matrix.png', dpi=300, facecolor='white',
                    edgecolor='black',
                    format='png', pad_inches=0.1, transparent=False)
        plt.savefig(f'{path}/confusion_matrix.svg', dpi=300, facecolor='white',
                    edgecolor='black',
                    format='svg', pad_inches=0.1, transparent=False)
        plt.show()

        plt.cla()  # clear the current axes
        plt.clf()  # clear the current figure
        plt.close()  # closes the current figure



    @staticmethod
    # plot index가 주어지면, 해당 plot index를 기준으로 리스트를 재정렬함.
    # ex) target_index = 60, plot_index = 30으로 주어지면, 60 프레임의 chunk들 중에서 중간 frame의 force들을 plot함.
    def lists_for_plot(f, pf, target_index, plot_index):
        # plot을 위해, 시퀀스 별로 predicted force 리스트를 정리함
        f_result = f[:len(f) - (target_index - plot_index)]

        predicted_force_sequence = []
        predicted_force_sequence.extend(pf[0][:plot_index])  # 첫 번째 데이터는 모두 저장
        for i in range(1, len(pf)):
            predicted_force_sequence.append(pf[i][plot_index - 1])  # 그 이후부턴 마지막 프레임만 저장

        mse = torch.nn.functional.mse_loss(torch.Tensor(predicted_force_sequence).float(),
                                           torch.Tensor(f_result).float(),
                                           reduction='none').tolist()
        mae = torch.nn.functional.l1_loss(torch.Tensor(predicted_force_sequence).float(),
                                          torch.Tensor(f_result).float(),
                                          reduction='none').tolist()

        return f_result, predicted_force_sequence, mse, mae


    @staticmethod
    # chunk들끼리 겹치는 부분은 평균 내어 리스트를 재정렬함.
    def lists_for_plot_mean(f, pf):
        # plot을 위해, 시퀀스 별로 predicted force 리스트를 정리함
        predicted_force_sequence = [[] for _ in range(len(f))]

        # predicted force의 모든 값을 리스트에 append함.
        for i in range(0, len(pf)):
            for j in range(0, len(pf[i])):
                predicted_force_sequence[i + j].append(pf[i][j])

        for i in range(0, len(predicted_force_sequence)):
            predicted_force_sequence[i] = np.mean(np.array(predicted_force_sequence[i]), axis=0)

        mse = torch.nn.functional.mse_loss(torch.Tensor(predicted_force_sequence).float(), torch.Tensor(f).float(), reduction='none').tolist()
        mae = torch.nn.functional.l1_loss(torch.Tensor(predicted_force_sequence).float(), torch.Tensor(f).float(), reduction='none').tolist()

        return f, predicted_force_sequence, mse, mae


    @staticmethod
    # 재정렬된 force와 predicted force가 주어지면, 이를 plot하여 저장함.
    # 때문에 force 리스트와 predicted force 리스트의 shape은 완전히 같아야 함. (# of sequence, frames of each sequence, 3)
    def plot(f, pf, save_dir, folder_name, use_wandb=False, label=None):
        # 모든 sequence에 대해서 수행
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for index in range(len(f)):
            fig, axes = plt.subplots(3, 1, figsize=(14, 12))  # plot 만듦

            _f = f[index]
            _pf = pf[index]

            if label is not None:
                plt.suptitle(f"{index}th test sequence data ({bodypart_label.get(label[index])})")
            else:
                plt.suptitle(f"{index}th test sequence data")

            # 해당 plot에 대해서만 MSE, MAE
            mse = torch.nn.functional.mse_loss(torch.Tensor(_f).float(), torch.Tensor(_pf).float(), reduction='none')
            mae = torch.nn.functional.l1_loss(torch.Tensor(_f).float(), torch.Tensor(_pf).float(), reduction='none')

            for num_data in range(3):  # 3개 Fx, Fy, Fz에 대하여
                axes[num_data].plot(np.array(_f)[:, num_data], color='r', label='force')
                axes[num_data].plot(np.array(_pf)[:, num_data], color='b', label='predicted force')
                axes[num_data].set_xlabel('Frame')
                axes[num_data].set_ylabel('Force (N)')
                if num_data == 0 or num_data == 1:
                    axes[num_data].set_ylim([-2, 2])
                elif num_data == 2:
                    axes[num_data].set_ylim([-5, 0.5])

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
            if not os.path.exists(f"{save_dir}/{folder_name}"): os.makedirs(f"{save_dir}/{folder_name}")

            if label is not None:
                plt.savefig(f'{save_dir}/{folder_name}/test_{index}th_data.png',
                            dpi=300, facecolor='white', edgecolor='black',
                            format='png', pad_inches=0.1, transparent=False)
                plt.savefig(f'{save_dir}/{folder_name}/test_{index}th_data.svg',
                            dpi=300,
                            facecolor='white', edgecolor='black',
                            format='svg', pad_inches=0.1, transparent=False)
            else:
                plt.savefig(f'{save_dir}/{folder_name}/test_{bodypart_label.get(label[index])}_{index}th_data.png', dpi=300,
                            facecolor='white', edgecolor='black',
                            format='png', pad_inches=0.1, transparent=False)
                plt.savefig(f'{save_dir}/{folder_name}/test_{bodypart_label.get(label[index])}_{index}th_data.svg', dpi=300,
                            facecolor='white', edgecolor='black',
                            format='svg', pad_inches=0.1, transparent=False)

            if use_wandb:
                img = wandb.Image(plt.imread(f'{save_dir}/{folder_name}/test_{index}th_data.png'))
                wandb.log({f"{folder_name}_test_{index}th_data": img})

            plt.cla()  # clear the current axes
            plt.clf()  # clear the current figure
            plt.close()  # closes the current figure




    @staticmethod
    # force와, chunk 상태의 predicted force가 주어지면, 각기 다른 방식으로 plot하여 저장하고 MSE & MAE를 계산함.
    def plot_for_lstm(f, pf, target_index, save_dir, train_txt_path, use_wandb=False, label=None):

        # ===============================================================
        # 중간 frame만 이어붙이는 식으로 예측
        f_for_plot = []  # shape: (# of sequence, frame of sequence, 3)
        pf_for_plot = []
        mse_for_plot = []
        mae_for_plot = []
        for seq_index in range(len(f)):
            _f, _pf, _mse, _mae = Visualization.lists_for_plot(f[seq_index], pf[seq_index], target_index, target_index // 2)
            f_for_plot.append(_f)
            pf_for_plot.append(_pf)
            mse_for_plot.extend(_mse)
            mae_for_plot.extend(_mae)

        with open(train_txt_path, 'a') as train_txt_file:
            train_txt_file.write("\nTest: median frame\n")
            train_txt_file.write(f'RMSE: {torch.sqrt(torch.mean(torch.Tensor(mse_for_plot)))}\n')
            train_txt_file.write(f'MSE: {torch.mean(torch.Tensor(mse_for_plot))}\n')
            train_txt_file.write(f'MAE: {torch.mean(torch.Tensor(mae_for_plot))}\n')

        Visualization.plot(f_for_plot, pf_for_plot, save_dir, "median_frame", use_wandb=use_wandb, label=label)
        # ===============================================================

        # ===============================================================
        # 가장 마지막 frame만 이어붙이는 식으로 예측
        f_for_plot = []  # shape: (# of sequence, frame of sequence, 3)
        pf_for_plot = []
        mse_for_plot = []
        mae_for_plot = []
        for seq_index in range(len(f)):
            _f, _pf, _mse, _mae = Visualization.lists_for_plot(f[seq_index], pf[seq_index], target_index, target_index)
            f_for_plot.append(_f)
            pf_for_plot.append(_pf)
            mse_for_plot.extend(_mse)
            mae_for_plot.extend(_mae)

        with open(train_txt_path, 'a') as train_txt_file:
            train_txt_file.write("\nTest: last frame\n")
            train_txt_file.write(f'RMSE: {torch.sqrt(torch.mean(torch.Tensor(mse_for_plot)))}\n')
            train_txt_file.write(f'MSE: {torch.mean(torch.Tensor(mse_for_plot))}\n')
            train_txt_file.write(f'MAE: {torch.mean(torch.Tensor(mae_for_plot))}\n')

        Visualization.plot(f_for_plot, pf_for_plot, save_dir, "last_frame", use_wandb=use_wandb, label=label)
        # ===============================================================

        # ===============================================================
        # chunk끼리 겹치는 frame은 stack하여 평균 냄
        f_for_plot = []  # shape: (# of sequence, frame of sequence, 3)
        pf_for_plot = []
        mse_for_plot = []
        mae_for_plot = []
        for seq_index in range(len(f)):
            _f, _pf, _mse, _mae = Visualization.lists_for_plot_mean(f[seq_index], pf[seq_index])
            f_for_plot.append(_f)
            pf_for_plot.append(_pf)
            mse_for_plot.extend(_mse)
            mae_for_plot.extend(_mae)

        with open(train_txt_path, 'a') as train_txt_file:
            train_txt_file.write("\nTest: mean frame\n")
            train_txt_file.write(f'RMSE: {torch.sqrt(torch.mean(torch.Tensor(mse_for_plot)))}\n')
            train_txt_file.write(f'MSE: {torch.mean(torch.Tensor(mse_for_plot))}\n')
            train_txt_file.write(f'MAE: {torch.mean(torch.Tensor(mae_for_plot))}\n\n')

        Visualization.plot(f_for_plot, pf_for_plot, save_dir, "mean_frame", use_wandb=use_wandb, label=label)
        # ===============================================================
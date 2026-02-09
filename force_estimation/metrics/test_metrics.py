import torch
import numpy as np
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class TestMetrics:

    def __init__(self, is_classification : bool = False):
        # =================== Regression Metrics ===================
        self.ground_truth_for_sequence = []

        # 각 시퀀스마다 하나의 리스트를 갖고, 해당 리스트 내에 sliding window를 순서대로 저장
        self.prediction_for_window = []
        self.ground_truth_for_window = []

        self.mean_force_for_axis = None # for calculate r2 score
        self.upper_bound_for_axis = None # for calculate NRMSE

        # =================== Classification Metrics ===================
        self.is_classification = is_classification
        if self.is_classification:
            # 현재로서는, 하나의 sliding window에서 하나의 값을 예측함.
            self.cross_entropy = []
            self.ground_truth_label = []
            self.prediction_label = []

        # =================== inference time ===================
        self.inference_times = []


    def init_new_sequence(self):
        self.prediction_for_window.append([])
        self.ground_truth_for_window.append([])


    def save_sequence(self, ground_truth : torch.Tensor) -> None:
        self.ground_truth_for_sequence.append(ground_truth)


    def save_regression_result_of_window(self, prediction : torch.Tensor, ground_truth : torch.Tensor) -> None:
        self.prediction_for_window[len(self.prediction_for_window) - 1].extend(prediction)
        self.ground_truth_for_window[len(self.ground_truth_for_window) - 1].extend(ground_truth)


    def save_classification_data_of_window(self, prediction : torch.Tensor, ground_truth : torch.Tensor) -> None:
        assert self.is_classification, "This function can only be called when is_classification is True."
        self.cross_entropy.extend(torch.nn.functional.cross_entropy(prediction, ground_truth.float(), reduction='none'))
        y_pred = torch.max(prediction, dim=1)[1].cpu().numpy()
        y_true = torch.argmax(ground_truth, dim=1).cpu().numpy()

        self.ground_truth_label.extend(y_true)
        self.prediction_label.extend(y_pred)


    def save_accuracy_metrics(self, accuracy : float) -> None:
        assert self.is_classification, "This function can only be called when is_classification is True."
        self.accuracy = accuracy


    def save_inference_time(self, inference_time : float) -> None:
        self.inference_times.append(inference_time)


    def set_mean_of_test(self, mean_force_for_axis):
        self.mean_force_for_axis = torch.Tensor(mean_force_for_axis)


    def set_upper_bound_of_test(self, upper_bound_for_axis):
        self.upper_bound_for_axis = torch.Tensor(upper_bound_for_axis)


    def get_metrics_for_filter(self, mask):

        # shape: (num_of_chunks, target_index, 3)
        filtered_prediction = [sequence for sequence, m in zip(self.prediction_for_window, mask) if m]
        filtered_ground_truth = [sequence for sequence, m in zip(self.ground_truth_for_window, mask) if m]

        filtered_prediction = torch.from_numpy(np.concatenate(filtered_prediction, axis=0))
        filtered_ground_truth = torch.from_numpy(np.concatenate(filtered_ground_truth, axis=0))

        mse = torch.nn.functional.mse_loss(filtered_prediction, filtered_ground_truth, reduction='none')
        mae = torch.nn.functional.l1_loss(filtered_prediction, filtered_ground_truth, reduction='none')

        return torch.mean(mse), torch.mean(mae), torch.sqrt(torch.mean(mse))  # MSE, MAE, RMSE


    def get_metrics_for_label(self):
        from force_estimation.with_classification.bodypart_label import bodypart_label

        for label in bodypart_label.keys():
            mask = (self.ground_truth_label == label)

            filtered_prediction = torch.from_numpy(np.concatenate(self.prediction_for_window, axis=0))[mask]
            filtered_ground_truth = torch.from_numpy(np.concatenate(self.ground_truth_for_window, axis=0))[mask]

            mse = torch.nn.functional.mse_loss(filtered_prediction, filtered_ground_truth, reduction='none')
            mae = torch.nn.functional.l1_loss(filtered_prediction, filtered_ground_truth, reduction='none')

            mse_for_axis = torch.mean(mse, dim=(0, 1))
            mae_for_axis = torch.mean(mae, dim=(0, 1))

            print(f"{bodypart_label[label]}: MSE: {torch.mean(mse)} (X: {mse_for_axis[0]:.6f}, Y: {mse_for_axis[1]:.6f}, Z: {mse_for_axis[2]:.6f}) ,\
            MAE: {torch.mean(mae)} (X: {mae_for_axis[0]:.6f}, Y: {mae_for_axis[1]:.6f}, Z: {mae_for_axis[2]:.6f})")


    def get_regression_metrics(self, export_r2_score : bool = False, export_nrmse : bool = False):
        if export_r2_score: assert self.mean_force_for_axis is not None, "R2 score is not supported yet."
        if export_nrmse: assert self.upper_bound_for_axis is not None, "NRMSE is not supported yet."

        message = ''


        # ================== Classification Result ==================
        ground_truth_label, prediction_label = None, None
        if self.is_classification:
            self.cross_entropy = torch.Tensor(self.cross_entropy)
            ground_truth_label = np.array(self.ground_truth_label) # 1-d array
            prediction_label = np.array(self.prediction_label) # 1-d array
            self.accuracy = accuracy_score(ground_truth_label, prediction_label)

            message += f"TEST: \tCE loss: {torch.mean(self.cross_entropy):.5f}\tAccuracy: {self.accuracy:.5f}\n"
            message += classification_report(ground_truth_label, prediction_label, digits=6)
            message += "\n\n"
        # ===========================================================



        # ================== Regression Result ==================
        # shape: (num_of_chunks, target_index, 3)
        self.prediction = torch.from_numpy(np.concatenate(self.prediction_for_window, axis=0))
        self.ground_truth = torch.from_numpy(np.concatenate(self.ground_truth_for_window, axis=0))

        # shape: (num_of_chunks, target_index, 3)
        mse = torch.nn.functional.mse_loss(self.prediction, self.ground_truth, reduction='none')
        mae = torch.nn.functional.l1_loss(self.prediction, self.ground_truth, reduction='none')

        mse_for_axis = torch.mean(mse, dim=(0, 1))
        mae_for_axis = torch.mean(mae, dim=(0, 1))

        message += f"TEST: \tMSE loss: {torch.mean(mse):.6f}\tL1 loss: {torch.mean(mae):.6f}\tRMSE loss: {torch.sqrt(torch.mean(mse)):.6f}\n"
        if export_r2_score:
            r2_score = 1 - (torch.sum((self.ground_truth - self.prediction) ** 2) / torch.sum((self.ground_truth - self.mean_force_for_axis) ** 2))
            message += f"(Coefficient of Determination, R^2) ALL: {torch.mean(r2_score):.6f}\n"
        if export_nrmse:
            nrmse_for_axis = torch.sqrt(torch.mean(mse, dim=(0, 1))) / self.upper_bound_for_axis
            message += f"(NRMSE) X: {nrmse_for_axis[0]:.6f}\tY: {nrmse_for_axis[1]:.6f}\tZ: {nrmse_for_axis[2]:.6f}\n"
        message += f"(TEST RMSE) X: {torch.sqrt(mse_for_axis[0])}\tY: {torch.sqrt(mse_for_axis[1])}\tZ: {torch.sqrt(mse_for_axis[2])}\n"
        message += f'(TEST MSE) X: {mse_for_axis[0]}\tY: {mse_for_axis[1]}\tZ: {mse_for_axis[2]}\n'
        message += f'(TEST MAE) X: {mae_for_axis[0]}\tY: {mae_for_axis[1]}\tZ: {mae_for_axis[2]}\n\n'
        # ===========================================================


        # =================== inference time ===================
        self.inference_times = np.array(self.inference_times)
        mean = self.inference_times.mean()
        std = self.inference_times.std()
        p95 = np.percentile(self.inference_times, 95)

        message += "====== Inference time statistics ======\n"
        message += f"Mean\t: {mean:.6f} ms\n"
        message += f"Std\t: {std:.6f} ms\n"
        message += f"P95\t: {p95:.6f} ms\n"
        # ===========================================================

        if not export_r2_score and not export_nrmse:
            return message, torch.mean(mse), torch.mean(mae)
        elif not export_r2_score and export_nrmse:
            return message, torch.mean(mse), torch.mean(mae), nrmse_for_axis
        elif export_r2_score and not export_nrmse:
            return message, torch.mean(mse), torch.mean(mae), torch.mean(r2_score)
        else:
            return message, torch.mean(mse), torch.mean(mae), torch.mean(r2_score), nrmse_for_axis

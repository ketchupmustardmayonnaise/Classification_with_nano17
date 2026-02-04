import torch
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class TrainMetrics:

    def __init__(self, is_classification : bool = False):
        # =================== Regression Metrics ===================
        self.mse = []
        self.mae = []

        # =================== Classification Metrics ===================
        self.is_classification = is_classification

        if self.is_classification:
            self.cross_entropy = []
            self.ground_truth_label = []
            self.prediction_label = []

    def initialize_metrics(self):
        self.mse = []
        self.mae = []

        if self.is_classification:
            self.cross_entropy = []
            self.ground_truth_label = []
            self.prediction_label = []

    def calculate_regression_metrics(self, prediction : torch.Tensor, ground_truth : torch.Tensor):
        mse = torch.nn.functional.mse_loss(prediction, ground_truth, reduction='none')
        mae = torch.nn.functional.l1_loss(prediction, ground_truth, reduction='none')

        self.mse.extend(mse)
        self.mae.extend(mae)


    def calculate_classification_metrics(self, prediction : torch.Tensor, ground_truth : torch.Tensor):
        assert self.is_classification, "This function can only be called when is_classification is True."
        self.cross_entropy.extend(torch.nn.functional.cross_entropy(prediction, ground_truth.float(), reduction='none'))
        y_pred = torch.max(prediction, dim=1)[1].cpu().numpy()
        y_true = torch.argmax(ground_truth, dim=1).cpu().numpy()

        self.ground_truth_label.extend(y_true)
        self.prediction_label.extend(y_pred)


    def report_metrics(self):
        self.mse, self.mae = torch.cat(self.mse), torch.cat(self.mae)

        if self.is_classification:
            self.cross_entropy = torch.Tensor(self.cross_entropy)

            ground_truth_label = np.array(self.ground_truth_label) # 1-d array
            prediction_label = np.array(self.prediction_label) # 1-d array

            self.accuracy = accuracy_score(ground_truth_label, prediction_label)

            return self.mse.mean(), self.mae.mean(), self.cross_entropy.mean(), self.accuracy
        else:
            return self.mse.mean(), self.mae.mean()
import pathlib
import sys
import time
import torchinfo
import yaml
from tqdm import tqdm
from utils import *
from dataset import RFTDataset
from torch.optim import lr_scheduler

import wandb
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from model import *
from utils.apply_img_augmentation import *

class ForceEstimation:
    def __init__(self, cfg):
        # parameters
        self.cfg = cfg
        self.best_epoch = 0

        # 1. 파라미터 초기화를 위한 헬퍼 메소드 정의
        def _init_param(key, p_type=None):
            value = self.cfg.get(key)

            if value is not None and p_type is not None:
                # --- 수정된 부분 ---
                # p_type이 ast.literal_eval이고, 실제 값이 문자열일 때만 변환을 수행합니다.
                if p_type is ast.literal_eval:
                    if isinstance(value, str):
                        value = p_type(value)
                else:  # int, float, str2bool 등 다른 타입 변환은 그대로 수행
                    value = p_type(value)
                # --- 수정 끝 ---

            setattr(self, key, value)

        # 2. 공통 파라미터 리스트 정의 (파라미터명, 타입)
        # 타입이 None이면 변환 없이 그대로 사용
        common_params = [
            ('train_mode', bool),  # train일 때만 학습 (bool)
            ('use_wandb', bool),  # wandb에 올릴 것인지 (bool)
            ('resume', bool),

            ('image_type', str),  # origin, rgb, diff (str)
            ('image_size', ast.literal_eval),  # 이미지를 몇 size로 할 것인지 (int, int)
            ('target_index', int),  # 몇 frame으로 자를 것인지 (bool)
            ('stride', int),  # 몇 frame만큼 띄워서 학습할 것인지 (int)
            # ('augmentation_angle', int),  # online augmentation 시, 각도 최대치 (int)

            ('encoder', str),  # 어떤 cnn을 쓸 것인지 (str)

            ('decoder', str),  # 어떤 시계열 모델을 사용할 것인지 (str)
            ('rnn_layer', int),  # 시계열 모델 내 layer의 개수 (int)
            ('decoder_output', int),  # 시계열 모델에서 결과적으로 나온 embedding_vector의 차원 (int)

            ('fc_num_layer', int),  # fully-connected layer의 개수 (int)
            ('fc_hidden_dim', int),  # fully-connected layer의 hidden layer의 dimension 개수

            ('batch_size', int),
            ('learning_rate', float),
            ('weight_decay', float),
            ('dropout', float),
            ('num_epoch', int),

            ('optimizer_type', str),
            ('scheduler_type', str),
            ('loss_type', str),

            ('model_dir', str),
            ('save_dir', str)
        ]

        for key, p_type in common_params:
            _init_param(key, p_type)
        
        # test dataset
        # self.test_dataset = RFTDataset(target_index=self.target_index, stride=self.stride, path=cfg['test_data_dir'], dataset_type='test')
        # self.test_dataloader = DataLoader(self.test_dataset, batch_size=1, shuffle=False)


        # train
        
        # train dataset
        self.train_dataset = RFTDataset(target_index=self.target_index, stride=self.stride, path=cfg['train_data_dir'], dataset_type='train')
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size,
                                                            shuffle=True)

        # validation dataset
        # self.val_dataset = RFTDataset(target_index=self.target_index, stride=self.stride, path=cfg['val_data_dir'], dataset_type='val')
        # self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size,
        #                                                     shuffle=False)


    def check_data(self):
        
        for batch_index,(inputs, forces ,classification) in enumerate(self.train_dataloader):
            print(f"---Batch #{batch_index}Shape 확인 ---")
            print(f"X (inputs) shape: {inputs.shape}")  # 예: torch.Size([32, 3, 224, 224])
            print(f"y (targets) shape: {forces.shape}") # 예: torch.Size([32])
            print("-" * 20)

            # 2. y (레이블) 값 확인하기
            print(f"--- Batch #{batch_index} target 값 ---")
            print(f"Batch #{batch_index}의 target 값: \n{forces}")
            print("-" * 20)




if __name__ == '__main__':
    with open("config.yaml", 'r+', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    force_estimation = ForceEstimation(config)
    force_estimation.check_data()
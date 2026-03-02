import argparse
import os
import shutil
import torch

from force_estimation.dataset.rftdataset import RFTDataset


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


def read(data_dir, split):
    filename = split + ".pt"
    x, y = torch.load(os.path.join(data_dir, filename))
    return RFTDataset(x, y)
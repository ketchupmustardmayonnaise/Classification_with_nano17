import argparse
import os
import shutil

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
    

def write_basic_information(self, save_dir,use_wandb, use_all_sequence, target_indices, stride,
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
    parser.add_argument('--train_mode', default=None, type=str2bool)
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
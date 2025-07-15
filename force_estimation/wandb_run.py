import argparse
import ast
import yaml
from train import ForceEstimation
import wandb

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_mode', default=None, type=str2bool)

    parser.add_argument('--encoder', default=None, type=ast.literal_eval)
    parser.add_argument('--encoder_output', default=None, type=str)

    parser.add_argument('--decoder', default=None, type=str)
    parser.add_argument('--rnn_layer', default=None, type=int)
    parser.add_argument('--decoder_output', default=None, type=int)

    parser.add_argument('--fc_num_layer', default=None, type=int)
    parser.add_argument('--fc_hidden_dim', default=None, type=int)

    parser.add_argument('--use_all_sequence', default=None, type=str2bool)
    parser.add_argument('--target_indices', default=None, type=int)
    parser.add_argument('--stride', default=None, type=int)

    parser.add_argument('--image_type', default=None, type=str)
    parser.add_argument('--image_resize', default=None, type=str2bool)
    parser.add_argument('--image_size', default=None, type=ast.literal_eval)

    parser.add_argument('--save_dir', default=None, type=str)
    parser.add_argument('--test_data_dir', default=None, type=str)
    parser.add_argument('--train_data_dir', default=None, type=str)
    parser.add_argument('--val_data_dir', default=None, type=str)

    parser.add_argument('--optimizer', default=None, type=str)
    parser.add_argument('--scheduler', default=None, type=str)
    parser.add_argument('--batch_size', default=None, type=int)
    parser.add_argument("--num_epoch", default=None, type=int)
    parser.add_argument("--learning_rate", default=None, type=float)
    parser.add_argument("--weight_decay", default=None, type=float)
    parser.add_argument('--dropout', default=None, type=float)
    parser.add_argument('--loss', default=None, type=str)
    args = parser.parse_args()

    return args

def wandb_sweep():
    wandb.init(config=config)
    cfg = wandb.config

    model = ForceEstimation(cfg)

if __name__ == '__main__':
    f = open("sweep_exp.yaml", 'r+', encoding='utf-8')
    config = yaml.load(f, Loader=yaml.FullLoader)

    sweep_id = wandb.sweep(sweep=config)
    wandb.agent(sweep_id, wandb_sweep, count=25)

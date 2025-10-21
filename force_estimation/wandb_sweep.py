import argparse
import ast
import yaml
from train import ForceEstimation
import wandb

def wandb_sweep():
    wandb.init(config=config)
    cfg = wandb.config

    model = ForceEstimation(cfg)

if __name__ == '__main__':
    f = open("sweep.yaml", 'r+', encoding='utf-8')
    config = yaml.load(f, Loader=yaml.FullLoader)

    sweep_id = wandb.sweep(sweep=config)
    wandb.agent(sweep_id, wandb_sweep, count=30)

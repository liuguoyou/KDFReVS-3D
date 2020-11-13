import argparse
import torch
from utils import setup_runtime
from trainer_gru import Trainer_GRU
from video_model_distill import Videosup3D_LSTM_GRU
from video_dataloader_gru import get_video_data_loaders_gru


## runtime arguments
parser = argparse.ArgumentParser(description='Training configurations.')
parser.add_argument('--config', default="./configs/test_single_image.yml", type=str, help='Specify a config file path')
parser.add_argument('--gpu', default=0, type=int, help='Specify a GPU device')
parser.add_argument('--num_workers', default=1, type=int, help='Specify the number of worker threads for models loaders')
parser.add_argument('--seed', default=0, type=int, help='Specify a random seed')
args = parser.parse_args()

## set up
cfgs = setup_runtime(args)

print(cfgs)

trainer = Trainer_GRU(cfgs, Videosup3D_LSTM_GRU, get_data_loader_func=get_video_data_loaders_gru)
run_train = cfgs.get('run_train', False)
run_test = cfgs.get('run_test', False)

## run
if run_train:
    trainer.train()
if run_test:
    trainer.test()

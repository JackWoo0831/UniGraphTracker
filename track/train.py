"""
main train code
"""

import sys 
import os 
import os.path as osp 
from loguru import logger
import torch 
import yaml 
import argparse


from utils.envs import select_device
from engine.trainer import Trainer

import warnings
warnings.filterwarnings('error', category=RuntimeWarning)   

def get_args_parser():
    """ get configs from terminal

    """

    parser = argparse.ArgumentParser(description='Tracker Training')
    
    # add config file paths
    parser.add_argument('--dataset_cfg', type=str, default='./track/cfgs/dataset.yaml', help='dataset config file path')
    parser.add_argument('--assc_graph_cfg', type=str, default='./track/cfgs/assc_graph.yaml', help='assc graph config file path')
    parser.add_argument('--frame_graph_cfg', type=str, default='./track/cfgs/frame_graph.yaml', help='frame graph config file path')
    parser.add_argument('--uni_graph_cfg', type=str, default='./track/cfgs/uni_graph.yaml', help='uni graph config file path')

    # training options 
    parser.add_argument('--train_cfg', type=str, default='./track/cfgs/train.yaml', help='train config file path')
    parser.add_argument('--device', type=str, default='2', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--epochs', type=int, default=15, help='number of total epochs to run')
    parser.add_argument('--resume_path', type=str, default='', help='resume training, value is the path of model')
    parser.add_argument('--save_interval', type=int, default=3, help='checkpoint saving interval, 0 for not save')
    parser.add_argument('--save_path', type=str, default='./track/saved_weights', help='model save path')    

    # for ablation studies 
    parser.add_argument('--wo_fg', action='store_true', help='without frame graph')

    return parser


def main(cfgs):
    """ main process
    
    """
    # get cfgs 
    with open(cfgs.train_cfg, 'r') as f:
        other_training_cfgs = yaml.safe_load(f)
        # merge training cfgs 
        cfgs_dict = vars(cfgs)
        cfgs_dict.update(other_training_cfgs)  # dict
        cfgs = cfgs_dict
    
    logger.info(f'training configs are:\n {cfgs_dict}')

    # set device
    device = select_device(cfgs['device'])

    trainer = Trainer(cfgs, device)

    trainer.train()

if __name__ == '__main__':
    cfgs = get_args_parser().parse_args()

    # debug
    # cfgs.resume_path = 'track/weights/UAVDT_epoch24_20230421.pth'
    main(cfgs)

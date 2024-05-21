"""
Calculate the flops and params use thop
"""
from thop import clever_format
from thop import profile

import os 
import os.path as osp 
import numpy as np 
import torch 
import torch.nn as nn
import cv2 

import random
import yaml 

from models.uni_graph import UniGraph

SEQ_ROOT = '/data/wujiapeng/datasets/UAVDT/images/train'

def get_gt_info_of_two_frams(seq_dir, ):
    """
    get random two nearby frames within a seq 

    Return:
        dict, dict
    """
    imgs = sorted(os.listdir(seq_dir))
    seq_length = len(imgs)
    frame_idx1 = random.randint(0, seq_length - 2)
    frame_idx2 = frame_idx1 + 1

    img_to_gt_path = lambda path: path.replace('images', 'labels').replace('jpg', 'txt')

    img1_path = os.path.join(seq_dir, imgs[frame_idx1 - 1])
    img2_path = os.path.join(seq_dir, imgs[frame_idx2 - 1])

    gt1_path = img_to_gt_path(img1_path)
    gt2_path = img_to_gt_path(img2_path)

    img1_info = _read_gt_to_dict(img1_path, gt1_path, frame_idx1)
    img2_info = _read_gt_to_dict(img2_path, gt2_path, frame_idx2)

    return img1_info, img2_info

def _read_gt_to_dict(img_path, gt_path, frame_id):
    """
    read the gt txt and read as dict
    """
    img = cv2.imread(img_path)
    img_info = {}
    img_info['image_path'] = img_path
    img_info['frame_id'] = frame_id 
    img_info['seq_name'] = ''
    img_info['ori_size'] = [img.shape[1], img.shape[0]]

    frame_gt = np.loadtxt(gt_path, dtype=int, delimiter=' ')

    if len(frame_gt.shape) == 1:
        frame_gt = frame_gt[None, :]  # avoid situation that only one line

    objects_info_list = []
    for frame_obj_gt in frame_gt:
        obj_info = {
            'id': frame_obj_gt[0],
            'bbox': frame_obj_gt[1: -1],
            'category': frame_obj_gt[-1]
        }

        objects_info_list.append(obj_info)

    img_info['objects'] = objects_info_list

    return img_info

def get_info_for_model_input(gt_info, device):
    """
    Corresponding to _get_dict_for_model in trainer.py
    """
    ret_dict = {}
    bboxes, ids, cat_ids = [], [], []
    for item in gt_info['objects']:
        bboxes.append(item['bbox'])
        ids.append(item['id'])
        cat_ids.append(item['category'])
    
    # fit the input of class UniGraph.forward()
    det_results = np.stack(bboxes, axis=0)
    det_results = np.concatenate(
        (det_results, np.ones((det_results.shape[0], 2)), np.array(cat_ids, dtype=int)[:, np.newaxis]),
        axis=1,
    )

    det_results = torch.from_numpy(det_results)
    ids = torch.tensor(ids)

    ret_dict['det_results'] = det_results.to(device)
    ret_dict['reid_features'] = torch.randn(size=(det_results.shape[0], 512)).to(device)

    gt_info = {
        'frame_id': gt_info['frame_id'],
        'id': ids, 
        'bbox': bboxes,
        'ori_img_size': gt_info['ori_size'], 
    }

    ret_dict['gt_info'] = gt_info
    return ret_dict

if __name__ == '__main__':

    with open('track/cfgs/frame_graph.yaml', 'r') as f:
        frame_graph_cfg = yaml.safe_load(f)
    with open('track/cfgs/assc_graph.yaml', 'r') as f:
        assc_graph_cfg = yaml.safe_load(f)
    with open('track/cfgs/uni_graph.yaml', 'r') as f:
        uni_graph_cfg = yaml.safe_load(f)

    # cuda = torch.cuda.is_available()
    # device = torch.device('cuda:0' if cuda else 'cpu')
    device = torch.device('cpu')
    print(device)

    model = UniGraph(frame_graph_cfg, assc_graph_cfg, uni_graph_cfg, 
                    ori_img_size=[1920, 1080], model_img_size=[1280, 1280], device=device)
    
    seqs = os.listdir(SEQ_ROOT)

    all_macs, all_params = [], []
    
    for seq in seqs:
        seq_dir = os.path.join(SEQ_ROOT, seq)

        t0_info, t1_info = get_gt_info_of_two_frams(seq_dir)

        input_t0 = get_info_for_model_input(t0_info, device)
        input_t1 = get_info_for_model_input(t1_info, device)

        macs, params = profile(model, inputs=(input_t0, input_t1), )
        macs_, params_ = clever_format([macs, params], "%.3f")

        print(f'seq: {seq}', f"num_obj {input_t0['det_results'].shape[0]}", macs_, params_)

        all_macs.append(macs)
        all_params.append(params)

    min_macs = min(all_macs)
    max_macs = max(all_macs)
    avg_macs = sum(all_macs) / len(all_macs)

    min_params = min(all_params)
    avg_params = sum(all_params) / len(all_params)

    avg_macs, avg_params = clever_format([avg_macs, avg_params], "%.3f")
    min_macs, min_params = clever_format([min_macs, min_params], "%.3f")
    # max_macs, max_params = clever_format([max_macs, min_params], "%.3f")
    print('avg: ', avg_macs, avg_params)
    print('min: ', min_macs, min_params)
    print('max: ', max_macs, avg_params)

    # python track/cal_flops_and_params.py
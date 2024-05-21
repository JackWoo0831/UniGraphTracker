"""
main track code 
"""

import sys 
import os 
import os.path as osp 
from loguru import logger
import torch 
from torch.utils.data import DataLoader
import yaml 
import argparse


from utils.envs import select_device
from engine.tracker import Tracker
from data.datasets import TestDataset
from utils.tracking_utils import *

from tqdm import tqdm

# YOLOX modules
sys.path.append(os.getcwd())
from yolox.exp import get_exp 
from yolox_utils.postprocess import postprocess_yolox
from yolox.utils import fuse_model

# YOLOv7 modules
try:
    sys.path.append(os.getcwd())
    from yolov7.models.experimental import attempt_load
    from yolov7.utils.torch_utils import select_device, time_synchronized, TracedModel
    from yolov7.utils.general import non_max_suppression, scale_coords, check_img_size
    from yolov7_utils.postprocess import postprocess as postprocess_yolov7

except:
    pass

# yolov8 models:
try:
    from ultralytics import YOLO
    from yolov8_utils.postprocess import postprocess as postprocess_yolov8
except:
    pass

def get_args_parser():
    """ get configs from terminal

    """

    parser = argparse.ArgumentParser(description='Tracker Inference')
    
    # add config file paths
    parser.add_argument('--dataset_cfg', type=str, default='./track/cfgs/dataset.yaml', help='dataset config file path')
    parser.add_argument('--assc_graph_cfg', type=str, default='./track/cfgs/assc_graph.yaml', help='assc graph config file path')
    parser.add_argument('--frame_graph_cfg', type=str, default='./track/cfgs/frame_graph.yaml', help='frame graph config file path')
    parser.add_argument('--uni_graph_cfg', type=str, default='./track/cfgs/uni_graph.yaml', help='uni graph config file path')

    # tracking options 
    parser.add_argument('--track_cfg', type=str, default='./track/cfgs/track.yaml', help='train config file path')
    parser.add_argument('--device', type=str, default='4', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    parser.add_argument('--save_images', action='store_true', help='save tracking results (image)')
    parser.add_argument('--save_videos', action='store_true', help='save tracking results (video)')  

    parser.add_argument('--save_data_type', type=str, default='mot_challenge', help='default, mot challenge or visdrone')

    # thresh
    parser.add_argument('--high_det_thresh', type=float, default=0.5, help='high detection thresh')
    parser.add_argument('--low_det_thresh', type=float, default=0.15, help='low detection thresh')
    parser.add_argument('--edge_cls_thresh', type=float, default=0.5, help='edge conf thresh(may not be used)')

    # for ablation studies 
    parser.add_argument('--wo_fg', action='store_true', help='without frame graph')

    return parser

def main(cfgs):
    # get cfgs 
    with open(cfgs.track_cfg, 'r') as f:
        other_tracking_cfgs = yaml.safe_load(f)
        # merge training cfgs 
        cfgs_dict = vars(cfgs)
        cfgs_dict.update(other_tracking_cfgs)  # dict
        cfgs = cfgs_dict
    
    logger.info(f'tracking configs are:\n {cfgs}')

    # set device
    device = select_device(cfgs['device'])


    # get configs
    det_config = cfgs['detector']
    det_config['name'] = det_config['name'].lower()

    with open(cfgs['dataset_cfg'], 'r') as f:
        dataset_cfg = yaml.safe_load(f)

    if cfgs['save_videos']:
        cfgs['save_images'] = True 

    """
    1. load detection model
    """ 

    if det_config['name'] == 'yolox':

        exp = get_exp(det_config['exp_file'], None)  # TODO: modify num_classes etc. for specific dataset
        model_img_size = exp.input_size
        model = exp.get_model()
        model.to(device)
        model.eval()

        logger.info(f"loading detector {det_config['name']} checkpoint {det_config['weights']}")
        ckpt = torch.load(det_config['weights'], map_location=device)
        model.load_state_dict(ckpt['model'])
        logger.info("loaded checkpoint done")
        model = fuse_model(model)

        stride = None  # match with yolo v7

        logger.info(f'Now detector is on device {next(model.parameters()).device}')

    elif det_config['name'] == 'yolov7':

        logger.info(f"loading detector {det_config['name']} checkpoint {det_config['weights']}")
        model = attempt_load(det_config['weights'], map_location=device)

        # get inference img size
        stride = int(model.stride.max())  # model stride
        model_img_size = check_img_size(det_config['img_size'], s=stride)  # check img_size

        # Traced model
        model = TracedModel(model, device=device, img_size=det_config['img_size'])
        model.half()

        logger.info("loaded checkpoint done")

        logger.info(f'Now detector is on device {next(model.parameters()).device}')

    elif det_config['name'] == 'yolov8':

        logger.info(f"loading detector {det_config['name']} checkpoint {det_config['weights']}")
        model = YOLO(det_config['weights'])

        model_img_size = [None, None]  
        stride = None 

        logger.info("loaded checkpoint done")

    else:
        logger.error(f"detector {det_config['name']} is not supprted")
        exit(0)

    """
    2. load dataset and track
    """
    # data_root/images/test
    logger.info(f"tracking {dataset_cfg['dataset_name']} dataset")
    dataset_cfg = dataset_cfg[dataset_cfg['dataset_name']]
    seqs = os.listdir(osp.join(dataset_cfg['path'], 'images', cfgs['tracker']['split']))
    seqs = sorted(seqs)

    # seqs = ['uav0000201_00000_v']

    # for MOT17 inference
    if dataset_cfg['name'] == 'mot17':
        seqs = [seq for seq in seqs if 'FRCNN' in seq]

    logger.info(f'following seqs will be tracked: \n{seqs}')

    folder_name = get_save_folder_name(dataset_cfg['name'])
    
    tracker = Tracker(cfgs, device=device, det_model=det_config['name'])  # init tracker every new seq

    for seq in seqs:

        logger.info(f'tracking seq {seq}')

        tracker.reset()

        dataset = TestDataset(dataset_cfg, seq, img_size=model_img_size, split=cfgs['tracker']['split'],
                              model=det_config['name'], stride=stride)
        
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        process_bar = enumerate(data_loader)
        process_bar = tqdm(process_bar, total=len(data_loader), ncols=150)

        results = []

        for frame_idx, (ori_img, img) in process_bar:

            # if frame_idx >= 130: break
            
            if det_config['name'] == 'yolov8':
                img = img.squeeze(0).cpu().numpy()

            else:
                img = img.to(device)  # (1, C, H, W)
                img = img.float()

            # get detector output 
            with torch.no_grad():
                if det_config['name'] == 'yolov8':
                    output = model.predict(img, conf=det_config['conf_thresh'], iou=det_config['nms_thresh'])
                else:
                    output = model(img)

            # postprocess output
            if det_config['name'] == 'yolox':
                output = postprocess_yolox(output, dataset_cfg['num_classes'], conf_thresh=det_config['conf_thresh'], 
                                           img=img, ori_img=ori_img)

            elif det_config['name'] == 'yolov7':
                output = postprocess_yolov7(output, det_config, img.shape[2:], ori_img.shape)

            elif det_config['name'] == 'yolov8':
                output = postprocess_yolov8(output)
            
            else: raise NotImplementedError


            # update tracker
            # output: (tlbr, conf, cls)
            current_tracks = tracker.update(output, img, ori_img)
        
            # save results
            cur_tlwh, cur_id, cur_cls, cur_score = [], [], [], []
            for trk in current_tracks:
                bbox = trk.tlwh
                id = trk.track_id
                cls = trk.cls
                score = trk.score

                # filter low area bbox
                vertical = bbox[2] / bbox[3] > cfgs['tracker']['wh_ratio_filter']
                if bbox[2] * bbox[3] > cfgs['tracker']['min_area'] and not vertical:
                    cur_tlwh.append(bbox)
                    cur_id.append(id)
                    cur_cls.append(cls)
                    cur_score.append(score)
                    # results.append((frame_id + 1, id, bbox, cls))

            results.append((frame_idx + 1, cur_id, cur_tlwh, cur_cls, cur_score))

            if cfgs['save_images']:
                plot_img(ori_img, frame_idx, [cur_tlwh, cur_id, cur_cls], 
                         save_dir=os.path.join(dataset_cfg['path'], 'result_images', seq))
                
        save_results(folder_name, seq, results, data_type=cfgs['save_data_type'])

        if cfgs['save_images'] and cfgs['save_videos']:
            save_videos(data_root=dataset_cfg['path'], seq_names=seq)

        
        # feature_to_save = {'frame_ids': torch.tensor(tracker.feature_frame_idx), 
        #                    'feats': torch.vstack(tracker.feature_to_save)}
        # torch.save(feature_to_save, '4_ugt.pt')

        # torch.save(tracker.obersve_cost_matrix, 'cm_reid.pt')

        



if __name__ == '__main__':
    cfgs = get_args_parser().parse_args()

    main(cfgs)

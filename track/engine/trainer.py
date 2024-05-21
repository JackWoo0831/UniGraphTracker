import sys 
import os 
import os.path as osp 
from loguru import logger
import numpy as np 
import torch 
from torch.utils.data import dataloader
import torchvision.transforms as T
import yaml 
import math 
import time 
from tqdm import tqdm
import cv2 

from yolox.exp import get_exp 
from yolox.utils import fuse_model, get_model_info, postprocess, vis

from torchreid.utils import FeatureExtractor

from models.uni_graph import UniGraph, UniGraph_wo_FG
from data.mapper import TwoFrameMapper
from data.datasets import TrainValDataset
from utils.general import increment_name

class Trainer:
    def __init__(self, cfgs, device) -> None:
        """
        Args:  
            cfgs: config dict 
            device: torch.device

        Return:    
            None 
        """
        self.cfgs = cfgs 
        self.training_opts = self.cfgs['tracker']
        self.device = device 

        # other params 
        self.start_epoch, self.end_epoch = 1, self.cfgs['epochs'] 
        self.save_path = increment_name(osp.join(self.cfgs['save_path'], 'runs'))
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        # load dataloader
        self.train_loader = self._get_data_loader()

        # load detector model 
        if self.cfgs['detector']['name'].lower() == 'yolox':
            self.detector = self._get_detector(self.cfgs['detector']['exp_file'], 
                                               self.cfgs['detector']['weights'])
            self.detector.eval()

        # load re-id model 
        self.reid = self._get_reid(self.cfgs['reid']['weights'])
        self.reid.model.eval() 

        # load tracker model 
        self.uni_graph = self._get_tracker()

        # load optimizer 
        self.optimizer = self._get_optimizer()

        # load lr scheduler
        self.lr_scheduler = self._get_lr_scheduler()

    
    def train(self, ):
        """ main train process
        
        """
        logger.info('Start training!')
        self.start_time = time.time()
        
        for self.epoch in range(self.start_epoch, self.end_epoch + 1):
            self.train_before_epoch()
            self.train_in_epoch(self.epoch)
            self.train_after_epoch(self.epoch)
            
    def train_before_epoch(self, ):
        """ set process bar, optimizer etc.

        Args:

        Return:
            None        
        """
        self.max_iternum = len(self.train_loader)

        # convert dataloader to iterable and set process bar
        self.process_bar = enumerate(self.train_loader)
        self.process_bar = tqdm(self.process_bar, 
                                total=self.max_iternum, 
                                ncols=200)
        
        # set lr scheduler 
        if self.epoch > self.start_epoch:
            self.lr_scheduler.step() 

        self.uni_graph.train() 
        self.optimizer.zero_grad() 

        self.mean_loss = 0.0  # cal mean loss during training
        self.mean_edge_loss, self.mean_node_loss = 0.0, 0.0

    def train_in_epoch(self, epoch):
        """ process in every epoch

        Args:
            epoch: int 

        Return:
            None
        """
        for self.step, self.batch_data in self.process_bar:
            # 0. (optional) get detector results 
            if self.cfgs['detector']['enabled']:
                pass
            # 1. get reid features
            if len(self.batch_data) == 1: self.batch_data = self.batch_data[0]  # for bs = 1

            features_all = self._get_reid_features(self.batch_data)
            
            # 2. construct frame info dict, see models.uni_graph.UniGraph
            t0_dict = self._get_dict_for_model(self.batch_data[0], features_all[0])
            t1_dict = self._get_dict_for_model(self.batch_data[1], features_all[1])

            # 3. forward and compute loss
            loss, loss_dict = self.uni_graph(t0_dict, t1_dict)
            self.loss = loss.item()
            self.loss_dict = loss_dict

            # 4. backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 5. print info
            self._print_info()

    def train_after_epoch(self, epoch):
        """ save weights...

        Args:
            epoch: int 

        Return:
            None      
        
        """

        if not self.cfgs['save_interval']: return

        if not epoch % self.cfgs['save_interval'] or epoch == self.end_epoch:
            save_dict = {
                'model': self.uni_graph.state_dict(),
                'optimizer': self.optimizer.state_dict(), 
                'epoch': epoch,
                # 'loss': self.loss, 
            }
            save_file = osp.join(self.save_path, f'epoch{epoch}.pth')
            torch.save(save_dict, save_file)
            logger.info(f'Saved ckpt file at {save_file}')

        torch.cuda.empty_cache()
    
    def _get_detector(self, exp_file, weight_path):
        """ load detector model of YOLO-X 

        Args:
            exp_file: str, exp file for YOLO-X model 
            weight_path: trained YOLO-X model path

        Return:
            torch.nn.Module            
        
        """
        logger.info(f'loading YOLOX model for {exp_file}')
    
        exp = get_exp(exp_file, None)  # TODO: modify num_classes etc. for specific dataset
        model = exp.get_model()
        model.to(self.device)
        
        logger.info(f'Now detector is on device {next(model.parameters()).device}')

        self.__model_img_size = exp.input_size  # for Uni graph construction, Tuple(height, width)

        logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

        if weight_path:
            ckpt = torch.load(weight_path, map_location=self.device)
            model.load_state_dict(ckpt['model'])
            logger.info('loaded detector checkpoint done.')

        return model 

    def _get_reid(self, weight_path):
        """ load re-id model

        Args:
            weight_path: trained re-id model path

        Return:
            torch.nn.Module 
        
        """
        logger.info(f"loading reid model {self.cfgs['reid']['name']}")

        if self.cfgs['reid']['name'].lower() == 'osnet':
            # get model name to load OSNet 
            # e.g., ./weights/osnet_x0_75_market1501.pt -> osnet_x0_75
            model_name = str(weight_path).rsplit('/', 1)[-1].split('.')[0]
            model_name = model_name.rsplit('_', 1)[0]            

            reid_model = FeatureExtractor(
                model_name=model_name,
                model_path=weight_path,
                image_size=self.cfgs['reid']['image_size'],
                device=str(self.device)
            )

        else: raise NotImplementedError            

        logger.info('loaded reid checkpoint done.')

        return reid_model


    def _get_tracker(self):
        """ load association graph model

        Args:

        Return:
            torch.nn.Module 
        
        """
        logger.info('loading graph model')
        # read config yamls 
        with open(self.cfgs['frame_graph_cfg'], 'r') as f:
            frame_graph_cfg = yaml.safe_load(f)
        with open(self.cfgs['assc_graph_cfg'], 'r') as f:
            assc_graph_cfg = yaml.safe_load(f)
        with open(self.cfgs['uni_graph_cfg'], 'r') as f:
            uni_graph_cfg = yaml.safe_load(f)
        
        if self.cfgs['wo_fg']:
            logger.warning('without framge graph')

            model = UniGraph_wo_FG(frame_graph_cfg, assc_graph_cfg, uni_graph_cfg, 
                            ori_img_size=[1920, 1080], model_img_size=self.__model_img_size, device=self.device)
        else:
            model = UniGraph(frame_graph_cfg, assc_graph_cfg, uni_graph_cfg, 
                            ori_img_size=[1920, 1080], model_img_size=self.__model_img_size, device=self.device)
            
        model.to(self.device)

        # resume 
        if self.cfgs['resume_path']:
            ckpt = torch.load(self.cfgs['resume_path'], map_location=self.device)
            logger.info(f"loaded ckpt in epoch {ckpt['epoch']} for resume training")

            model.load_state_dict(ckpt['model'])
            self.start_epoch += ckpt['epoch']

        logger.info(f'Now tracker is on device {next(model.parameters()).device}')

        logger.info('loaded graph model done')

        return model 

    def _get_data_loader(self, ):
        """ get data loader 

        Args:

        Return:
            torch.utils.data.dataloader.Dataloader
        """
        logger.info('loading data loader')

        with open(self.cfgs['dataset_cfg'], 'r') as f:
            dataset_cfg = yaml.safe_load(f)

        if dataset_cfg['mapper'] == 'TwoFrameMapper':
            data_dict = TrainValDataset(dataset_cfg[dataset_cfg['dataset_name']], 
                                        split='train',
                                        cache_dataset=dataset_cfg['cache_dataset'],
                                        cache_path=dataset_cfg['cache_path'],
                                        prob_add_fp=dataset_cfg['prob_add_fp'],
                                        max_fp_number=dataset_cfg['max_fp_number'])
            
            train_loader = TwoFrameMapper(dataset=data_dict, 
                                          cfgs=dataset_cfg)
            
            my_collect_fn = lambda x: x
            train_loader = dataloader.DataLoader(
                dataset=train_loader, 
                batch_size=self.cfgs['batch_size'],
                shuffle=True,
                collate_fn=my_collect_fn                
            )

        else: raise NotImplementedError

        logger.info('loaded data loader done')

        return train_loader
    
    def _get_optimizer(self, ):
        """ get optimizer

        Args:

        Return:
            torch.optim.Adam | torch.optim.SGD
        
        """
        logger.info(f"using optimizer {self.training_opts['optimizer']}")
        if self.training_opts['optimizer'].lower() == 'adam':
            optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, self.uni_graph.parameters()), 
                                         lr=self.training_opts['lr0'],
                                         )
        elif self.training_opts['optimizer'].lower() == 'sgd':
            optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, self.uni_graph.parameters()), 
                                        lr=self.training_opts['lr0'],
                                        momentum=self.training_opts['momentum'],
                                        )
        else:
            optimizer = None

        # resume 
        if self.cfgs['resume_path']:
            ckpt = torch.load(self.cfgs['resume_path'], map_location=self.device)
            logger.info(f"loaded ckpt in epoch {ckpt['epoch']} for resume training")

            optimizer.load_state_dict(ckpt['optimizer'])

        return optimizer
    
    def _get_lr_scheduler(self, ):
        """ get lr scheduler

        Args:

        Return:
            torch.optim.lr_scheduler
        
        """
        logger.info(f"using learning rate scheduler {self.training_opts['lr_scheduler']}")
        epochs = self.cfgs['epochs']
        if self.training_opts['lr_scheduler'] == 'Cosine':  
            lr_lambda = \
                lambda x: ((1 - math.cos(x * math.pi / epochs)) / 2) * (self.training_opts['lr0'] - 1) + 1
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)

        elif self.training_opts['lr_scheduler'] == 'Constant':
            lr_lambda = \
                lambda x: 1.0
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)

        elif self.training_opts['lr_scheduler'] == 'Step':
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=[int(epochs) - 9, int(epochs) - 3],
                gamma=0.2
            )

        else: lr_scheduler = None 

        return lr_scheduler


    def _print_info(self, ):
        """ print some info
        
        """
        self.mean_edge_loss = (self.mean_edge_loss * self.step + self.loss_dict['edge_loss']) / (self.step + 1)
        self.mean_node_loss = (self.mean_node_loss * self.step + self.loss_dict['node_loss']) / (self.step + 1)
        self.mean_loss = (self.mean_loss * self.step + self.loss) / (self.step + 1)

        self.process_bar.set_description(
            desc='epoch: {:d}, cur_loss: {:.6f}, mean_loss: {:.6f}, mean_e_loss: {:.6f}, mean_n_loss: {:.6f}, current_lr: {:.6f}'\
                .format(self.epoch, self.loss, self.mean_loss, self.mean_edge_loss, self.mean_node_loss, self.optimizer.param_groups[0]['lr'])
        )


    def _get_reid_features(self, img_info_list):
        """ get reid features from list of img info

        Args:
            img_info_list: List[dict], see data.datasets.TrainValDataset

        Returns:
            List[torch.Tensor], each tensor with shape (num of objs, feature_dims)
        """

        features_all = []

        # set norm transform
        pixel_mean = [0.485, 0.456, 0.406]
        pixel_std = [0.229, 0.224, 0.225]

        self.reid_toPIL = T.ToPILImage()
        self.reid_norm = T.Compose([
            T.Resize(size=self.cfgs['reid']['image_size']), 
            T.ToTensor(),
            T.Normalize(pixel_mean, pixel_std),
        ])
 
        for img_info in img_info_list:
            img_path = img_info['image_path']
            ori_img = cv2.imread(img_path)

            crop_batch = []

            for item in img_info['objects']:
                bbox = item['bbox']
                crop = ori_img[bbox[1]: bbox[3], bbox[0]: bbox[2]]
                crop = self.reid_toPIL(crop)
                crop = self.reid_norm(crop)

                crop_batch.append(crop)

            crop_batch = torch.stack(crop_batch, dim=0).to(self.device)
            features = self.reid(crop_batch)


            features_all.append(features)

        return features_all
            
    def _get_dict_for_model(self, img_info_dict, reid_features):
        """ convert dict format from dataloader to UniGraph

        Args:
            img_info_dict: dict 
            reid_features: torch.Tensor

        Return:
            dict
        
        """

        ret_dict = {}
        if not self.cfgs['detector']['enabled']:
            bboxes, ids, cat_ids = [], [], []
            for item in img_info_dict['objects']:
                bboxes.append(item['bbox'])
                ids.append(item['id'].item())
                cat_ids.append(item['category'].item())
            
            # fit the input of class UniGraph.forward()
            det_results = np.stack(bboxes, axis=0)
            det_results = np.concatenate(
                (det_results, np.ones((det_results.shape[0], 2)), np.array(cat_ids, dtype=int)[:, np.newaxis]),
                axis=1,
            )

            det_results = torch.from_numpy(det_results)
            ids = torch.tensor(ids)

            ret_dict['det_results'] = det_results.to(self.device)
            ret_dict['reid_features'] = reid_features 

            gt_info = {
                'frame_id': img_info_dict['frame_id'],
                'id': ids, 
                'bbox': bboxes,
                'ori_img_size': img_info_dict['ori_size'], 
            }

            ret_dict['gt_info'] = gt_info

        else:
            pass

        return ret_dict
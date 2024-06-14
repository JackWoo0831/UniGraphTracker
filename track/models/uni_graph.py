"""
build unified graph: frame graph + assc graph
"""

import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F

from models.frame_graph import FrameGraph
from models.association_graph import AssociationGraph 

class UniGraph(nn.Module):
    """ Unified Graph
    
    In every timestamp, do this steps:
    1. build frame graph and update it (weight sharing in different frames)
    2. build assc graph with features output from frame graph
        and update it 
    3. cal loss(training) or track(inference)
    
    """
    def __init__(self, frame_graph_cfg, assc_graph_cfg, uni_graph_cfg, ori_img_size,
                 model_img_size, device, *args, **kwargs) -> None:
        """ Construct function of class UniGraph
        
        Args:
            frame_graph_cfg, assc_graph_cfg, uni_graph_cfg: dict that read from yaml
            ori_img_size: List[int, int] | Tuple(int, int), raw image w and h
            model_img_size: List[int, int] | Tuple(int, int), image w and h resized in detector

        """
        super().__init__()

        self.frame_graph = FrameGraph(frame_graph_cfg, ori_img_size, device=device)
        self.assc_graph = AssociationGraph(assc_graph_cfg, ori_img_size, device=device)

        self.cfg = uni_graph_cfg
        self.ori_img_size, self.model_img_size = ori_img_size, model_img_size

        self.device = device

    def forward(self, t0_dict, t1_dict):
        """ forward process

        Args:
            t0_dict, t1_dict, dict:
            { 'det_results': torch.Tensor, shape: (num of objects, 7), which means t-l-b-r, obj_conf, class_conf, class_pred
              'reid_features': torch.Tensor, shape: (num of objects, feature dims)
              'gt_info': dict: 
                {'frame_id': int,
                 'id': torch.Tensor, (num of objects, )
                 'bbox': torch.Tensor, (num of objects, 4)
                }
            }

        Return:
            loss: float
        
        """

        # update origin image size 
        self.ori_img_size = t0_dict['gt_info']['ori_img_size']

        # step1. construct frame graph for last frame
        node_features_t0, edge_features_t0, edge_index_t0, id_map_t0, occ_label_t0 = \
            self.frame_graph.forward(t0_dict['reid_features'], t0_dict['det_results'][:, :4], 
                                     t0_dict['det_results'][:, 4], self.ori_img_size, 
                                     is_training=True, gt_info=t0_dict['gt_info'])

        # step2. construct frame graph for cur frame

        node_features_t1, edge_features_t1, edge_index_t1, id_map_t1, occ_label_t1 = \
            self.frame_graph.forward(t1_dict['reid_features'], t1_dict['det_results'][:, :4], 
                                     t1_dict['det_results'][:, 4], self.ori_img_size, 
                                     is_training=True, gt_info=t1_dict['gt_info'])

        # step3. set t0 features as tracklet graph 
        self.assc_graph.set_tracklet_graph(node_features_t0[-1], edge_index_t0, edge_features_t0[-1],
                                           t0_dict['det_results'][:, :4], id_map_t0, occ_label_t0)

        # step4. construct assc graph and compute loss
        loss, loss_dict = self.assc_graph.forward(node_features_t1[-1], edge_index_t1, 
                                       t1_dict['det_results'][:, :4], self.ori_img_size, 
                                       gt_info=None, node_target_id_cur_frame=id_map_t1,
                                       occ_label_cur_frame=occ_label_t1)

        return loss, loss_dict

    def inference(self, trk_dict, det_dict, ori_img_size):
        """ inference process

        Args:
            trk_dict, dict:
            { 'id_map': torch.Tensor, shape: (num of objects, 2)
              'bboxes': torch.Tensor, shape: (num of objects, 4)
              'node_features': torch.Tensor, shape: (num of objects, feat dim)
              'edge_index': torch.Tensor, shape: (2, num of edges)
              'edge_features': torch.Tensor, shape: (num of edges, feat dim)
              'num': int

            }

            det_dict, dict:
            { 'det_results': torch.Tensor, shape: (num of objects, 7), which means t-l-b-r, obj_conf, class_conf, class_pred
              'reid_features': torch.Tensor, shape: (num of objects, feature dims),
              'num': int
            }

        Returns:
            confs, edge_index, node_features_t1, node_occ_prob       
        """
        self.ori_img_size = ori_img_size 
        det_socres = det_dict['det_results'][:, 4]

        node_features_t1, edge_features_t1, edge_index_t1, = \
            self.frame_graph.inference(det_dict['reid_features'], det_dict['det_results'][:, :4],
                                       det_socres, ori_img_size=self.ori_img_size)
        
        if not trk_dict:  # only have detections
            return None, None, node_features_t1, torch.zeros(size=(det_dict['num'], ))
        
        self.assc_graph.set_tracklet_graph(trk_dict['node_features'], trk_dict['edge_index'], 
                                           trk_dict['edge_features'], x_position=trk_dict['bboxes'], id_map=trk_dict['id_map'],
                                           occ_label=None)
        
        confs, edge_index, confs_node = self.assc_graph.inference(node_features_t1, edge_index_t1, det_dict['det_results'][:, :4], 
                                          ori_img_size=self.ori_img_size)
        
        # filter low-conf but no-occ objs and relative edges
        confs_node_cur = confs_node[-det_dict['num']: ]
        if self.cfg['mask_occ']:
            tau = torch.exp(-1.2 * confs_node_cur) - 0.5 
            tau[tau < 0.1] = 0.1
            mask_occ = det_socres < tau.flatten()

            discard_dets = torch.nonzero(mask_occ).squeeze()
            mask_edge = torch.isin(edge_index[1, :], discard_dets + trk_dict['num'])
            confs[mask_edge] = -1       
        
        return confs, edge_index, node_features_t1, confs_node_cur

    def inference_wo_AG(self, trk_dict, det_dict, ori_img_size):
        """ inference with out assc graph, for ablation study
        
        """

        self.ori_img_size = ori_img_size 
        det_socres = det_dict['det_results'][:, 4]

        node_features_t1, edge_features_t1, edge_index_t1, = \
            self.frame_graph.inference(det_dict['reid_features'], det_dict['det_results'][:, :4],
                                       det_socres, ori_img_size=self.ori_img_size)
        
        # cal node similarity as conf 

        node_features_t0 = trk_dict['node_features'] 

        # norm 
        node_features_t0 /= torch.norm(node_features_t0, dim=1, keepdim=True) 
        node_features_t1 /= torch.norm(node_features_t1, dim=1, keepdim=True)

        confs = torch.matmul(node_features_t0, node_features_t1.transpose(0, 1))

        return confs.cpu(), None, node_features_t1


class UniGraph_wo_FG(nn.Module):
    """ Unified Graph without frame graph
    
    """

    def __init__(self, frame_graph_cfg, assc_graph_cfg, uni_graph_cfg, ori_img_size,
                 model_img_size, device, *args, **kwargs) -> None:
        """ Construct function of class UniGraph
        
        Args:
            frame_graph_cfg, assc_graph_cfg, uni_graph_cfg: dict that read from yaml
            ori_img_size: List[int, int] | Tuple(int, int), raw image w and h
            model_img_size: List[int, int] | Tuple(int, int), image w and h resized in detector

        """
        super().__init__()

        self.frame_graph = None 
        self.assc_graph = AssociationGraph(assc_graph_cfg, ori_img_size, device=device)

        self.cfg = uni_graph_cfg
        self.ori_img_size, self.model_img_size = ori_img_size, model_img_size

        self.device = device

    def forward(self, t0_dict, t1_dict):
        """ forward process

        """

        # update origin image size 
        self.ori_img_size = t0_dict['gt_info']['ori_img_size']
        
        # set reid feature as node features
        node_features_t0 = F.adaptive_avg_pool1d(t0_dict['reid_features'], output_size=160)
        node_features_t1 = F.adaptive_avg_pool1d(t1_dict['reid_features'], output_size=160)

        edge_index_t0 = torch.Tensor([])
        edge_index_t1 = torch.Tensor([])
        edge_features_t0 = torch.Tensor([])

        id_map_t0 = torch.cat([torch.arange(0, node_features_t0.shape[0]).reshape(-1, 1),
                              t0_dict['gt_info']['id'].reshape(-1, 1)], dim=1)  # (num_of_detections, 2)
        
        id_map_t1 = torch.cat([torch.arange(0, node_features_t1.shape[0]).reshape(-1, 1),
                              t1_dict['gt_info']['id'].reshape(-1, 1)], dim=1)  # (num_of_detections, 2)
        
        occluded_labels_t0 = torch.zeros(size=(1, node_features_t0.shape[0]), dtype=torch.long, device=self.device)
        occluded_labels_t1 = torch.zeros(size=(1, node_features_t1.shape[0]), dtype=torch.long, device=self.device)

        # set t0 features as tracklet graph 
        self.assc_graph.set_tracklet_graph(node_features_t0, edge_index_t0, edge_features_t0,
                                           t0_dict['det_results'][:, :4], id_map_t0, occluded_labels_t0)

        # construct assc graph and compute loss
        loss = self.assc_graph.forward(node_features_t1, edge_index_t1, 
                                       t1_dict['det_results'][:, :4], self.ori_img_size, 
                                       gt_info=None, node_target_id_cur_frame=id_map_t1, occ_label_cur_frame=occluded_labels_t1)

        return loss 

    def inference(self, trk_dict, det_dict, ori_img_size):
        """ inference process

        """
        self.ori_img_size = ori_img_size 
        det_socres = det_dict['det_results'][:, 4]
        
        node_features_t1 = F.adaptive_avg_pool1d(det_dict['reid_features'], 160)
        edge_index_t1 = torch.Tensor([])
        
        if not trk_dict:  # only have detections
            return None, None, node_features_t1, torch.zeros(size=(det_dict['num'], ))
        
        # TODO: id_map
        self.assc_graph.set_tracklet_graph(trk_dict['node_features'], trk_dict['edge_index'], 
                                           trk_dict['edge_features'], x_position=trk_dict['bboxes'], id_map=trk_dict['id_map'], 
                                           occ_label=None)
        
        confs, edge_index, confs_node = self.assc_graph.inference(node_features_t1, edge_index_t1, det_dict['det_results'][:, :4], 
                                          ori_img_size=self.ori_img_size)
        
        confs_node_cur = confs_node[-det_dict['num']: ]
        if self.cfg['mask_occ']:
            tau = torch.exp(-1.2 * confs_node_cur) - 0.5 
            tau[tau < 0.1] = 0.1
            mask_occ = det_socres < tau.flatten()

            discard_dets = torch.nonzero(mask_occ).squeeze()
            mask_edge = torch.isin(edge_index[1, :], discard_dets + trk_dict['num'])
            confs[mask_edge] = -1       
        
        return confs, edge_index, node_features_t1, confs_node_cur
    
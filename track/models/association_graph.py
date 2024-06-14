"""
build cross frame(association) graph
"""

import time 

import numpy as np
import torch 
import torch.nn as nn
from torch_geometric.data import Data, DataLoader, Batch
from torch_geometric.nn import GATConv, HEATConv, GeneralConv, GATv2Conv
from torch_geometric.nn import MetaLayer
import torchvision.ops as ops
import torch.nn.functional as F

import utils.cal_distance as cal_distance
from models.graph_net import NodeEncodeNetAG, EdgeEncodeNetAG, \
    EdgeUpdateNetAG, NodeUpdateNetAG, myMetaLayer, EdgeClassifier, NodeClassifier, MultiLoss

class AssociationGraph(nn.Module):
    """ Association Graph Class 

    Association graph consists of tracklet graph(that is frame graph in past time)
    and frame graph at current frame

    Association graph omits edges in tracklet graph and frame graph, and build edges
    between tracklet graph and framge graph
    
    """
    def __init__(self, assc_graph_cfg, ori_img_size, device, *args, **kwargs) -> None:
        """ Construct function of class AssociationGraph

        Args:
            assc_graph_cfg: dict that read from yaml
            ori_img_size: List[int, int] | Tuple(int, int), image w and h
        
        """
        super().__init__()

        self.cfg = assc_graph_cfg
        self.device = device
        
        self.TrackletGraph = Data(x=None, edge_index=None, edge_attr=None, 
                                  active_mask=None, x_position=None, id_map=None)  # torch_geometric.data.Data,
        # tracklet graph that stores info of tracklets 

        self.edge_classifier = None if not self.cfg['edge_cls'] \
            else EdgeClassifier(self.cfg['edge_cls'])
        
        self.node_classifier = None if not self.cfg['node_cls'] \
           else NodeClassifier(self.cfg['node_cls'])
        
        self.beta = [1.0, 0.2]
        self.multi_loss = MultiLoss(method=assc_graph_cfg['multi_loss_method'])

        # params
        self.edge_build_method = self.cfg['edge_build_method']
        self.edge_feature_parts = self.cfg['edge_feature_parts']

        self.net_type = self.cfg['forward_method']

        self.build_net()

    def build_graph(self, det_node_feat, det_edge_index, det_position, ori_img_size):
        """ build association graph

        Args:
            det_node_feat: torch.Tensor, shape: (num_of_detections, feature_dim), 
            node feature at current frame, including high conf and low conf detections

            det_edge_index: torch.Tensor, shape: (2, num_of_edges), 
            edge index at current frame

            det_position: torch.Tensor, shape: (num_of_detections, 4), 
            bbox of detections, format: tlbr

            ori_img_size: List[int, int] | Tuple(int, int), image w and h

        Return:
            torch_geometric.data.Data
        
        """

        # tracklet node and edge feat
        tracklet_node_feat = self.TrackletGraph.x
        tracklet_node_edge_index = self.TrackletGraph.edge_index 
        # tracklet_active_mask = self.TrackletGraph.active_mask
        tracklet_position = self.TrackletGraph.x_position

        tracklet_num = tracklet_node_feat.shape[0]
        det_num = det_node_feat.shape[0]

        node_feature_tensor = torch.cat([tracklet_node_feat, det_node_feat], dim=0)  # shape: (num of tracklets + num of detections, feature dim)

        max_connect_edges = min(self.cfg['max_connect_edges'], det_num)  # max number
        # of how many a tarcklet node can connect with detection nodes

        # convert tlbr to xywh 
        tracklet_position_  = self._convert_position_feature(tracklet_position)
        det_position_ = self._convert_position_feature(det_position)

        # for each tracklet node, cal dist of every detection node
        # in order to speed up the process, use matrix to parallel

        edge_index_tracklet, edge_index_det = torch.Tensor([]), torch.Tensor([])

        # [0, 1, 2, 3] -> [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, ...]
        edge_index_helper = torch.arange(0, tracklet_num).unsqueeze(1).\
            repeat(1, max_connect_edges).flatten()

        if 'iou_dist' in self.edge_build_method:
            iou_mat = ops.box_iou(boxes1=tracklet_position, boxes2=det_position)  # shape: (num of tracklets, num of detections)
            max_iou_values, max_iou_indices = iou_mat.topk(k=max_connect_edges, dim=1)  # shape: (num of tracklets, k)

            edge_index_tracklet = torch.concat([edge_index_tracklet, edge_index_helper], dim=0)
            edge_index_det = torch.concat([edge_index_det, max_iou_indices.cpu().flatten()], dim=0)

        if 'ecu_dist' in self.edge_build_method:
            point_tracklet = tracklet_position_[:, :2]
            point_det = det_position_[:, :2]

            dist_mat = torch.cdist(point_tracklet, point_det, p=2)  # cal Eculid dist
            max_dist_values, max_dist_indices = dist_mat.topk(k=max_connect_edges, dim=1)  # shape: (num of tracklets, k)

            edge_index_tracklet = torch.concat([edge_index_tracklet, edge_index_helper], dim=0)
            edge_index_det = torch.concat([edge_index_det, max_dist_indices.cpu().flatten()], dim=0)

        if 'sim_dist' in self.edge_build_method:
            # cos_sim(x, y) = <x, y> / (||x|| · ||y||)
            tracklet_node_feat_normed = F.normalize(tracklet_node_feat, p=2, dim=1)
            det_node_feat_normed = F.normalize(det_node_feat, p=2, dim=1)
            sim_mat = torch.matmul(tracklet_node_feat_normed, det_node_feat_normed.transpose(1, 0))
            max_sim_values, max_sim_indices = sim_mat.topk(k=max_connect_edges, dim=1)

            edge_index_tracklet = torch.concat([edge_index_tracklet, edge_index_helper], dim=0)
            edge_index_det = torch.concat([edge_index_det, max_sim_indices.cpu().flatten()], dim=0)

        # edge_index_tracklet, edge_index_det: shape: (len(self.edge_build_method) * tracklet_num * max_connect_edges, )
        
        edge_index = torch.unique(torch.stack([edge_index_tracklet,
                                               edge_index_det]), dim=1).long()
        edge_index_half = edge_index.clone().detach()  # half, only tracklet -> det

        # undirected graph, tracklet -> det && det -> tracklet
        edge_index = torch.cat([edge_index, edge_index.flip(dims=[0])], dim=1)  # shape: (2, ~tracklet_num * max_connect_edges * 2)
        edge_index = edge_index.to(self.device)
    

        # cal edge feat according to edge_index
        edge_feature_list = []  # List[torch.Tensor], store every part to a list
        if 'xy_diff' in self.edge_feature_parts:
            scaled_x_diff = tracklet_position_[edge_index_half[0], 0] - det_position_[edge_index_half[1], 0]  # shape: (~tracklet_num*max_connect_edges, max_connect_edges)
            scaled_y_diff = tracklet_position_[edge_index_half[0], 1] - det_position_[edge_index_half[1], 1]

            scaled_x_diff, scaled_y_diff = scaled_x_diff / ori_img_size[0], scaled_y_diff / ori_img_size[1]

            scaled_x_diff = torch.cat([scaled_x_diff, -1 * scaled_x_diff], dim=0)  # shape: (2*tracklet_num*max_connect_edges, max_connect_edges)
            scaled_y_diff = torch.cat([scaled_y_diff, -1 * scaled_y_diff], dim=0)

            edge_feature_list.append(scaled_x_diff)
            edge_feature_list.append(scaled_y_diff)

        if 'iou' in self.edge_feature_parts:
            iou_feature = iou_mat[edge_index_half[0], edge_index_half[1]]  # shape: (~tracklet_num*max_connect_edges, )
            iou_feature = torch.cat([iou_feature, iou_feature], dim=0)
            edge_feature_list.append(iou_feature)

        if 'wass' in self.edge_feature_parts:
            wass_mat = cal_distance.bbox_wassertein(bboxes1=tracklet_position, bboxes2=det_position)

            wass_feature = wass_mat[edge_index_half[0], edge_index_half[1]]
            wass_feature = torch.cat([wass_feature, wass_feature], dim=0)
            edge_feature_list.append(wass_feature)

        if 'buffered_iou' in self.edge_feature_parts:
            pass 

        if 'log_wh' in self.edge_feature_parts:
            vi_width, vi_height = tracklet_position_[edge_index_half[0], 2], tracklet_position_[edge_index_half[0], 3]
            vj_width, vj_height = det_position_[edge_index_half[1], 2], det_position_[edge_index_half[1], 3]

            # log(wi / wj), log(hi / hj)
            log_wh = torch.cat([torch.log(vi_width / vj_width).unsqueeze(1),  
                                torch.log(vi_height / vj_height).unsqueeze(1)], dim=1)
            log_wh = torch.cat([log_wh, -1 * log_wh], dim=0)

            edge_feature_list.append(log_wh)

        if 'feature' in self.edge_feature_parts:
            node_feature_merge_method = self.cfg['node_feature_merge_method']
            vi_features = tracklet_node_feat[edge_index_half[0]]  # shape: (num_of_edges / 2, feature_dims)
            vj_features = det_node_feat[edge_index_half[1]]  # shape: (num_of_edges / 2, feature_dims)

            if node_feature_merge_method == 'avg':
                node_feature_merge = 0.5 * (vi_features + vj_features)  # shape: (num_of_edges / 2, feature_dims)
            elif node_feature_merge_method == 'concat':
                node_feature_merge = torch.concat([vi_features, vj_features], dim=1)  # shape: (num_of_edges / 2, 2 * feature_dims)
            elif node_feature_merge_method == 'sim':
                node_feature_merge = sim_mat[edge_index_half[0], edge_index_half[1]]  # shape: (num_of_edges / 2, )
            else: 
                raise NotImplementedError
            
            edge_feature_list.append(torch.cat([node_feature_merge, node_feature_merge], dim=0))

        # we need to unify the dimention
        for idx in range(len(edge_feature_list)):
            if len(edge_feature_list[idx].shape) == 1: 
                edge_feature_list[idx] = edge_feature_list[idx][:, None]

        edge_feature_tensor = torch.concat(edge_feature_list, dim=1)  # edge_feature, shape: (num_of_edges, edge_dim)

        # in assc graph, tarcklet node idx: 0 ~ n - 1, detection node idx: 0 ~ m - 1 
        # we need to map detection node idx to n ~ n + m - 1
        self._map_det_node_idx(edge_index, tracklet_num)

        # graph
        frame_graph = Data(x=node_feature_tensor.float(), edge_index=edge_index, edge_attr=edge_feature_tensor.float())

        return frame_graph
    
    def build_net(self):
        """ build net to update the features of assc graph

        Args:

        Return:
            torch.nn.Module
        
        """

        node_mlp_cfg = self.cfg['node_mlp']
        edge_mlp_cfg = self.cfg['edge_mlp']
        msg_pass_cfg = self.cfg['message_passing']
        gnn_cfg = self.cfg['GNN']
        
        self.node_encoder = None if not node_mlp_cfg['node_encode'] \
                else NodeEncodeNetAG(node_mlp_cfg['node_encode']).to(self.device)
        self.edge_encoder = None if not edge_mlp_cfg['edge_encode'] \
            else EdgeEncodeNetAG(edge_mlp_cfg['edge_encode']).to(self.device)
        

        if self.net_type == 'message_passing':
           
            edge_update_net = EdgeUpdateNetAG(edge_mlp_cfg['edge_update']).to(self.device)
            node_update_net = NodeUpdateNetAG(node_mlp_cfg['node_update'], 
                                        msg_pass_cfg['node_aggregate_func']).to(self.device)
            
            self.net = MetaLayer(edge_update_net, node_update_net)

        else: 
            edge_update_net = EdgeUpdateNetAG(edge_mlp_cfg['edge_update']).to(self.device)

            if gnn_cfg['GNN_type'] == 'GATConv':
                node_update_net = GATConv(gnn_cfg['node_dim_input'], gnn_cfg['node_dim_output'], heads=gnn_cfg['heads'],
                                        edge_dim=gnn_cfg['edge_dim'], add_self_loops=False).to(self.device)
                
            elif gnn_cfg['GNN_type'] == 'GeneralConv':
                node_update_net = GeneralConv(gnn_cfg['node_dim_input'], gnn_cfg['node_dim_output'], in_edge_channels=gnn_cfg['edge_dim'],
                                            heads=gnn_cfg['heads'], attention=True, ).to(self.device)
                
            elif gnn_cfg['GNN_type'] == 'HEATConv':
                node_update_net = HEATConv()

            else: raise NotImplementedError

            self.net = myMetaLayer(edge_update_net, node_update_net, gnn_type=gnn_cfg['GNN_type'])

        if self.cfg['layer_norm']:
            iters = self.cfg[self.net_type]['depth']

            if self.net_type == 'GNN':
                node_hidden_dim = gnn_cfg['node_dim_output']
            else:
                node_hidden_dim = node_mlp_cfg['node_update']['dims'][-1]

            edge_hidden_dim = edge_mlp_cfg['edge_update']['dims'][-1]

            self.node_layer_norm = nn.ModuleList([nn.LayerNorm(node_hidden_dim) for _ in range(iters)])
            self.edge_layer_norm = nn.ModuleList([nn.LayerNorm(edge_hidden_dim) for _ in range(iters)])

    
    def forward(self, det_node_feat, det_edge_index, det_position, ori_img_size, gt_info, node_target_id_cur_frame, 
                occ_label_cur_frame=None, is_traning=True):
        """ forward process for training

        Args:
            det_node_feat, det_edge_index, det_position, ori_img_size: same as self.build_graph()

            node_target_id_cur_frame: torch.Tensor, shape: (num of nodes, 2)

        Return:
            loss(training) or confs(inference)     
        """
        
        # build graph
        graph_data = self.build_graph(det_node_feat, det_edge_index, det_position, ori_img_size)

        # extract features
        initial_node_features = graph_data.x 
        edge_index = graph_data.edge_index 
        initial_edge_features = graph_data.edge_attr

        alpha = None 

        node_features_list, edge_feature_list = [], []  # List[torch.Tensor], store features in middle layers

        # Encode node and edge
        if self.node_encoder is not None:
            initial_node_features = self.node_encoder(initial_node_features)
        if self.edge_encoder is not None:
            initial_edge_features = self.edge_encoder(initial_edge_features)

        middle_node_features, middle_edge_features = initial_node_features, initial_edge_features

        depth = self.cfg[self.net_type]['depth']
        residual = self.cfg[self.net_type]['residual']

        # forward
        for iter in range(depth):
            middle_node_features, middle_edge_features, alpha = \
                self.net(middle_node_features, edge_index, middle_edge_features)

            # residual connection
            if residual:
                middle_node_features = middle_node_features + initial_node_features
                middle_edge_features = middle_edge_features + initial_edge_features

            # Layer Norm
            if self.cfg['layer_norm']:
                middle_node_features = self.node_layer_norm[iter](middle_node_features)
                middle_edge_features = self.edge_layer_norm[iter](middle_edge_features)

            node_features_list.append(middle_node_features)
            edge_feature_list.append(middle_edge_features)

        if is_traning:
            # edge classify
            # cal node -> target id map for two frames
            node_target_id_last_frame = self.TrackletGraph.id_map
            # shape: (num of nodes, 2)
                        
            edge_loss = self.edge_classifier(edge_feature_list, edge_index, alpha, gt_info=gt_info,
                node_target_id_dicts=[node_target_id_last_frame, node_target_id_cur_frame])
            
            occ_labels = [self.TrackletGraph.occ_label, 
                          occ_label_cur_frame]  # TODO 
            node_loss = self.node_classifier(node_features_list[-1], edge_index=edge_index, occ_labels=occ_labels, 
                node_target_id_dicts=[node_target_id_last_frame, node_target_id_cur_frame])

            loss, loss_dict = self.multi_loss.forward(edge_loss, node_loss)
            
            return loss, loss_dict
        
        else:
            confs, edge_index_merged = self.edge_classifier.inference(edge_feature_list, edge_index, None)
            confs_node = self.node_classifier.inference(node_feature_cur=node_features_list[-1])
            return confs, edge_index_merged, confs_node


    def inference(self, det_node_feat, det_edge_index, det_position, ori_img_size,):
        """
        Args:
            same as self.forward()

        Returns:
            confs, edge_index
        """
        return self.forward(det_node_feat, det_edge_index, det_position, ori_img_size,
                            gt_info=None, node_target_id_cur_frame=None, is_traning=False)

    def set_tracklet_graph(self, x, edge_index, edge_attr, x_position, id_map, occ_label):
        """
        
        """
        self.TrackletGraph.x = x 
        self.TrackletGraph.edge_index = edge_index 
        self.TrackletGraph.edge_attr = edge_attr 
        self.TrackletGraph.x_position = x_position 
        self.TrackletGraph.id_map = id_map
        self.TrackletGraph.occ_label = occ_label

    def _update_tracklet_graph(self, det_node_feat, det_edge_index, det_position, node_target_id_cur_frame):
        """ update tracklet graph for training

        Args:
            same as self.forward()
        
        """

        self.TrackletGraph.x = det_node_feat
        self.TrackletGraph.edge_index = det_edge_index
        self.TrackletGraph.x_position = det_position 
        self.id_map = node_target_id_cur_frame[:, 1]
        

    def _map_det_node_idx(self, edge_index, tracklet_num):
        """ map detection node idx from 0 ~ m - 1 to n ~ n + m - 1
        
        Args:
            edge_index: torch.Tensor, shape: (2, num_of_edges)
            tracklet_num: int, value of n
        
        Return:
            None
        
        """
        num_of_edges = edge_index.shape[1]
        
        edge_index[1, :num_of_edges >> 1] += tracklet_num
        edge_index[0, num_of_edges >> 1:] += tracklet_num

    def _convert_position_feature(self, positions, method='xywh'):
        """ Convert bbox to a position feature
        
        Args:
            positions: torch.Tensor, shape: (num_of_objs, 4), format: tlbr
            method: 
                'xywh': convert tlbr to [xc, yc, w, h]
                'xyar': convert tlbr to [xc, yc, w / h, w * h]

        Return:
            torch.Tensor, shape: (num_of_objs, 4)
        """

        positions_ = positions.clone()
        positions_[:, 0] = 0.5 * (positions[:, 0] + positions[:, 2])
        positions_[:, 1] = 0.5 * (positions[:, 1] + positions[:, 3])
        positions_[:, 2] = positions[:, 2] - positions[:, 0]
        positions_[:, 3] = positions[:, 3] - positions[:, 1]

        if method == 'xyar':
            positions_[:, 2] /= positions_[:, 3]
            positions_[:, 3] *= positions_[:, 2] * positions_[:, 3]

        return positions_
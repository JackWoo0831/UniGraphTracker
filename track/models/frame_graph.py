"""
build inner frame graph
"""

import time 

import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader, Batch
from torch_geometric.nn import GATConv, GENConv, DeepGCNLayer
from torch_geometric.nn import MetaLayer
from torch_scatter import scatter
import torchvision.ops as ops

import utils.cal_distance as cal_distance
from models.graph_net import NodeEncodeNetFG, NodeUpdateNetFG, EdgeEncodeNetFG, EdgeUpdateNetFG, myMetaLayer

class FrameGraph(nn.Module):
    """ Frame Graph Class

    node: Re-ID feature, motion and topology feature
    edge: feature similarity between two objects
    
    """
    def __init__(self, frame_graph_cfg, ori_img_size, device, *args, **kwargs) -> None:
        """ Construct function of class FrameGraph

        Args:
            frame_graph_cfg: dict that read from yaml
        
        """
        super().__init__()

        self.cfg = frame_graph_cfg
        self.device = device
        self.max_near_obj_num = self.cfg['max_near_obj_num']
        self.distance_thresh = self.cfg['max_distance'] * min(ori_img_size[0], ori_img_size[1])
        self.edge_build_method = self.cfg['edge_build_method']
        self.edge_feature_parts = self.cfg['edge_feature_parts']

        self.net_type = self.cfg['forward_method']

        # build net
        self.build_net()

    def build_graph(self, reid_features, positions, confs, ori_img_size, 
                    gen_id_map=False, gt_info=None):
        """ build frame graph

        Args:
            reid_features: torch.Tensor, shape: (num_of_objs, feature_dims)
            positions: torch.Tensor, shape: (num_of_objs, 4), format: tlbr
            confs: torch.Tensor, shape: (num_of_objs, )
            ori_img_size: List[int, int] | Tuple(int, int), image w and h

            gen_id_map: bool, True when training
            gt_info: dict: 
                {'frame_id': int,
                 'id': torch.Tensor, (num of objects, )
                 'bbox': torch.Tensor, (num of objects, 4)
                }

        Return:
            torch_geometric.data.Data
        """

        num_of_detections = reid_features.shape[0]  # num of detections in this frame

        # Norm reid feature 
        reid_features = F.normalize(reid_features, dim=1)

        # Pooling reid features
        if self.cfg['pooling']:
            if 'avg' in self.cfg['pooling_method']:
                reid_features = F.adaptive_avg_pool1d(reid_features, 
                                                      output_size=self.cfg['pooling_shape'])
            else:
                reid_features = F.adaptive_max_pool1d(reid_features,
                                                      output_size=self.cfg['pooling_shape'])
                

        
        positions_ = positions.clone()  # tlbr
        positions = self._convert_position_feature(positions, method='xywh')  # tlbr -> xywh

        # calculate every node's topology measurement 
        topology_dim = 2 * self.max_near_obj_num  # dist + angle
        topology_feature = []  # List[Torch.Tensor], store topology feature from idx 0 ~ num_of_detections - 1

        dists_list_all = []  # List[List[List[float, int]]] = List[dists_list], len(dists_list_all) = num_of_detections
        # it stores info of distances of near objs of all detections, in order to simplify the calculation
        # of edge feature(avoid duplicate calculation) 

        occluded_labels = torch.zeros(size=(1, num_of_detections), dtype=torch.long, device=self.device)  # torch.Tensor, label each obj occluded
        # when \max{overlap_area} / area > \tau, mark occluded
        
        for i in range(num_of_detections):
            
            current_topology_feature = torch.zeros(size=(topology_dim, ))
            dists_list = []  # List[List[float, int]], list to store distance with other objs
            angles_list = []  # List[int], list to store angle with other objs

            current_bbox = positions[i]  # bbox: xywh | tlbr
            for j in range(num_of_detections):
                # find nearest self.max_near_obj_num objs

                if j == i: continue  # do not compare itself
                nearset_obj_bbox = positions[j]
                
                dist = cal_distance.cal_bbox_distance(current_bbox, nearset_obj_bbox)

                if dist > self.distance_thresh: continue

                dists_list.append([dist, j])  # append a pair to remember obj id

            dists_list = sorted(dists_list, key=lambda x : x[0])  # sorted by dist

            # cal overlap area
            
            if len(dists_list):
                nearest_obj = dists_list[0][1]
                overlap_area = self._get_overlap_area(positions_[i], positions_[nearest_obj])
                area = positions[i, 2] * positions[i, 3]
                if overlap_area > area * self.cfg['occ_thresh']:
                    occluded_labels[0, i] = int(1)

            if len(dists_list) < self.max_near_obj_num:
                # fill with 0
                cur_length = len(dists_list)
                for _ in range(self.max_near_obj_num - cur_length):
                    dists_list.append([0.0, -1])
            elif len(dists_list) > self.max_near_obj_num:
                # cut
                dists_list = dists_list[: self.max_near_obj_num]

            dists_list_all.append(dists_list)

            # calculate angles
            # dists_list = [[dist_0, idx0], [dist1, idx1], ...], where dist0 <= dist1 <= ...
            # calculate angles: [angle(idx0, idx1), angle(idx1, idx2), ...]
            for j in range(1, len(dists_list)):
                if dists_list[j][1] == -1: break  # only consider valid idx

                p1, p2 = positions[dists_list[j - 1][1]], positions[dists_list[j][1]]
                pc = current_bbox
                # cal ∠P1PcP2
                angles_list.append(cal_distance.cal_bbox_angle(pc, p1, p2))

            # make len(angels_list) == self.max_near_obj_num
            if len(angles_list) < self.max_near_obj_num:
                # fill with 0
                cur_length = len(angles_list)
                for _ in range(self.max_near_obj_num - cur_length):
                    angles_list.append(0.0)  

            for j in range(topology_dim):
                if j < topology_dim >> 1:
                    current_topology_feature[j] = dists_list[j][0]
                else:
                    current_topology_feature[j] = angles_list[j - (topology_dim >> 1)]

            topology_feature.append(current_topology_feature)
        
        topology_feature_tensor = torch.stack(topology_feature).to(self.device)  # shape: (num_of_detections, topology_dim)

        # norm positions and topology_feature_tensor
        positions_normed = positions.clone()
        positions_normed[:, 0] /= ori_img_size[0]
        positions_normed[:, 2] /= ori_img_size[0]
        positions_normed[:, 1] /= ori_img_size[1]
        positions_normed[:, 3] /= ori_img_size[1]

        topology_feature_tensor[:, :(topology_dim >> 1)] /= ori_img_size[1]
        topology_feature_tensor[:, (topology_dim >> 1):] /= 360.0

        # node feature, shape: (num_of_detections, reid_dim + position_dim + topology_dim)
        node_feature_tensor = torch.cat([reid_features, 
                                         positions_normed, 
                                         topology_feature_tensor], dim=1) 
        
        edge_feature_tensor = None  # edge_feature, shape: (num_of_edges, edge_dim)

        if self.edge_build_method == 'nearest_k':
            edge_index = self._get_edge_index_nearestK(dists_list_all).to(self.device)
        elif self.edge_build_method == 'fixed_range':
            edge_index = self._get_edge_index_fixed_range(dists_list_all).to(self.device)

        else:
            raise NotImplementedError
        
        # get edge feature by edge index
        edge_feature_list = []  # List[torch.Tensor], store every part to a list
        no_edge = 0 in edge_index.shape  # TODO: no edge situation
        if no_edge: edge_index = torch.Tensor([[0], [0]]).long().to(self.device)
        if 'xy_diff' in self.edge_feature_parts:
            # edge: (vi, vj) -> (xc_i - xc_j, yc_i -yc_j)
            scaled_x_diff = positions[edge_index[0]][:, 0] - positions[edge_index[1]][:, 0]
            scaled_y_diff = positions[edge_index[0]][:, 1] - positions[edge_index[1]][:, 1]
            # norm by image weight and height
            # shape: (num_of_edges, )
            scaled_x_diff, scaled_y_diff = scaled_x_diff / ori_img_size[0], scaled_y_diff / ori_img_size[1]

            edge_feature_list.append(scaled_x_diff)
            edge_feature_list.append(scaled_y_diff)

        if 'iou' in self.edge_feature_parts:
            # cal iou matrix first
            iou_mat = ops.box_iou(boxes1=positions_, boxes2=positions_)

            iou_feature = iou_mat[edge_index[0], edge_index[1]]  # shape: (num_of_edges, )
            edge_feature_list.append(iou_feature)

        if 'wass' in self.edge_feature_parts:
            wass_mat = cal_distance.bbox_wassertein(bboxes1=positions_, bboxes2=positions_)

            wass_feature = wass_mat[edge_index[0], edge_index[1]]
            edge_feature_list.append(wass_feature)

        if 'buffered_iou' in self.edge_feature_parts:
            pass 

        if 'log_wh' in self.edge_feature_parts:
            vi_width, vi_height = positions[edge_index[0]][:, 2], positions[edge_index[0]][:, 3]
            vj_width, vj_height = positions[edge_index[1]][:, 2], positions[edge_index[1]][:, 3]
            # log(wi / wj), log(hi / hj)

            log_wh = torch.cat([torch.log(vi_width / vj_width).unsqueeze(1),  
                                torch.log(vi_height / vj_height).unsqueeze(1)], dim=1)
            edge_feature_list.append(log_wh)

        if 'feature' in self.edge_feature_parts:
            node_feature_merge_method = self.cfg['node_feature_merge_method']
            vi_features = reid_features[edge_index[0]]  # shape: (num_of_edges, feature_dims)
            vj_features = reid_features[edge_index[1]]  # shape: (num_of_edges, feature_dims)

            if node_feature_merge_method == 'avg':
                node_feature_merge = 0.5 * (vi_features + vj_features)  # shape: (num_of_edges, feature_dims)
            elif node_feature_merge_method == 'concat':
                node_feature_merge = torch.concat([vi_features, vj_features], dim=1)  # shape: (num_of_edges, 2 * feature_dims)
            elif node_feature_merge_method == 'sim':
                sim_mat = torch.matmul(reid_features, reid_features.t())  # shape: (num_of_objs, num_of_objs)
                node_feature_merge = sim_mat[edge_index[0], edge_index[1]]  # shape: (num_of_edges, )
            else: 
                raise NotImplementedError
            
            edge_feature_list.append(node_feature_merge)

        # we need to unify the dimention
        for idx in range(len(edge_feature_list)):
            if len(edge_feature_list[idx].shape) == 1: 
                edge_feature_list[idx] = edge_feature_list[idx][:, None]

        edge_feature_tensor = torch.concat(edge_feature_list, dim=1)  # edge_feature, shape: (num_of_edges, edge_dim)

        # graph
        id_map = torch.tensor([])
        if gen_id_map:
            assert gt_info is not None 
            id_map = torch.cat([torch.arange(0, num_of_detections).reshape(-1, 1),
                              gt_info['id'].reshape(-1, 1)], dim=1)  # (num_of_detections, 2)


        frame_graph = Data(x=node_feature_tensor.float(), edge_index=edge_index, edge_attr=edge_feature_tensor.float(), id_map=id_map,
                           occ_label=occluded_labels, border=topology_dim + 4)

        return frame_graph
        
    def build_net(self):
        """ build net to update the features of frame graph
        Args:

        Return:
            torch.nn.Module
        """   

        node_mlp_cfg = self.cfg['node_mlp']
        edge_mlp_cfg = self.cfg['edge_mlp']
        msg_pass_cfg = self.cfg['message_passing']
        gnn_cfg = self.cfg['GNN']

        # NOTE: v2: separate net, appearance and topology
        # Encoders
        self.node_encoder_app = None if not node_mlp_cfg['node_encode_app'] \
            else NodeEncodeNetFG(node_mlp_cfg['node_encode_app'])
        self.node_encoder_tpy = None if not node_mlp_cfg['node_encode_tpy'] \
            else NodeEncodeNetFG(node_mlp_cfg['node_encode_tpy'])
        
        self.edge_encoder = None if not edge_mlp_cfg['edge_encode'] \
            else EdgeEncodeNetFG(edge_mlp_cfg['edge_encode']).to(self.device)
        
        # Edge update module
        edge_update_net_app = None if not edge_mlp_cfg['edge_update_app'] \
            else EdgeUpdateNetFG(edge_mlp_cfg['edge_update_app']).to(self.device)
        edge_update_net_tpy = None if not edge_mlp_cfg['edge_update_tpy'] \
            else EdgeUpdateNetFG(edge_mlp_cfg['edge_update_tpy']).to(self.device)

        if self.net_type == 'message_passing':
        
            node_update_net_app = None if not node_mlp_cfg['node_update_app'] \
                else NodeUpdateNetFG(node_mlp_cfg['node_update_app'], 
                                     msg_pass_cfg['node_aggregate_func']).to(self.device)            
            self.net_app = MetaLayer(edge_update_net_app, node_update_net_app).to(self.device)
            
            node_update_net_tpy = None if not node_mlp_cfg['node_update_tpy'] \
                else NodeUpdateNetFG(node_mlp_cfg['node_update_tpy'], 
                                     msg_pass_cfg['node_aggregate_func']).to(self.device)            
            self.net_tpy = MetaLayer(edge_update_net_tpy, node_update_net_tpy).to(self.device)

        else: 
            node_update_net_app = GENConv(in_channels=gnn_cfg['node_dim_input_app'], out_channels=gnn_cfg['node_dim_output_app'], 
                                          norm='layer', edge_dim=gnn_cfg['edge_dim'], )
            node_update_net_tpy = GENConv(in_channels=gnn_cfg['node_dim_input_tpy'], out_channels=gnn_cfg['node_dim_output_tpy'], 
                                          norm='layer', edge_dim=gnn_cfg['edge_dim'], )
            
            self.net_app = myMetaLayer(edge_update_net_app, node_update_net_app, gnn_type='GENConv', )
            self.net_tpy = myMetaLayer(edge_update_net_tpy, node_update_net_tpy, gnn_type='GENConv', )
        
        # set layer norm

        if self.cfg['layer_norm']:
            iters = self.cfg[self.net_type]['depth']

            node_hidden_dim_app = node_mlp_cfg['node_encode_app']['dims'][-1]
            node_hidden_dim_tpy = node_mlp_cfg['node_encode_tpy']['dims'][-1]
            edge_hidden_dim = edge_mlp_cfg['edge_encode']['dims'][-1]

            self.node_layer_norm_app = nn.ModuleList([nn.LayerNorm(node_hidden_dim_app) for _ in range(iters)])
            self.node_layer_norm_tpy = nn.ModuleList([nn.LayerNorm(node_hidden_dim_tpy) for _ in range(iters)])

            self.edge_layer_norm_app = nn.ModuleList([nn.LayerNorm(edge_hidden_dim) for _ in range(iters)])
            self.edge_layer_norm_tpy = nn.ModuleList([nn.LayerNorm(edge_hidden_dim) for _ in range(iters)])
            
        
    def forward(self, reid_features, positions, confs, ori_img_size, is_training=False, gt_info=None):
        """ forward propagation. Steps:
            1. build graph
            2. encode node and edge
            3. pass message passing net

        Args:
            reid_features: torch.Tensor, shape: (num_of_objs, feature_dims)
            positions: torch.Tensor, shape: (num_of_objs, 4), format: tlbr
            confs: torch.Tensor, shape: (num_of_objs, )
            ori_img_size: List[int, int] | Tuple[int, int], image w and h

            is_training: bool, if is training, generate node -> target id map
            gt_info: dict: 
                {'frame_id': int,
                 'id': torch.Tensor, (num of objects, )
                 'bbox': torch.Tensor, (num of objects, 4)

                }
        
        Return:
            'message_passing': node and edge features in every layer, List[torch.Tensor]
            'GNN': 
        """
        # build graph
        graph_data = self.build_graph(reid_features, positions, confs, ori_img_size, 
                                      gen_id_map=is_training, gt_info=gt_info)
        # print(f'build frame graph: {(cur_end_time - cur_start_time) * 1e3}')

        border = graph_data.border
        initial_node_features_app = graph_data.x[:, :-border]
        initial_node_features_tpy = graph_data.x[:, -border:]
        edge_index = graph_data.edge_index 
        initial_edge_features = graph_data.edge_attr

        node_features_list, edge_feature_list = [], []  # List[torch.Tensor], store features in middle layers

        # Encode node and edge
        if self.node_encoder_app is not None:
            initial_node_features_app = self.node_encoder_app(initial_node_features_app)
        if self.node_encoder_tpy is not None:
            initial_node_features_tpy = self.node_encoder_tpy(initial_node_features_tpy)
        if self.edge_encoder is not None:
            initial_edge_features = self.edge_encoder(initial_edge_features)

        # features in middle layers
        middle_node_features_app, middle_edge_features_app = initial_node_features_app, initial_edge_features
        middle_node_features_tpy, middle_edge_features_tpy = initial_node_features_tpy, initial_edge_features

        depth = self.cfg[self.net_type]['depth']
        residual = self.cfg[self.net_type]['residual']
                     
        # forward
        for iter in range(depth):
            # apperance forward
            middle_node_features_app, middle_edge_features_app, _ = \
                self.net_app(middle_node_features_app, edge_index, middle_edge_features_app)
            
            # topology forward
            middle_node_features_tpy, middle_edge_features_tpy, _ = \
                self.net_tpy(middle_node_features_tpy, edge_index, middle_edge_features_tpy)

            # residual connection
            if residual:
                middle_node_features_app = middle_node_features_app + initial_node_features_app
                middle_edge_features_app = middle_edge_features_app + initial_edge_features

                middle_node_features_tpy = middle_node_features_tpy + initial_node_features_tpy
                middle_edge_features_tpy = middle_edge_features_tpy + initial_edge_features

            # Layer Norm
            if self.cfg['layer_norm']:
                middle_node_features_app = self.node_layer_norm_app[iter](middle_node_features_app)
                middle_node_features_tpy = self.node_layer_norm_tpy[iter](middle_node_features_tpy)

                middle_edge_features_app = self.edge_layer_norm_app[iter](middle_edge_features_app)
                middle_edge_features_tpy = self.edge_layer_norm_tpy[iter](middle_edge_features_tpy)
        
        # combine apperance and topology
        final_node_features = torch.cat([middle_node_features_app, middle_node_features_tpy,], dim=1)

        return [final_node_features], [middle_edge_features_app + middle_edge_features_tpy], edge_index, graph_data.id_map, graph_data.occ_label
    
    def inference(self, reid_features, positions, confs, ori_img_size):
        """ inference process

        Args:
            same as self.forward()

        Returns:
            node feature, (num of nodes, node feat dim)
            edge feature, (num of edges, edge feat dim)
            edge_index, (2, num of edges)
        
        """
        node_features_list, edge_feature_list, edge_index, _, _ \
            = self.forward(reid_features, positions, confs, ori_img_size, )
        
        return node_features_list[-1], edge_feature_list[-1], edge_index

    def _get_overlap_area(self, bbox1, bbox2):
        """ cal overlap area between two boxes

        Args:
            bbox1, bbox2: torch.Tensor, tlbr

        Returns:
            float        
        """
        olap_x1 = max(bbox1[0], bbox2[0])
        olap_y1 = max(bbox1[1], bbox2[1])
        olap_x2 = min(bbox1[2], bbox2[2])
        olap_y2 = min(bbox1[3], bbox2[3])

        olap_w = max(0, olap_x2 - olap_x1)
        olap_h = max(0, olap_y2 - olap_y1)

        return olap_w * olap_h


    def _get_edge_index_nearestK(self, dists_list_all):
        """ get edge index
            The rule of edge construction: a central vertex and K vertexs connected to form an edge
            K = self.max_near_obj_num

        Args:
            dists_list_all: list that stores all the info of distances

        Return:
            torch.Tensor, shape: (~self.max_near_obj_num * num_of_detections, 2)
        """
        # find edge_index
        edge_index_center, edge_index_near = [], []  # List[int], nth edge: edge_index_center[n] -> edge_index_near[n]
        for idx_center, item in enumerate(dists_list_all):
            # item: List[List[float, int]], list of [distances, obj_id]

            for idx_obj, near_obj in enumerate(item):
                # near_obj: List[float, int]
                if near_obj[1] == -1:  # only consider valid idx
                    continue
                if not idx_obj < self.max_near_obj_num:
                    break
                edge_index_center.append(idx_center)
                edge_index_near.append(near_obj[1])
        
        # unique edge index
        edge_index = torch.stack([
            torch.Tensor(edge_index_center),
            torch.Tensor(edge_index_near)
            ])  # shape: (~self.max_near_obj_num * num_of_detections, 2)
        edge_index_unqiue = torch.unique(edge_index, dim=1)

        return edge_index_unqiue.long()
    
    def _get_edge_index_fixed_range(self, dists_list_all):
        """ get edge index
            The rule of edge construction: choose objs that in a fixed range of distance, no matter how many
        
        Args:
            dists_list_all: list that stores all the info of distances

        Return:
            torch.Tensor, shape: (~self.max_near_obj_num * num_of_detections, 2)
        """
        # find edge_index
        edge_index_center, edge_index_near = [], []  # List[int], nth edge: edge_index_center[n] -> edge_index_near[n]
        for idx_center, item in enumerate(dists_list_all):
            # item: List[List[float, int]], list of [distances, obj_id]

            for idx_obj, near_obj in enumerate(item):
                # near_obj: List[float, int]
                edge_index_center.append(idx_center)
                edge_index_near.append(near_obj[1])
        
        # unique edge index
        edge_index = torch.stack([
            torch.Tensor(edge_index_center),
            torch.Tensor(edge_index_near)
            ])  # shape: (~self.max_near_obj_num * num_of_detections, 2)
        edge_index_unqiue = torch.unique(edge_index, dim=1)

        return edge_index_unqiue.long()

        
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


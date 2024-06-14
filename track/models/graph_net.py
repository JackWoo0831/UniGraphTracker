"""
Some net to encode or update node feature use in frame graph, association graph and uni graph
"""

import math 
import torch 
import torch.nn as nn
from torch_geometric.data import Data, DataLoader, Batch
from torch_scatter import scatter
from torch_geometric.nn import GATConv

from models.loss import FocalLoss, FocalLossWithSigmoid, FocalLossAndDiceLoss

def build_layers(input_dim, fc_dims, dropout_p, norm_layer, act_func, bias=True):
    """ Build MLP

    Args:
        input_dim: int, input tensor dim
        fc_dims: List[int], dims in intermediate layer
        dropout_p: float, ratio to drop out layer
        norm_layer: nn.Module
        act_func: nn.Nodule, activation func
        bias: bool, whether use bias in Linear layer

    Return: 
        List[nn.Module]
    
    """
    layers = []
    for fc_dim in fc_dims:
        layers.append(nn.Linear(input_dim, fc_dim, bias=bias))
        layers.append(norm_layer(fc_dim))
        layers.append(act_func)
        if dropout_p > 0.0:
            layers.append(nn.Dropout(p=dropout_p))
        input_dim = fc_dim
    return layers


"""
Net used in frame graph
"""

class NodeEncodeNetFG(nn.Module):
    """
    net to encode node feature in frame graph
    """
    def __init__(self, cfgs) -> None:
        super().__init__()

        self.mlp_dims = cfgs['dims'] 
        
        drop_out_p = cfgs['drop_out_p']
        norm_layer = nn.LayerNorm

        act_func = nn.LeakyReLU(negative_slope=0.01)

        self.mlp = nn.Sequential(*build_layers(self.mlp_dims[0], self.mlp_dims[1: ], drop_out_p, norm_layer, act_func))

    def forward(self, x):
        """
        Args:
            x: torch.Tensor, shape: (num_of_nodes, node_feature_dim)
        """
        return self.mlp(x)
    

class NodeUpdateNetFG(nn.Module):
    """
    net to update node feature in frame graph
    """
    def __init__(self, cfgs, aggregate_func) -> None:
        """ constuctor
        Args:
            cfgs: dict, node_mlp['node_update'] in frame_graph.yaml
            aggregate_func: str, 'mean' 'sum' 'max' 'min'  
        
        Return:
            None
        """
        super().__init__()

        self.aggregate_func = lambda src, index, dim_size: \
            scatter(src, index, dim=0, dim_size=dim_size, reduce=aggregate_func)
        
        self.mlp_dims = cfgs['dims'] 
        
        drop_out_p = cfgs['drop_out_p']
        norm_layer = nn.LayerNorm
        act_func = nn.LeakyReLU(negative_slope=0.01)
        self.act_func = act_func
        mlp_before_agg = build_layers(self.mlp_dims[0], self.mlp_dims[1: ], drop_out_p, norm_layer, act_func)
        mlp_after_agg = build_layers(self.mlp_dims[-1], [self.mlp_dims[-1]], drop_out_p, norm_layer, act_func)

        self.mlp_before_agg = nn.Sequential(*mlp_before_agg)
        self.mlp_after_agg = nn.Sequential(*mlp_after_agg)

    def forward(self, x, edge_index, edge_attr, u=None, batch=None):
        """ forward propagation. Steps:
            1. concat node and edge feature
            2. forward a MLP
            3. aggregate features with neighbors
            4. forward another MLP
        
        Args:
            x: torch.Tensor, shape: (num_of_nodes, node_feature_dim)
            edge_index: torch.Tensor, shape: (2, num_of_edges)
            edge_attr: torch.Tensor, shape: (num_of_edges, edge_feature_dim)
        
        Return:
            torch.Tensor
        """
        row, col = edge_index  # vi -> vj
        # concat node feature and edge feature, why??
        out = torch.cat([x[row], edge_attr], dim=1)  # shape: (num_of_edges, node_feature_dim + edge_feature_dim)
        out = self.mlp_before_agg(out)  # shape: (num_of_edges, self.mlp_dims[-1])
        out = self.aggregate_func(out, col, x.shape[0])
        out = self.mlp_after_agg(out)
        # TODO: add residual 

        out += x
        return self.act_func(out)

class EdgeEncodeNetFG(nn.Module):
    """
    net to encode edge feature in graph
    """
    def __init__(self, cfgs) -> None:
        """
        Args:
            cfgs: dict, edge_mlp['edge_encode'] in frame_graph.yaml
        """
        super().__init__()

        self.mlp_dims = cfgs['dims'] 
        
        drop_out_p = cfgs['drop_out_p']
        norm_layer = nn.LayerNorm
        act_func = nn.LeakyReLU(negative_slope=0.01)

        self.mlp = nn.Sequential(*build_layers(self.mlp_dims[0], self.mlp_dims[1: ], drop_out_p, norm_layer, act_func))

    def forward(self, edge_attr):
        """
        Args:
            edge_attr: torch.Tensor, shape: (num_of_edges, edge_feature_dim)
        """
        return self.mlp(edge_attr)
    
class EdgeUpdateNetFG(nn.Module):
    """
    net to update edge feature in graph
    """
    def __init__(self, cfgs) -> None:
        """
        Args:
            cfgs: dict, edge_mlp['edge_update'] in frame_graph.yaml
        """
        super().__init__()

        self.mlp_dims = cfgs['dims'] 
        
        drop_out_p = cfgs['drop_out_p']
        norm_layer = nn.LayerNorm
        act_func = nn.LeakyReLU(negative_slope=0.01)

        self.mlp = nn.Sequential(*build_layers(self.mlp_dims[0], self.mlp_dims[1: ], drop_out_p, norm_layer, act_func))

    def forward(self, src, dest, edge_attr, u=None, batch=None):
        """
        Args:
            src: torch.Tensor, shape: (num_of_edges, node_feature_dim), that is x[edge_index[0]]
            dest: torch.Tensor, shape: (num_of_edges, node_feature_dim), that is x[edge_index[1]]
            edge_attr: torch.Tensor, shape: (num_of_edges, edge_feature_dim)
        """
        # TODO: find other method to aggregate node features
        out = torch.cat([src, dest, edge_attr], dim=1)
        return self.mlp(out)
    

"""
Net used in assc graph
"""

class NodeEncodeNetAG(nn.Module):
    """
    net to encode node feature in graph
    """
    def __init__(self) -> None:
        super().__init__()

class NodeUpdateNetAG(nn.Module):
    """
    net to update node feature in frame graph
    """
    def __init__(self, cfgs, aggregate_func) -> None:
        """ constuctor
        Args:
            cfgs: dict, node_mlp['node_update'] in frame_graph.yaml
            aggregate_func: str, 'mean' 'sum' 'max' 'min'  
        
        Return:
            None
        """
        super().__init__()

        self.aggregate_func = lambda src, index, dim_size: \
            scatter(src, index, dim=0, dim_size=dim_size, reduce=aggregate_func)
        
        self.mlp_dims = cfgs['dims'] 
        
        drop_out_p = cfgs['drop_out_p']
        norm_layer = nn.LayerNorm
        act_func = nn.LeakyReLU(negative_slope=0.01)
        mlp_before_agg = build_layers(self.mlp_dims[0], self.mlp_dims[1: ], drop_out_p, norm_layer, act_func)
        mlp_after_agg = build_layers(self.mlp_dims[-1], [self.mlp_dims[-1]], drop_out_p, norm_layer, act_func)

        self.mlp_before_agg = nn.Sequential(*mlp_before_agg)
        self.mlp_after_agg = nn.Sequential(*mlp_after_agg)

    def forward(self, x, edge_index, edge_attr, u=None, batch=None):
        """ forward propagation. Steps:
            1. concat node and edge feature
            2. forward a MLP
            3. aggregate features with neighbors
            4. forward another MLP
        
        Args:
            x: torch.Tensor, shape: (num_of_nodes, node_feature_dim)
            edge_index: torch.Tensor, shape: (2, num_of_edges)
            edge_attr: torch.Tensor, shape: (num_of_edges, edge_feature_dim)
        
        Return:
            torch.Tensor
        """
        row, col = edge_index  # vi -> vj
        # concat node feature and edge feature, why??
        out = torch.cat([x[row], edge_attr], dim=1)  # shape: (num_of_edges, node_feature_dim + edge_feature_dim)
        out = self.mlp_before_agg(out)  # shape: (num_of_edges, self.mlp_dims[-1])
        out = self.aggregate_func(out, col, x.shape[0])
        out = self.mlp_after_agg(out)
        # TODO: add residual 

        return out

class EdgeEncodeNetAG(nn.Module):
    """
    net to encode edge feature in graph
    """
    def __init__(self, cfgs) -> None:
        """
        Args:
            cfgs: dict, edge_mlp['edge_encode'] in frame_graph.yaml
        """
        super().__init__()

        self.mlp_dims = cfgs['dims'] 
        
        drop_out_p = cfgs['drop_out_p']
        norm_layer = nn.LayerNorm
        act_func = nn.LeakyReLU(negative_slope=0.01)

        self.mlp = nn.Sequential(*build_layers(self.mlp_dims[0], self.mlp_dims[1: ], drop_out_p, norm_layer, act_func))

    def forward(self, edge_attr):
        """
        Args:
            edge_attr: torch.Tensor, shape: (num_of_edges, edge_feature_dim)
        """
        return self.mlp(edge_attr)
    
class EdgeUpdateNetAG(nn.Module):
    """
    net to update edge feature in graph
    """
    def __init__(self, cfgs) -> None:
        """
        Args:
            cfgs: dict, edge_mlp['edge_update'] in frame_graph.yaml
        """
        super().__init__()

        self.mlp_dims = cfgs['dims'] 
        
        drop_out_p = cfgs['drop_out_p']
        norm_layer = nn.LayerNorm
        act_func = nn.LeakyReLU(negative_slope=0.01)

        self.mlp = nn.Sequential(*build_layers(self.mlp_dims[0], self.mlp_dims[1: ], drop_out_p, norm_layer, act_func))

    def forward(self, src, dest, edge_attr, u=None, batch=None):
        """
        Args:
            src: torch.Tensor, shape: (num_of_edges, node_feature_dim), that is x[edge_index[0]]
            dest: torch.Tensor, shape: (num_of_edges, node_feature_dim), that is x[edge_index[1]]
            edge_attr: torch.Tensor, shape: (num_of_edges, edge_feature_dim)
        """
        # TODO: find other method to aggregate node features
        out = torch.cat([src, dest, edge_attr], dim=1)
        return self.mlp(out)
    

class myMetaLayer(nn.Module):
    """ rewrite torch_geometric.nn.MetaLayer to better use Graph Attn network
    
    """
    def __init__(self, edge_model, node_model, gnn_type='GATConv', 
                 update_edge_with_attn_weights=True, ):
        super().__init__()
        self.edge_model = edge_model
        self.node_model = node_model

        self.reset_parameters()

        self.gnn_type = gnn_type
        self.update_edge_with_attn_weights = update_edge_with_attn_weights

    def reset_parameters(self):
        for item in [self.node_model, self.edge_model]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        """"""
        row = edge_index[0]
        col = edge_index[1]

        alpha = None 

        if self.edge_model is not None:
            edge_attr = self.edge_model(x[row], x[col], edge_attr)

        if self.node_model is not None:
            if self.gnn_type == 'GATConv':
                x, (edge_index, alpha) = self.node_model(x, edge_index, edge_attr, return_attention_weights=True)
            elif self.gnn_type in ['GeneralConv', 'GENConv']:
                x = self.node_model(x, edge_index, edge_attr,)
            else: raise NotImplementedError

        if self.update_edge_with_attn_weights:  # TODO find how to use attn weights
            pass 
        

        return x, edge_attr, alpha
    

class EdgeClassifier(nn.Module):
    """
    net to calculate the conf score of an edge (which represents an assc relationship) in assc graph
    """
    def __init__(self, cfgs, with_sigmoid=True, add_dice_loss=True, use_sinkhorn=False, show_attn_weights=False) -> None:
        super().__init__()

        self.mlp_dims = cfgs['dims']
        self.forward_method = cfgs['method']
        drop_out_p = cfgs['drop_out_p']
        norm_layer = nn.LayerNorm
        act_func = nn.LeakyReLU(negative_slope=0.01)

        self.mlp = nn.Sequential(*build_layers(self.mlp_dims[0], self.mlp_dims[1: ], drop_out_p, norm_layer, act_func),
                                 nn.Linear(self.mlp_dims[-1], 1))
        
        self.with_sigmoid = with_sigmoid
        
        if add_dice_loss:
            self.loss_func = FocalLossAndDiceLoss(
                alpha=0.25, gamma=2, reduction="sum", return_weighted_sum=False
            )

        else:
            if not with_sigmoid:
                self.loss_func = FocalLoss(
                    alpha=0.25, gamma=2, num_classes=2, size_average=False
                )
            else:
                self.loss_func = FocalLossWithSigmoid(
                    alpha=0.25, gamma=2, reduction="sum"
                )

        self.add_dice_loss = add_dice_loss
        self.use_sinkhorn = use_sinkhorn

        self.use_gt_in_train = cfgs['use_gt_in_train']

        self._show_attn_weights = show_attn_weights
        self._cnt = 0

    def forward(self, edge_feature_list, edge_index, alpha, gt_info, node_target_id_dicts):
        """
        Args:
            edge_feature_list: List[torch.Tensor], len == layer number of GNN in assc graph, 
                item.shape: (num of edges, edge feature dim)

            edge_index: torch.Tensor, shape: (2, num of edges) tracklet_graph_id -> detection_graph_id

            alpha: None | torch.Tensor, attn weights used in GATConv

            gt_info: torch.Tensor, shape: (num of targets, 7) which means 
                frame id, obj id, x-y-w-h, cls

            node_target_id_dicts: List[torch.Tensor], len == 2, id_dict for last frame and id_dict 
                for current frame. each tensor with a shape (num of nodes, 2), 2 means node id in graph -> gt target id:
                [[0, t_id_0],
                 [1, t_id_1], 
                 ...]
        """

        # merge bi-directional edges as one
        border = edge_index.shape[1] >> 1
        edge_index_merged = edge_index[:, :border]
        edge_feature_list_merged = []

        self._cnt += 1
        
        for item in edge_feature_list:
            edge_feature_list_merged.append(
                0.5 * (item[:border, :] + item[border:, :])
            ) 
        
        if self.forward_method == 'last':
            edge_feature = edge_feature_list_merged[-1]  # shape: (num of edges(single-direction), feature dims)
            confs = self.mlp(edge_feature)
        else:
            raise NotImplementedError
        
        if not self.add_dice_loss:
            if not self.with_sigmoid:
                loss = self.get_loss(confs, gt_info, edge_index_merged, node_target_id_dicts)
            else:
                loss = self.get_loss_v2(confs, gt_info, edge_index_merged, node_target_id_dicts)
        else:
            loss = self.get_loss_focal_and_dice(confs, alpha, gt_info, edge_index_merged, node_target_id_dicts)

        return loss
    
    def inference(self, edge_feature_list, edge_index, node_target_id_last_frame):
        """
        Args:
            see self.forward()
        
        Returns:
            confs of edges to be a positive match
            torch.Tensor, shape(num of edges, )
        """
        # merge bi-directional edges as one
        border = edge_index.shape[1] >> 1
        edge_index_merged = edge_index[:, :border]
        edge_feature_list_merged = []
        
        for item in edge_feature_list:
            edge_feature_list_merged.append(
                0.5 * (item[:border, :] + item[border:, :])
            ) 
        
        if self.forward_method == 'last':
            edge_feature = edge_feature_list_merged[-1]  # shape: (num of edges(single-direction), feature dims)
            confs = self.mlp(edge_feature)
        else:
            raise NotImplementedError
        
        return torch.sigmoid(confs), edge_index_merged
        
    def get_loss(self, confs, gt_info, edge_index_merged, node_target_id_dicts):
        """ cal loss value
        Args:
            confs: shape: (num of edges, ) | (bs, num of edges)
            gt_info: gt info of two frames
            edge_index_merged: shape: (2, num of edges) | (bs, 2, num of edges)

            node_target_id_dicts: List[torch.Tensor], len == 2, id_dict for last frame and id_dict 
                for current frame. each tensor with a shape (num of nodes, 2), 2 means node id in graph -> gt target id:
                [[0, t_id_0],
                 [1, t_id_1], 
                 ...]

        Return:`
            float
        """

        if self.use_gt_in_train:

            if len(confs.shape) == 1:
                confs = confs.unsqueeze(0) # (num of edges, ) -> (1, num of edges)
            
            confs = confs.unsqueeze(2)  # (1, num of edges) -> (1, num of edges, 1)
            confs_ = confs.clone().detach()
            confs_[:, :, 0] = 1 - confs_[:, :, 0]
            confs = torch.cat([confs_, confs], dim=2)  # (1, num of edges, 1) -> (1, num of edges, 2)
            # 2 means prob belong to 0(not same id), prob belong to 1(same id)

            # construct target
            # edge connection means inherit the ID 
            
            # map node id to target id
            target_id_last_frame = self._map_node_target_id(node_target_id_dicts[0], edge_index_merged[0], 
                                                            id_offset=0)
            target_id_cur_frame = self._map_node_target_id(node_target_id_dicts[1], edge_index_merged[1], 
                                                           id_offset=node_target_id_dicts[0].shape[0])

            # for gt mode, same target id means correct connection
            labels = torch.zeros(size=(1, edge_index_merged.shape[1]))  # (1, num of edges)
            labels[0, target_id_last_frame == target_id_cur_frame] = 1
            labels = labels.long().to(confs.device)

            loss = self.loss_func(confs, labels)
            

        else:
            raise NotImplementedError
        
        return loss
    
    def get_loss_v2(self, confs, gt_info, edge_index_merged, node_target_id_dicts):
        """ get loss use FocalLossWithSigmoid

        Args:
            same as get_loss
        
        Returns:
            float
        """
        if self.use_gt_in_train:

            if len(confs.shape) == 1:
                confs = confs.unsqueeze(0)  # (num of edges, ) -> (1, num of edges)
            if confs.shape[1] == 1: 
                confs = confs.view(1, -1)  # (num_of_edges, 1) -> (1, num_of_edges)
            
            # construct target
            # edge connection means inherit the ID 
            
            # map node id to target id
            target_id_last_frame = self._map_node_target_id(node_target_id_dicts[0], edge_index_merged[0], 
                                                            id_offset=0)
            target_id_cur_frame = self._map_node_target_id(node_target_id_dicts[1], edge_index_merged[1], 
                                                           id_offset=node_target_id_dicts[0].shape[0])

            # for gt mode, same target id means correct connection
            labels = torch.zeros(size=(1, edge_index_merged.shape[1]))  # (1, num of edges)
            labels[0, target_id_last_frame == target_id_cur_frame] = 1
            labels = labels.long().to(confs.device)

            loss = self.loss_func(confs, labels)

            loss = loss / max(1.0, torch.sum(labels))  # divide by positive connection; +1 for the case of no positive connection
            

        else:
            raise NotImplementedError
        
        return loss
    
    def get_loss_focal_and_dice(self, confs, alpha, gt_info, edge_index_merged, node_target_id_dicts, 
                                beta=[1.0, 0.2], rand_perm=True):
        """
        Args:
            same as get_loss
            beta: List[float, float], weight of two losses
            rand_perm: randomly permute the confs and labels to prevent overfitting
        
        Returns:
            float
        """
        if self.use_gt_in_train:
            
            # num of tracklet and num of detection
            trk_num, det_num = node_target_id_dicts[0].shape[0], node_target_id_dicts[1].shape[0]

            if self.use_sinkhorn:
                confs = self._get_sinkhorn_score(confs, edge_index_merged, trk_num=trk_num, det_num=det_num)

            if len(confs.shape) == 1:
                confs = confs.unsqueeze(0)  # (num of edges, ) -> (1, num of edges)
            if confs.shape[1] == 1: 
                confs = confs.view(1, -1)  # (num_of_edges, 1) -> (1, num_of_edges)

            if alpha is not None:
                alpha = alpha.view(1, -1)
            
            # construct target
            # edge connection means inherit the ID 
            
            # map node id to target id
            target_id_last_frame = self._map_node_target_id(node_target_id_dicts[0], edge_index_merged[0], 
                                                            id_offset=0)
            target_id_cur_frame = self._map_node_target_id(node_target_id_dicts[1], edge_index_merged[1], 
                                                           id_offset=trk_num)

            # for gt mode, same target id means correct connection
            labels = torch.zeros(size=(1, edge_index_merged.shape[1]))  # (1, num of edges)
            labels[0, target_id_last_frame == target_id_cur_frame] = 1
            labels = labels.long().to(confs.device)

            if rand_perm:
                rand_idx = torch.randperm(confs.shape[1])
                confs = confs[:, rand_idx]
                labels = labels[:, rand_idx]

                if alpha is not None: alpha = alpha[:, rand_idx]

            if alpha is not None and self._show_attn_weights and not (self._cnt % 100):
                print(alpha[labels == 1])
                print(alpha[labels == 0])
            

            focal_loss, dice_loss = self.loss_func(confs, labels)
            # print('{:.6f}, {:.6f}'.format(focal_loss, dice_loss))

            loss = beta[0] * focal_loss / max(1.0, torch.sum(labels)) + beta[1] * dice_loss
            

        else:
            raise NotImplementedError
        
        return loss


    def _map_node_target_id(self, map_dict, node_id, id_offset=0): 
        """ map node id to gt target id

        Args:
            map_dict: torch.Tensor, (num of nodes, 2), 2 means node id in graph -> gt target id
            node_id: current node id in edge, shape: (num of edges, )

        Return:
            target_id, torch.Tensor, (num of edges, )
        
        """

        map_dict_ = map_dict.clone().detach().cpu()
        map_dict_[:, 0] += id_offset

        num_of_nodes = map_dict.shape[0]
        target_id = node_id.clone().detach().cpu()

        for idx in range(num_of_nodes):
            # node_id idx -> map_dict[idx]
            mask = (node_id == map_dict_[idx, 0]).cpu()

            value_to_replace = torch.zeros((mask.shape[0], )).long() + map_dict_[idx, 1]
            target_id.masked_scatter_(mask, value_to_replace)

        return target_id
    
    def _get_sinkhorn_score(self, confs, edge_index_merged, trk_num, det_num):
        """ cal sinkhorn matrix (score) to force satisfy the constrain of matching

        Args:
            confs: torch.Tensor, shape: (num of edges, )
            edge_index_merged: torch.Tensor, shape: (2, num of edges // 2)
            trk_num, det_num: int
        
        Return:
            new conf: torch.Tensor, shape: (num of edges, )
        """
        # init matrix
        device = confs.device

        if trk_num == 0 or det_num == 0:
            raise NotImplementedError
        
        if not confs.shape[0] == trk_num * det_num:
            return confs

        matrix = torch.zeros(size=(trk_num, det_num), requires_grad=False).to(device)
        matrix[edge_index_merged[0], edge_index_merged[1] - trk_num] = torch.exp(confs.view(1, -1) * 5)

        # add dummy cols and rows: (trk_num + 1, det_num + 1)
        matrix = torch.cat([matrix, torch.ones((trk_num, 1), device=device) * math.exp(-0.2 * 5)], dim=1)
        matrix = torch.cat([matrix, torch.ones((1, det_num + 1), device=device) * math.exp(-0.2 * 5)], dim=0)

        # apply sinkhorn
        desired_row = torch.ones((trk_num + 1, ), device=device)
        desired_row[-1] = det_num

        desired_col = torch.ones((det_num + 1, ), device=device)
        desired_col[-1] = trk_num

        for _ in range(8):  # 8 iters
            matrix *= (desired_row / matrix.sum(dim=1)).reshape(-1, 1)
            matrix *= (desired_col / matrix.sum(dim=0)).reshape(1, -1)

        # cut last row and col 
        matrix = matrix[0:-1, 0:-1]

        # convert matrix value to origin conf
        confs = matrix.flatten()  # NOTE: check here TODO TODO TODO
        return confs.view(-1, 1)
    

class NodeClassifier(nn.Module):
    """
    net to calculate the conf score of occlusion of a object in assc graph
    """

    def __init__(self, cfgs) -> None:
        super().__init__()

        self.mlp_dims = cfgs['dims']
        self.forward_method = cfgs['method']
        drop_out_p = cfgs['drop_out_p']
        norm_layer = nn.LayerNorm
        act_func = nn.LeakyReLU(negative_slope=0.01)

        self.mlp = nn.Sequential(*build_layers(self.mlp_dims[0], self.mlp_dims[1: ], drop_out_p, norm_layer, act_func),
                                 nn.Linear(self.mlp_dims[-1], 1))
        
        self.loss_func = FocalLossWithSigmoid(
                    alpha=0.25, gamma=2, reduction="sum"
                )
        
        self.use_gt_in_train = cfgs['use_gt_in_train']


    def forward(self, node_features, edge_index, occ_labels, node_target_id_dicts):
        """
        Args:
            node_features: torch.Tensor, shape: (num_of_objs_last + num_of_objs_cur, feat_dims)
            edge_index: torch.Tensor, shape: (2, num_of_edges)
            occ_labels: list[torch.Tensor, torch.Tensor], occ labels in last and cur frame, elem shape: (1, num_of_objs)
            node_target_id_dicts: same as EdgeClassifier

        Returns:
            float

        """

        # merge bi-directional edges as one
        border = edge_index.shape[1] >> 1
        edge_index_merged = edge_index[:, :border]

        node_border = node_target_id_dicts[0].shape[0]
        node_confs_last = self.mlp(node_features[:node_border, :])
        node_confs_cur = self.mlp(node_features[node_border:, :])

        
        loss = self.get_loss(node_confs_last, node_confs_cur, edge_index_merged, occ_labels, node_target_id_dicts)

        return loss 
    
    def inference(self, node_feature_cur, ):
        """
        Args:
            node_feature_cur: torch.Tensor, shape (num_of_objs_cur, feat_dims)

        Returns:
            torch.Tensor, (num_of_objs_cur, )
        """
        confs = self.mlp(node_feature_cur)

        return torch.sigmoid(confs)

    def get_loss(self, confs_last, confs_cur, edge_index_merged, occ_labels, node_target_id_dicts, beta=0.2):

        if self.use_gt_in_train:

            if len(confs_last.shape) == 1:
                confs_last = confs_last.unsqueeze(0)  # (num of edges, ) -> (1, num of edges)
                confs_cur = confs_cur.unsqueeze(0)

            if confs_last.shape[1] == 1: 
                confs_last = confs_last.view(1, -1)  # (num_of_edges, 1) -> (1, num_of_edges)
                confs_cur = confs_cur.view(1, -1)

            target_id_last_frame = self._map_node_target_id(node_target_id_dicts[0], edge_index_merged[0], 
                                                            id_offset=0)
            target_id_cur_frame = self._map_node_target_id(node_target_id_dicts[1], edge_index_merged[1], 
                                                           id_offset=node_target_id_dicts[0].shape[0])

            focal_loss_last = self.loss_func(confs_last, occ_labels[0])
            focal_loss_cur = self.loss_func(confs_cur, occ_labels[1])

            # node id that belongs to same obj (NOTE: not the real id)
            edge_index_selected = edge_index_merged[:, target_id_last_frame == target_id_cur_frame]
            pos_sample_num = edge_index_selected.shape[1]  # num of positive samples

            conf_diff_loss = confs_last[0, edge_index_selected[0]] - confs_cur[0, edge_index_selected[1] - node_target_id_dicts[0].shape[0]]
            conf_diff_loss = torch.sum(0.5 * conf_diff_loss ** 2)

            loss = (focal_loss_last + focal_loss_cur + beta * conf_diff_loss) / max(1.0, pos_sample_num)
            

        else: raise NotImplementedError
        
        return loss 

    def _map_node_target_id(self, map_dict, node_id, id_offset=0): 
        """ map node id to gt target id

        Args:
            map_dict: torch.Tensor, (num of nodes, 2), 2 means node id in graph -> gt target id
            node_id: current node id in edge, shape: (num of edges, )

        Return:
            target_id, torch.Tensor, (num of edges, )
        
        """
        
        map_dict_ = map_dict.clone().detach().cpu()
        map_dict_[:, 0] += id_offset

        num_of_nodes = map_dict.shape[0]
        target_id = node_id.clone().detach().cpu()

        for idx in range(num_of_nodes):
            # node_id idx -> map_dict[idx]
            mask = (node_id == map_dict_[idx, 0]).cpu()

            value_to_replace = torch.zeros((mask.shape[0], )).long() + map_dict_[idx, 1]
            target_id.masked_scatter_(mask, value_to_replace)

        return target_id
    
class MultiLoss(nn.Module):
    """
    combine edge and node loss
    """
    def __init__(self, method='fixed') -> None:
        """
        Args:
            method: 'fixed' or 'uncertainty'
        """
        super().__init__()

        self.method = method
        # for fixed
        self.beta = [1.0, 0.2]

        # for uncertainty
        self.w1 = nn.Parameter(-1.85 * torch.ones(1, ))
        self.w2 = nn.Parameter(-1.05 * torch.ones(1, ))
    
    def forward(self, edge_loss, node_loss):
        """
        edge_loss, node_loss: torch.Tensor
        """

        loss_dict = {'edge_loss': edge_loss.item(),
                     'node_loss': node_loss.item()}  # loss dict to show

        if self.method == 'fixed':
            loss = self.beta[0] * edge_loss + self.beta[1] * node_loss 
        else:
            loss = torch.exp(-self.w1) * edge_loss + torch.exp(-self.w2) * node_loss + \
                    (self.w1 + self.w2)
            
            loss *= 0.5
            # print([torch.exp(-self.w1).item(), torch.exp(-self.w2).item()])
            loss_dict['extra'] = [torch.exp(-self.w1).item(), torch.exp(-self.w2).item()]

        return loss, loss_dict, 
"""
calculate distances
"""

import numpy as np 
import torch
import math 

from loguru import logger

def cal_bbox_distance(bbox1, bbox2, format='xywh'):
    """ calculate bbox distance
    
    Args:
        bbox1, bbox2: np.ndarray | torch.Tensor, shape: (4, )
        format: str, 'tlbr' | 'xywh'
    
    Return:
        float
    """
    if format == 'tlbr':
        cx1, cy1 = 0.5 * (bbox1[0] + bbox1[2]), 0.5 * (bbox1[1] + bbox1[3])
        cx2, cy2 = 0.5 * (bbox2[0] + bbox2[2]), 0.5 * (bbox2[1] + bbox2[3])

    elif format == 'xywh':
        cx1, cy1 = bbox1[0], bbox1[1]
        cx2, cy2 = bbox2[0], bbox2[1]
    else:
        raise NotImplementedError
    
    if isinstance(cx1, torch.Tensor):
            cx1, cy1 = cx1.item(), cy1.item()
            cx2, cy2 = cx2.item(), cy2.item()
    return np.linalg.norm(np.array([cx1, cy1]) - np.array([cx2, cy2]))
    
def cal_bbox_angle(bbox_center, bbox1, bbox2, format='xywh', method='atan', radians=False):
    """ calculate angle of ∠(bbox1 bbox_center bbox2)

    Args:
        bbox_center, bbox_1, bbox_2: np.ndarray | torch.Tensor, shape: (4, )
        format: str, 'tlbr' | 'xywh'
        method: str, 'acos' | 'atan', when use 'acos', use dot product and norm to cal angle
            when use 'atan', split the Angle into two parts divided by the x axis
        radians: bool, whether to use radians

    Return:
        float
    """

    # P_c: (cx, cy), P_1: (cx1, cy1), P_2: (cx2, cy2)
    # calculate: ∠P_1P_cP_2
    if format == 'tlbr':
        cx, cy = 0.5 * (bbox_center[0] + bbox_center[2]), 0.5 * (bbox_center[1] + bbox_center[3])
        cx1, cy1 = 0.5 * (bbox1[0] + bbox1[2]), 0.5 * (bbox1[1] + bbox1[3])
        cx2, cy2 = 0.5 * (bbox2[0] + bbox2[2]), 0.5 * (bbox2[1] + bbox2[3])
    elif format == 'xywh':
        cx, cy = bbox_center[0], bbox_center[1]
        cx1, cy1 = bbox1[0], bbox1[1]
        cx2, cy2 = bbox2[0], bbox2[1]
    else:
        raise NotImplementedError
    
    if isinstance(cx1, torch.Tensor):
        cx, cy = cx.item(), cy.item()
        cx1, cy1 = cx1.item(), cy1.item()
        cx2, cy2 = cx2.item(), cy2.item()

    P_cP_1 = np.array([cy1 - cy, cx1 - cx])
    P_cP_2 = np.array([cy2 - cy, cx2 - cx])

    if method == 'acos':
        try:
            cos_theta = np.dot(P_cP_1, P_cP_2) / (np.linalg.norm(P_cP_1) * np.linalg.norm(P_cP_2))
            # scale 
            cos_theta = min(cos_theta, 1.0)
            cos_theta = max(-1.0, cos_theta)
        except:
            logger.warning('\nError or Warning Occured! the values are:')
            logger.warning(f'PcP1: {np.linalg.norm(P_cP_1)}, PcP2: {np.linalg.norm(P_cP_2)}')
            exit(-1)
        
        assert cos_theta <= 1 and cos_theta >= -1, logger.error(f'Value Error, cos_theta={cos_theta},\n vectors are \
                                                                {P_cP_1}, {P_cP_2}')
        theta = math.acos(cos_theta)
        if not radians:
            theta = theta / math.pi * 180

    elif method == 'atan':
        # first cal angle w.r.t. positive X-axis
        theta_PcP1_x = math.atan2(P_cP_1[1], P_cP_1[0])
        theta_PcP2_x = math.atan2(P_cP_2[1], P_cP_2[0])

        if not radians:
            theta_PcP1_x = theta_PcP1_x * 180 / math.pi
            theta_PcP2_x = theta_PcP2_x * 180 / math.pi 

            if theta_PcP1_x * theta_PcP2_x > 0:  # 
                theta = abs(theta_PcP1_x - theta_PcP2_x)
            else:
                theta = abs(theta_PcP1_x) + abs(theta_PcP2_x)
                if theta > 180:
                    theta = 360 - theta

    else: raise NotImplementedError

    return theta

def bbox_wassertein(bboxes1, bboxes2, eps=1e-7, constant=50):
    """ calculate wass dist matrix
        Paper: A Normalized GaussianWasserstein Distance for Tiny Object Detection

    Args:
        bboxes1, bboxes2: torch.Tensor, shape: (num_of_bboxes, 4) or (batch_size, num_of_bboxes, 4), format: tlbr
        eps: small float to avoid zero
        constant: a param in wass dist
    
    Return:
        dist matrix, shape: (num_of_bboxes1, num_of_bboxes2)
    """
    center1 = (bboxes1[..., :, None, :2] + bboxes1[..., :, None, 2:]) / 2  # shape: (N1, 1, 2)
    center2 = (bboxes2[..., None, :, :2] + bboxes2[..., None, :, 2:]) / 2  # shape: (1, N2, 2)
    whs = center1[..., :2] - center2[..., :2]  # shape: (N1, N2, 2)

    center_distance = whs[..., 0] * whs[..., 0] + whs[..., 1] * whs[..., 1] + eps  # shape: (N1, N2)

    w1 = bboxes1[..., :, None, 2] - bboxes1[..., :, None, 0] + eps  # shape: (N1, 1)
    h1 = bboxes1[..., :, None, 3] - bboxes1[..., :, None, 1] + eps
    w2 = bboxes2[..., None, :, 2] - bboxes2[..., None, :, 0] + eps  # shape: (1, N2)
    h2 = bboxes2[..., None, :, 3] - bboxes2[..., None, :, 1] + eps

    wh_distance = ((w1 - w2)**2 + (h1 - h2) ** 2) / 4  # shape: (N1, N2)

    wassersteins = torch.sqrt(center_distance + wh_distance)

    normalized_wasserstein = torch.exp(-wassersteins/constant)

    return normalized_wasserstein
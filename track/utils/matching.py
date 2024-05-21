"""
Some match functions
"""
import lap 
import math 
import numpy as np 
import torch 
from cython_bbox import bbox_overlaps as bbox_ious

def linear_assignment(cost_matrix, thresh):
    """ Solve linear assignment problem use lap.lapjv

    Args:
        cost_matrix: np.ndarray
        thresh: float, max value of cost in matching
    
    Returns:
        matches: List[List[int, int]], matched pair idx 
        unmatched_a: unmatched row idx 
        unmathced_b: unmatched col idx
    
    """
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b

def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray
    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=float),
        np.ascontiguousarray(btlbrs, dtype=float)
    )

    return ious

    
def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]
    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix

def wass(bboxes1, bboxes2, eps=1e-7, constant=50):
    """
    
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

    wassersteins = np.sqrt(center_distance + wh_distance)

    normalized_wasserstein = np.exp(-wassersteins/constant)

    return normalized_wasserstein

def wass_distance(atracks, btracks):
    """
    Compute cost based on Wass
    """
    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]

    _wass = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if _wass.size == 0:
        return 1. - _wass
    
    _wass = wass(
        np.ascontiguousarray(atlbrs, dtype=float), 
        np.ascontiguousarray(btlbrs, dtype=float), 
    )
    cost_matrix = 1 - _wass

    return cost_matrix

def sinkhorn_norm(matrix, match_thresh=0.25):
    """
    Sinkhorn used for inference
    """
    matrix = torch.exp(5 * matrix)
    device = matrix.device 
    trk_num, det_num = matrix.shape[0], matrix.shape[1]

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
    matrix[matrix > match_thresh] = 1.

    return matrix

from scipy.spatial.distance import cdist

def cal_cosine_distance(mat1, mat2):
    """
    simple func to calculate cosine distance between 2 matrixs
    
    :param mat1: np.ndarray, shape(M, dim)
    :param mat2: np.ndarray, shape(N, dim)
    :return: np.ndarray, shape(M, N)
    """
    # result = mat1·mat2^T / |mat1|·|mat2|
    # norm mat1 and mat2
    mat1 = mat1 / np.linalg.norm(mat1, axis=1, keepdims=True)
    mat2 = mat2 / np.linalg.norm(mat2, axis=1, keepdims=True)

    return np.dot(mat1, mat2.T)    

def embedding_distance(tracks, detections, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[STrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.features[-1] for track in detections], dtype=np.float)
    track_features = np.asarray([track.features[-1] for track in tracks], dtype=np.float)
    if metric == 'euclidean':
        cost_matrix = np.maximum(0.0, cdist(track_features, det_features)) # Nomalized features
    elif metric == 'cosine':
        cost_matrix = 1. - cal_cosine_distance(track_features, det_features)
    else:
        raise NotImplementedError
    return cost_matrix
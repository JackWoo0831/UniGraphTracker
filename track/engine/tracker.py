"""
track framework 
get detection result -> construct graph -> solve association
"""

import torch 
import torchvision.transforms as T
import numpy as np 
import os 
import os.path as osp 
from loguru import logger
import yaml

from models.uni_graph import UniGraph, UniGraph_wo_FG
from utils.basetrack import BaseTrack, TrackState, STrack
import utils.matching as matching

from torchreid.utils import FeatureExtractor

class Tracker:
    def __init__(self, cfgs, frame_rate=30, device=None, det_model=None, *args, **kwargs) -> None:
        
        self.cfgs = cfgs  # options 

        # model 
        self.device = device

        model_img_size = cfgs['detector']['img_size']
        self.__model_img_size = [model_img_size, model_img_size]
        
        self.uni_graph = self._get_tracker()
        self.uni_graph.eval()

        self.reid = self._get_reid(self.cfgs['reid']['weights'])
        self.reid.model.eval() 

        # Tracklet handler
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.trk_dict = {
            'id_map': None,
            'bboxes': None, 
            'node_features': None,
            'edge_index': None,
            'edge_features': None,
        }  # store info of prev frame
        self.det_dict = {}  # store info of cur frame

        # Other params
        self.buffer_size = int(frame_rate / 30.0 * cfgs['tracker']['track_buffer'])
        self.max_time_lost = self.buffer_size

        self.frame_id = 0

        # set reid norm transform
        pixel_mean = [0.485, 0.456, 0.406]
        pixel_std = [0.229, 0.224, 0.225]

        self.reid_toPIL = T.ToPILImage()
        self.reid_norm = T.Compose([
            T.Resize(size=self.cfgs['reid']['image_size']), 
            T.ToTensor(),
            T.Normalize(pixel_mean, pixel_std),
        ])

        # detection model 
        self.det_model = det_model

        # for debug
        self.feature_frame_idx = [] 
        self.feature_to_save = []
    

    def reset(self, ):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        self.trk_dict = {
            'id_map': None,
            'bboxes': None, 
            'node_features': None,
            'edge_index': None,
            'edge_features': None,
        }  # store info of prev frame
        self.det_dict = {}  # store info of cur frame


    @torch.no_grad()
    def update(self, det_results, img, ori_img):
        """ Update tracks, this func should be called every timestamp 

        Args:
            det_results: np.ndarray | torch.Tensor, detection results, maybe (N, 6)
            img: np.ndarray, image used to inference 
            ori_img: np.ndarray, original image (original size)
        
        Return:

        """

        """Step 1. post process detection results"""
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if self.det_model in ['yolov8']:
            ori_img = ori_img.squeeze(0)
            img_size = [img.shape[0], img.shape[1]]
            ori_img_size = img_size.copy()

        else:
            img, ori_img = img.squeeze(0), ori_img.squeeze(0)  # (c, h, w), (h0, w0, c)
            img_size = [img.shape[-2], img.shape[-1]]  # h, w
            ori_img_size = [ori_img.shape[0], ori_img.shape[1]]  # h0, w0
        
        scores = det_results[:, 4]
        bboxes = det_results[:, :4]
        classes = det_results[:, -1]

        img_h, img_w = img_size[0], img_size[1]


        # filter low conf det
        remain_ids = scores > self.cfgs['low_det_thresh']
        scores = scores[remain_ids]
        bboxes = bboxes[remain_ids]
        classes = classes[remain_ids].int()
        
        if len(bboxes) > 0:
            detections = [STrack(tlbr.detach().cpu().numpy(), s.item(), cls.item()) for
                          (tlbr, s, cls) in zip(bboxes, scores, classes)]
        else:
            detections = [] 

        """Step 2. Add newly detected tracklets to tracked_stracks"""

        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)  # merge tracked and lost tracks

        """Step 3. First association, use uni graph"""

        reid_features = self._get_reid_features(ori_img, bboxes, ori_img_size)
        # set dict that store info of current detection 
        self._get_det_dict(bboxes, scores, reid_features)
        self._get_trk_dict(strack_pool)

        if not len(detections):  # no detections
            confs_cost_matrix = np.zeros((len(strack_pool), 0))
            matches, u_track, u_detection = matching.linear_assignment(confs_cost_matrix, thresh=0.9)

        elif not len(strack_pool):  # no tracks
            confs_cost_matrix = np.zeros((0, len(detections)))
            matches, u_track, u_detection = matching.linear_assignment(confs_cost_matrix, thresh=0.9)

            _, _, det_features, confs_node_cur = self.uni_graph.inference({}, self.det_dict, ori_img_size)

        else:           

            confs, edge_index, det_features, confs_node_cur = self.uni_graph.inference(self.trk_dict, self.det_dict, ori_img_size)
            edge_index[1, :] -= self.trk_dict['num']

            # convert confs to a matrix
            confs_cost_matrix = torch.zeros((self.trk_dict['num'], self.det_dict['num']))

            if not self.cfgs['tracker']['sinkhorn']:
                confs_cost_matrix -= 0x3f3f3f3f  # set to -inf
                confs_cost_matrix[edge_index[0], edge_index[1]] = 1 - confs.view(1, -1).cpu()
            else:
                confs_cost_matrix += 10  # set to +inf
                confs_cost_matrix[edge_index[0], edge_index[1]] = confs.view(1, -1).cpu()
                confs_cost_matrix = 1. - matching.sinkhorn_norm(confs_cost_matrix)

            # match
            matches, u_track, u_detection = matching.linear_assignment(confs_cost_matrix.numpy(), thresh=0.9)

        # NOTE: update det features
        for idx, det in enumerate(detections):
            # det: STrack
            det.features.append(det_features[idx])

        # update tracks
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]

            # if track.track_id == 4:
            #     self.feature_to_save.append(track.features[-1])
            #     self.feature_frame_idx.append(self.frame_id)

            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id, confs_node_cur[idet])
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)


        """Step 4. Second association, use wass"""
        detections = [detections[i] for i in u_detection]
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]

        dists = matching.wass_distance(r_tracked_stracks, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.5)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id, None)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # mark lost
        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        """Step 5. Deal with unconfirmed tracks, usually tracks with only one beginning frame"""
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)

        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id, None)
            activated_starcks.append(unconfirmed[itracked])

        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 6: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.cfgs['high_det_thresh']:
                continue
            track.activate(self.frame_id, track.features[-1])
            activated_starcks.append(track)

        """ Step 7: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        return output_stracks
    
        
    def _get_det_dict(self, bboxes, scores, reid_features):
        """ get det dict in cur frame

        Args:
            bboxes: torch.Tensor, tlbr, (N, 4)
            scores: torch.Tensor, tlbr, (N, )
            reid_features: torch.Tensor, tlbr, (N, 512)

        Returns:
        """
        self.det_dict['det_results'] = torch.cat([bboxes, scores.unsqueeze(1)], dim=1)
        self.det_dict['reid_features'] = reid_features 
        self.det_dict['num'] = bboxes.shape[0]

    def _get_trk_dict(self, strack_pool):
        """ get trk dict from active and lost tracks
        
        Args:
            strack_pool: List[STrack]
        """

        if not strack_pool: return 

        ids = [trk.track_id for trk in strack_pool]
        bboxes = [torch.from_numpy(trk.bbox) for trk in strack_pool]
        node_features = [trk.features[-1] for trk in strack_pool]

        self.trk_dict['id_map'] = torch.concat([torch.arange(0, len(ids)).reshape(-1, 1), 
                                                  torch.tensor(ids).reshape(-1, 1)], dim=1).to(self.device)
        
        self.trk_dict['bboxes'] = torch.stack(bboxes).to(self.device)
        self.trk_dict['node_features'] = torch.stack(node_features).to(self.device)
        self.trk_dict['num'] = len(ids)



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

        # load ckpt
        if self.cfgs['tracker']['model_path']:
            ckpt = torch.load(self.cfgs['tracker']['model_path'], map_location=self.device)
            logger.info(f"loaded ckpt in epoch {ckpt['epoch']} for resume training")

            model.load_state_dict(ckpt['model'])

        logger.info(f'Now tracker is on device {next(model.parameters()).device}')

        logger.info('loaded graph model done')

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
    
    def _get_reid_features(self, ori_img, bboxes, ori_img_size):
        """ get reid features from list of img info

        Args:
            ori_img: torch.Tensor, shape (H, W, C)
            bboxes: torch.Tensor, shape (N, 4)
            ori_img_size: List[int, int], [h0, w0]

        Returns:
            List[torch.Tensor], each tensor with shape (num of objs, feature_dims)
        """

        crop_batch = []

        # get legal coordinate in origin img
        get_coor = lambda x, is_h_axis=False: torch.clamp(x, 0, ori_img_size[0]).int().item() if is_h_axis else \
            torch.clamp(x, 0, ori_img_size[1]).int().item()

        for bbox in bboxes:
            
            x1, x2 = get_coor(bbox[0], is_h_axis=False), get_coor(bbox[2], is_h_axis=False)
            y1, y2 = get_coor(bbox[1], is_h_axis=True), get_coor(bbox[3], is_h_axis=True)
            crop = ori_img[y1: y2, x1: x2]  # (h, w, c)
            try:
                crop = self.reid_toPIL(crop.numpy())
            except:
                logger.warning(f'\nbbox {crop} has 0 w or h! Use zero array instead. ')
                crop = self.reid_toPIL(np.zeros((32, 64, 3), dtype=np.uint8))

            crop = self.reid_norm(crop)

            crop_batch.append(crop)

        if not len(crop_batch): return [torch.Tensor([])]

        crop_batch = torch.stack(crop_batch, dim=0).to(self.device)
        features = self.reid(crop_batch)

        return features


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
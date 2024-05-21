import numpy as np
from collections import OrderedDict


class TrackState(object):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class BaseTrack(object):
    _count = 0

    track_id = 0
    is_activated = False
    state = TrackState.New

    history = OrderedDict()
    features = []
    curr_feature = None
    score = 0
    start_frame = 0
    frame_id = 0
    time_since_update = 0

    # multi-camera
    location = (np.inf, np.inf)

    @property
    def end_frame(self):
        return self.frame_id

    @staticmethod
    def next_id():
        BaseTrack._count += 1
        return BaseTrack._count

    def activate(self, *args):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed


class STrack(BaseTrack):

    def __init__(self, tlbr, score, cls):

        # wait activate
        
        self.is_activated = False

        self.bbox = tlbr  # tlbr
        self.score = score
        self.cls = cls 
        
        self.features = []
        self.store_features_budget = 1
        self.smooth_feature_weight = [0.9, 0.1]  # w0 * f_{old} + w1 * f_{new}
        
        self.tracklet_len = 0
        self.track_id = None 

        self._feature_update_manner = 'dynamic_occ'  # 'linear' 'dynamic' 'dynamic_occ'
        # NOTE: for ablation SET HERE
       

    def predict(self):
        pass

    @staticmethod
    def multi_predict(stracks):
        pass

    def activate(self, frame_id, node_feature):
        """Start a new tracklet"""
        self.track_id = self.next_id()

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.features.append(node_feature)

    def re_activate(self, new_track, frame_id, new_id=False):
        
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.bbox = new_track.bbox
        self.features.append(new_track.features[-1])

    def update(self, new_track, frame_id, conf_node):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1
        
        self.state = TrackState.Tracked
        self.is_activated = True

        self.bbox = new_track.bbox 

        if self._feature_update_manner == 'linear':
            # update feature linearly
            self.features.append(
                self.smooth_feature_weight[0] * self.features[-1] + \
                self.smooth_feature_weight[1] * new_track.features[-1]
            )
            self.score = new_track.score

        elif self._feature_update_manner == 'dynamic':
            # SGT manner
            # f_{t2} = f_{t1} * (a1 / (a1 + a2)) + d_{t2} * (a2 / (a1 + a2))
            sum_score = self.score + new_track.score 
            norm_prev_score = self.score / sum_score 
            norm_cur_score = new_track.score / sum_score 
            self.features.append(
                norm_prev_score * self.features[-1] + \
                norm_cur_score * new_track.features[-1]
            )

            self.score = new_track.score

        elif self._feature_update_manner == 'dynamic_occ':
            if conf_node is not None:
                self.features.append(
                    (1. - 2 * conf_node) * self.features[-1] + \
                    2 * conf_node * new_track.features[-1]
                )
            else:
                self.features.append(
                    new_track.features[-1]
                )

        else: raise NotImplementedError
        self.features = self.features[-self.store_features_budget: ]


    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        ret = self.bbox.copy()
        # tlbr -> tlwh
        ret[2:] -= ret[:2]
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        return self.bbox

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)
"""
map origin datasets to the input fit for GNN
"""

from torch.utils.data import Dataset
from loguru import logger
import random 


class TwoFrameMapper(Dataset):
    def __init__(self, dataset, cfgs):
        """
        Args:
            dataset: torch.utils.data.Dataset
                containing origin data 
            cfgs: dict

        Return:
            None
        """
        self.dataset = dataset 
        self.seq_intervals = dataset.get_seq_intervals()

        self.cfgs = cfgs
        self.random_range = cfgs['random_range']  # List[int, int]
        self.reduced_scale = int(cfgs['reduced_scale'])

        self.ori_length = len(self.dataset)

    def __getitem__(self, idx):
        """ get two frames within a seq randomly

        Args:
            idx: int, index of whole data sample

        Return:
            List[dict, dict]
            
        """
        if not self.reduced_scale == 1: 
            # l = self.ori_length, r = self.reduced_scale
            # randomly set idx from [idx, idx + l / r, idx + 2 * l / r, ..., idx + (r - 1) * l / r], len == r
            idx_random_value = [idx + k * self.ori_length // self.reduced_scale for k in range(self.reduced_scale)]
            idx = idx_random_value[random.randint(0, self.reduced_scale - 1)]

        img_info = self.dataset[idx] 
        seq_name = img_info['seq_name']
        seq_idx_interval = self.seq_intervals[seq_name]  # List[int, int]

        if idx == seq_idx_interval[-1]: idx -= 1  # end frame is illegal

        idx_next_max = min(seq_idx_interval[-1], idx + self.random_range[-1])
        idx_next = idx + self.random_range[0] if random.random() < self.cfgs['prob_adj_sampling'] \
            else random.randint(idx + self.random_range[0], idx_next_max)
        
        # merge frame id and frame id_next

        ret = [img_info, self.dataset[idx_next]]  # List[dict, dict]

        return ret 

    def __len__(self, ):
        return len(self.dataset) // self.reduced_scale
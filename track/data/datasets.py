"""
contains Dataset class
"""

import numpy as np
import torch 
import cv2 
import os 
import os.path as osp
import copy
import pickle
import random

from torch.utils.data import Dataset
from loguru import logger

class TrainValDataset(Dataset):
    """ This class reads the dataset and generates a list of a specific format in order. 
        Further reading is achieved through a Mapper.
    
    """
    def __init__(self, cfgs, split='train', copy=False, cache_dataset=False, cache_path='', 
                 prob_add_fp=0.1, max_fp_number=10) -> None:
        """
        Args:
            cfgs: config dict
            split: str, train or val
            copy: whether to deepcopy the element when producing it
            cache_dataset: bool, whether generate cache to speed up 
            cache_path: str, path of cache file
            max_fp_number: randomly generate FP boxes, 0 for abandon

        """
        super().__init__()

        self.cfgs = cfgs
        self.copy = copy
        self.cache_dataset = cache_dataset
        self.cache_path = cache_path
        self.prob_add_fp = prob_add_fp
        self.max_fp_number = max_fp_number

        self.label_list = self._get_label_list(split)

    
    def _get_label_list(self, split='train'):
        """ get label info

        Args:
            split: str, train or val

        Return:
            List[dict], each dict contains the info of an image, format:
                # key name    # value description
            {
                'image_path': str, abs path of the image 
                'frame_id': int, frame id, start from 1 
                'seq_name': str, name of the sequnece
                'ori_size': List[int, int], width and height of image
                'objects': List[dict], objects info in this frame, dict format:
                {
                    'id': obj id 
                    'category': category id
                    'bbox': tl-br

                }
            }
        
        """
        logger.info('loading labels')

        label_list = []
        
        # check cache files 
        cache_file_name = self.cache_path.format(self.cfgs['name'], split)
        if (not self.max_fp_number) and self.cache_dataset and osp.isfile(cache_file_name):
            logger.info('dataset cache exists, load cache instead')
            with open(cache_file_name, 'rb') as f:
                cache_file = pickle.load(f)
                label_list = cache_file['label_list']
                self.seq_intervals = cache_file['seq_intervals']

            logger.info('loading labels Done!')
            self.cache_dataset = False  # do not save cache when use random FPs
            return label_list

        cur_dir = osp.join(self.cfgs['path'], 'images', split)
        seqs = os.listdir(cur_dir)

        if self.cfgs['IGNORE_SEQS'] is not None:
            seqs = [seq for seq in seqs if seq not in self.cfgs['IGNORE_SEQS']]
        if self.cfgs['CERTRAIN_SEQS'] is not None:
            seqs = [seq for seq in seqs if seq in self.cfgs['CERTRAIN_SEQS']]

        logger.info(f'there are {len(seqs)} seqs will be used: \n{seqs}')

        img_to_gt_path = lambda path: path.replace('images', 'labels').replace('jpg', 'txt')

        seq_intervals = {}  # Dict{seq_name: [start_idx, end_idx]}, mark start and end idx in label_list for every seq
        seq_start_idx = 0
        prefix_length = self.cfgs['img_name_prefix_length']  # non-digit prefix in img file name

        for seq in seqs:
            logger.info(f'loading labels: for seq {seq}')

            imgs_in_seq = sorted(os.listdir(osp.join(cur_dir, seq)))

            # get img size 
            img_eg = cv2.imread(osp.join(cur_dir, seq, imgs_in_seq[0]))
            w0, h0 = img_eg.shape[1], img_eg.shape[0]

            # iter all images 
            valid_seq_length = 0           

            for idx, img_name in enumerate(imgs_in_seq):

                img_info = {}
                img_info['image_path'] = osp.join(cur_dir, seq, img_name)
                frame_id = int(img_name.split('.')[0][prefix_length: ])
                img_info['frame_id'] = frame_id 
                img_info['seq_name'] = seq 
                img_info['ori_size'] = [w0, h0]

                # check if the txt is empty 
                img_gt_path = img_to_gt_path(osp.join(cur_dir, seq, imgs_in_seq[frame_id - 1]))

                if osp.getsize(img_gt_path):
                    frame_gt = np.loadtxt(img_gt_path,
                                        dtype=int, delimiter=' ')
                    
                    if len(frame_gt.shape) == 1:
                        frame_gt = frame_gt[None, :]  # avoid situation that only one line

                    valid_seq_length += 1
                    
                else: continue

                objects_info_list = []

                for frame_obj_gt in frame_gt:
                    obj_info = {
                        'id': frame_obj_gt[0],
                        'bbox': frame_obj_gt[1: -1],
                        'category': frame_obj_gt[-1]
                    }

                    objects_info_list.append(obj_info)

                objects_info_list.extend(self._gen_random_boxes([w0, h0]))  # add randomly FP boxes

                img_info['objects'] = objects_info_list

                label_list.append(img_info)

            seq_intervals[seq] = [seq_start_idx, seq_start_idx + valid_seq_length - 1]
            seq_start_idx += valid_seq_length

        self.seq_intervals = seq_intervals
        
        logger.info('loading labels Done!')

        # save cache 
        if (not self.max_fp_number) and self.cache_dataset:
            logger.info('caching dataset info...')
            with open(cache_file_name, 'wb') as f:
                cache_file = {'label_list': label_list,
                              'seq_intervals': seq_intervals}
                pickle.dump(cache_file, f)
            logger.info(f'cache saved to {cache_file_name}')

        return label_list
    
    def _gen_random_boxes(self, img_size):
        """
        gen random FP boxes

        Args:
            img_size: List, [w, h]

        Returns:
            List[dict]
        """
        fp_box_number = random.randint(0, self.max_fp_number)
        ret = []
        if random.random() < self.prob_add_fp:
            return ret 
        
        w, h = img_size[0], img_size[1]
        is_legal = lambda x, y, width, height: \
            width > 0 and height > 0 and 0 <= x - width/2 < w \
            and 0 <= x + width/2 < w and 0 <= y - height/2 < h and 0 <= y + height/2 < h
        
        for idx in range(fp_box_number):
            id = random.randint(1e5, 1e8)
            
            x = np.random.normal(loc=w/2, scale=w/4) 
            y = np.random.normal(loc=h/2, scale=h/4) 
            width = np.random.normal(loc=w/16, scale=w/32)  
            height = np.random.normal(loc=h/16, scale=h/32) 

            if is_legal(x, y, width, height):
                bbox = np.array([x - width / 2, y - height / 2, x + width / 2, y + height / 2], dtype=int)
            else: continue 

            ret.append({
                'id': id,
                'bbox': bbox,
                'category': -1
            })

        return ret

        

    def __getitem__(self, idx):
        if self.copy:
            return copy.deepcopy(self.label_list[idx])
        else: 
            return self.label_list[idx]
            
    def __len__(self):
        return len(self.label_list)
    
    def get_seq_intervals(self):
        return self.seq_intervals
    

class TestDataset(Dataset):
    """ This class generate origin image, preprocessed image for inference
        NOTE: for every sequence, initialize a TestDataset class

    """

    def __init__(self, cfgs, seq_name, img_size=[640, 640], split='test', legacy_yolox=True, model='yolox', **kwargs) -> None:
        """
        Args:
            cfgs: dataset config 
            seq_name: name of sequence
            img_size: List[int, int] | Tuple[int, int] image size for detection model 
            legacy_yolox: bool, to be compatible with older versions of yolox
            model: detection model, currently support x, v7, v8
        """
        super().__init__()

        self.model = model

        self.cfgs = cfgs 
        self.seq_name = seq_name
        self.img_size = img_size 
        self.split = split 

        self.seq_path = osp.join(self.cfgs['path'], 'images', self.split, self.seq_name)
        self.imgs_in_seq = sorted(os.listdir(self.seq_path))
        
        self.legacy = legacy_yolox

        self.other_param = kwargs

    def __getitem__(self, idx):
        
        if self.model == 'yolox':
            return self._getitem_yolox(idx)
        elif self.model == 'yolov7':
            return self._getitem_yolov7(idx)
        elif self.model == 'yolov8':
            return self._getitem_yolov8(idx)
    
    def _getitem_yolox(self, idx):

        img = cv2.imread(osp.join(self.seq_path, self.imgs_in_seq[idx])) 
        img_resized, _ = self._preprocess_yolox(img, self.img_size, )
        if self.legacy:
            img_resized = img_resized[::-1, :, :].copy()  # BGR -> RGB
            img_resized /= 255.0
            img_resized -= np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            img_resized /= np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

        return torch.from_numpy(img), torch.from_numpy(img_resized)

    def _getitem_yolov7(self, idx):

        img = cv2.imread(osp.join(self.seq_path, self.imgs_in_seq[idx])) 

        img_resized = self._preprocess_yolov7(img, )  # torch.Tensor

        return torch.from_numpy(img), img_resized
    
    def _getitem_yolov8(self, idx):

        img = cv2.imread(osp.join(self.seq_path, self.imgs_in_seq[idx]))  # (h, w, c)
        # img = self._preprocess_yolov8(img)

        return torch.from_numpy(img), torch.from_numpy(img)


    def _preprocess_yolox(self, img, size, swap=(2, 0, 1)):
        """ convert origin image to resized image, YOLOX-manner

        Args:
            img: np.ndarray
            size: List[int, int] | Tuple[int, int]
            swap: (H, W, C) -> (C, H, W)

        Returns:
            np.ndarray, float
        
        """
        if len(img.shape) == 3:
            padded_img = np.ones((size[0], size[1], 3), dtype=np.uint8) * 114
        else:
            padded_img = np.ones(size, dtype=np.uint8) * 114

        r = min(size[0] / img.shape[0], size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

        padded_img = padded_img.transpose(swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img, r

    def _preprocess_yolov7(self, img, ):
        
        img_resized = self._letterbox(img, new_shape=self.img_size, stride=self.other_param['stride'], )[0]
        img_resized = img_resized[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img_resized = np.ascontiguousarray(img_resized)

        img_resized = torch.from_numpy(img_resized).float()
        img_resized /= 255.0

        return img_resized
    
    def _preprocess_yolov8(self, img, ):

        img = img.transpose((2, 0, 1))
        img = np.ascontiguousarray(img) 

        return img


    def _letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)

    def __len__(self, ):
        return len(self.imgs_in_seq)
    

class DemoDataset(Dataset):
    """
    dataset for demo
    """
    def __init__(self, file_name, img_size=[640, 640], model='yolox', legacy_yolox=True) -> None:
        super().__init__()

        self.file_name = file_name
        self.model = model 
        self.img_size = img_size

        self.is_video = '.mp4' in file_name or '.avi' in file_name 

        if not self.is_video:
            self.imgs_in_seq = sorted(os.listdir(file_name))
        else:
            self.imgs_in_seq = []
            self.cap = cv2.VideoCapture(file_name)

            while True:
                ret, frame = self.cap.read()
                if not ret: break

                self.imgs_in_seq.append(frame)

        self.legacy = legacy_yolox


    def __getitem__(self, idx):

        if not self.is_video:
            img = cv2.imread(osp.join(self.file_name, self.imgs_in_seq[idx]))
        else:
            img = self.imgs_in_seq[idx]
        
        if self.model == 'yolox':
            return self._getitem_yolox(img)
        elif self.model == 'yolov7':
            return self._getitem_yolov7(img)
        elif self.model == 'yolov8':
            return self._getitem_yolov8(img)

        
    def _getitem_yolox(self, img):

        img_resized, _ = self._preprocess_yolox(img, self.img_size, )
        if self.legacy:
            img_resized = img_resized[::-1, :, :].copy()  # BGR -> RGB
            img_resized /= 255.0
            img_resized -= np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            img_resized /= np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

        return torch.from_numpy(img), torch.from_numpy(img_resized)

    def _getitem_yolov7(self, img):

        img_resized = self._preprocess_yolov7(img, )  # torch.Tensor

        return torch.from_numpy(img), img_resized
    
    def _getitem_yolov8(self, img):

        # img = self._preprocess_yolov8(img)

        return torch.from_numpy(img), torch.from_numpy(img)


    def _preprocess_yolox(self, img, size, swap=(2, 0, 1)):
        """ convert origin image to resized image, YOLOX-manner

        Args:
            img: np.ndarray
            size: List[int, int] | Tuple[int, int]
            swap: (H, W, C) -> (C, H, W)

        Returns:
            np.ndarray, float
        
        """
        if len(img.shape) == 3:
            padded_img = np.ones((size[0], size[1], 3), dtype=np.uint8) * 114
        else:
            padded_img = np.ones(size, dtype=np.uint8) * 114

        r = min(size[0] / img.shape[0], size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

        padded_img = padded_img.transpose(swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img, r

    def _preprocess_yolov7(self, img, ):
        
        img_resized = self._letterbox(img, new_shape=self.img_size, stride=self.other_param['stride'], )[0]
        img_resized = img_resized[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img_resized = np.ascontiguousarray(img_resized)

        img_resized = torch.from_numpy(img_resized).float()
        img_resized /= 255.0

        return img_resized
    
    def _preprocess_yolov8(self, img, ):

        img = img.transpose((2, 0, 1))
        img = np.ascontiguousarray(img) 

        return img


    def _letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)

    def __len__(self, ):
        return len(self.imgs_in_seq)

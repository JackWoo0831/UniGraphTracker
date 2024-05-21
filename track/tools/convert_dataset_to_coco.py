"""
将具有**YOLO格式目录的数据集**生成COCO格式的json文件

数据集目录格式:

$data_root/$dataset_name
    |______images
              |____train/val/test
                         |_____$seq_name
                                   |________xxxxx.jpg
    |______labels
            |____train/val/test
                        |_____$seq_name
                                |________xxxxx.txt

txt format:
obj id, x1, y1, x2, y2, cat_id
"""

import os
import os.path as osp
import numpy as np
import json
import cv2

import skimage.io as io
from matplotlib import pyplot as plt
from pycocotools.coco import COCO

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

import argparse
from loguru import logger

DATA_ROOT = '/data/wujiapeng/datasets/VisDrone2019'
CATEGORY_DICT = {
    'MOT17': [{'id': 1, 'name': 'pedestrian'}], 
    'UAVDT': [{'id': 1, 'name': 'car'}],
    'VisDrone2019': [{'id': 1, 'name': 'pedestrain'},
                 {'id': 2, 'name': 'car'},
                 {'id': 3, 'name': 'van'},
                 {'id': 4, 'name': 'truck'},
                 {'id': 5, 'name': 'bus'}]
}


def convert_a_split_to_coco(dataset_name, split_name='train', debug=False):
    """ 将train val test转换成COCO

    Args:
        dataset_name: str, 'MOT17', 'UAVDT', 'VisDrone'
        split_name: str, 'train', 'test', 'val'
    """

    annotations = {}
    annotations['images'] = [] 
    annotations['annotations'] = [] 
    annotations['videos'] = []
    annotations['categories'] = CATEGORY_DICT[dataset_name]

    root_split_path = osp.join(DATA_ROOT, dataset_name, 'images', split_name)
    logger.info(f'seq path is {root_split_path}')
    seqs = sorted(os.listdir(root_split_path))  # 序列名称的列表

    logger.info(f'All seqs:\n {seqs}')

    img_id = 0 # 图像id 初始化为0
    video_id = 0  # 视频序列id
    anno_id = 0  # 注释id
    tid_curr = 0
    tid_last = -1

    for seq in seqs:
        logger.info(f'processing video {seq}')

        images_in_seq = sorted(os.listdir(osp.join(root_split_path, seq)))  # 图像名称的列表
        num_of_imgs = len(images_in_seq)
        
        # 读其中一个图像 获取信息
        img_eg = cv2.imread(os.path.join(root_split_path, seq, images_in_seq[0]))

        img_width, img_height = img_eg.shape[1], img_eg.shape[0]

        # 加入video信息
        video_id += 1
        annotations['videos'].append({'id': video_id, 'file_name': seq})

        # 加入image信息
        for idx, img in enumerate(images_in_seq):
            img_anno = {
                'file_name': f'{seq}/{img}',
                'id': img_id + idx + 1,  # 整个数据集中的id 从1开始
                'frame_id': idx + 1,  # 在当前序列中的id
                'prev_image_id': img_id + idx if idx > 0 else -1, 
                'next_image_id': img_id + idx + 2 if idx < num_of_imgs - 1 else -1,
                'video_id': video_id, 
                'height': img_height, 
                'width': img_width, 
            }

            annotations['images'].append(img_anno)

        logger.info('{}: {} images'.format(seq, num_of_imgs))
        
        # 加入object信息
        for idx, img in enumerate(images_in_seq):
            img_abs_path = osp.join(root_split_path, seq, img)

            txt_name = img_abs_path.replace('images', 'labels').replace('.jpg', '.txt').replace('.png', '.txt')
            assert osp.isfile(txt_name), f'{txt_name} is a wrong path! '

            # txt anno format: 
            # obj_id x1 y1 x2 y2 cat_id
            anno_of_img = np.loadtxt(txt_name, dtype=float, delimiter=' ')

            if not anno_of_img.shape[0]:  # empty in file
                continue

            if anno_of_img.ndim == 1:
                anno_of_img = anno_of_img[np.newaxis, :]

            for row_idx in range(anno_of_img.shape[0]):
                anno_id += 1

                try:
                    obj_id = anno_of_img[row_idx][0]

                except:
                    logger.error(f'{anno_of_img.shape}')
                tlbr = anno_of_img[row_idx][1: 5].tolist()
                tlwh = [tlbr[0], tlbr[1], tlbr[2] - tlbr[0], tlbr[3] - tlbr[1]]
                cat_id = anno_of_img[row_idx][-1]
                
                if not obj_id == tid_last:
                    tid_curr += 1
                    tid_last = obj_id


                obj_anno = {
                    'id': anno_id,
                    'category_id': cat_id + 1,
                    'image_id': img_id + idx + 1,
                    'track_id': tid_curr,
                    'bbox': tlwh,
                    'conf': 1.0,
                    'iscrowd': 0,
                    'area': float(tlwh[2] * tlwh[3]),
                }

                annotations['annotations'].append(obj_anno)
            
        img_id += num_of_imgs 

        if video_id > 1 and debug:
            break

    logger.info('saving json...')

    # '{data_root}/{dataset_name}/annotations/{split_name}.json'
    out_file = osp.join(DATA_ROOT, dataset_name, 'annotations', f'{split_name}.json')

    with open(out_file, 'w') as f:
        json.dump(annotations, f)

    logger.info(f'Done! json file dumped to {out_file}')

    return out_file, root_split_path

    
""" denug funcs"""

def check(json_file, root_split_path):
    """
    Visualize generated COCO data. Only used for debugging.
    """

    coco = COCO(json_file)
    cat_ids = coco.getCatIds(catNms=['car'])
    img_ids = coco.getImgIds()

    index = np.random.randint(0, len(img_ids))
    img = coco.loadImgs(img_ids[index])[0]

    i = io.imread(os.path.join(root_split_path, img['file_name']))

    plt.imshow(i)
    plt.axis('off')
    ann_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
    anns = coco.loadAnns(ann_ids)
    # logger.info(anns)
    # exit(0)
    # coco.showAnns(anns, draw_bbox=True)
    show_anns(anns)
    plt.savefig('annotations.png')

def show_anns(anns):
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in anns:
        c = (np.random.random((1, 3))*0.6+0.4).tolist()[0]
        [bbox_x, bbox_y, bbox_w, bbox_h] = ann['bbox']
        poly = [[bbox_x, bbox_y], [bbox_x, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y]]
        np_poly = np.array(poly).reshape((4,2))
        polygons.append(Polygon(np_poly))
        color.append(c)
    p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.4)
    ax.add_collection(p)
    p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=2)
    ax.add_collection(p)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate COCO format annotations')
    parser.add_argument('--dataset_name', type=str, default='UAVDT', )
    parser.add_argument('--split_name', type=str, default='train', help='trian, val, test')
    parser.add_argument('--debug_mode', action='store_true')

    args = parser.parse_args()

    out_file, root_split_path = convert_a_split_to_coco(dataset_name=args.dataset_name, split_name=args.split_name, debug=args.debug_mode)

    check(out_file, root_split_path)
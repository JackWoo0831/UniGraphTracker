"""
将VisDrone-DET转换为yolo v5格式
obj_id, x0, y0, x1, y1, cls_id

# 本转换代码包含五类 car van truck bus pedestrian

"""
import os
import os.path as osp
import argparse
import cv2
import glob
import numpy as np
from tqdm import tqdm

DATA_ROOT = '/data/wujiapeng/datasets/VisDrone2019_DET/'


def generate_imgs_and_labels(opts):
    """
    产生图片路径的txt文件以及yolo格式真值文件
    """

    txt_name_dict = {'VisDrone2019-DET-train': 'train',
                        'VisDrone2019-DET-val': 'val', 
                        'VisDrone2019-DET-test-dev': 'test'}  # 产生txt文件名称对应关系
    

    if opts.category == 'pedestrain':
        category_list = [1]
    elif opts.category == 'car':
        category_list = [4, 5, 6, 9]
    else:
        category_list = [1, 4, 5, 6, 9]

    if not osp.exists('./visdrone_det/'):
        os.makedirs('./visdrone_det/')

    # 类别ID 从0开始
    category_dict = {category_list[idx]: idx for idx in range(len(category_list))}
    
    # 如果已经存在就不写了
    write_txt = False if os.path.isfile(os.path.join('./visdrone_det', txt_name_dict[opts.split_name] + '.txt')) else True
    print(f'write txt is {write_txt}')
    
    # 所有图片
    imgs = os.listdir(osp.join(DATA_ROOT, opts.split_name, 'images'))
    imgs = sorted(imgs)

    # 初始化要写入的真值txt(yolo 格式)
    gt_to_path = osp.join(DATA_ROOT, 'labels', txt_name_dict[opts.split_name])
    if not osp.exists(gt_to_path):
        os.makedirs(gt_to_path)

    # 初始化要写入的图片(软链接)
    img_to_path = osp.join(DATA_ROOT, 'images', txt_name_dict[opts.split_name]) 
    if not osp.exists(img_to_path):
        os.makedirs(img_to_path)

    # 定义进度条
    pbar = tqdm(enumerate(imgs))
    for _, img_name in pbar:
        img = cv2.imread(osp.join(DATA_ROOT, opts.split_name, 'images', img_name))
        h0, w0 = img.shape[0], img.shape[1]

        anno = np.loadtxt(
            fname=osp.join(DATA_ROOT, opts.split_name, 'annotations', img_name).replace('.jpg', '.txt'),
            dtype=np.float32, 
            delimiter=','
        )
        if anno.ndim == 1: anno = anno[np.newaxis, :]

        # 第一步 产生软链接
        if opts.generate_imgs:
            os.symlink(osp.join(DATA_ROOT, opts.split_name, 'images', img_name),
                        osp.join(img_to_path, img_name))  # 创建软链接
            
        # 第二步 读取GT 
        gt_to_file = osp.join(gt_to_path, img_name[:-4] + '.txt')
        with open(gt_to_file, 'a') as f_gt:

            for i in range(anno.shape[0]):
                category = int(anno[i][5])

                if category in category_list:
                    x0, y0 = int(anno[i][0]), int(anno[i][1])
                    w, h = int(anno[i][2]), int(anno[i][3])

                    x1, y1 = x0 + w, y0 + h

                    category_id = category_dict[category]
                    xc, yc = x0 + w // 2, y0 + h // 2  # 中心点 x y
                    xc, yc = xc / w0, yc / h0
                    w, h = w / w0, h / h0

                    write_line = '{:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                    category_id, xc, yc, w, h)

                    f_gt.write(write_line)

        f_gt.close()

        if write_txt:
            to_file = os.path.join('./visdrone_det', txt_name_dict[opts.split_name] + '.txt')
            with open(to_file, 'a') as f:
                
                f.write(osp.join(DATA_ROOT, 'images', txt_name_dict[opts.split_name], img_name) + '\n')

            f.close()

    
    print('All done!!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_name', type=str, default='VisDrone2019-MOT-train', help='train or test')
    parser.add_argument('--generate_imgs', action='store_true', help='whether generate soft link of imgs')
    parser.add_argument('--category', type=str, default='all', help='pedestrain(1) or car(4, 5, 6, 9) or all(1, 4, 5, 6, 9)')

    opts = parser.parse_args()

    generate_imgs_and_labels(opts)
    # python track/tools/convert_VisDroneDET_to_yolo.py --split_name VisDrone2019-DET-train --generate_imgs
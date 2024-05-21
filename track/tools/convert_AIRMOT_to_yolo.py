"""
convert AIRMOT to yolo

AIRMOT dataset format
frame id, obj id, x0, y0, w, h, 1, category id, 1

category id 1: plane 2: ship
"""

import os
import os.path as osp
import argparse
import cv2
import glob
import numpy as np
import random

DATA_ROOT = '/data/wujiapeng/datasets/AIRMOT'

TRAIN_SEQ = ['6', '59', '14', '39', '29', '23', '38', '51', '71', '57', '19', '26', '56', '33', '40', '64', '60', '24', '61', '10', '65', '47', '54', '21', '68', '44', '52', '1', '78', '55', '58', '67', '73', '13', '79', '63', '50', '3', '66', '32', '34', '17', '80', '9', '74', '45', '15', '4', '18', '8', '16', '49', '28', '72', '12', '2', '48', '25', '30', '76', '43', '42', '53', '31']
TEST_SEQ = ['5', '62', '36', '22', '7', '41', '35', '70', '69', '77', '37', '20', '75', '27', '46', '11']

def generate_imgs_and_labels(opts):
    seq_list = os.listdir(osp.join(DATA_ROOT, 'trainData', 'video'))
    print('--------------------------')
    print(f'Total {len(seq_list)} seqs!!')
    # split train test
    if opts.random: 
        random.shuffle(seq_list)

        bound = int(opts.ratio * len(seq_list))
        train_seq_list = seq_list[: bound]
        test_seq_list = seq_list[bound:]
        del bound

    else:
        train_seq_list = [seq for seq in seq_list if seq in TRAIN_SEQ]
        test_seq_list = [seq for seq in seq_list if seq in TEST_SEQ]

    print(f'train dataset: {train_seq_list}')
    print(f'test dataset: {test_seq_list}')
    print('--------------------------')
    
    if not osp.exists('./airmot/'):
        os.makedirs('./airmot/')

    # category-id dict
    CATEGOTY_ID = {1: 0, 2: 1}

    frame_range = {'start': 0.0, 'end': 1.0}
    if opts.half:  
        frame_range['end'] = 0.5

    process_train_test(train_seq_list, frame_range, CATEGOTY_ID, 'train', norm_for_yolo=opts.norm)
    process_train_test(test_seq_list, {'start': 0.0, 'end': 1.0}, CATEGOTY_ID, 'test', norm_for_yolo=opts.norm)
    print('All Done!!')
                

def process_train_test(seqs: list, frame_range: dict, cat_id_dict: dict, split: str = 'trian', norm_for_yolo: bool = False) -> None:


    for seq in seqs:
        print('Dealing with train dataset...')

        img_dir = osp.join(DATA_ROOT, 'trainData', 'video', seq, 'img') 
        imgs = sorted(os.listdir(img_dir))  
        seq_length = len(imgs)  

        w0, h0 = 1920, 1080  

        ann_of_seq_path = os.path.join(DATA_ROOT, 'trainData', 'gt', f'{seq}.txt') 
        ann_of_seq = np.loadtxt(ann_of_seq_path, dtype=np.float32, delimiter=',') 

        if ann_of_seq.shape[0] == 0:
            print(f'warning: seq {seq} is empty')
            continue
        elif ann_of_seq.ndim == 1:
            ann_of_seq = ann_of_seq[None, :]


        gt_to_path = osp.join(DATA_ROOT, 'labels', split, seq)

        if not osp.exists(gt_to_path):
            os.makedirs(gt_to_path)

        exist_gts = []  

        for idx, img in enumerate(imgs):

            if idx < int(seq_length * frame_range['start']) or idx > int(seq_length * frame_range['end']):
                continue
            
            # Step 1. gen symlink
            # print('step1, creating imgs symlink...')
            if opts.generate_imgs:
                img_to_path = osp.join(DATA_ROOT, 'images', split, seq)  

                if not osp.exists(img_to_path):
                    os.makedirs(img_to_path)

                os.symlink(osp.join(img_dir, img),
                                osp.join(img_to_path, img)) 
            
            # Step 2. gen anno file
            # print('step2, generating gt files...')
            ann_of_current_frame = ann_of_seq[ann_of_seq[:, 0] == float(idx + 1), :]  
            exist_gts.append(True if ann_of_current_frame.shape[0] != 0 else False)

            gt_to_file = osp.join(gt_to_path, img[:-4] + '.txt')

            with open(gt_to_file, 'a') as f_gt:
                for i in range(ann_of_current_frame.shape[0]):    

                    # bbox xywh 
                    obj_id = int(ann_of_current_frame[i][1])
                    x0, y0 = int(ann_of_current_frame[i][2]), int(ann_of_current_frame[i][3])
                    w, h = int(ann_of_current_frame[i][4]), int(ann_of_current_frame[i][5])
                    cat_id = cat_id_dict[int(ann_of_current_frame[i][7])]

                    x1, y1 = x0 + w, y0 + h

                    if not norm_for_yolo:
                            write_line = '{:d} {:d} {:d} {:d} {:d} {:d}\n'.format(
                                obj_id, x0, y0, x1, y1, 0)
                    else:
                        xc, yc = x0 + w // 2, y0 + h // 2 
                        # norm
                        xc, yc = xc / w0, yc / h0
                        w, h = w / w0, h / h0

                        write_line = '{:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                            cat_id, xc, yc, w, h)

                    f_gt.write(write_line)

            f_gt.close()

        # Step 3. gen txt
        print(f'generating img index file of {seq}')        
        to_file = os.path.join('./airmot/', split + '.txt')
        with open(to_file, 'a') as f:
            for idx, img in enumerate(imgs):
                if idx < int(seq_length * frame_range['start']) or idx > int(seq_length * frame_range['end']):
                    continue
                
                if exist_gts[idx]:
                    # f.write('UAVDT/' + 'images/' + split + '/' \
                    #         + seq + '/' + img + '\n')
                    
                    f.write(osp.join(DATA_ROOT, 'images', split, seq, img) + '\n')

            f.close()

    

if __name__ == '__main__':
    if not osp.exists('./uavdt'):
        os.system('mkdir ./uavdt')
    else:
        os.system('rm -rf ./uavdt/*')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate_imgs', action='store_true', help='whether generate soft link of imgs')
    parser.add_argument('--certain_seqs', action='store_true', help='for debug')
    parser.add_argument('--half', action='store_true', help='half frames')
    parser.add_argument('--ratio', type=float, default=0.8, help='ratio of test dataset devide train dataset')
    parser.add_argument('--random', action='store_true', help='random split train and test')

    parser.add_argument('--norm', action='store_true', help='only true when used in yolo training')

    opts = parser.parse_args()
    
    generate_imgs_and_labels(opts)
    # python track/tools/convert_AIRMOT_to_yolo.py --generate_imgs --random --norm
"""
Statistical test (student-t) for ensuring the confidence of results
"""

import os 
import numpy as np 
from scipy import stats
import math 

def read_txt(txt_file, num_of_seqs):

    f = open(txt_file, 'r')
    lines = f.readlines()

    part_1 = [1, num_of_seqs + 1]
    part_2 = [num_of_seqs + 4, 2 * num_of_seqs + 4]
    part_3 = [2 * num_of_seqs + 7, 3 * num_of_seqs + 7]

    hota, mota, idf1 = [], [], []

    for idx, line in enumerate(lines):
        if idx in part_1:
            l = line.split(' ')
            l = [item for item in l if item != '']
            hota.append(float(l[1]) / 100)

        if idx in part_2:
            l = line.split(' ')
            l = [item for item in l if item != '']
            mota.append(float(l[1]) / 100)

        if idx in part_3:
            l = line.split(' ')
            l = [item for item in l if item != '']
            idf1.append(float(l[1]) / 100)
            

    f.close()

    return hota, mota, idf1


if __name__ == '__main__':

    dataset = 'visdrone'

    num_of_seqs = 17 if dataset == 'visdrone' else 9

    our_method_txt = 'runs/detect/student_t/ugt_visdrone.txt'

    other_methods_txt = os.listdir('runs/detect/student_t')

    ori_hota, ori_mota, ori_idf1 = read_txt(our_method_txt, num_of_seqs)

    metrics = ['hota', 'mota', 'idf1']

    for txt in other_methods_txt:
        print(f'processing {txt}')
        
        hota, mota, idf1 = read_txt(os.path.join('runs/detect/student_t', txt), num_of_seqs)

        assert len(hota) == len(ori_hota)

        cnt = 0

        for ori_metric, metric in zip([ori_hota, ori_mota, ori_idf1], [hota, mota, idf1]):
            
            ori_metric = np.array(ori_metric)
            metric = np.array(metric)
            diff = abs(ori_metric - metric)

            diff_mean = np.mean(diff)
            diff_std = np.std(diff)

            t = math.sqrt(len(ori_hota)) * diff_mean / diff_std

            print(f'{metrics[cnt]}: {t} \n\n')

            cnt += 1


            """
            ori_metric = np.array(ori_metric)
            ori_metric_mean = np.mean(ori_metric)
            ori_metric_std = np.std(ori_metric) ** 2

            metric = np.array(metric)
            metric_mean = np.mean(metric)
            metric_std = np.std(metric) ** 2

            t = abs(ori_metric_mean - metric_mean)
            t = t / (np.sqrt(ori_metric_std / (2 * num_of_seqs) + metric_std / (2 * num_of_seqs)))

            print(t)

            if t > 2.037:
                print('OK')
            else:
                print('Not OK')

            print('\n\n\n')

            """



    # python track/utils/student_t_test.py
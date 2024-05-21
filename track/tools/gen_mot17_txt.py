"""
copy result txt file for MOT17 submission
"""

import os 
import shutil

if __name__ == '__main__':
    folder_dir = '/data/wujiapeng/codes/DroneGraphTracker/tracker/results/after_interp/mot17_07_01_23_00'

    txt_names = os.listdir(folder_dir)

    for txt in txt_names:
        shutil.copy(os.path.join(folder_dir, txt), os.path.join(folder_dir, txt.replace('FRCNN', 'SDP')))
        shutil.copy(os.path.join(folder_dir, txt), os.path.join(folder_dir, txt.replace('FRCNN', 'DPM')))

    # python track/tools/gen_mot17_txt.py
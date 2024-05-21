import os
import sys 
import fileinput
import numpy as np
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from PyQt5.QtCore import QThread, pyqtSignal


from collections import deque

from loguru import logger
import numpy as np 
import torch 
from torch.utils.data import DataLoader
import yaml 
import argparse

from utils.envs import select_device
from engine.tracker import Tracker
from data.datasets import TestDataset, DemoDataset
from utils.tracking_utils import *

from tqdm import tqdm

# YOLOX modules
sys.path.append(os.getcwd())
from yolox.exp import get_exp 
from yolox_utils.postprocess import postprocess_yolox
from yolox.utils import fuse_model

# YOLOv7 modules
try:
    sys.path.append(os.getcwd())
    from yolov7.models.experimental import attempt_load
    from yolov7.utils.torch_utils import select_device, time_synchronized, TracedModel
    from yolov7.utils.general import non_max_suppression, scale_coords, check_img_size
    from yolov7_utils.postprocess import postprocess as postprocess_yolov7

except:
    pass

# yolov8 models:
try:
    from ultralytics import YOLO
    from yolov8_utils.postprocess import postprocess as postprocess_yolov8
except:
    pass

def get_args_parser():
    """ get configs from terminal

    """

    parser = argparse.ArgumentParser(description='Tracker Inference')
    
    # add config file paths
    parser.add_argument('--dataset_cfg', type=str, default='./track/cfgs/dataset.yaml', help='dataset config file path')
    parser.add_argument('--assc_graph_cfg', type=str, default='./track/cfgs/assc_graph.yaml', help='assc graph config file path')
    parser.add_argument('--frame_graph_cfg', type=str, default='./track/cfgs/frame_graph.yaml', help='frame graph config file path')
    parser.add_argument('--uni_graph_cfg', type=str, default='./track/cfgs/uni_graph.yaml', help='uni graph config file path')

    # tracking options 
    parser.add_argument('--track_cfg', type=str, default='./track/cfgs/track.yaml', help='train config file path')
    parser.add_argument('--device', type=str, default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    parser.add_argument('--save_images', action='store_true', help='save tracking results (image)')
    parser.add_argument('--save_videos', action='store_true', help='save tracking results (video)')  

    parser.add_argument('--save_data_type', type=str, default='mot_challenge', help='default, mot challenge or visdrone')

    # thresh
    parser.add_argument('--high_det_thresh', type=float, default=0.5, help='high detection thresh')
    parser.add_argument('--low_det_thresh', type=float, default=0.15, help='low detection thresh')
    parser.add_argument('--edge_cls_thresh', type=float, default=0.5, help='edge conf thresh(may not be used)')


    parser.add_argument('--wo_fg', action='store_true', help='without frame graph')
    return parser


class TrackThread(QThread):

    track_result_signal = pyqtSignal(np.ndarray)
    track_process_signal = pyqtSignal(str)

    def __init__(self, 
                 cfgs, 
                 device, 
                 det_config, 
                 dataset_cfg, 
                 data_loader, 
                 model, ) -> None:
        super().__init__()

        self.cfgs = cfgs 
        self.device = device 
        self.det_config = det_config
        self.dataset_cfg = dataset_cfg
        self.data_loader = data_loader
        self.model = model 

        self.tracker = Tracker(self.cfgs, device=self.device, det_model=self.det_config['name'])  # init tracker every new seq

        self.running_flag = True

    def run(self, ):
        frame_idx = 1
        results = []

        for ori_img, img in self.data_loader:

            if not self.running_flag: return 

            self.track_process_signal.emit(f'Tracking Frame {frame_idx}')

            if self.det_config['name'] == 'yolov8':
                img = img.squeeze(0).cpu().numpy()

            else:
                img = img.to(self.device)  # (1, C, H, W)
                img = img.float()

            # get detector output 
            with torch.no_grad():
                if self.det_config['name'] == 'yolov8':
                    output = self.model.predict(img, conf=self.det_config['conf_thresh'], iou=self.det_config['nms_thresh'])
                else:
                    output = self.model(img)

            # postprocess output
            if self.det_config['name'] == 'yolox':
                output = postprocess_yolox(output, self.dataset_cfg['num_classes'], conf_thresh=self.det_config['conf_thresh'], 
                                           img=img, ori_img=ori_img)

            elif self.det_config['name'] == 'yolov7':
                output = postprocess_yolov7(output, self.det_config, img.shape[2:], ori_img.shape)

            elif self.det_config['name'] == 'yolov8':
                output = postprocess_yolov8(output)
            
            else: raise NotImplementedError

            # update tracker
            # output: (tlbr, conf, cls)
            current_tracks = self.tracker.update(output, img, ori_img)
        
            # save results
            cur_tlwh, cur_id, cur_cls, cur_score = [], [], [], []
            for trk in current_tracks:
                bbox = trk.tlwh
                id = trk.track_id
                cls = trk.cls
                score = trk.score

                # filter low area bbox
                vertical = bbox[2] / bbox[3] > self.cfgs['tracker']['wh_ratio_filter']
                if bbox[2] * bbox[3] > self.cfgs['tracker']['min_area'] and not vertical:
                    cur_tlwh.append(bbox)
                    cur_id.append(id)
                    cur_cls.append(cls)
                    cur_score.append(score)
                    # results.append((frame_id + 1, id, bbox, cls))

            results.append((frame_idx, cur_id, cur_tlwh, cur_cls, cur_score))

            
            result_img = self.plot_img(ori_img, [cur_tlwh, cur_id, cur_cls], )
            
            self.track_result_signal.emit(result_img)  # send result image 
                
            frame_idx += 1

    def plot_img(self, img, results, ):
        """
        img: np.ndarray: (H, W, C)
        results: [tlwhs, ids, clses]

        plot images with bboxes of a seq
        """


        if len(img.shape) > 3:
            img = img.squeeze(0)

        img_ = np.ascontiguousarray(np.copy(img))

        tlwhs, ids, clses = results[0], results[1], results[2]
        for tlwh, id, cls in zip(tlwhs, ids, clses):

            # convert tlwh to tlbr
            tlbr = tuple([int(tlwh[0]), int(tlwh[1]), int(tlwh[0] + tlwh[2]), int(tlwh[1] + tlwh[3])])
            # draw a rect
            cv2.rectangle(img_, tlbr[:2], tlbr[2:], get_color(id), thickness=3, )
            # note the id and cls
            text = f'{cls}_{id}'
            cv2.putText(img_, text, (tlbr[0], tlbr[1]), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, 
                            color=(255, 164, 0), thickness=2)
            
        img_ = cv2.resize(img_, (800, 576))  # fix the display weight size
        img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)

        return img_

    def stop(self, ):
        self.running_flag = False


class Ui_MainWindow(object):
    def __init__(self, MainWindow) -> None:
        self.setupUi(MainWindow)
        self.retranslateUi(MainWindow)

        self.cfgs = get_args_parser().parse_args()

        with open(self.cfgs.track_cfg, 'r') as f:
            other_tracking_cfgs = yaml.safe_load(f)
            # merge training cfgs 
            cfgs_dict = vars(self.cfgs)
            cfgs_dict.update(other_tracking_cfgs)  # dict
            self.cfgs = cfgs_dict

        self.device = select_device(self.cfgs['device'])

        self.load_detector()
        self.load_dataset(fileName_choose='demo_videos/uav0000088_00290_v.mp4')
        self.textEdit_file.setText('demo_videos/uav0000088_00290_v.mp4')

        self.checkBox_topo.setChecked(True)
        self.use_topo = True 

        self.lineEdit_filter_thresh.setText('0.1')
        self.lineEdit_receptive_r.setText('0.3')
        
        self.path = os.getcwd()

        # link funcs with ui
        self.slot_init()

        self.track_thread = None
    
    def clear_all(self, ):
        self.dets = []
        self.boxes = []
        self.indexIDs = []
        self.cls_IDs = []

    def slot_init(self, ):
        self.toolButton_choose_file.clicked.connect(self.choose_file)
        self.pushButton_start.clicked.connect(self.track)
        self.checkBox_topo.stateChanged.connect(self.set_topo)
        
    """
    def choose_model(self, ):
        self.clear_all()

        fileName_choose, filetype = QFileDialog.getOpenFileName(self.centralwidget,
                                                                "choose model file", getcwd(), 
                                                                "Model File (*.pt, *.pth)") 
        if fileName_choose != '':
            self.model_path = fileName_choose
        else:
            self.model_path = "track/weights/v2/UAVDT_epoch6_GENConv_20230611_best.pth"  

        self.textEdit_model.setText(self.model_path)
    """

    def choose_file(self, ):
        self.clear_all()

        fileName_choose, filetype = QFileDialog.getOpenFileName(
            self.centralwidget, "choose mp4 file or folder",
            self.path, )  
        
        if fileName_choose != '':
            self.video_path = fileName_choose
        else:
            self.video_path = 'uav0000088_00290_v.mp4'
        
        self.textEdit_file.setText(self.video_path)

        self.load_dataset(fileName_choose)
        

    def track(self, ):       

        # if one is running, kill it
        if self.track_thread is not None:
            print('Stopping current tracking process...')
            self.textEdit_show_process.setText('Stopping current tracking process...')
            self.track_thread.stop()

        self.textEdit_show_process.setText('Start Tracking!')

        # check params 
        det_conf_thresh = self.lineEdit_filter_thresh.text()
        if det_conf_thresh != '':
            det_conf_thresh = float(det_conf_thresh)
            det_conf_thresh = max(0.01, min(det_conf_thresh, 0.9))
            self.det_config['conf_thresh'] = det_conf_thresh

        receptive_r = self.lineEdit_receptive_r.text()

        if receptive_r != '':
            receptive_r = float(receptive_r)
            receptive_r = max(0, min(receptive_r, 0.9))
            self._modify_yaml(file_path=self.cfgs['frame_graph_cfg'], key='max_distance', new_value=receptive_r)
        else:
            self.lineEdit_receptive_r.setText('0.3')
            self._modify_yaml(file_path=self.cfgs['frame_graph_cfg'], key='max_distance', new_value=0.3)

        if not self.use_topo:
            self.cfgs['wo_fg'] = True
            self.cfgs['tracker']['model_path'] = 'track/weights/ugt_wo_fm_20230615.pth'  
            
        else:
            self.cfgs['wo_fg'] = False
            self.cfgs['tracker']['model_path'] = 'track/weights/ugt_best_20230611.pth'  

        # sub thread
        self.track_thread = TrackThread(
            self.cfgs, self.device, self.det_config, self.dataset_cfg, self.data_loader, self.model
        )
        self.track_thread.track_process_signal.connect(lambda msg: self.textEdit_show_process.setText(msg))
        self.track_thread.track_result_signal.connect(self._show_result_image)

        self.track_thread.start()

    def _show_result_image(self, result_image):
        showImage = QtGui.QImage(result_image.data, result_image.shape[1], result_image.shape[0], QtGui.QImage.Format_RGB888)
        self.label_display.setPixmap(QtGui.QPixmap.fromImage(showImage))

    def _modify_yaml(self, file_path, key, new_value, key2=None):
        with open(file_path, 'r') as f:
            cfg = yaml.safe_load(f)
            if key in cfg.keys():
                if key2 is not None:
                    old_value = cfg[key][key2]
                else:
                    old_value = cfg[key]

        f.close()

        with fileinput.FileInput(file_path, inplace=True) as f:
            for line in f:
                if key2 is not None: key_check = key2
                else: key_check = key

                if key_check in line and ':' in line:
                    line = line.replace(str(old_value), str(new_value))
                    print(line, end='')
                else:
                    print(line, end='')
        
        f.close()


    def load_detector(self, ):
        det_config = self.cfgs['detector']
        det_config['name'] = det_config['name'].lower()
        

        if det_config['name'] == 'yolox':

            exp = get_exp(det_config['exp_file'], None)  # TODO: modify num_classes etc. for specific dataset
            model_img_size = exp.input_size
            model = exp.get_model()
            model.to(self.device)
            model.eval()

            logger.info(f"loading detector {det_config['name']} checkpoint {det_config['weights']}")
            ckpt = torch.load(det_config['weights'], map_location=self.device)
            model.load_state_dict(ckpt['model'])
            logger.info("loaded checkpoint done")
            model = fuse_model(model)

            stride = None  # match with yolo v7

            logger.info(f'Now detector is on device {next(model.parameters()).device}')

        elif det_config['name'] == 'yolov7':

            logger.info(f"loading detector {det_config['name']} checkpoint {det_config['weights']}")
            model = attempt_load(det_config['weights'], map_location=self.device)

            # get inference img size
            stride = int(model.stride.max())  # model stride
            model_img_size = check_img_size(det_config['img_size'], s=stride)  # check img_size

            # Traced model
            model = TracedModel(model, device=self.device, img_size=det_config['img_size'])
            model.half()

            logger.info("loaded checkpoint done")

            logger.info(f'Now detector is on device {next(model.parameters()).device}')

        elif det_config['name'] == 'yolov8':

            logger.info(f"loading detector {det_config['name']} checkpoint {det_config['weights']}")
            model = YOLO(det_config['weights'])

            model_img_size = [None, None]  
            stride = None 

            logger.info("loaded checkpoint done")

        else:
            logger.error(f"detector {det_config['name']} is not supprted")
            exit(0)

        self.det_config = det_config
        self.model = model 
        self.model_img_size = model_img_size


    def load_dataset(self, fileName_choose=None, ):
        with open(self.cfgs['dataset_cfg'], 'r') as f:
            dataset_cfg = yaml.safe_load(f)
            dataset_cfg = dataset_cfg[dataset_cfg['dataset_name']]

        self.dataset = DemoDataset(fileName_choose, img_size=self.model_img_size, model=self.det_config['name'], )
        self.data_loader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        self.dataset_cfg = dataset_cfg

    def set_topo(self, state):
        if state == 2:
            self.use_topo = True 
        else:
            self.use_topo = False


    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1305, 966)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(410, 60, 541, 71))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(22)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.pushButton_start = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_start.setGeometry(QtCore.QRect(750, 850, 181, 51))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(16)
        self.pushButton_start.setFont(font)
        self.pushButton_start.setObjectName("pushButton_start")
        self.toolButton_choose_file = QtWidgets.QToolButton(self.centralwidget)
        self.toolButton_choose_file.setGeometry(QtCore.QRect(70, 430, 181, 41))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(16)
        self.toolButton_choose_file.setFont(font)
        self.toolButton_choose_file.setObjectName("toolButton_choose_file")
        self.label_display = QtWidgets.QLabel(self.centralwidget)
        self.label_display.setGeometry(QtCore.QRect(450, 270, 800, 576))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.label_display.setFont(font)
        self.label_display.setObjectName("label_display")
        self.lineEdit_filter_thresh = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_filter_thresh.setGeometry(QtCore.QRect(200, 740, 71, 31))
        self.lineEdit_filter_thresh.setObjectName("lineEdit_filter_thresh")
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(50, 670, 241, 16))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.label_filter_thresh = QtWidgets.QLabel(self.centralwidget)
        self.label_filter_thresh.setGeometry(QtCore.QRect(60, 740, 131, 31))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.label_filter_thresh.setFont(font)
        self.label_filter_thresh.setObjectName("label_filter_thresh")
        self.label_receptive_r = QtWidgets.QLabel(self.centralwidget)
        self.label_receptive_r.setGeometry(QtCore.QRect(70, 800, 121, 31))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.label_receptive_r.setFont(font)
        self.label_receptive_r.setObjectName("label_receptive_r")
        self.lineEdit_receptive_r = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_receptive_r.setGeometry(QtCore.QRect(200, 800, 71, 31))
        self.lineEdit_receptive_r.setObjectName("lineEdit_receptive_r")
        self.label_filter_params = QtWidgets.QLabel(self.centralwidget)
        self.label_filter_params.setGeometry(QtCore.QRect(130, 630, 111, 31))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        font.setBold(False)
        font.setWeight(50)
        self.label_filter_params.setFont(font)
        self.label_filter_params.setObjectName("label_filter_params")
        self.textEdit_file = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit_file.setGeometry(QtCore.QRect(20, 480, 281, 31))
        self.textEdit_file.setObjectName("textEdit_file")
        self.textEdit_show_process = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit_show_process.setGeometry(QtCore.QRect(660, 210, 281, 31))
        self.textEdit_show_process.setObjectName("textEdit_show_process")
        self.label_filter_info = QtWidgets.QLabel(self.centralwidget)
        self.label_filter_info.setGeometry(QtCore.QRect(590, 210, 51, 31))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        font.setBold(False)
        font.setWeight(50)
        self.label_filter_info.setFont(font)
        self.label_filter_info.setObjectName("label_filter_info")
        self.checkBox_topo = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_topo.setGeometry(QtCore.QRect(60, 690, 211, 31))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.checkBox_topo.setFont(font)
        self.checkBox_topo.setObjectName("checkBox_topo")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1305, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "Demo for UniGraphTracker"))
        self.pushButton_start.setText(_translate("MainWindow", "Start Track"))
        self.toolButton_choose_file.setText(_translate("MainWindow", "Choose File"))
        self.label_display.setText(_translate("MainWindow", "Visualization Results Here"))
        self.label_filter_thresh.setText(_translate("MainWindow", "det filter thresh"))
        self.label_receptive_r.setText(_translate("MainWindow", "receptive r"))
        self.label_filter_params.setText(_translate("MainWindow", "Params"))
        self.label_filter_info.setText(_translate("MainWindow", "Info"))
        self.checkBox_topo.setText(_translate("MainWindow", "  topological modeling"))
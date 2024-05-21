

CUDA_VISIBLE_DEVICES=4,5 python yolox_utils/train_yolox.py -f yolox_utils/yolox_m.py -d 2 -b 16 --fp16 -c ./track/weights/yolox_m.pth.tar
# CUDA_VISIBLE_DEVICES=0 python yolox_utils/train_yolox.py -f yolox_utils/yolox_m.py -d 1 -b 16 --fp16 -o -c ./track/weights/yolox_s.pth

# CUDA_VISIBLE_DEVICES=4,5 python yolox_utils/train_yolox.py -f exps/mot/yolox_m.py -d 2 -b 16 --fp16 -c ./track/weights/yolox_m.pth
# CUDA_VISIBLE_DEVICES=4,5 python yolox_utils/train_yolox.py -f exps/mot/yolox_m.py -d 2 -b 16 --fp16 -c ./track/weights/yolox_m.pth 
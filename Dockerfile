FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

# Install Linux packages
RUN apt update \
    && apt install --no-install-recommends -y gcc git zip curl htop libgl1 libglib2.0-0 libpython3-dev gnupg g++ libusb-1.0-0 libsm6

RUN apt upgrade --no-install-recommends -y openssl tar

# Create working directory
WORKDIR /usr/src/unigraphtracker

# Install pip packages
RUN python3 -m pip install --upgrade pip wheel

# Install dependencies
# basic
RUN pip3 install cython
RUN pip3 install numpy==1.23.5
# torch
RUN pip3 install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
# pyg
RUN pip3 install assets/pyg_wheels/torch_scatter-2.1.0+pt112cu113-cp39-cp39-linux_x86_64.whl 
RUN pip3 install assets/pyg_wheels/torch_sparse-0.6.16+pt112cu113-cp39-cp39-linux_x86_64.whl
RUN pip3 install assets/pyg_wheels/torch_cluster-1.6.0+pt112cu113-cp39-cp39-linux_x86_64.whl
RUN pip3 install assets/pyg_wheels/torch_spline_conv-1.2.1+pt112cu113-cp39-cp39-linux_x86_64.whl
RUN pip3 install torch-geometric==2.2.0
# others
RUN pip3 install --no-cache-dir -r requirements.txt
# yolox
RUN git clone -b 0.1.0 git@github.com:Megvii-BaseDetection/YOLOX.git
RUN cd YOLOX
RUN pip3 install -v -e .
# pycocotools
RUN pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
RUN cd ..
# torchreid
RUN cd deep-person-reid
RUN python setup.py develop

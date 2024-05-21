# automatically create the env

echo "----------1. Creating conda virtual env----------"

conda create -n myenv python==3.9
conda activate myenv

echo "----------2. Installing basic packages----------"

pip3 install cython
pip3 install numpy==1.23.5

echo "----------3. Installing torch----------"
pip3 install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

echo "----------4. Installing pyg----------"
pip3 install assets/pyg_wheels/torch_scatter-2.1.0+pt112cu113-cp39-cp39-linux_x86_64.whl 
pip3 install assets/pyg_wheels/torch_sparse-0.6.16+pt112cu113-cp39-cp39-linux_x86_64.whl
pip3 install assets/pyg_wheels/torch_cluster-1.6.0+pt112cu113-cp39-cp39-linux_x86_64.whl
pip3 install assets/pyg_wheels/torch_spline_conv-1.2.1+pt112cu113-cp39-cp39-linux_x86_64.whl
pip3 install torch-geometric==2.2.0

echo "----------5. Installing other dependencies----------"
pip3 install --no-cache-dir -r requirements.txt

echo "----------6. Installing yolox & torchreid----------"
git clone -b 0.1.0 git@github.com:Megvii-BaseDetection/YOLOX.git
cd YOLOX
pip3 install -v -e .

pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
cd ..

cd deep-person-reid
python setup.py develop
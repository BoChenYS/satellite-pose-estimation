# High-resolution networks (HRNets) for object detection
## Introduction
This is a simple instruction to use the HRNet object detection module. For more details visit the original repo [https://github.com/HRNet/HRNet-Object-Detection](https://github.com/HRNet/HRNet-Object-Detection)

<div align=center>

![](images/hrnetv2p.png)

</div>


## Quick start
#### Environment
This code is developed using on Python 3.6 and PyTorch 1.0.0 on Ubuntu 16.04 with NVIDIA GPUs. Training and testing are 
performed using 4 NVIDIA P100 GPUs with CUDA 9.0 and cuDNN 7.0. Other platforms or GPUs are not fully tested.

#### Install
1. Install PyTorch 1.0 following the [official instructions](https://pytorch.org/)
2. Install `mmcv`
````bash
pip install mmcv
````
3. Install `pycocotools`
````bash
git clone https://github.com/cocodataset/cocoapi.git \
 && cd cocoapi/PythonAPI \
 && python setup.py build_ext install \
 && cd ../../
````
4. Install `mmdetection-hrnet`
````bash
git clone https://github.com/HRNet/HRNet-Object-Detection.git

cd mmdetection-hrnet
# compile CUDA extensions.
chmod +x compile.sh
./compile.sh

# run setup
python setup.py install 

# or install locally
python setup.py install --user
````
For more details, see [INSTALL.md](INSTALL.md)

#### Train (multi-gpu training)
Please specify the configuration file in `configs` (learning rate should be adjusted when the number of GPUs is changed).
````bash
python -m torch.distributed.launch --nproc_per_node <GPUS NUM> tools/train.py <CONFIG-FILE> --launcher pytorch
# example:
python -m torch.distributed.launch --nproc_per_node 4 tools/train.py configs/hrnet/faster_rcnn_hrnetv2p_w18_1x.py --launcher pytorch
````

#### Test
````bash
python tools/test.py <CONFIG-FILE> <MODEL WEIGHT> --gpus <GPUS NUM> --eval bbox --out result.pkl
# example:
python tools/test.py configs/hrnet/faster_rcnn_hrnetv2p_w18_1x.py work_dirs/faster_rcnn_hrnetv2p_w18_1x/model_final.pth --gpus 4 --eval bbox --out result.pkl
````


## Reference
[1] Deep High-Resolution Representation Learning for Human Pose Estimation. Ke Sun, Bin Xiao, Dong Liu, and Jingdong Wang. CVPR 2019. [download](https://arxiv.org/pdf/1902.09212.pdf)

[2] Cascade R-CNN: Delving into High Quality Object Detection. Zhaowei Cai, and Nuno Vasconcetos. CVPR 2018.

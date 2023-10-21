# MRFEDepth

<p float="left">
  <img src="assets\main.jpg" width="100%" />
</p>

## Datasets

1. [UAVid 2020](https://uavid.nl/)
2. [WildUAV](https://github.com/hrflr/wuav)
3. [UAV ula](https://github.com/takisu0916/UAV_ula)


## Setup

Assuming a fresh [Anaconda](https://www.anaconda.com/download/) distribution, you can install the dependencies with:
```shell
pip3 install torch==1.7.0+cu110 torchvision==0.8.1+cu110 
pip install Pillow==8.4.0 visdom==0.1.8.9
pip install opencv-python  matplotlib scikit-image 
pip install timm tqdm einops=0.4.1 IPython
```
We ran our experiments with PyTorch 1.7.0, CUDA 11.0, Python 3.6 and Ubuntu 18.04.


## Acknowledgement
Thanks the authors for their works:

[Monodepth2](https://github.com/nianticlabs/monodepth2)

[DIFFNet](https://github.com/brandleyzhou/DIFFNet)

#!/bin/sh

conda create -n group3 python=3.7 -y

conda activate group3

conda install pytorch=1.13.1 torchvision cudatoolkit=11.7 -c pytorch -c conda-forge

conda install pytorch torchvision torchaudio pytorch-cuda=11.3.1 -c pytorch -c nvidia

conda install -c fvcore -c iopath -c conda-forge fvcore

pip install simplejson

pip install einops

pip install timm

conda install av -c conda-forge

pip install psutil

pip install scikit-learn

pip install opencv-python

pip install tensorboard

pip install chardet




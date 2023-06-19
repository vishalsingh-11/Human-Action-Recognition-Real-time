#!/bin/sh

conda create -n action_rec python=3.8 -y

conda activate action_rec

conda install -c fvcore -c iopath -c conda-forge fvcore

pip install simplejson

pip install einops

pip install timm

conda install av -c conda-forge

pip install psutil

pip install scikit-learn

pip install opencv-python

pip install chardet

conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia



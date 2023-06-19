#!/bin/bash

eval "$(conda shell.bash hook)"

conda activate action_rec

python HAR_Main.py

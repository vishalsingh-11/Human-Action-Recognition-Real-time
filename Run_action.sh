#!/bin/bash

eval "$(conda shell.bash hook)"

conda activate ActionRecog

python HAR_Main.py

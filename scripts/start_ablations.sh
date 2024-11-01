#!/bin/bash

python train.py -m --config-name ablation_train_token
python train.py -m --config-name ablation_train_resample
python train.py -m --config-name ablation_train_encodings
python train.py -m --config-name ablation_train_backbone

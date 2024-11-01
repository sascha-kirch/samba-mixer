#!/bin/bash

python train.py -m --config-name experiment_train_data_splits
python train.py -m --config-name experiment_train_model_scaling
python eval.py -m --config-name experiment_eval_different_start_cycles

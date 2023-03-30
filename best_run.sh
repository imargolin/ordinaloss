#!/usr/bin/env bash

# This script is tagged as the best score so far. 
base_path="./constraints_configs/"
device_id=$1
config_file=$2
config_path="$base_path""$config_file"".json"

echo Training a model with the config path $config_path on device_id $device_id

python train_model.py --constraints_path $config_path \
                      --batch_size 16 \
                      --cost_distance 3.0 \
                      --device_id $device_id \
                      --diagonal_value 5.0 \
                      --is_mock 0 \
                      --loss_type "Sinim" \
                      --lr 5.0e-4 \
                      --meta_lr 37.5 \
                      --min_delta 1.0 \
                      --model_architecture vgg19 \
                      --momentum 0.9 \
                      --n_epochs 16 \
                      --n_procs 1 \
                      --num_classes 5 \
                      --optim SGD \
                      --patience 10 \
                      --sch_gamma 0.9 \
                      --sch_step_size 5 \
                      --weight_decay 5.0e-4
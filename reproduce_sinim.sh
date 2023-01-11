#!/usr/bin/env bash

python train_model.py --n_epochs 16 --batch_size 24 --device_id 2 --lr 5.0e-4 --model_architecture vgg19 --patience 16 --loss_type "Sinim" --weight_decay 5.0e-4 --sch_gamma 0.8 --momentum 0.9
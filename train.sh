#!/usr/bin/env bash


#source /data/lily/af726/tools/envs/pytorch2/bin/activate

CUDA_VISIBLE_DEVICES=3 python train.py \
    --history full \
    --train_component multi_sql \
    --epoch 200 > train__multi_sql.out.txt 2>&1

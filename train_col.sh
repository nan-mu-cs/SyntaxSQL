#!/bin/bash
#source /data/lily/af726/tools/envs/pytorch2/bin/activate


cuda=1 #tangra

history=full
save_dir=saved_models/ours_fullhs
log_dir=${save_dir}/train_log

DATE=`date '+%Y-%m-%d-%H:%M:%S'`

CUDA_VISIBLE_DEVICES=${cuda} python train.py \
    --save_dir ${save_dir} \
    --history ${history} \
    --train_component col \
    --epoch 300 > "${log_dir}/train__col__${DATE}.txt" 2>&1 &

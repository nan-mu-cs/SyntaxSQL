#!/bin/bash
#source /data/lily/af726/tools/envs/pytorch2/bin/activate


cuda=3 # morana

history=full
save_dir=saved_models/ours_fullhs
log_dir=${save_dir}/train_log

DATE=`date '+%Y-%m-%d-%H:%M:%S'`

CUDA_VISIBLE_DEVICES=${cuda} python train.py \
    --save_dir ${save_dir} \
    --history ${history} \
    --train_component multi_sql \
    --epoch 200 > "${log_dir}/train__multi_sql__${DATE}.txt" 2>&1 &

CUDA_VISIBLE_DEVICES=${cuda} python train.py \
    --save_dir ${save_dir} \
    --history ${history} \
    --train_component keyword \
    --epoch 200 > "${log_dir}/train__keyword__${DATE}.txt" 2>&1 &

CUDA_VISIBLE_DEVICES=${cuda} python train.py \
    --save_dir ${save_dir} \
    --history ${history} \
    --train_component op \
    --epoch 200 > "${log_dir}/train__op__${DATE}.txt" 2>&1 &

CUDA_VISIBLE_DEVICES=${cuda} python train.py \
    --save_dir ${save_dir} \
    --history ${history} \
    --train_component agg \
    --epoch 200 > "${log_dir}/train__agg__${DATE}.txt" 2>&1 &

CUDA_VISIBLE_DEVICES=${cuda} python train.py \
    --save_dir ${save_dir} \
    --history ${history} \
    --train_component root_tem \
    --epoch 200 > "${log_dir}/train__root_tem__${DATE}.txt" 2>&1 &

CUDA_VISIBLE_DEVICES=${cuda} python train.py \
    --save_dir ${save_dir} \
    --history ${history} \
    --train_component des_asc \
    --epoch 200 > "${log_dir}/train__des_asc__${DATE}.txt" 2>&1 &

CUDA_VISIBLE_DEVICES=${cuda} python train.py \
    --save_dir ${save_dir} \
    --history ${history} \
    --train_component having \
    --epoch 200 > "${log_dir}/train__having__${DATE}.txt" 2>&1 &

CUDA_VISIBLE_DEVICES=${cuda} python train.py \
    --save_dir ${save_dir} \
    --history ${history} \
    --train_component andor \
    --epoch 200 > "${log_dir}/train__andor__${DATE}.txt" 2>&1 &

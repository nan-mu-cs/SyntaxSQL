#!/bin/bash

#source /data/lily/af726/tools/envs/pytorch2/bin/activate


## --part for part of history
export CUDA_VISIBLE_DEVICES=0

# python test.py \
#     --test_data_path /data/projects/nl2sql/hold_out/test.json \
#     --models ./saved_models/before_emnlp/ours_fullhs/ \
#     --output_path ./test_results/before_emnlp/ours_fullhs_result.txt \
#     > ./test_results/before_emnlp/ours_fullhs_result.out.txt 2>&1
               # --toy

python test.py \
    --test_data_path /data/projects/nl2sql/hold_out/test.json \
    --models         ./saved_models/before_emnlp/ours_fullhs_aug/ \
    --output_path    ./test_results/before_emnlp/ours_fullhs_aug_result.txt \
    > ./test_results/before_emnlp/ours_fullhs_aug_result.out.txt 2>&1


# python test.py --models /home/lily/rz268/nl2sql/datasets/generated_data_part/saved_models/ \
#                --test_data_path /data/projects/nl2sql/hold_out/test.json \
#                --output_path ./results/ours_parths_result.txt \
#                --part
#
# python test.py --models /home/lily/rz268/nl2sql/datasets/generated_data_wikisql/saved_models/ \
#                --test_data_path /data/projects/nl2sql/hold_out/test.json \
#                --output_path ./results/ours_fullhs_wikisql_result.txt

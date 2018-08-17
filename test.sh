#!/bin/bash

## --part for part of history
export CUDA_VISIBLE_DEVICES=1

# python test.py --models ./saved_models/ \
#                --test_data_path /data/projects/nl2sql/hold_out/test.json \
#                --output_path ./results/ours_fullhs_result.txt \
#                # --toy

python test.py --models ./saved_models_data_augment/ \
               --test_data_path /data/projects/nl2sql/hold_out/test.json \
               --output_path ./results/ours_fullhs_aug_result.txt \
               # --toy


# python test.py --models /home/lily/rz268/nl2sql/datasets/generated_data_part/saved_models/ \
#                --test_data_path /data/projects/nl2sql/hold_out/test.json \
#                --output_path ./results/ours_parths_result.txt \
#                --part
#
# python test.py --models /home/lily/rz268/nl2sql/datasets/generated_data_wikisql/saved_models/ \
#                --test_data_path /data/projects/nl2sql/hold_out/test.json \
#                --output_path ./results/ours_fullhs_wikisql_result.txt

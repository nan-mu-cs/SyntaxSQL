#!/usr/bin/env bash

## --part for part of history
python test.py --toy --models /data/projects/nl2sql/models/ours_no_hs/saved_models \
               --test_data_path /data/projects/nl2sql/hold_out/test.json \
               --output_path ./ours_no_hs_result.txt
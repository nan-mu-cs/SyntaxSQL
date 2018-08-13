## Documentation for SyntaxSQLNet


## Environment
``source /data/lily/af726/tools/envs/pytorch2/bin/activate``

## Folder structure
- ``data/`` contains raw train/dev/test data
- ``generated_data/`` contains preprocessed train/dev data with full/partial history path
- ``models/`` contains each module file.
- `` modules.py`` ?? what is the difference from ``models`` directory?

- ``hierachical_col_emb_version/`` contains the version with hierarchical table embeddings (already merged into main files, so this is folder deprecated)
- ``trainable_version/`` contains the version with trainable embeddings for SQL keywords




## Training
Example:
```
python train.py \
    --history full \
    --train_component multi_sql \
    --epoch 200 > train__multi_sql.out.txt 2>&1
```
The model save location can be configured in ``train.py``
The train/dev location is specified in ``utils.py``


## Testing
Example:
```
python test.py
    --models <saved model location> \
    --test_data_path /data/projects/nl2sql/hold_out/test.json \
    --output_path ./results/ours_fullhs_result.txt
```

## Evaluation
Use ``evaluation.py`` script

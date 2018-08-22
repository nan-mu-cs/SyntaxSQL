python preprocess.py -root_dir ../data_model/ -dataset conala -src_words_min_frequency 2 -tgt_words_min_frequency 2


CUDA_VISIBLE_DEVICES=4 python train.py -root_dir ../data_model/ -dataset conala -rnn_size 300 -word_vec_size 250 -decoder_input_size 200 -layers 1 -start_checkpoint_at 30 -learning_rate 0.002 -epochs 100 -global_attention "dot" -attn_hidden 0 -dropout 0.1 -dropout_i 0.1 -lock_dropout -copy_prb hidden

CUDA_VISIBLE_DEVICES=6 python evaluate.py -root_dir ../data_model/ -dataset conala -split dev -model_path "../data_model/conala/run.5/m_*.pt"


CUDA_VISIBLE_DEVICES=6 python evaluate.py -root_dir ../data_model/ -dataset conala -split test -model_path "../data_model/conala/dev_best.pt"

#################
# initial results
#################

DEV: tgt: 0.26585131650257576, 3.02%
TEST: tgt: 0.25629099418611, 3.00%
use gold_layout: tgt: 0.7022770445213729, 31.60%


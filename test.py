import json
import torch
import datetime
import argparse
import numpy as np
from utils import *
from supermodel import SuperModel

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_emb', action='store_true',
            help='Train word embedding.')
    args = parser.parse_args()

    N_word=300
    B_word=42
    N_h = 300
    N_depth=2
    if args.toy:
        USE_SMALL=True
        GPU=True
        BATCH_SIZE=20
    else:
        USE_SMALL=False
        GPU=True
        BATCH_SIZE=64
    # TRAIN_ENTRY=(False, True, False)  # (AGG, SEL, COND)
    # TRAIN_AGG, TRAIN_SEL, TRAIN_COND = TRAIN_ENTRY
    learning_rate = 1e-4

    #TODO
    data = load_test_dataset()

    word_emb = load_word_emb('glove/glove.%dB.%dd.txt'%(B_word,N_word), \
            load_used=args.train_emb, use_small=USE_SMALL)
    #word_emb = load_concat_wemb('glove/glove.42B.300d.txt', "/data/projects/paraphrase/generation/para-nmt-50m/data/paragram_sl999_czeng.txt")

    model = SuperModel(word_emb, N_word=N_word, gpu=GPU, trainable_emb = args.train_emb)

    agg_m, sel_m, cond_m = best_model_name(args)
    torch.save(model.state_dict(), "saved_models/{}_models.dump".format(args.train_component))

    print "Loading from modules..."
    model.multi_sql.load_state_dict(torch.load("saved_models/multi_sql_models.dump"))
    model.key_word.load_state_dict(torch.load("saved_models/keyword_models.dump"))
    model.col.load_state_dict(torch.load("saved_models/col_models.dump"))
    model.op.load_state_dict(torch.load("saved_models/op_models.dump"))
    model.agg.load_state_dict(torch.load("saved_models/agg_models.dump"))
    model.root_teminal.load_state_dict(torch.load("saved_models/root_tem_models.dump"))
    model.des_asc.load_state_dict(torch.load("saved_models/des_asc_models.dump"))
    model.having.load_state_dict(torch.load("saved_models/having_models.dump"))

    test_acc(model, batch_size, data)
    #test_exec_acc()

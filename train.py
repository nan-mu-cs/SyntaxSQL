import json
import torch
import datetime
import argparse
import numpy as np
from utils import *
# from model.sqlnet import SQLNet
from word_embedding import WordEmbedding
# from models import MultiSqlPredictor,KeyWordPredictor,ColPredictor,OpPredictor,RootTeminalPredictor,DesAscLimitPredictor,AggPredictor
from models.agg_predictor import AggPredictor
from models.col_predictor import ColPredictor
from models.desasc_limit_predictor import DesAscLimitPredictor
from models.having_predictor import HavingPredictor
from models.keyword_predictor import KeyWordPredictor
from models.multisql_predictor import MultiSqlPredictor
from models.op_predictor import OpPredictor
from models.root_teminal_predictor import RootTeminalPredictor
from models.andor_predictor import AndOrPredictor

TRAIN_COMPONENTS = ('multi_sql','keyword','col','op','agg','root_tem','des_asc','having','andor')
SQL_TOK = ['<UNK>', '<END>', 'WHERE', 'AND', 'EQL', 'GT', 'LT', '<BEG>']
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--toy', action='store_true',
            help='If set, use small data; used for fast debugging.')
    parser.add_argument('--suffix', type=str, default='',
            help='The suffix at the end of saved model name.')
    parser.add_argument('--sd', type=str, default='',
            help='set model save directory.')
    parser.add_argument('--dataset', type=int, default=0,
            help='0: original dataset, 1: re-split dataset, 2: new complex dataset')
    parser.add_argument('--train_emb', action='store_true',
            help='Train word embedding.')
    parser.add_argument('--hier_col', action='store_true',
            help='Use hierarchical table/column embedding.')
    parser.add_argument('--train_component',type=str,default='',
                        help='set train components,available:[multi_sql,keyword,col,op,agg,root_tem,des_asc,having,andor]')
    parser.add_argument('--epoch',type=int,default=500,
                        help='number of epoch for training')
    parser.add_argument('--history', type=str, default='full',
                        help='part or full history')
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
    if args.train_component not in TRAIN_COMPONENTS:
        print("Invalid train component")
        exit(1)
    train_data = load_train_dev_dataset(args.train_component,"train",args.history)
    dev_data = load_train_dev_dataset(args.train_component, "dev",args.history)
    # sql_data, table_data, val_sql_data, val_table_data, \
    #         test_sql_data, test_table_data, \
    #         TRAIN_DB, DEV_DB, TEST_DB = load_dataset(args.dataset, use_small=USE_SMALL)

    word_emb = load_word_emb('glove/glove.%dB.%dd.txt'%(B_word,N_word), \
            load_used=args.train_emb, use_small=USE_SMALL)
    print("finished load word embedding")
    #word_emb = load_concat_wemb('glove/glove.42B.300d.txt', "/data/projects/paraphrase/generation/para-nmt-50m/data/paragram_sl999_czeng.txt")
    model = None
    if args.train_component == "multi_sql":
        model = MultiSqlPredictor(N_word=N_word,N_h=N_h,N_depth=N_depth,gpu=GPU)
    elif args.train_component == "keyword":
        model = KeyWordPredictor(N_word=N_word,N_h=N_h,N_depth=N_depth,gpu=GPU)
    elif args.train_component == "col":
        model = ColPredictor(N_word=N_word,N_h=N_h,N_depth=N_depth,gpu=GPU,hier_col=args.hier_col)
    elif args.train_component == "op":
        model = OpPredictor(N_word=N_word,N_h=N_h,N_depth=N_depth,gpu=GPU)
    elif args.train_component == "agg":
        model = AggPredictor(N_word=N_word,N_h=N_h,N_depth=N_depth,gpu=GPU)
    elif args.train_component == "root_tem":
        model = RootTeminalPredictor(N_word=N_word,N_h=N_h,N_depth=N_depth,gpu=GPU)
    elif args.train_component == "des_asc":
        model = DesAscLimitPredictor(N_word=N_word,N_h=N_h,N_depth=N_depth,gpu=GPU)
    elif args.train_component == "having":
        model = HavingPredictor(N_word=N_word,N_h=N_h,N_depth=N_depth,gpu=GPU)
    elif args.train_component == "andor":
        model = AndOrPredictor(N_word=N_word, N_h=N_h, N_depth=N_depth, gpu=GPU)
    # model = SQLNet(word_emb, N_word=N_word, gpu=GPU, trainable_emb=args.train_emb)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = 0)
    print("finished build model")
    # agg_m, sel_m, cond_m = best_model_name(args)
    #
    # if args.train_emb: # Load pretrained model.
    #     agg_lm, sel_lm, cond_lm = best_model_name(args, for_load=True)
    #     print "Loading from %s"%agg_lm
    #     model.agg_pred.load_state_dict(torch.load(agg_lm))
    #     print "Loading from %s"%sel_lm
    #     model.selcond_pred.load_state_dict(torch.load(sel_lm))
    #     print "Loading from %s"%cond_lm
    #     model.cond_pred.load_state_dict(torch.load(cond_lm))


    #initial accuracy
    # init_acc = epoch_acc(model, BATCH_SIZE, val_sql_data, val_table_data, TRAIN_ENTRY)
    # if TRAIN_AGG:
    #     torch.save(model.agg_pred.state_dict(), agg_m)
    # if TRAIN_SEL:
    #     torch.save(model.selcond_pred.state_dict(), sel_m)
    # if TRAIN_COND:
    #     torch.save(model.op_str_pred.state_dict(), cond_m)

    print_flag = False
    embed_layer = WordEmbedding(word_emb, N_word, gpu=GPU,
                                SQL_TOK=SQL_TOK, trainable=args.train_emb)
    print("start training")
    best_acc = 0.0
    for i in range(args.epoch):
        print('Epoch %d @ %s'%(i+1, datetime.datetime.now()))
        print(' Loss = %s'%epoch_train(
                model, optimizer, BATCH_SIZE,args.train_component,embed_layer,train_data,hier_col=args.hier_col))
        acc = epoch_acc(model, BATCH_SIZE, args.train_component,embed_layer,dev_data,hier_col=args.hier_col)
        if acc > best_acc:
            best_acc = acc
            print("Save model...")
            torch.save(model.state_dict(),"saved_models/{}_models.dump".format(args.train_component))
        # print '\nTrain sel acc: %s, sel # acc: %s' % (train_bkd_acc[1], train_bkd_acc[0])
        #print ' Breakdown results: agg #: %s, agg: %s, sel: %s, cond: %s, sel #: %s, cond #: %s, cond col: %s, cond op: %s, cond val: %s, group #: %s, group: %s, order #: %s, order: %s, order agg: %s, order par: %s'\
        #    % (train_bkd_acc[0], train_bkd_acc[1], train_bkd_acc[2], train_bkd_acc[3], train_bkd_acc[4], train_bkd_acc[5], train_bkd_acc[6], train_bkd_acc[7], train_bkd_acc[8], train_bkd_acc[9], train_bkd_acc[10], train_bkd_acc[11], train_bkd_acc[12], train_bkd_acc[13], train_bkd_acc[14])
        # if i > 497:
        #     print_flag = True
        # val_tot_acc, val_bkd_acc = epoch_acc(model, BATCH_SIZE, val_sql_data, val_table_data, TRAIN_ENTRY, error_print = print_flag, train_flag = False) #for detailed error analysis, pass True to error_print
        # print '\nDev sel acc: %s, sel # acc: %s' % (val_bkd_acc[1], val_bkd_acc[0])
        #print ' Breakdown results: agg #: %s, agg: %s,  sel: %s, cond: %s, sel #: %s, cond #: %s, cond col: %s, cond op: %s, cond val: %s, group #: %s, group: %s, order #: %s, order: %s, order agg: %s, order par: %s'\
        #    % (val_bkd_acc[0], val_bkd_acc[1], val_bkd_acc[2], val_bkd_acc[3], val_bkd_acc[4], val_bkd_acc[5], val_bkd_acc[6], val_bkd_acc[7], val_bkd_acc[8], val_bkd_acc[9], val_bkd_acc[10], val_bkd_acc[11], val_bkd_acc[12], val_bkd_acc[13], val_bkd_acc[14])

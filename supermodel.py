import json
import torch
import datetime
import argparse
import numpy as np
from utils import *
from word_embedding import WordEmbedding
from models.agg_predictor import AggPredictor
from models.col_predictor import ColPredictor
from models.desasc_limit_predictor import DesAscLimitPredictor
from models.having_predictor import HavingPredictor
from models.keyword_predictor import KeyWordPredictor
from models.multisql_predictor import MultiSqlPredictor
from models.root_teminal_predictor import RootTeminalPredictor


class Stack:
     def __init__(self):
         self.items = []

     def isEmpty(self):
         return self.items == []

     def push(self, item):
         self.items.append(item)

     def pop(self):
         return self.items.pop()

     def peek(self):
         return self.items[len(self.items)-1]

     def size(self):
         return len(self.items)


class SuperModel(nn.Module):
    def __init__(self, word_emb, N_word, N_h=300, N_depth=2, gpu=True, trainable_emb=False):
        super(SuperModel, self).__init__()
        self.gpu = gpu
        self.N_h = N_h
        self.N_depth = N_depth
        self.trainable_emb = trainable_emb

        self.SQL_TOK = ['<UNK>', '<END>', 'WHERE', 'AND', 'EQL', 'GT', 'LT', '<BEG>']

        # word embedding layer
        self.embed_layer = WordEmbedding(word_emb, N_word, gpu,
                self.SQL_TOK, trainable=trainable_emb)

        # initial all modules
        self.multi_sql = MultiSqlPredictor(N_word=N_word,N_h=N_h,N_depth=N_depth,gpu=GPU)

        self.key_word = KeyWordPredictor(N_word=N_word,N_h=N_h,N_depth=N_depth,gpu=GPU)

        self.col = ColPredictor(N_word=N_word,N_h=N_h,N_depth=N_depth,gpu=GPU)

        self.op = OpPredictor(N_word=N_word,N_h=N_h,N_depth=N_depth,gpu=GPU)

        self.agg = AggPredictor(N_word=N_word,N_h=N_h,N_depth=N_depth,gpu=GPU)

        self.root_teminal = RootTeminalPredictor(N_word=N_word,N_h=N_h,N_depth=N_depth,gpu=GPU)

        self.des_asc = DesAscLimitPredictor(N_word=N_word,N_h=N_h,N_depth=N_depth,gpu=GPU)

        self.having = HavingPredictor(N_word=N_word,N_h=N_h,N_depth=N_depth,gpu=GPU)

        self.softmax = nn.Softmax() #dim=1
        self.CE = nn.CrossEntropyLoss()
        self.log_softmax = nn.LogSoftmax()
        self.mlsml = nn.MultiLabelSoftMarginLoss()
        self.bce_logit = nn.BCEWithLogitsLoss()
        self.sigm = nn.Sigmoid()
        if gpu:
            self.cuda()


    def forward(self, q_seq, history, tables):
        B = len(q_seq)
        q_emb_var, q_len = embed_layer.gen_x_q_batch(q_seq)
        hs_emb_var, hs_len = embed_layer.gen_x_history_batch(history)
        col_emb_var,col_len = embed_layer.gen_table_embedding(tables)

        mkw_emb_var = embed_layer.gen_word_list_embedding(["none","except","intersect","union"],(B))
        mkw_len = np.full(q_len.shape, 4,dtype=np.int64)
        kw_emb_var = embed_layer.gen_word_list_embedding(["where", "group by", "order by"], (B))
        mkw_len = np.full(q_len.shape, 3, dtype=np.int64)

        stack = Stack()
        stack.push("ROOT")
        history = []
        while stack:
            vet = stack.pop()
            history.append(vet)
            if vet == "ROOT":
                pass
            elif vet == "multi_sql":
                pass
            elif vet == "":
                pass

        return history


    def gen_sql(self, history):
        sql = []

        return sql

    def check_acc(self, pred_sql, gt_sql):
        pass

import json
import torch
import datetime
import argparse
import numpy as np
import torch.nn as nn
from utils import *
from word_embedding import WordEmbedding
from models.agg_predictor import AggPredictor
from models.col_predictor import ColPredictor
from models.desasc_limit_predictor import DesAscLimitPredictor
from models.having_predictor import HavingPredictor
from models.keyword_predictor import KeyWordPredictor
from models.multisql_predictor import MultiSqlPredictor
from models.root_teminal_predictor import RootTeminalPredictor
from models.andor_predictor import AndOrPredictor
from models.op_predictor import OpPredictor
from gen_partical_module import index_to_column_name


SQL_OPS = ('none','intersect', 'union', 'except')
KW_OPS = ('where','groupBy','orderBy')
AGG_OPS = ('max', 'min', 'count', 'sum', 'avg')
ROOT_TERM_OPS = ("root","terminal")
COND_OPS = ("and","or")
DEC_ASC_OPS = (("asc",True),("asc",False),("desc",True),("desc",False))
NEW_WHERE_OPS = ('=','>','<','>=','<=','!=','like','not in','in','between')
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
        self.multi_sql = MultiSqlPredictor(N_word=N_word,N_h=N_h,N_depth=N_depth,gpu=gpu)

        self.key_word = KeyWordPredictor(N_word=N_word,N_h=N_h,N_depth=N_depth,gpu=gpu)

        self.col = ColPredictor(N_word=N_word,N_h=N_h,N_depth=N_depth,gpu=gpu)

        self.op = OpPredictor(N_word=N_word,N_h=N_h,N_depth=N_depth,gpu=gpu)

        self.agg = AggPredictor(N_word=N_word,N_h=N_h,N_depth=N_depth,gpu=gpu)

        self.root_teminal = RootTeminalPredictor(N_word=N_word,N_h=N_h,N_depth=N_depth,gpu=gpu)

        self.des_asc = DesAscLimitPredictor(N_word=N_word,N_h=N_h,N_depth=N_depth,gpu=gpu)

        self.having = HavingPredictor(N_word=N_word,N_h=N_h,N_depth=N_depth,gpu=gpu)

        self.andor = AndOrPredictor(N_word=N_word, N_h=N_h, N_depth=N_depth, gpu=gpu)

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
        # print("Batch size:{}".format(B))
        q_emb_var, q_len = self.embed_layer.gen_x_q_batch(q_seq)
        col_emb_var,col_len = self.embed_layer.gen_table_embedding([[tables["table_names"],tables["column_names"],tables["column_types"]]]*B)

        mkw_emb_var = self.embed_layer.gen_word_list_embedding(["none","except","intersect","union"],(B))
        mkw_len = np.full(q_len.shape, 4,dtype=np.int64)
        kw_emb_var = self.embed_layer.gen_word_list_embedding(["where", "group by", "order by"], (B))
        kw_len = np.full(q_len.shape, 3, dtype=np.int64)

        stack = Stack()
        stack.push("root")
        history = [["root"]]*B
        andor_cond = ""
        has_limit = False
        # sql = {}
        current_sql = {}
        sql_stack = []
        idx_stack = []
        kw = ""
        while not stack.isEmpty():
            vet = stack.pop()
            hs_emb_var, hs_len = self.embed_layer.gen_x_history_batch(history)
            if len(idx_stack) > 0 and stack.size() < idx_stack[-1]:
                idx_stack.pop()
                current_sql = sql_stack.pop()
            # history.append(vet)
            # print("hs_emb:{} hs_len:{}".format(hs_emb_var.size(),hs_len.size()))
            if vet == "root":
                idx_stack.append(stack.size())
                sql_stack.append(current_sql)
                current_sql["sql"] = {}
                current_sql = current_sql["sql"]
                # print("q_emb_var:{} hs_emb_var:{} mkw_emb_var:{}".format(q_emb_var.size(),hs_emb_var.size(),mkw_emb_var.size()))
                score = self.multi_sql.forward(q_emb_var,q_len,hs_emb_var,hs_len,mkw_emb_var,mkw_len)
                label = np.argmax(score[0].data.cpu().numpy())
                label = SQL_OPS[label]
                history[0].append(label)
                stack.push(label)
                if label != "none":
                    current_sql["nested_label"] = label

            elif vet in ('intersect', 'except', 'union'):
                stack.push("root")
                stack.push("root")
            elif vet == "none":
                score = self.key_word.forward(q_emb_var,q_len,hs_emb_var,hs_len,kw_emb_var,kw_len)
                kw_num_score, kw_score = [x.data.cpu().numpy() for x in score]
                num_kw = np.argmax(kw_num_score[0])
                kw_score = list(np.argsort(-kw_score[0])[:num_kw])
                kw_score.sort(reverse=True)
                print("num_kw:{}".format(num_kw))
                for kw in kw_score:
                    stack.push(KW_OPS[kw])
                stack.push("select")
            elif vet in ("select","orderBy","where","groupBy","having"):
                kw = vet
                current_sql[kw] = []
                history[0].append(vet)
                stack.push(("col",vet))
                # score = self.andor.forward(q_emb_var,q_len,hs_emb_var,hs_len)
                # label = score[0].data.cpu().numpy()
                # andor_cond = COND_OPS[label]
                # history.append("")
            # elif vet == "groupBy":
            #     score = self.having.forward(q_emb_var,q_len,hs_emb_var,hs_len,col_emb_var,col_len,)
            elif isinstance(vet,tuple) and vet[0] == "col":
                # print("q_emb_var:{} hs_emb_var:{} col_emb_var:{}".format(q_emb_var.size(), hs_emb_var.size(),col_emb_var.size()))
                score = self.col.forward(q_emb_var, q_len, hs_emb_var, hs_len, col_emb_var, col_len)
                col_num_score, col_score = [x.data.cpu().numpy() for x in score]
                col_num = np.argmax(col_num_score[0]) + 1  # double check
                cols = np.argsort(-col_score[0])[:col_num]
                for col in cols:
                    if vet[1] == "where":
                        stack.push(("op","where",col))
                    elif vet[1] != "groupBy":
                        stack.push(("agg",vet[1],col))
                    elif vet[1] == "groupBy":
                        history[0].append(index_to_column_name(col, tables))
                        current_sql[kw].append(col)
                #predict and or or when there is multi col in where condition
                if col_num > 1 and vet[1] == "where":
                    score = self.andor.forward(q_emb_var,q_len,hs_emb_var,hs_len)
                    label = np.argmax(score[0].data.cpu().numpy())
                    andor_cond = COND_OPS[label]
                    current_sql[kw].append(andor_cond)
                if vet[1] == "groupBy" and col_num > 0:
                    score = self.having.forward(q_emb_var, q_len, hs_emb_var, hs_len, col_emb_var, col_len, np.full(B, cols[0],dtype=np.int64))
                    label = np.argmax(score[0].data.cpu().numpy())
                    if label == 1:
                        stack.push("having")
                # history.append(index_to_column_name(cols[-1], tables[0]))
            elif isinstance(vet,tuple) and vet[0] == "agg":
                history[0].append(index_to_column_name(vet[2], tables))
                current_sql[kw].append(index_to_column_name(vet[2], tables))
                hs_emb_var, hs_len = self.embed_layer.gen_x_history_batch(history)
                score = self.agg.forward(q_emb_var,q_len,hs_emb_var,hs_len,col_emb_var,col_len,np.full(B, vet[2],dtype=np.int64))
                agg_num_score, agg_score = [x.data.cpu().numpy() for x in score]
                agg_num = np.argmax(agg_num_score[0])  # double check
                agg_idxs = np.argsort(-agg_score[0])[:agg_num]
                # print("agg:{}".format([AGG_OPS[agg] for agg in agg_idxs]))
                if len(agg_idxs) > 0:
                    history[0].append(AGG_OPS[agg_idxs[0]])
                for agg in agg_idxs[1:]:
                    history[0].append(index_to_column_name(vet[2], tables))
                    history[0].append(AGG_OPS[agg])
                current_sql[kw].append([AGG_OPS[agg] for agg in agg_idxs])
                if vet[1] == "having":
                    stack.push(("op","having",vet[2],agg_idxs))
                if vet[1] == "orderBy":
                    stack.push(("des_asc",vet[2],agg_idxs))
            elif isinstance(vet,tuple) and vet[0] == "op":
                current_sql[kw].append(index_to_column_name(vet[2], tables))
                if vet[1] == "where":
                    history[0].append(index_to_column_name(vet[2], tables))
                    hs_emb_var, hs_len = self.embed_layer.gen_x_history_batch(history)
                score = self.op.forward(q_emb_var,q_len,hs_emb_var,hs_len,col_emb_var,col_len,np.full(B, vet[2],dtype=np.int64))
                op_num_score, op_score = [x.data.cpu().numpy() for x in score]
                op_num = np.argmax(op_num_score[0]) + 1  # num_score 0 maps to 1 in truth, must have at least one op
                ops = np.argsort(-op_score[0])[:op_num]
                current_sql[kw].append([NEW_WHERE_OPS[op] for op in ops])
                if op_num > 0:
                    history[0].append(NEW_WHERE_OPS[ops[0]])
                for op in ops[1:]:
                    history[0].append(index_to_column_name(vet[2], tables))
                    history[0].append(NEW_WHERE_OPS[op])
                stack.push(("root_teminal",vet[2]))
            elif isinstance(vet,tuple) and vet[0] == "root_teminal":
                score = self.root_teminal.forward(q_emb_var,q_len,hs_emb_var,hs_len,col_emb_var,col_len,np.full(B, vet[1],dtype=np.int64))
                label = np.argmax(score[0].data.cpu().numpy())
                label = ROOT_TERM_OPS[label]
                if label == "root":
                    stack.push("root")
                    history[0].append("root")
                else:
                    current_sql[kw].append("terminal")
            elif isinstance(vet,tuple) and vet[0] == "des_asc":
                score = self.des_asc.forward(q_emb_var,q_len,hs_emb_var,hs_len,col_emb_var,col_len,np.full(B, vet[1],dtype=np.int64))
                label = np.argmax(score[0].data.cpu().numpy())
                dec_asc,has_limit = DEC_ASC_OPS[label]
                history[0].append(dec_asc)
                current_sql[kw].append(dec_asc)
                current_sql[kw].append(has_limit)
        print("sql:{}".format(current_sql))
        # print("history:{}".format(history[0]))
        return history


    def gen_sql(self, history):
        sql = []

        return sql

    def check_acc(self, pred_sql, gt_sql):
        pass

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

AGG_OPS = ('none', 'maximum', 'minimum', 'count', 'sum', 'average')
class WordEmbedding(nn.Module):
    def __init__(self, word_emb, N_word, gpu, SQL_TOK,
            trainable=False):
        super(WordEmbedding, self).__init__()
        self.trainable = trainable
        self.N_word = N_word
        self.gpu = gpu
        self.SQL_TOK = SQL_TOK

        if trainable:
            print "Using trainable embedding"
            self.w2i, word_emb_val = word_emb
            # tranable when using pretrained model, init embedding weights using prev embedding
            self.embedding = nn.Embedding(len(self.w2i), N_word)
            self.embedding.weight = nn.Parameter(torch.from_numpy(word_emb_val.astype(np.float32)))
        else:
            # else use word2vec or glove
            self.word_emb = word_emb
            print "Using fixed embedding for words but trainable embedding for types"


    def gen_xc_type_batch(self, xc_type, is_col=False, is_list=False):
        B = len(xc_type)
        val_embs = []
        val_len = np.zeros(B, dtype=np.int64)
        for i, one_q in enumerate(xc_type):
            if is_list:
                q_val = map(lambda x:self.w2i.get(" ".join(sorted(x)), 0), one_q)
            else:
                q_val = map(lambda x:self.w2i.get(x, 0), one_q)
            if is_col:
                val_embs.append(q_val)  #<BEG> and <END>
                val_len[i] = len(q_val)
            else:
                val_embs.append([1] + q_val + [2])  #<BEG> and <END>
                val_len[i] = 1 + len(q_val) + 1
        max_len = max(val_len)
        val_tok_array = np.zeros((B, max_len), dtype=np.int64)
        for i in range(B):
            for t in range(len(val_embs[i])):
                val_tok_array[i,t] = val_embs[i][t]
        val_tok = torch.from_numpy(val_tok_array)
        if self.gpu:
            val_tok = val_tok.cuda()
        val_tok_var = Variable(val_tok)
        val_inp_var = self.embedding(val_tok_var)

        return val_inp_var, val_len


    def gen_x_q_batch(self, q):
        B = len(q)
        val_embs = []
        val_len = np.zeros(B, dtype=np.int64)
        for i, one_q in enumerate(q):
            q_val = []
            for ws in one_q:
                q_val.append(self.word_emb.get(ws, np.zeros(self.N_word, dtype=np.float32)))

            val_embs.append([np.zeros(self.N_word, dtype=np.float32)] + q_val + [np.zeros(self.N_word, dtype=np.float32)])  #<BEG> and <END>
            val_len[i] = 1 + len(q_val) + 1
        max_len = max(val_len)

        val_emb_array = np.zeros((B, max_len, self.N_word), dtype=np.float32)
        for i in range(B):
            for t in range(len(val_embs[i])):
                val_emb_array[i, t, :] = val_embs[i][t]
        val_inp = torch.from_numpy(val_emb_array)
        if self.gpu:
            val_inp = val_inp.cuda()
        val_inp_var = Variable(val_inp)

        return val_inp_var, val_len

    def gen_table_embedding(self,tables):
        B = len(tables)
        val_embs = []
        val_len = np.zeros(B, dtype=np.int64)
        # val_emb_array = np.zeros((B, self.N_word*3), dtype=np.float32)
        for i,table in enumerate(tables):
            tnames = []
            table_embs = []
            # print(table)
            for tname in table[0]:
                # print(tname)
                tname = tname.split()
                tname_emb = []
                for w in tname:
                    tname_emb.append(self.word_emb.get(w, np.zeros(self.N_word, dtype=np.float32)))
                if len(tname_emb) == 0:
                    raise Exception("col name should not be empty!")
                tnames.append(sum(tname_emb)/len(tname_emb))
            # print("tnames {}".format(len(tnames)))

            for idx,col in table[1]:
                # print(col)
                col = col.split()
                col_emb = []
                for w in col:
                    col_emb.append(self.word_emb.get(w, np.zeros(self.N_word, dtype=np.float32)))
                if len(col_emb) == 0:
                    raise Exception("col name should not be empty!")
                if idx == -1:
                    emb = np.concatenate((sum(col_emb)/len(col_emb), np.zeros(self.N_word, dtype=np.float32),np.zeros(self.N_word, dtype=np.float32)), axis=0)
                else:
                    emb = np.concatenate((sum(col_emb) / len(col_emb), tnames[idx],
                                    self.word_emb.get(table[2][idx], np.zeros(self.N_word, dtype=np.float32))), axis=0)
                table_embs.append(emb)
            val_embs.append(table_embs)
            val_len[i] = len(table_embs)

        max_len = max(val_len)

        val_emb_array = np.zeros((B, max_len, self.N_word*3), dtype=np.float32)
        for i in range(B):
            for t in range(len(val_embs[i])):
                val_emb_array[i, t, :] = val_embs[i][t]
        val_inp = torch.from_numpy(val_emb_array)
        if self.gpu:
            val_inp = val_inp.cuda()
        val_inp_var = Variable(val_inp)

        return val_inp_var, val_len

    def gen_hier_table_embedding(self,tables):
        B = len(tables)
        t_embs_batch = []
        col_embs_batch = []
        col_t_idxs_batch = []
        for i,table in enumerate(tables):
            n_tables = len(table[0])
            t_embs = [np.zeros(self.N_word, dtype=np.float32)] # list of table name emb; [n_tables +1, ]
            col_embs = [] # [total_n_cols, ]
            col_t_idxs = [] # [total_n_cols, ]
            for tname in table[0]:
                tname = tname.split()
                tname_emb = []
                for w in tname:
                    tname_emb.append(self.word_emb.get(w, np.zeros(self.N_word, dtype=np.float32)))
                if len(tname_emb) == 0:
                    raise Exception("col name should not be empty!")
                t_embs.append(sum(tname_emb)/len(tname_emb)) # take average emb

            for idx,col in table[1]: # idx: which table
                table_idx = idx + 1
                col = col.split()
                col_emb = []
                for w in col:
                    col_emb.append(self.word_emb.get(w, np.zeros(self.N_word, dtype=np.float32)))
                if len(col_emb) == 0:
                    col_emb.append(np.zeros(self.N_word, dtype=np.float32))
                    #raise Exception("col name should not be empty!")
                if idx == -1:
                    emb = np.concatenate((sum(col_emb)/len(col_emb), np.zeros(self.N_word, dtype=np.float32)), axis=0)
                else:
                    emb = np.concatenate((sum(col_emb) / len(col_emb), self.word_emb.get(table[2][idx], np.zeros(self.N_word, dtype=np.float32))), axis=0)
                col_embs.append(emb)
                col_t_idxs.append(table_idx)

            t_embs_batch.append(t_embs)
            col_embs_batch.append(col_embs)
            col_t_idxs_batch.append(col_t_idxs)

        # len related
        t_len_arr = np.asarray([len(t_embs) for t_embs in t_embs_batch]) # [B, ]
        col_len_arr = np.asarray([len(col_embs) for col_embs in col_embs_batch])
        max_t_len = max(t_len_arr)
        max_col_len = max(col_len_arr)

        col_t_map_matrix = np.zeros((B, max_col_len, max_t_len), dtype=np.float32)
        for i in range(B):
            col_t_idxs = col_t_idxs_batch[i]
            for c in range(len(col_t_idxs)): # total_n_cols
                col_t_map_matrix[i, c, col_t_idxs[c]] = 1.

        # emb related
        t_embs_batch_array = np.zeros((B, max_t_len, self.N_word), dtype=np.float32)
        for i in range(B):
            for t in range(len(t_embs_batch[i])): # for each table
                t_embs_batch_array[i, t, :] = t_embs_batch[i][t]
        col_embss_batch_array = np.zeros((B, max_col_len, self.N_word*2), dtype=np.float32)
        for i in range(B):
            for c in range(len(col_embs_batch[i])): # for each table
                col_embss_batch_array[i, c, :] = col_embs_batch[i][c]

        val_inp_t = torch.from_numpy(t_embs_batch_array)
        val_inp_col = torch.from_numpy(col_embss_batch_array)
        col_t_map_matrix = torch.from_numpy(col_t_map_matrix)
        if self.gpu:
            val_inp_col = val_inp_col.cuda()
            val_inp_t = val_inp_t.cuda()
            col_t_map_matrix = col_t_map_matrix.cuda()
        val_inp_col_var = Variable(val_inp_col)
        val_inp_t_var = Variable(val_inp_t)
        col_t_map_matrix = Variable(col_t_map_matrix, requires_grad=False)

        return val_inp_t_var, val_inp_col_var, t_len_arr, col_len_arr, col_t_map_matrix
        

    def gen_word_list_embedding(self,words,B):
        val_emb_array = np.zeros((B,len(words), self.N_word), dtype=np.float32)
        for i,word in enumerate(words):
            if len(word.split()) == 1:
                emb = self.word_emb.get(word, np.zeros(self.N_word, dtype=np.float32))
            else:
                word = word.split()
                emb = (self.word_emb.get(word[0], np.zeros(self.N_word, dtype=np.float32))
                                       +self.word_emb.get(word[1], np.zeros(self.N_word, dtype=np.float32)) )/2
            for b in range(B):
                val_emb_array[b,i,:] = emb
        val_inp = torch.from_numpy(val_emb_array)
        if self.gpu:
            val_inp = val_inp.cuda()
        val_inp_var = Variable(val_inp)
        return val_inp_var
    def gen_x_history_batch(self, history):
        B = len(history)
        val_embs = []
        val_len = np.zeros(B, dtype=np.int64)
        for i, one_history in enumerate(history):
            history_val = []
            for item in one_history:
                #col
                if isinstance(item, list) or isinstance(item, tuple):
                    emb_list = []
                    ws = item[0].split() + item[1].split()
                    ws_len = len(ws)
                    for w in ws:
                        emb_list.append(self.word_emb.get(w, np.zeros(self.N_word, dtype=np.float32)))
                    if ws_len == 0:
                        raise Exception("word list should not be empty!")
                    elif ws_len == 1:
                        history_val.append(emb_list[0])
                    else:
                        history_val.append(sum(emb_list) / float(ws_len))
                #ROOT
                elif isinstance(item,basestring):
                    if item == "ROOT":
                        item = "root"
                    elif item == "asc":
                        item = "ascending"
                    elif item == "desc":
                        item == "descending"
                    if item in (
                    "none", "select", "from", "where", "having", "limit", "intersect", "except", "union", 'not',
                    'between', '=', '>', '<', 'in', 'like', 'is', 'exists', 'root', 'ascending', 'descending'):
                        history_val.append(self.word_emb.get(item, np.zeros(self.N_word, dtype=np.float32)))
                    elif item == "orderBy":
                        history_val.append((self.word_emb.get("order", np.zeros(self.N_word, dtype=np.float32)) +
                                            self.word_emb.get("by", np.zeros(self.N_word, dtype=np.float32))) / 2)
                    elif item == "groupBy":
                        history_val.append((self.word_emb.get("group", np.zeros(self.N_word, dtype=np.float32)) +
                                            self.word_emb.get("by", np.zeros(self.N_word, dtype=np.float32))) / 2)
                    elif item in ('>=', '<=', '!='):
                        history_val.append((self.word_emb.get(item[0], np.zeros(self.N_word, dtype=np.float32)) +
                                            self.word_emb.get(item[1], np.zeros(self.N_word, dtype=np.float32))) / 2)
                elif isinstance(item,int):
                    history_val.append(self.word_emb.get(AGG_OPS[item], np.zeros(self.N_word, dtype=np.float32)))
                else:
                    print("Warning: unsupported data type in history! {}".format(item))

            val_embs.append(history_val)
            val_len[i] = len(history_val)
        max_len = max(val_len)

        val_emb_array = np.zeros((B, max_len, self.N_word), dtype=np.float32)
        for i in range(B):
            for t in range(len(val_embs[i])):
                val_emb_array[i, t, :] = val_embs[i][t]
        val_inp = torch.from_numpy(val_emb_array)
        if self.gpu:
            val_inp = val_inp.cuda()
        val_inp_var = Variable(val_inp)

        return val_inp_var, val_len

    def gen_col_batch(self, cols):
        ret = []
        col_len = np.zeros(len(cols), dtype=np.int64)

        names = []
        for b, one_cols in enumerate(cols):
            names = names + one_cols
            col_len[b] = len(one_cols)
        #TODO: what is the diff bw name_len and col_len?
        name_inp_var, name_len = self.str_list_to_batch(names)
        return name_inp_var, name_len, col_len


    def gen_agg_batch(self, q):
        B = len(q)
        ret = []
        agg_ops = ['none', 'maximum', 'minimum', 'count', 'total', 'average']
        for b in range(B):
            if self.trainable:
                ct_val = map(lambda x:self.w2i.get(x, 0), agg_ops)
            else:
                ct_val = map(lambda x:self.word_emb.get(x, np.zeros(self.N_word, dtype=np.float32)), agg_ops)
            ret.append(ct_val)

        agg_emb_array = np.zeros((B, 6, self.N_word), dtype=np.float32)
        for i in range(B):
            for t in range(len(ret[i])):
                agg_emb_array[i,t,:] = ret[i][t]
        agg_inp = torch.from_numpy(agg_emb_array)
        if self.gpu:
            agg_inp = agg_inp.cuda()
        agg_inp_var = Variable(agg_inp)

        return agg_inp_var


    def str_list_to_batch(self, str_list):
        """get a list var of wemb of words in each column name in current bactch"""
        B = len(str_list)

        val_embs = []
        val_len = np.zeros(B, dtype=np.int64)
        for i, one_str in enumerate(str_list):
            if self.trainable:
                val = [self.w2i.get(x, 0) for x in one_str]
            else:
                val = [self.word_emb.get(x, np.zeros(
                    self.N_word, dtype=np.float32)) for x in one_str]
            val_embs.append(val)
            val_len[i] = len(val)
        max_len = max(val_len)

        if self.trainable:
            val_tok_array = np.zeros((B, max_len), dtype=np.int64)
            for i in range(B):
                for t in range(len(val_embs[i])):
                    val_tok_array[i,t] = val_embs[i][t]
            val_tok = torch.from_numpy(val_tok_array)
            if self.gpu:
                val_tok = val_tok.cuda()
            val_tok_var = Variable(val_tok)
            val_inp_var = self.embedding(val_tok_var)
        else:
            val_emb_array = np.zeros(
                    (B, max_len, self.N_word), dtype=np.float32)
            for i in range(B):
                for t in range(len(val_embs[i])):
                    val_emb_array[i,t,:] = val_embs[i][t]
            val_inp = torch.from_numpy(val_emb_array)
            if self.gpu:
                val_inp = val_inp.cuda()
            val_inp_var = Variable(val_inp)

        return val_inp_var, val_len

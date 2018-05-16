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
            # print "Using trainable embedding for sql keywords. Other words are fixed"
            self.w2i, word_emb_val, _ = word_emb
            # tranable when using pretrained model, init embedding weights using prev embedding
            self.embedding = nn.Embedding(len(self.w2i), N_word)
            word_emb_val = torch.from_numpy(word_emb_val.astype(np.float32))
            if self.gpu:
                word_emb_val = word_emb_val.cuda()
            self.embedding.weight = nn.Parameter(word_emb_val)
        else:
            # else use word2vec or glove
            _, _, self.word_emb = word_emb
            # print "Using fixed embedding"

    """# not active #
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

        return val_inp_var, val_len"""

    ### active ###
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

    ### active ###
    def gen_table_embedding(self,tables):
        B = len(tables)
        val_embs = []
        val_len = np.zeros(B, dtype=np.int64)
        # val_emb_array = np.zeros((B, self.N_word*3), dtype=np.float32)
        for i,table in enumerate(tables):
            tnames = []
            table_embs = []
            for tname in table[0]:
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

    ### active ###
    def gen_word_list_embedding(self,words,B):
        if self.trainable:
            val_emb_array = np.zeros((B,len(words)), dtype=np.int64)
            for i,word in enumerate(words):
                emb = self.w2i.get(word, 0) # actually word id
                for b in range(B):
                    val_emb_array[b,i] = emb
            val_inp = torch.from_numpy(val_emb_array)
            if self.gpu:
                val_inp = val_inp.cuda()
            val_inp_var = Variable(val_inp)
            val_inp_var = self.embedding(val_inp_var)
        else:
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

    ### active ###
    def gen_x_history_batch(self, history):
        B = len(history)
        val_embs = []
        val_len = np.zeros(B, dtype=np.int64)
        for i, one_history in enumerate(history):
            history_val = []
            for item in one_history:
                #col
                if isinstance(item, list):
                    if self.trainable:
                        history_val.append(self.w2i.get("column", 0))
                    else:
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
                    if self.trainable:
                        history_val.append(self.w2i.get(item.lower(), 0))
                    else:
                        if item == "ROOT":
                            item = "root"
                        elif item == "asc":
                            item = "ascending"
                        elif item == "desc":
                            item = "descending"
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
                    if self.trainable:
                        history_val.append(self.w2i.get(AGG_OPS[item], 0))
                    else:
                        history_val.append(self.word_emb.get(AGG_OPS[item], np.zeros(self.N_word, dtype=np.float32)))
                else:
                    print("Warning: unsupported data type in history! {}".format(item))

            val_embs.append(history_val)
            val_len[i] = len(history_val)
        max_len = max(val_len)

        if self.trainable:
            val_emb_array = np.zeros((B, max_len), dtype=np.int64)
            for i in range(B):
                for t in range(len(val_embs[i])):
                    val_emb_array[i, t] = val_embs[i][t]
            val_inp = torch.from_numpy(val_emb_array)
            if self.gpu:
                val_inp = val_inp.cuda()
            val_inp_var = Variable(val_inp)
            val_inp_var = self.embedding(val_inp_var)
        else:
            val_emb_array = np.zeros((B, max_len, self.N_word), dtype=np.float32)
            for i in range(B):
                for t in range(len(val_embs[i])):
                    val_emb_array[i, t, :] = val_embs[i][t]
            val_inp = torch.from_numpy(val_emb_array)
            if self.gpu:
                val_inp = val_inp.cuda()
            val_inp_var = Variable(val_inp)

        return val_inp_var, val_len


"""
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
        # get a list var of wemb of words in each column name in current bactch
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
"""

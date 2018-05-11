import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from net_utils import run_lstm, col_name_encode


class MultiSqlPredictor(nn.Module):
    '''Predict if the next token is (multi SQL key words):
        NONE, EXCEPT, INTERSECT, or UNION.'''
    def __init__(self, N_word, N_h, N_depth, gpu):
        super(MultiSqlPredictor, self).__init__()
        self.N_h = N_h
        self.gpu = gpu

        self.q_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)

        self.hs_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)

        self.mkw_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)

        self.q_att = nn.Linear(N_h, N_h)
        self.hs_att = nn.Linear(N_h, N_h)
        self.multi_out_q = nn.Linear(N_h, N_h)
        self.multi_out_hs = nn.Linear(N_h, N_h)
        self.multi_out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, 1))

        self.softmax = nn.Softmax() #dim=1

    def forward(self, q_emb_var, q_len, hs_emb_var, hs_len, mkw_emb_var, mkw_len):
        max_q_len = max(q_len)
        max_hs_len = max(hs_len)
        B = len(q_len)

        # q_enc: (B, max_q_len, hid_dim)
        # hs_enc: (B, max_hs_len, hid_dim)
        # mkw: (B, 4, hid_dim)
        q_enc, _ = run_lstm(self.q_lstm, q_emb_var, q_len)
        hs_enc, _ = run_lstm(self.hs_lstm, hs_emb_var, hs_len)
        mkw_enc, _ = run_lstm(self.mkw_lstm, mkw_emb_var, mkw_len)

        # Compute attention values between multi SQL key words and question tokens.
        # qmkw_att(q_enc).transpose(1, 2): (B, hid_dim, max_q_len)
        # att_val_qmkw: (B, 4, max_q_len)
        att_val_qmkw = torch.mm(mkw_enc, self.q_att(q_enc).transpose(1, 2))
        # assign appended positions values -100
        for idx, num in enumerate(q_len):
            if num < max_q_len:
                att_val_qmkw[idx, :, num:] = -100
        # att_prob_qmkw: (B, 4, max_q_len)
        att_prob_qmkw = self.softmax(att_val_qmkw.view((-1, max_q_len))).view(B, -1, max_q_len)
        # q_enc.unsqueeze(1): (B, 1, max_q_len, hid_dim)
        # att_prob_qmkw.unsqueeze(3): (B, 4, max_q_len, 1)
        # q_weighted: (B, 4, hid_dim)
        q_weighted = (q_enc.unsqueeze(1) * att_prob_qmkw.unsqueeze(3)).sum(2)

        # Same as the above, compute SQL history embedding weighted by key words attentions
        att_val_hsmkw = torch.mm(mkw_enc, self.hs_att(hs_enc).transpose(1, 2))
        for idx, num in enumerate(hs_len):
            if num < max_hs_len:
                att_val_hsmkw[idx, :, num:] = -100
        att_prob_hsmkw = self.softmax(att_val_hsmkw.view((-1, max_hs_len))).view(B, -1, max_hs_len)
        hs_weighted = (hs_enc.unsqueeze(1) * att_prob_hsmkw.unsqueeze(3)).sum(2)

        # Compute prediction scores
        # self.multi_out.squeeze(): (B, 4, 1) -> (B, 4)
        mulit_score = self.multi_out(self.multi_out_q(q_weighted) + self.multi_out_hs(hs_weighted)).squeeze()



class KeyWordPredictor(nn.Module):
    '''Predict if the next token is (SQL key words):
        WHERE, GROUP BY, ORDER BY. excluding SELECT (it is a must)'''
    def __init__(self, N_word, N_h, N_depth, gpu):
        super(KeyWordPredictor, self).__init__()
        self.N_h = N_h
        self.gpu = gpu

        self.q_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)

        self.hs_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)

        self.kw_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)

        self.q_num_att = nn.Linear(N_h, N_h)
        self.hs_num_att = nn.Linear(N_h, N_h)
        self.kw_num_out_q = nn.Linear(N_h, N_h)
        self.kw_num_out_hs = nn.Linear(N_h, N_h)
        self.kw_num_out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, 4)) # num of key words: 0-3

        self.q_att = nn.Linear(N_h, N_h)
        self.hs_att = nn.Linear(N_h, N_h)
        self.kw_out_q = nn.Linear(N_h, N_h)
        self.kw_out_hs = nn.Linear(N_h, N_h)
        self.kw_out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, 1))

        self.softmax = nn.Softmax() #dim=1

    def forward(self, q_emb_var, q_len, hs_emb_var, hs_len, kw_emb_var, kw_len):
        max_q_len = max(q_len)
        max_hs_len = max(hs_len)
        B = len(q_len)

        q_enc, _ = run_lstm(self.q_lstm, q_emb_var, q_len)
        hs_enc, _ = run_lstm(self.hs_lstm, hs_emb_var, hs_len)
        kw_enc, _ = run_lstm(self.kw_lstm, kw_emb_var, kw_len)

        # Predict key words number: 0-3
        att_val_qkw_num = torch.mm(kw_enc, self.q_num_att(q_enc).transpose(1, 2))
        for idx, num in enumerate(q_len):
            if num < max_q_len:
                att_val_qkw_num[idx, :, num:] = -100
        att_prob_qkw_num = self.softmax(att_val_qkw_num.view((-1, max_q_len))).view(B, -1, max_q_len)
        # q_weighted: (B, hid_dim)
        q_weighted_num = (q_enc.unsqueeze(1) * att_prob_qkw_num.unsqueeze(3)).sum(2).sum(1)

        # Same as the above, compute SQL history embedding weighted by key words attentions
        att_val_hskw_num = torch.mm(kw_enc, self.hs_num_att(hs_enc).transpose(1, 2))
        for idx, num in enumerate(hs_len):
            if num < max_hs_len:
                att_val_hskw_num[idx, :, num:] = -100
        att_prob_hskw_num = self.softmax(att_val_hskw_num.view((-1, max_hs_len))).view(B, -1, max_hs_len)
        hs_weighted_num = (hs_enc.unsqueeze(1) * att_prob_hskw_num.unsqueeze(3)).sum(2).sum(1)
        # Compute prediction scores
        # self.kw_num_out: (B, 4)
        kw_num_score = self.kw_num_out(self.kw_num_out_q(q_weighted_num) + self.kw_num_out_hs(hs_weighted_num))

        # Predict key words: WHERE, GROUP BY, ORDER BY.
        att_val_qkw = torch.mm(kw_enc, self.q_att(q_enc).transpose(1, 2))
        for idx, num in enumerate(q_len):
            if num < max_q_len:
                att_val_qkw[idx, :, num:] = -100
        att_prob_qkw = self.softmax(att_val_qkw.view((-1, max_q_len))).view(B, -1, max_q_len)
        # q_weighted: (B, 3, hid_dim)
        q_weighted = (q_enc.unsqueeze(1) * att_prob_qkw.unsqueeze(3)).sum(2)

        # Same as the above, compute SQL history embedding weighted by key words attentions
        att_val_hskw = torch.mm(kw_enc, self.hs_att(hs_enc).transpose(1, 2))
        for idx, num in enumerate(hs_len):
            if num < max_hs_len:
                att_val_hskw[idx, :, num:] = -100
        att_prob_hskw = self.softmax(att_val_hskw.view((-1, max_hs_len))).view(B, -1, max_hs_len)
        hs_weighted = (hs_enc.unsqueeze(1) * att_prob_hskw.unsqueeze(3)).sum(2)
        # Compute prediction scores
        # self.kw_out.squeeze(): (B, 3)
        kw_score = self.kw_out(self.kw_out_q(q_weighted) + self.kw_out_hs(hs_weighted)).squeeze()



class ColPredictor(nn.Module):
    def __init__(self, N_word, N_h, N_depth, gpu):
        super(ColPredictor, self).__init__()
        self.N_h = N_h
        self.gpu = gpu

        self.q_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)

        self.hs_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)

        self.col_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)

        self.q_num_att = nn.Linear(N_h, N_h)
        self.hs_num_att = nn.Linear(N_h, N_h)
        self.col_num_out_q = nn.Linear(N_h, N_h)
        self.col_num_out_hs = nn.Linear(N_h, N_h)
        self.col_num_out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, 3)) # num of cols: 1-3

        self.q_att = nn.Linear(N_h, N_h)
        self.hs_att = nn.Linear(N_h, N_h)
        self.col_out_q = nn.Linear(N_h, N_h)
        self.col_out_hs = nn.Linear(N_h, N_h)
        self.col_out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, 1))

        self.softmax = nn.Softmax() #dim=1

    def forward(self, q_emb_var, q_len, hs_emb_var, hs_len, col_emb_var, col_len):
        max_q_len = max(q_len)
        max_hs_len = max(hs_len)
        max_col_len = max(col_len)
        B = len(q_len)

        q_enc, _ = run_lstm(self.q_lstm, q_emb_var, q_len)
        hs_enc, _ = run_lstm(self.hs_lstm, hs_emb_var, hs_len)
        col_enc, _ = run_lstm(self.col_lstm, col_emb_var, col_len)

        # Predict column number: 1-3
        # att_val_qc_num: (B, max_col_len, max_q_len)
        att_val_qc_num = torch.mm(col_enc, self.q_num_att(q_enc).transpose(1, 2))
        for idx, num in enumerate(col_len):
            if num < max_col_len:
                att_val_qc_num[idx, num:, :] = -100
        for idx, num in enumerate(q_len):
            if num < max_q_len:
                att_val_qc_num[idx, :, num:] = -100
        att_prob_qc_num = self.softmax(att_val_qc_num.view((-1, max_q_len))).view(B, -1, max_q_len)
        # q_weighted_num: (B, hid_dim)
        q_weighted_num = (q_enc.unsqueeze(1) * att_prob_qc_num.unsqueeze(3)).sum(2).sum(1)

        # Same as the above, compute SQL history embedding weighted by column attentions
        # att_val_hc_num: (B, max_col_len, max_hs_len)
        att_val_hc_num = torch.mm(col_enc, self.hs_num_att(hs_enc).transpose(1, 2))
        for idx, num in enumerate(hs_len):
            if num < max_hs_len:
                att_val_hc_num[idx, :, num:] = -100
        att_prob_hc_num = self.softmax(att_val_hc_num.view((-1, max_hs_len))).view(B, -1, max_hs_len)
        hs_weighted_num = (hs_enc.unsqueeze(1) * att_prob_hc_num.unsqueeze(3)).sum(2).sum(1)
        # self.col_num_out: (B, 3)
        col_num_score = self.col_num_out(self.col_num_out_q(q_weighted_num) + self.col_num_out_hs(hs_weighted_num))

        # Predict columns.
        att_val_qc = torch.mm(col_enc, self.q_att(q_enc).transpose(1, 2))
        for idx, num in enumerate(q_len):
            if num < max_q_len:
                att_val_qc[idx, :, num:] = -100
        att_prob_qc = self.softmax(att_val_qc.view((-1, max_q_len))).view(B, -1, max_q_len)
        # q_weighted: (B, max_col_len, hid_dim)
        q_weighted = (q_enc.unsqueeze(1) * att_prob_qc.unsqueeze(3)).sum(2)

        # Same as the above, compute SQL history embedding weighted by column attentions
        att_val_hc = torch.mm(col_enc, self.hs_att(hs_enc).transpose(1, 2))
        for idx, num in enumerate(hs_len):
            if num < max_hs_len:
                att_val_hc[idx, :, num:] = -100
        att_prob_hc = self.softmax(att_val_hc.view((-1, max_hs_len))).view(B, -1, max_hs_len)
        hs_weighted = (hs_enc.unsqueeze(1) * att_prob_hc.unsqueeze(3)).sum(2)
        # Compute prediction scores
        # self.col_out.squeeze(): (B, max_col_len)
        col_score = self.col_out(self.col_out_q(q_weighted) + self.col_out_hs(hs_weighted)).squeeze()


class OpPredictor(nn.Module):
    def __init__(self, N_word, N_h, N_depth, gpu):
        super(OpPredictor, self).__init__()
        self.N_h = N_h
        self.gpu = gpu

        self.q_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)

        self.hs_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)

        self.col_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)

        self.q_att = nn.Linear(N_h, N_h)
        self.hs_att = nn.Linear(N_h, N_h)
        self.op_out_q = nn.Linear(N_h, N_h)
        self.op_out_hs = nn.Linear(N_h, N_h)
        self.op_out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, 10)) #for 10 operators

        self.softmax = nn.Softmax() #dim=1

    def forward(self, q_emb_var, q_len, hs_emb_var, hs_len, col_emb_var, col_len, gt_col):
        max_q_len = max(q_len)
        max_hs_len = max(hs_len)
        max_col_len = max(col_len)
        B = len(q_len)

        q_enc, _ = run_lstm(self.q_lstm, q_emb_var, q_len)
        hs_enc, _ = run_lstm(self.hs_lstm, hs_emb_var, hs_len)
        col_enc, _ = run_lstm(self.col_lstm, col_emb_var, col_len)
        # get target/predicted column's embedding
        # col_emb: (B, hid_dim)
        col_emb = []
        for b in range(B):
            col_emb.append(col_enc[b, gt_col[b]])
        col_emb = torch.stack(col_emb)
        # Compute attention values between selected column and question tokens.
        # q_enc.transpose(1, 2): (B, hid_dim, max_q_len)
        # col_emb.squeeze(1): (B, 1, hid_dim)
        # att_val_qc: (B, max_q_len)
        att_val_qc = torch.mm(col_emb.squeeze(1), self.q_att(q_enc).transpose(1, 2)).unsqueeze()
        # assign appended positions values -100
        for idx, num in enumerate(q_len):
            if num < max_q_len:
                att_val_qc[idx, num:] = -100
        # att_prob_qc: (B, max_q_len)
        att_prob_qc = self.softmax(att_val_qc)
        # q_enc: (B, max_q_len, hid_dim)
        # att_prob_qc.unsqueeze(2): (B, max_q_len, 1)
        # q_weighted: (B, hid_dim)
        q_weighted = (q_enc * att_prob_qc.unsqueeze(2)).sum(1)

        # Same as the above, compute SQL history embedding weighted by column attentions
        att_val_hc = torch.mm(col_emb.squeeze(1), self.hs_att(hs_enc).transpose(1, 2)).unsqueeze()
        for idx, num in enumerate(hs_len):
            if num < max_hs_len:
                att_val_hc[idx, num:] = -100
        att_prob_hc = self.softmax(att_val_hc)
        hs_weighted = (hs_enc * att_prob_hc.unsqueeze(2)).sum(1)

        # Compute prediction scores
        # op_score: (B, 10)
        op_score = self.op_out(self.op_out_q(q_weighted) + self.op_out_hs(hs_weighted))


class AggPredictor(nn.Module):
    def __init__(self, N_word, N_h, N_depth, gpu):
        super(AggPredictor, self).__init__()
        self.N_h = N_h
        self.gpu = gpu

        self.q_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)

        self.hs_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)

        self.col_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)

        self.q_num_att = nn.Linear(N_h, N_h)
        self.hs_num_att = nn.Linear(N_h, N_h)
        self.agg_num_out_q = nn.Linear(N_h, N_h)
        self.agg_num_out_hs = nn.Linear(N_h, N_h)
        self.agg_num_out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, 4)) #for 0-3 agg num

        self.q_att = nn.Linear(N_h, N_h)
        self.hs_att = nn.Linear(N_h, N_h)
        self.agg_out_q = nn.Linear(N_h, N_h)
        self.agg_out_hs = nn.Linear(N_h, N_h)
        self.agg_out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, 5)) #for 1-5 aggregators

        self.softmax = nn.Softmax() #dim=1

    def forward(self, q_emb_var, q_len, hs_emb_var, hs_len, col_emb_var, col_len, gt_col):
        max_q_len = max(q_len)
        max_hs_len = max(hs_len)
        max_col_len = max(col_len)
        B = len(q_len)

        q_enc, _ = run_lstm(self.q_lstm, q_emb_var, q_len)
        hs_enc, _ = run_lstm(self.hs_lstm, hs_emb_var, hs_len)
        col_enc, _ = run_lstm(self.col_lstm, col_emb_var, col_len)

        col_emb = []
        for b in range(B):
            col_emb.append(col_enc[b, gt_col[b]])
        col_emb = torch.stack(col_emb)

        # Predict agg number
        att_val_qc_num = torch.mm(col_emb.squeeze(1), self.q_num_att(q_enc).transpose(1, 2)).unsqueeze()
        for idx, num in enumerate(q_len):
            if num < max_q_len:
                att_val_qc_num[idx, num:] = -100
        att_prob_qc_num = self.softmax(att_val_qc_num)
        q_weighted_num = (q_enc * att_prob_qc_num.unsqueeze(2)).sum(1)

        # Same as the above, compute SQL history embedding weighted by column attentions
        att_val_hc_num = torch.mm(col_emb.squeeze(1), self.hs_num_att(hs_enc).transpose(1, 2)).unsqueeze()
        for idx, num in enumerate(hs_len):
            if num < max_hs_len:
                att_val_hc_num[idx, num:] = -100
        att_prob_hc_num = self.softmax(att_val_hc_num)
        hs_weighted_num = (hs_enc * att_prob_hc_num.unsqueeze(2)).sum(1)
        # agg_num_score: (B, 4)
        agg_num_score = self.agg_num_out(self.agg_num_out_q(q_weighted_num) + self.agg_num_out_hs(hs_weighted_num))

        # Predict aggregators
        att_val_qc = torch.mm(col_emb.squeeze(1), self.q_att(q_enc).transpose(1, 2)).unsqueeze()
        for idx, num in enumerate(q_len):
            if num < max_q_len:
                att_val_qc[idx, num:] = -100
        att_prob_qc = self.softmax(att_val_qc)
        q_weighted = (q_enc * att_prob_qc.unsqueeze(2)).sum(1)

        # Same as the above, compute SQL history embedding weighted by column attentions
        att_val_hc = torch.mm(col_emb.squeeze(1), self.hs_att(hs_enc).transpose(1, 2)).unsqueeze()
        for idx, num in enumerate(hs_len):
            if num < max_hs_len:
                att_val_hc[idx, num:] = -100
        att_prob_hc = self.softmax(att_val_hc)
        hs_weighted = (hs_enc * att_prob_hc.unsqueeze(2)).sum(1)
        # agg_score: (B, 5)
        agg_score = self.agg_out(self.agg_out_q(q_weighted) + self.agg_out_hs(hs_weighted))


class RootTeminalPredictor(nn.Module):
    def __init__(self, N_word, N_h, N_depth, gpu):
        super(RootTeminalPredictor, self).__init__()
        self.N_h = N_h
        self.gpu = gpu

        self.q_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)

        self.hs_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)

        self.col_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)

        self.q_att = nn.Linear(N_h, N_h)
        self.hs_att = nn.Linear(N_h, N_h)
        self.rt_out_q = nn.Linear(N_h, N_h)
        self.rt_out_hs = nn.Linear(N_h, N_h)
        self.rt_out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, 2)) #for 10 operators

        self.softmax = nn.Softmax() #dim=1

    def forward(self, q_emb_var, q_len, hs_emb_var, hs_len, col_emb_var, col_len, gt_col):
        max_q_len = max(q_len)
        max_hs_len = max(hs_len)
        max_col_len = max(col_len)
        B = len(q_len)

        q_enc, _ = run_lstm(self.q_lstm, q_emb_var, q_len)
        hs_enc, _ = run_lstm(self.hs_lstm, hs_emb_var, hs_len)
        col_enc, _ = run_lstm(self.col_lstm, col_emb_var, col_len)
        # get target/predicted column's embedding
        # col_emb: (B, hid_dim)
        col_emb = []
        for b in range(B):
            col_emb.append(col_enc[b, gt_col[b]])
        col_emb = torch.stack(col_emb)
        att_val_qc = torch.mm(col_emb.squeeze(1), self.q_att(q_enc).transpose(1, 2)).unsqueeze()
        for idx, num in enumerate(q_len):
            if num < max_q_len:
                att_val_qc[idx, num:] = -100
        att_prob_qc = self.softmax(att_val_qc)
        q_weighted = (q_enc * att_prob_qc.unsqueeze(2)).sum(1)

        # Same as the above, compute SQL history embedding weighted by column attentions
        att_val_hc = torch.mm(col_emb.squeeze(1), self.hs_att(hs_enc).transpose(1, 2)).unsqueeze()
        for idx, num in enumerate(hs_len):
            if num < max_hs_len:
                att_val_hc[idx, num:] = -100
        att_prob_hc = self.softmax(att_val_hc)
        hs_weighted = (hs_enc * att_prob_hc.unsqueeze(2)).sum(1)
        # rt_score: (B, 2)
        rt_score = self.rt_out(self.rt_out_q(q_weighted) + self.rt_out_hs(hs_weighted))



class DesAscLimitPredictor(nn.Module):
    def __init__(self, N_word, N_h, N_depth, gpu):
        super(DesAscLimitPredictor, self).__init__()
        self.N_h = N_h
        self.gpu = gpu

        self.q_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)

        self.hs_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)

        self.col_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)

        self.q_att = nn.Linear(N_h, N_h)
        self.hs_att = nn.Linear(N_h, N_h)
        self.rt_out_q = nn.Linear(N_h, N_h)
        self.rt_out_hs = nn.Linear(N_h, N_h)
        self.rt_out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, 2)) #for 10 operators

        self.softmax = nn.Softmax() #dim=1

    def forward(self, q_emb_var, q_len, hs_emb_var, hs_len, col_emb_var, col_len, gt_col):
        max_q_len = max(q_len)
        max_hs_len = max(hs_len)
        max_col_len = max(col_len)
        B = len(q_len)

        q_enc, _ = run_lstm(self.q_lstm, q_emb_var, q_len)
        hs_enc, _ = run_lstm(self.hs_lstm, hs_emb_var, hs_len)
        col_enc, _ = run_lstm(self.col_lstm, col_emb_var, col_len)
        # get target/predicted column's embedding
        # col_emb: (B, hid_dim)
        col_emb = []
        for b in range(B):
            col_emb.append(col_enc[b, gt_col[b]])
        col_emb = torch.stack(col_emb)
        att_val_qc = torch.mm(col_emb.squeeze(1), self.q_att(q_enc).transpose(1, 2)).unsqueeze()
        for idx, num in enumerate(q_len):
            if num < max_q_len:
                att_val_qc[idx, num:] = -100
        att_prob_qc = self.softmax(att_val_qc)
        q_weighted = (q_enc * att_prob_qc.unsqueeze(2)).sum(1)

        # Same as the above, compute SQL history embedding weighted by column attentions
        att_val_hc = torch.mm(col_emb.squeeze(1), self.hs_att(hs_enc).transpose(1, 2)).unsqueeze()
        for idx, num in enumerate(hs_len):
            if num < max_hs_len:
                att_val_hc[idx, num:] = -100
        att_prob_hc = self.softmax(att_val_hc)
        hs_weighted = (hs_enc * att_prob_hc.unsqueeze(2)).sum(1)
        # rt_score: (B, 2)
        rt_score = self.rt_out(self.rt_out_q(q_weighted) + self.rt_out_hs(hs_weighted))

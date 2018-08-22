import json
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from net_utils import run_lstm, col_name_encode


class ValPredictor(nn.Module):
    def __init__(self, N_word, N_h, N_depth, gpu, hier_col):
        super(ValPredictor, self).__init__()
        self.N_h = N_h
        self.gpu = gpu

        self.q_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)

        self.hs_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)

        self.col_lstm = nn.LSTM(input_size=N_word*3, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)
        if hier_col:
            print "Using hier_col for val module"
            self.t_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)
            self.t_col_concat_layer = nn.Sequential(nn.Linear(N_word*3, N_word*3), nn.Tanh())

        self.q_att = nn.Linear(N_h, N_h)
        self.hs_att = nn.Linear(N_h, N_h)
        self.rt_out_q = nn.Linear(N_h, N_h)
        self.rt_out_hs = nn.Linear(N_h, N_h)
        self.rt_out_c = nn.Linear(N_h, N_h)
        self.q_att_final = nn.Linear(N_h, N_h)
        self.rt_out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, 2)) #for 2 operators

        self.softmax = nn.Softmax() #dim=1
        self.CE = nn.CrossEntropyLoss()
        self.log_softmax = nn.LogSoftmax()
        self.mlsml = nn.MultiLabelSoftMarginLoss()
        self.bce_logit = nn.BCEWithLogitsLoss()
        self.sigm = nn.Sigmoid()
        if gpu:
            self.cuda()

    def forward(self, q_emb_var, q_len, hs_emb_var, hs_len, col_emb_var, col_len, gt_col, t_emb_var=None, t_len=None, col_t_map_matrix=None):
        max_q_len = max(q_len)
        max_hs_len = max(hs_len)
        max_col_len = max(col_len)
        B = len(q_len)

        q_enc, _ = run_lstm(self.q_lstm, q_emb_var, q_len)
        hs_enc, _ = run_lstm(self.hs_lstm, hs_emb_var, hs_len)

        if t_emb_var is None:
            col_enc, _ = run_lstm(self.col_lstm, col_emb_var, col_len)
        else: # hier_col
            max_t_len = max(t_len)
            t_enc, _ = run_lstm(self.t_lstm, t_emb_var, t_len) # (B, max_t_len, N_word)
            t_enc_for_col = torch.bmm(col_t_map_matrix, t_enc) # (B, max_col_len, N_word)
            t_col_concat = self.t_col_concat_layer(
                  torch.cat((col_emb_var, t_enc_for_col), dim=2)) # (B, max_c_len, N_word*3)
            col_enc, _ = run_lstm(self.col_lstm, t_col_concat, col_len)


        # get target/predicted column's embedding
        # col_emb: (B, hid_dim)
        col_emb = []
        for b in range(B):
            col_emb.append(col_enc[b, gt_col[b]])
        col_emb = torch.stack(col_emb)
        att_val_qc = torch.bmm(col_emb.unsqueeze(1), self.q_att(q_enc).transpose(1, 2)).squeeze() # (B, max_q_len)
        for idx, num in enumerate(q_len):
            if num < max_q_len:
                att_val_qc[idx, num:] = -100
        att_prob_qc = self.softmax(att_val_qc) # (B, max_q_len)
        q_weighted = (q_enc * att_prob_qc.unsqueeze(2)).sum(1) # (B, hid_dim)

        # Same as the above, compute SQL history embedding weighted by column attentions
        att_val_hc = torch.bmm(col_emb.unsqueeze(1), self.hs_att(hs_enc).transpose(1, 2)).squeeze()
        for idx, num in enumerate(hs_len):
            if num < max_hs_len:
                att_val_hc[idx, num:] = -100
        att_prob_hc = self.softmax(att_val_hc)
        hs_weighted = (hs_enc * att_prob_hc.unsqueeze(2)).sum(1)

        sum_hid = self.rt_out_q(q_weighted) + self.rt_out_hs(hs_weighted) + self.rt_out_c(col_emb) # (B, hid_dim)
        final_att_val_q = torch.bmm(sum_hid.unsqueeze(1), self.q_att_final(q_enc).transpose(1, 2)).squeeze() # (B, max_q_len)
        #TODO: this time, q_enc should take care of ngram phrases

        return final_att_val_q


    def loss(self, score, truth):
        loss = 0
        data = torch.from_numpy(np.array(truth))
        truth_var = Variable(data.cuda())
        loss = self.CE(score, truth_var)

        return loss


    def check_acc(self, score, truth):
        err = 0
        B = len(score)
        pred = []
        for b in range(B):
            pred.append(np.argmax(score[b].data.cpu().numpy()))
        for b, (p, t) in enumerate(zip(pred, truth)):
            if p != t:
                err += 1

        return err

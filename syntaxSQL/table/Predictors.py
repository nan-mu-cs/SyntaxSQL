from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


class AggPredictor(nn.Module):
    def __init__(self, input_size, score_size, dropout):
        super(AggPredictor, self).__init__()

        self.agg_num_out = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_size, score_size),
            nn.Tanh(),
            nn.Linear(score_size, 4)) #for 0-3 agg num

        self.agg_out = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_size, score_size),
            nn.Tanh(),
            nn.Linear(score_size, 5)) #for 1-5 aggregators

        self.softmax = nn.Softmax() #dim=1
        self.CE = nn.CrossEntropyLoss()
        self.log_softmax = nn.LogSoftmax()
        self.mlsml = nn.MultiLabelSoftMarginLoss()
        self.bce_logit = nn.BCEWithLogitsLoss()
        self.sigm = nn.Sigmoid()

    def forward(self, q_hs):
        # agg_num_score: (B, max_sql_hs_len, 4)
        agg_num_score = self.agg_num_out(q_hs)
        # agg_score: (B, max_sql_hs_len, 5)
        agg_score = self.agg_out(q_hs)
        
        score = (agg_num_score, agg_score)

        return score

    def loss(self, score, truth):
        loss = 0
        B = len(truth)
        agg_num_score, agg_score = score
        #loss for the column number
        truth_num = [len(t) for t in truth] # double check truth format and for test cases
        data = torch.from_numpy(np.array(truth_num))
        truth_num_var = Variable(data.cuda())
        loss += self.CE(agg_num_score, truth_num_var)
        #loss for the key words
        T = len(agg_score[0])
        truth_prob = np.zeros((B, T), dtype=np.float32)
        for b in range(B):
            truth_prob[b][truth[b]] = 1
        data = torch.from_numpy(truth_prob)
        truth_var = Variable(data.cuda())
        #loss += self.mlsml(agg_score, truth_var)
        #loss += self.bce_logit(agg_score, truth_var) # double check no sigmoid
        pred_prob = self.sigm(agg_score)
        bce_loss = -torch.mean( 3*(truth_var * \
                torch.log(pred_prob+1e-10)) + \
                (1-truth_var) * torch.log(1-pred_prob+1e-10) )
        loss += bce_loss

        return loss

    def check_acc(self, score, truth):
        num_err, err, tot_err = 0, 0, 0
        B = len(truth)
        pred = []
        agg_num_score, agg_score = [x.data.cpu().numpy() for x in score]
        for b in range(B):
            cur_pred = {}
            agg_num = np.argmax(agg_num_score[b]) #double check
            cur_pred['agg_num'] = agg_num
            cur_pred['agg'] = np.argsort(-agg_score[b])[:agg_num]
            pred.append(cur_pred)

        for b, (p, t) in enumerate(zip(pred, truth)):
            agg_num, agg = p['agg_num'], p['agg']
            flag = True
            if agg_num != len(t): # double check truth format and for test cases
                num_err += 1
                flag = False
            if flag and set(agg) != set(t):
                err += 1
                flag = False
            if not flag:
                tot_err += 1

        return np.array((num_err, err, tot_err))

    
class AndOrPredictor(nn.Module):
    def __init__(self, input_size, score_size, dropout):
        super(AndOrPredictor, self).__init__()
        
        self.ao_out = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_size, score_size),
            nn.Tanh(),
            nn.Linear(score_size, 2)) # for and/or

        self.softmax = nn.Softmax() #dim=1
        self.CE = nn.CrossEntropyLoss()
        self.log_softmax = nn.LogSoftmax()
        self.mlsml = nn.MultiLabelSoftMarginLoss()
        self.bce_logit = nn.BCEWithLogitsLoss()
        self.sigm = nn.Sigmoid()

    def forward(self, q_hs):
        # ao_score: (B, max_sql_hs_len, 2)
        ao_score = self.ao_out(q_hs)

        return ao_score

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

    
class DesAscLimitPredictor(nn.Module):
    def __init__(self, input_size, score_size, dropout):
        super(DesAscLimitPredictor, self).__init__()
        
        self.dat_out = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_size, score_size),
            nn.Tanh(),
            nn.Linear(score_size, 4)) # for 4 desc/asc limit/none combinations
        
        self.softmax = nn.Softmax() #dim=1
        self.CE = nn.CrossEntropyLoss()
        self.log_softmax = nn.LogSoftmax()
        self.mlsml = nn.MultiLabelSoftMarginLoss()
        self.bce_logit = nn.BCEWithLogitsLoss()
        self.sigm = nn.Sigmoid()

    def forward(self, q_hs):
        # dat_score: (B, max_sql_hs_len, 4)
        dat_score = self.dat_out(q_hs)

        return dat_score

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

    
class HavingPredictor(nn.Module):
    def __init__(self, input_size, score_size, dropout):
        super(HavingPredictor, self).__init__()
        
        self.hv_out = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_size, score_size),
            nn.Tanh(),
            nn.Linear(score_size, 2)) #for having/none
        
        self.softmax = nn.Softmax() #dim=1
        self.CE = nn.CrossEntropyLoss()
        self.log_softmax = nn.LogSoftmax()
        self.mlsml = nn.MultiLabelSoftMarginLoss()
        self.bce_logit = nn.BCEWithLogitsLoss()
        self.sigm = nn.Sigmoid()
        
    def forward(self, q_hs):
        # hv_score: (B, max_sql_hs_len, 2)
        hv_score = self.hv_out(q_hs)

        return hv_score

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

    
class KeyWordPredictor(nn.Module):
    '''Predict if the next token is (SQL key words):
        WHERE, GROUP BY, ORDER BY. excluding SELECT (it is a must)'''
    def __init__(self, input_size, score_size, dropout):
        super(KeyWordPredictor, self).__init__()
        
        self.kw_num_out = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_size, score_size),
            nn.Tanh(),
            nn.Linear(score_size, 4)) # num of key words: 0-3
        
        self.kw_out = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_size, score_size),
            nn.Tanh(),
            nn.Linear(score_size, 3)) # TODO: for 3 keywords: where, group, order
        
        self.softmax = nn.Softmax() #dim=1
        self.CE = nn.CrossEntropyLoss()
        self.log_softmax = nn.LogSoftmax()
        self.mlsml = nn.MultiLabelSoftMarginLoss()
        self.bce_logit = nn.BCEWithLogitsLoss()
        self.sigm = nn.Sigmoid()


    def forward(self, q_hs):
        # self.kw_num_out: (B, max_sql_hs_len, 4)
        kw_num_score = self.kw_num_out(q_hs)        
        # kw_score: (B, max_sql_hs_len, 3)
        kw_score = self.kw_out(q_hs)
        score = (kw_num_score, kw_score)

        return score

    def loss(self, score, truth):
        loss = 0
        B = len(truth)
        kw_num_score, kw_score = score
        #loss for the key word number
        truth_num = [len(t) for t in truth] # double check to exclude select
        data = torch.from_numpy(np.array(truth_num))
        truth_num_var = Variable(data.cuda())
        loss += self.CE(kw_num_score, truth_num_var)
        #loss for the key words
        T = len(kw_score[0])
        truth_prob = np.zeros((B, T), dtype=np.float32)
        for b in range(B):
            truth_prob[b][truth[b]] = 1
        data = torch.from_numpy(truth_prob)
        truth_var = Variable(data.cuda())
        #loss += self.mlsml(kw_score, truth_var)
        #loss += self.bce_logit(kw_score, truth_var) # double check no sigmoid for kw
        pred_prob = self.sigm(kw_score)
        bce_loss = -torch.mean( 3*(truth_var * \
                torch.log(pred_prob+1e-10)) + \
                (1-truth_var) * torch.log(1-pred_prob+1e-10) )
        loss += bce_loss

        return loss

    def check_acc(self, score, truth):
        num_err, err, tot_err = 0, 0, 0
        B = len(truth)
        pred = []
        kw_num_score, kw_score = [x.data.cpu().numpy() for x in score]
        for b in range(B):
            cur_pred = {}
            kw_num = np.argmax(kw_num_score[b])
            cur_pred['kw_num'] = kw_num
            cur_pred['kw'] = np.argsort(-kw_score[b])[:kw_num]
            pred.append(cur_pred)

        for b, (p, t) in enumerate(zip(pred, truth)):
            kw_num, kw = p['kw_num'], p['kw']
            flag = True
            if kw_num != len(t): # double check to excluding select
                num_err += 1
                flag = False
            if flag and set(kw) != set(t):
                err += 1
                flag = False
            if not flag:
                tot_err += 1

        return np.array((num_err, err, tot_err))

    
class MultiSqlPredictor(nn.Module):
    '''Predict if the next token is (multi SQL key words):
        NONE, EXCEPT, INTERSECT, or UNION.'''
    def __init__(self, input_size, score_size, dropout):
        super(MultiSqlPredictor, self).__init__()
        
        self.multi_out = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_size, score_size),
            nn.Tanh(),
            nn.Linear(score_size, 4)) # for none, intersect, union, expect
        
        self.softmax = nn.Softmax() #dim=1
        self.CE = nn.CrossEntropyLoss()
        self.log_softmax = nn.LogSoftmax()
        self.mlsml = nn.MultiLabelSoftMarginLoss()
        self.bce_logit = nn.BCEWithLogitsLoss()
        self.sigm = nn.Sigmoid()

    def forward(self, q_hs):
        # mulit_score: (B, max_sql_hs_len, 4)
        mulit_score = self.multi_out(q_hs)

        return mulit_score

    def loss(self, score, truth):
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
    
    
class OpPredictor(nn.Module):
    def __init__(self, input_size, score_size, dropout):
        super(OpPredictor, self).__init__()
        
        self.op_num_out = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_size, score_size),
            nn.Tanh(),
            nn.Linear(score_size, 2)) #for 1-2 op num, could be changed
        
        self.op_out = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_size, score_size),
            nn.Tanh(),
            nn.Linear(score_size, 11)) #for 11 operators

        self.softmax = nn.Softmax() #dim=1
        self.CE = nn.CrossEntropyLoss()
        self.log_softmax = nn.LogSoftmax()
        self.mlsml = nn.MultiLabelSoftMarginLoss()
        self.bce_logit = nn.BCEWithLogitsLoss()
        self.sigm = nn.Sigmoid()

    def forward(self, q_hs):
        # op_num_score: (B, max_sql_hs_len, 2)
        op_num_score = self.op_num_out(q_hs)
        # op_score: (B, max_sql_hs_len, 10) # TODO: double check 10/11?
        op_score = self.op_out(q_hs)
        score = (op_num_score, op_score)

        return score

    def loss(self, score, truth):
        loss = 0
        B = len(truth)
        op_num_score, op_score = score
        truth = [t if len(t) <= 2 else t[:2] for t in truth]
        # loss for the op number
        truth_num = [len(t)-1 for t in truth] #num_score 0 maps to 1 in truth
        data = torch.from_numpy(np.array(truth_num))
        truth_num_var = Variable(data.cuda())
        loss += self.CE(op_num_score, truth_num_var)
        # loss for op
        T = len(op_score[0])
        truth_prob = np.zeros((B, T), dtype=np.float32)
        for b in range(B):
            truth_prob[b][truth[b]] = 1
        data = torch.from_numpy(np.array(truth_prob))
        truth_var = Variable(data.cuda())
        #loss += self.mlsml(op_score, truth_var)
        #loss += self.bce_logit(op_score, truth_var)
        pred_prob = self.sigm(op_score)
        bce_loss = -torch.mean( 3*(truth_var * \
                torch.log(pred_prob+1e-10)) + \
                (1-truth_var) * torch.log(1-pred_prob+1e-10) )
        loss += bce_loss

        return loss

    def check_acc(self, score, truth):
        num_err, err, tot_err = 0, 0, 0
        B = len(truth)
        pred = []
        op_num_score, op_score = [x.data.cpu().numpy() for x in score]
        for b in range(B):
            cur_pred = {}
            op_num = np.argmax(op_num_score[b]) + 1 #num_score 0 maps to 1 in truth, must have at least one op
            cur_pred['op_num'] = op_num
            cur_pred['op'] = np.argsort(-op_score[b])[:op_num]
            pred.append(cur_pred)

        for b, (p, t) in enumerate(zip(pred, truth)):
            op_num, op = p['op_num'], p['op']
            flag = True
            if op_num != len(t):
                num_err += 1
                flag = False
            if flag and set(op) != set(t):
                err += 1
                flag = False
            if not flag:
                tot_err += 1

        return np.array((num_err, err, tot_err))
    
    
class RootTeminalPredictor(nn.Module):
    def __init__(self, input_size, score_size, dropout):
        super(RootTeminalPredictor, self).__init__()
        
        self.rt_out = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_size, score_size),
            nn.Tanh(),
            nn.Linear(score_size, 2)) #for new root or terminate
        
        self.softmax = nn.Softmax() #dim=1
        self.CE = nn.CrossEntropyLoss()
        self.log_softmax = nn.LogSoftmax()
        self.mlsml = nn.MultiLabelSoftMarginLoss()
        self.bce_logit = nn.BCEWithLogitsLoss()
        self.sigm = nn.Sigmoid()

    def forward(self, q_hs):
        # rt_score: (B, max_sql_hs_len, 2)
        rt_score = self.rt_out(q_hs)

        return rt_score

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


class ColPredictor(nn.Module):
    def __init__(self, input_size, score_size, dropout):
        super(MatchScorer, self).__init__()
        
        self.col_num_out = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_size*2/3, score_size),
            nn.Tanh(),
            nn.Linear(score_size, 3)) # for num of cols: 1-3
        
        self.col_out = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_size, score_size),
            nn.Tanh(),
            nn.Linear(score_size, 1))
                
        self.softmax = nn.Softmax() #dim=1
        self.CE = nn.CrossEntropyLoss()
        self.log_softmax = nn.LogSoftmax()
        self.mlsml = nn.MultiLabelSoftMarginLoss()
        self.bce_logit = nn.BCEWithLogitsLoss()
        self.sigm = nn.Sigmoid()

    def forward(self, q_hs, q_hs_col, tbl_mask):
        # q_hs: (batch, max_sql_hs_len, input_size*2/3)
        # q_hs_col: (batch, max_sql_hs_len, max_col_len, input_size)
        # col_num_score: (B, max_sql_hs_len, 3)
        col_num_score = self.col_num_out(q_hs)
        
        # col_score: (B, max_sql_hs_len, max_col_len)
        col_score_unmask = self.col_out(q_hs_col).squeeze()
        # tbl_mask_expand: (B, max_sql_hs_len, max_col_len)
        tbl_mask_expand = tbl_mask.unsqueeze(1).expand(tbl_mask.size(0), col_score_unmask.size(1), tbl_mask.size(1))
        # mask scores
        col_score = col_score_unmask.masked_fill(tbl_mask_expand, -float('inf'))
        score = (col_num_score, col_score)
        
        return score
    
    def loss(self, score, truth):
        #here suppose truth looks like [[[1, 4], 3], [], ...]
        # score = (col_num_score(B, 3), col_score(B, max_col_len)) 
        loss = 0
        B = len(truth)
        col_num_score, col_score = score
        #loss for the column number
        truth_num = [len(t) - 1 for t in truth] # double check truth format and for test cases
        data = torch.from_numpy(np.array(truth_num))
        truth_num_var = Variable(data.cuda())
        loss += self.CE(col_num_score, truth_num_var)
        #loss for the key words
        T = len(col_score[0])
        # print("T {}".format(T))
        truth_prob = np.zeros((B, T), dtype=np.float32)
        for b in range(B):
            gold_l = []
            for t in truth[b]:
                if isinstance(t, list):
                    gold_l.extend(t)
                else:
                    gold_l.append(t)
            truth_prob[b][gold_l] = 1
        data = torch.from_numpy(truth_prob)
        # print("data {}".format(data))
        # print("data {}".format(data.cuda()))
        truth_var = Variable(data.cuda())
        #loss += self.mlsml(col_score, truth_var)
        #loss += self.bce_logit(col_score, truth_var) # double check no sigmoid
        pred_prob = self.sigm(col_score)
        bce_loss = -torch.mean( 3*(truth_var * \
                torch.log(pred_prob+1e-10)) + \
                (1-truth_var) * torch.log(1-pred_prob+1e-10) )
        loss += bce_loss

        return loss

    def check_acc(self, score, truth):
        num_err, err, tot_err = 0, 0, 0
        B = len(truth)
        pred = []
        col_num_score, col_score = [x.data.cpu().numpy() for x in score]
        for b in range(B):
            cur_pred = {}
            col_num = np.argmax(col_num_score[b]) + 1 #double check
            cur_pred['col_num'] = col_num
            cur_pred['col'] = np.argsort(-col_score[b])[:col_num]
            pred.append(cur_pred)

        for b, (p, t) in enumerate(zip(pred, truth)):
            col_num, col = p['col_num'], p['col']
            flag = True
            if col_num != len(t): # double check truth format and for test cases
                num_err += 1
                flag = False
            #to eval col predicts, if the gold sql has JOIN and foreign key col, then both fks are acceptable
            fk_list = []
            regular = []
            for l in t:
                if isinstance(l, list):
                    fk_list.append(l)
                else:
                    regular.append(l)

            if flag: #double check
                for c in col:
                    for fk in fk_list:
                        if c in fk:
                            fk_list.remove(fk)
                    for r in regular:
                        if c == r:
                            regular.remove(r)

                if len(fk_list) != 0 or len(regular) != 0:
                    err += 1
                    flag = False

            if not flag:
                tot_err += 1

        return np.array((num_err, err, tot_err))
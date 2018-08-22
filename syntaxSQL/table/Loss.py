"""
This file handles the details of the loss function during training.

This includes: LossComputeBase and the standard NMTLossCompute, and
               sharded loss compute stuff.
"""
from __future__ import division
from itertools import count
import torch
import torch.nn as nn
from torch.autograd import Variable
import random as rnd

import table
from table.modules.cross_entropy_smooth import CrossEntropyLossSmooth

class LossCompute(nn.Module):
    def __init__(self, smooth_eps=0):
        super(LossCompute, self).__init__()
        self.criterion = {}
        if smooth_eps > 0:
            self.criterion['lay'] = CrossEntropyLossSmooth(
                size_average=False, ignore_index=table.IO.PAD, smooth_eps=smooth_eps)
            self.criterion['tgt'] = CrossEntropyLossSmooth(
                size_average=False, ignore_index=table.IO.PAD, smooth_eps=smooth_eps)
        else:
            self.criterion['lay'] = nn.NLLLoss(
                size_average=False, ignore_index=table.IO.PAD)
            # self.criterion['tgt'] = nn.CrossEntropyLoss(
            self.criterion['tgt'] = nn.NLLLoss(
                size_average=False, ignore_index=table.IO.PAD)
        self.criterion['token'] = nn.BCEWithLogitsLoss(size_average=False)
        self.criterion['cover'] = nn.KLDivLoss(size_average=False)
        
    
    def compute_loss(self, pred, gold, mask, predictors):
        loss_list = []
#         mulit_sql, keyword, col, op, agg, root_tem, des_asc, having, andor = predictors
        for loss_name in ('lay', 'mod_scores'):
            if loss_name not in gold:
                continue
            
            if loss_name == 'mod_scores':
                # col_num_score: (B, sql_len, 3), col_score: (B, sql_len, max_col_len)
#                 mulit_sql_score, keyword_score, col_score, op_score, agg_score,
#                   root_tem_score, des_asc_score, having_score, andor_score = pred[loss_name]
                label = gold[loss_name] # (B, sql_len) list of lists like [[1], [9], [7], [[1], []], [4], [[3, 5] [0]], ...]
                mod_mask = mask[loss_name] # (B, sql_len) list of lists like [[1], [2], [3], [4, 8], [5], ...]
                for i, ls, ms in zip(count(), label, mod_mask):
                    # ls, ms: [max_sql_len]; i: for B
                    for j, l, m in zip(count(), ls, ms):
                        # l label: [1] or [[3, 5], [0]] or [] m mask: [1] or [2, 4] or [-1]; j for max_sql_len
                        for mod_label, mod_id in zip(l, m):
                            if mod_id == -1 or len(mod_label) == 0: # to skip [] in [[1], []]
                                continue
                                
                            scores = pred[loss_name][mod_id]
                            mod_score = []
                            for score in scores:
                                mod_score.append(score[i][j].unsqueeze(0)) #append (1, num_of_classes)
                            # mod_label： 1 or [3, 5]
                            mod_label = [mod_label]
                            loss = predictors[mod_id].loss(mod_score, mod_label)
                            loss_list.append(loss)
            else:
                # pred lay: (dec_seq_len, batch_size, vocab_size), gold lay: (dec_seq_len, batch_size)
                for i, p, g in zip(count(), pred[loss_name], gold[loss_name]):
                    if (loss_name in mask) and mask[loss_name][i] == 1:
                        continue
                    # p: (batch_size, vocab_size), g: (batch_size)
                    loss = self.criterion[loss_name](p, g)
                    loss_list.append(loss)
                

        # sum up the loss functions
        return sum(loss_list)
    
    
    

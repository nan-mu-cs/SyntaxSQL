"""
This is the loadable seq2seq trainer library that is
in charge of training details, loss compute, and statistics.
"""
from __future__ import division
import os
import time
import sys
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from itertools import chain, count
from copy import deepcopy

import table
import table.modules
from table.Utils import argmax


class Statistics(object):
    def __init__(self, loss, eval_result):
        self.loss = loss
        self.eval_result = eval_result
        self.start_time = time.time()

    def update(self, stat):
        self.loss += stat.loss
        for k, v in stat.eval_result.items():
            if k in self.eval_result:
                v0 = self.eval_result[k][0] + v[0]
                v1 = self.eval_result[k][1] + v[1]
                self.eval_result[k] = (v0, v1)
            else:
                self.eval_result[k] = (v[0], v[1])

    def accuracy(self, return_str=False):
        d = sorted([(k, v)
                    for k, v in self.eval_result.items()], key=lambda x: x[0])
        if return_str:
            return '; '.join((('{}: {:.2%}'.format(k, v[0] / v[1],)) for k, v in d))
        else:
            return dict([(k, 100.0 * v[0] / v[1]) for k, v in d])

    def elapsed_time(self):
        return time.time() - self.start_time

    def output(self, epoch, batch, n_batches, start):
        print(("Epoch %2d, %5d/%5d; %s; %.0f s elapsed") %
              (epoch, batch, n_batches, self.accuracy(True), time.time() - start))
        sys.stdout.flush()

    def log(self, split, logger, lr, step):
        pass


def count_accuracy(scores, target, mask=None, row=False):
    pred = argmax(scores)
    if mask is None:
        m_correct = pred.eq(target)
        num_all = m_correct.numel()
    elif row:
        m_correct = pred.eq(target).masked_fill_(
            mask, 1).prod(0, keepdim=False)
        num_all = m_correct.numel()
    else:
        non_mask = mask.ne(1)
        m_correct = pred.eq(target).masked_select(non_mask)
        num_all = non_mask.sum()
    
    return (m_correct, num_all)


def count_accuracy_for_predictors(pred, gold, mod_mask, predictors):
    # (m_correct, num_all)
    mod_id_dict = {0: 'mulit_sql', 1: 'keyword', 2: 'col', 3: 'op', 4: 'agg',
                   5: 'root_tem', 6: 'des_asc', 7: 'having', 8: 'andor'}
    r_dict = {'mulit_sql': (0, 0), 'keyword': (0, 0), 'col': (0, 0), 'op': (0, 0),
              'agg': (0, 0), 'root_tem': (0, 0), 'des_asc': (0, 0), 'having': (0, 0), 'andor': (0, 0), "all": (0, 0)}
    label = gold["mod_scores"] # (B, sql_len) list of lists like [[1], [9], [7], [[1], []], [4], [[3, 5] [0]], ...]
    for i, ls, ms in zip(count(), label, mod_mask):
        # ls, ms: [max_sql_len]; i: for B
        r_dict["all"][1] += 1
        one_example_res = []
        for j, l, m in zip(count(), ls, ms):
            # l label: [1] or [[3, 5], [0]] or [] m mask: [1] or [2, 4] or [-1]; j for max_sql_len
            for mod_label, mod_id in zip(l, m):
                if mod_id == -1 or len(mod_label) == 0: # to skip [] in [[1], []]
                    continue
                    
                mod_name = mod_id_dict[mod_id]
                scores = pred["mod_scores"][mod_id]
                model = predictors[mod_id]
                mod_score = []
                for score in scores:
                    mod_score.append(score[i][j].unsqueeze(0)) #append (1, num_of_classes)
                # mod_label： 1 or [3, 5]
                mod_label = [mod_label]
                if mod_name in ("agg", "col", "keyword", "op"):
                    num_err, p_err, err = model.check_acc(score, label)
                else:
                    err = model.check_acc(score, label)
                    
                one_example_res.append(1 - err)
                r_dict[mod_name][0] += (1 - err)
                r_dict[mod_name][1] += 1
          
        if any(one_example_res) != 0:
            r_dict["all"][0] += 1
    
    return r_dict


def aggregate_accuracy(r_dict, metric_name_list):
    m_list = []
    for metric_name in metric_name_list:
        m_list.append(r_dict[metric_name][0])
    agg = torch.stack(m_list, 0).prod(0, keepdim=False)
    return (agg.sum(), agg.numel())


def reconstruct_mask_label_one(mask_one, mask_inds_one, label_list_one, label_nums_one):

    # mask_one/mi/ln: (max_len) mask_list, mask_inds, label_nums
    module_mask_one = []
    module_label_one = []
    module_label_one_temp = []
    two_mods = []
    label_index = 0
    for j, mlj, mij in zip(count(), mask_one, mask_inds_one):
        if mij != 0:
            if len(two_mods) > 0:
                module_mask_one.append(two_mods)
                two_mods = []   
            module_mask_one.append([mlj])
        elif mij == 0:
            two_mods.append(mlj)
        
    ln = label_nums_one.copy()
    for lnj in ln:
        label_add = []
        for _ in range(lnj):
            label_add.append(label_list_one[label_index])
            label_index += 1
        module_label_one_temp.append(label_add)
        
    label_num_tmp = []
    for j, mij in zip(count(), mask_inds_one):
        if mij < 2:
            label_num_tmp.append(ln.pop(0))
        else:
            lbs = []
            num_count = 0
            while num_count < mij:
                lnp = ln.pop(0)
                lbs.append(lnp)
                num_count += lnp
            assert num_count == mij
            label_num_tmp.append(lbs)
    
    two_labels = []
    assert len(mask_inds_one) == len(label_num_tmp)
    for k, mlk, mik, lnk in zip(count(), mask_one, mask_inds_one, label_num_tmp):
        if mik == 1:
            if len(two_labels) > 0:
                module_label_one.append(two_labels)
                two_labels = []
            if mlk in [0, 7, -1, 5, 6]:
                module_label_one.append(module_label_one_temp.pop(0))
            else:
                module_label_one.append([module_label_one_temp.pop(0)])
        elif mik == 0:
            label = module_label_one_temp.pop(0)
            if mlk == 5: # for case [1, ['VALUE_0', None]]
                two_labels.append(label[0])
            else:
                two_labels.append(label)
        else:
            col_labels = []
            for lnki in lnk:
                col_label = module_label_one_temp.pop(0)
                assert len(col_label) == lnki
                if len(col_label) == 1:
                    col_labels.append(col_label[0])
                else:
                    col_labels.append(col_label)
            module_label_one.append([col_labels])
            
    return module_mask_one, module_label_one


def reconstruct_mask_label(mask_list, mask_inds, label_list, label_nums):
    module_mask = []
    module_label = []
    # mask_list/mask_inds/label_nums: (max_len, B)
#     print("mask_list: ", mask_list, mask_list.size())
#     print("mask_inds: ", mask_inds, mask_inds.size())
#     print("label_list: ", label_list, label_list.size())
#     print("label_nums: ", label_nums, label_nums.size())
    
    mask_list = [[i for i in ll if i != -2] for ll in mask_list.transpose(0, 1).data.cpu().tolist()]
    mask_inds = [[i for i in ll if i != -2] for ll in mask_inds.transpose(0, 1).data.cpu().tolist()]
    label_list = [[i for i in ll if i != -2] for ll in label_list.transpose(0, 1).data.cpu().tolist()]
    label_nums = [[i for i in ll if i != -2] for ll in label_nums.transpose(0, 1).data.cpu().tolist()]
        
    for i, ml, mi, ln, ll in zip(count(), mask_list, mask_inds, label_nums, label_list):
        module_mask_one, module_label_one = reconstruct_mask_label_one(ml, mi, ll, ln)
        module_mask.append(module_mask_one)
        module_label.append(module_label_one)
        
    return module_mask, module_label
        

def _debug_batch_content(vocab, ts_batch):
    seq_len = ts_batch.size(0)
    batch_size = ts_batch.size(1)
    for b in range(batch_size):
        tk_list = []
        for i in range(seq_len):
            tk = vocab.itos[ts_batch[i, b]]
            tk_list.append(tk)
        print(tk_list)


class Trainer(object):
    def __init__(self, model, train_iter, valid_iter,
                 train_loss, valid_loss, optim):
        """
        Args:
            model: the seq2seq model.
            train_iter: the train data iterator.
            valid_iter: the validate data iterator.
            train_loss: the train side LossCompute object for computing loss.
            valid_loss: the valid side LossCompute object for computing loss.
            optim: the optimizer responsible for lr update.
        """
        # Basic attributes.
        self.model = model
        self.train_iter = train_iter
        self.valid_iter = valid_iter
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.optim = optim

        if self.model.opt.moving_avg > 0:
            self.moving_avg = deepcopy(
                list(p.data for p in model.parameters()))
        else:
            self.moving_avg = None

        # Set model in training mode.
        self.model.train()

    def forward(self, epoch, batch, criterion, fields):
        # 1. F-prop.
        q, q_len = batch.src
        tbl, tbl_len = batch.tbl
        sql, sql_len = batch.sql_history
        mask_list = batch.mask_list
        mask_inds = batch.mask_inds
        label_list = batch.label_list
        label_nums = batch.label_nums
        tbl_split = batch.tbl_split
        tbl_mask = batch.tbl_mask
        
        # suppose mask_list: (max_len, B) # TODO: check size!
        # mod_mask, mod_label: (B, max_len) # TODO: check size! and lens of mod_mask and mod_label are same
        mod_mask, mod_label = reconstruct_mask_label(mask_list, mask_inds, label_list, label_nums)
        
        #print("\n-------------------")
        #print("q: ", q)
        #print("sql_len: ", sql_len)
        #print("tbl: ", tbl)
        #print("mask_list: ", mask_list)
        #print("mask_inds: ", mask_inds)
        #print("label_list: ", label_list)
        #print("label_nums: ", label_nums)
        #print("mod_mask: ", mod_mask)
        #print("mod_label: ", mod_label)
        
        # TODO: add src_type and col_type : None
        sql_out, mod_scores, predictors = self.model(q, q_len, None, tbl, tbl_len, tbl_split, tbl_mask, sql, sql_len)

        _debug_batch_content(fields['sql_history'].vocab, argmax(sql_out.data))

        # 2. Compute loss.
        pred = {'sql': sql_out, 'mod_scores': mod_scores}
        gold = {}
        mask_loss = {'mod_scores': mod_mask} # TODO: mod_mask -> mask_loss
        gold['sql'] = sql[1:]
        gold['mod_scores'] = mod_label
        
        loss = criterion.compute_loss(pred, gold, mask_loss, predictors)

        # 3. Get the batch statistics.
        r_dict = {}
        for metric_name in ('sql',):
            p = pred[metric_name].data
            g = gold[metric_name].data
            r_dict[metric_name + '-token'] = count_accuracy(
                p, g, mask=g.eq(table.IO.PAD), row=False)
            r_dict[metric_name] = count_accuracy(
                p, g, mask=g.eq(table.IO.PAD), row=True)
        sql_res_dict = dict([(k, (v[0].sum(), v[1])) for k, v in r_dict.items()])
        mod_res_dict = count_accuracy_for_predictors(pred, gold, mod_mask, predictors)
        res_dict = sql_res_dict + mod_res_dict
        
        batch_stats = Statistics(loss.data[0], res_dict)

        return loss, batch_stats

    def train(self, epoch, fields, report_func=None):
        """ Called for each epoch to train. """
        total_stats = Statistics(0, {})
        report_stats = Statistics(0, {})

        for i, batch in enumerate(self.train_iter):
            self.model.zero_grad()
	    
            loss, batch_stats = self.forward(
                epoch, batch, self.train_loss, fields)
            # _debug_batch_content(fields['lay'].vocab, batch.lay.data)

            # Update the parameters and statistics.
            loss.backward()
            self.optim.step()
            total_stats.update(batch_stats)
            report_stats.update(batch_stats)

            if report_func is not None:
                report_stats = report_func(
                    epoch, i, len(self.train_iter),
                    total_stats.start_time, self.optim.lr, report_stats)

            if self.model.opt.moving_avg > 0:
                decay_rate = min(self.model.opt.moving_avg,
                                 (1 + epoch) / (1.5 + epoch))
                for p, avg_p in zip(self.model.parameters(), self.moving_avg):
                    avg_p.mul_(decay_rate).add_(1.0 - decay_rate, p.data)

        return total_stats

    def validate(self, epoch, fields):
        """ Called for each epoch to validate. """
        # Set model in validating mode.
        self.model.eval()

        stats = Statistics(0, {})
        for batch in self.valid_iter:
            loss, batch_stats = self.forward(
                epoch, batch, self.valid_loss, fields)

            # Update statistics.
            stats.update(batch_stats)

        # Set model back to training mode.
        self.model.train()

        return stats

    def epoch_step(self, eval_metric, epoch):
        """ Called for each epoch to update learning rate. """
        return self.optim.updateLearningRate(eval_metric, epoch)

    def drop_checkpoint(self, opt, epoch, fields, valid_stats):
        """ Called conditionally each epoch to save a snapshot. """

        model_state_dict = self.model.state_dict()
        model_state_dict = {k: v for k, v in model_state_dict.items()
                            if 'generator' not in k}
        checkpoint = {
            'model': model_state_dict,
            'vocab': table.IO.TableDataset.save_vocab(fields),
            'opt': opt,
            'epoch': epoch,
            'optim': self.optim,
            'moving_avg': self.moving_avg
        }
        eval_result = valid_stats.accuracy()
        torch.save(checkpoint, os.path.join(
            opt.save_path, 'm_%d.pt' % (epoch)))

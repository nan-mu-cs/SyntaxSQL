# -*- coding: utf-8 -*-

import codecs
import json
import random as rnd
import numpy as np
from collections import Counter, defaultdict
from itertools import chain, count
from six import string_types

import torch
import torchtext.data
import torchtext.vocab
from tree import SCode

UNK_WORD = '<unk>'
UNK = 0
PAD_WORD = '<blank>'
PAD = 1
BOS_WORD = '<s>'
BOS = 2
EOS_WORD = '</s>'
EOS = 3
SKP_WORD = '<sk>'
SKP = 4
RIG_WORD = '<]>'
RIG = 5
LFT_WORD = '<[>'
LFT = 6
SPLIT_WORD = '<|>'
special_token_list = [UNK_WORD, PAD_WORD, BOS_WORD, EOS_WORD, SKP_WORD, RIG_WORD, LFT_WORD, SPLIT_WORD]


def get_parent_index(tk_list):
    stack = [0]
    r_list = []
    for i, tk in enumerate(tk_list):
        r_list.append(stack[-1])
        if tk.startswith('('):
            # +1: because the parent of the top level is 0
            stack.append(i+1)
        elif tk ==')':
            stack.pop()
    # for EOS (</s>)
    r_list.append(0)
    return r_list


def get_tgt_mask(lay_skip):
    # 0: use layout encoding vectors; 1: use target word embeddings;
    # with a <s> token at the first position
    return [1] + [1 if tk in (SKP_WORD, RIG_WORD) else 0 for tk in lay_skip]


def get_lay_index(lay_skip):
    # with a <s> token at the first position
    r_list = [0]
    k = 0
    for tk in lay_skip:
        if tk in (SKP_WORD, RIG_WORD):
            r_list.append(0)
        else:
            r_list.append(k)
            k += 1
    return r_list


def get_tgt_loss(line, mask_target_loss):
    r_list = []
    for tk_tgt, tk_lay_skip in zip(line['tgt'], line['lay_skip']):
        if tk_lay_skip in (SKP_WORD, RIG_WORD):
            r_list.append(tk_tgt)
        else:
            if mask_target_loss:
                r_list.append(PAD_WORD)
            else:
                r_list.append(tk_tgt)
    return r_list


def __getstate__(self):
    return dict(self.__dict__, stoi=dict(self.stoi))


def __setstate__(self, state):
    self.__dict__.update(state)
    self.stoi = defaultdict(lambda: 0, self.stoi)


torchtext.vocab.Vocab.__getstate__ = __getstate__
torchtext.vocab.Vocab.__setstate__ = __setstate__


def filter_counter(freqs, min_freq):
    cnt = Counter()
    for k, v in freqs.items():
        if (min_freq is None) or (v >= min_freq):
            cnt[k] = v
    return cnt


def merge_vocabs(vocabs, min_freq=0, vocab_size=None):
    """
    Merge individual vocabularies (assumed to be generated from disjoint
    documents) into a larger vocabulary.

    Args:
        vocabs: `torchtext.vocab.Vocab` vocabularies to be merged
        vocab_size: `int` the final vocabulary size. `None` for no limit.
    Return:
        `torchtext.vocab.Vocab`
    """
    merged = Counter()
    for vocab in vocabs:
        merged += filter_counter(vocab.freqs, min_freq)
    return torchtext.vocab.Vocab(merged,
                                 specials=list(special_token_list),
                                 max_size=vocab_size, min_freq=min_freq)


def join_dicts(*args):
    """
    args: dictionaries with disjoint keys
    returns: a single dictionary that has the union of these keys
    """
    return dict(chain(*[d.items() for d in args]))


class OrderedIterator(torchtext.data.Iterator):
    def create_batches(self):
        if self.train:
            self.batches = torchtext.data.pool(
                self.data(), self.batch_size,
                self.sort_key, self.batch_size_fn,
                random_shuffler=self.random_shuffler)
        else:
            self.batches = []
            for b in torchtext.data.batch(self.data(), self.batch_size,
                                          self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))


def unlist_mask(mask_list):
    mask_nolist = []
    mask_inds = []
    for ml in mask_list:
        if len(ml) == 1:
            mask_nolist.append(ml[0])
            mask_inds.append(1)
        elif len(ml) == 2:
            mask_nolist.extend(ml)
            mask_inds.extend([0, 0])
        else:
            print("\nWarning: NOT expected length of the mask list greater than 2!")
            exit()
        
    assert len(mask_nolist) == len(mask_inds)
    
    return mask_nolist, mask_inds


def unlist_label(label_list):
    label_nolist = []
    label_nums = []
    for ll in label_list:
        if not isinstance(ll[0], list):
            label_nolist.extend(ll)
            label_nums.append(len(ll))
        else:
            assert len(ll) == 2
            label_nolist.extend(ll[0])
            label_nums.append(len(ll[0]))
            label_nolist.extend(ll[1])
            label_nums.append(len(ll[1]))
            
    assert sum(label_nums) == len(label_nolist)
    
    return label_nolist, label_nums


def read_anno_json(anno_path, opt):
    with codecs.open(anno_path, "r", "utf-8") as corpus_file:
        js_list = [json.loads(line) for line in corpus_file]
        for line in js_list:
            line["mask_list"], line["mask_inds"] = unlist_mask(line["module_mask"])
            line["label_list"], line["label_nums"] = unlist_label(line["module_label"])
    return js_list


class TableDataset(torchtext.data.Dataset):
    """Defines a dataset for machine translation."""

    @staticmethod
    def sort_key(ex):
        "Sort in reverse size order"
        return -len(ex.src)

    def __init__(self, anno, fields, permute_order, opt, filter_ex, **kwargs):
        """
        Create a TranslationDataset given paths and fields.

        anno: location of annotated data / js_list
        filter_ex: False - keep all the examples for evaluation (should not have filtered examples); True - filter examples with unmatched spans;
        """
        if isinstance(anno, string_types):
            js_list = read_anno_json(anno, opt)
        else:
            js_list = anno

        src_data = self._read_annotated_file(opt, js_list, 'src', filter_ex)
        src_examples = self._construct_examples(src_data, 'src')
                
        tbl_data = self._read_annotated_file(opt, js_list, 'tbl', filter_ex)
        tbl_examples = self._construct_examples(tbl_data, 'tbl')
        
        tbl_split_data = self._read_annotated_file(
            opt, js_list, 'tbl_split', filter_ex)
        tbl_split_examples = self._construct_examples(
            tbl_split_data, 'tbl_split')

        tbl_mask_data = self._read_annotated_file(
            opt, js_list, 'tbl_mask', filter_ex)
        tbl_mask_examples = self._construct_examples(
            tbl_mask_data, 'tbl_mask')
        
        sql_data = self._read_annotated_file(opt, js_list, 'sql_history', filter_ex) # TODO: replace col in sql hisotry
        sql_examples = self._construct_examples(sql_data, 'sql_history')
        
        mod_mask_data = self._read_annotated_file(
            opt, js_list, 'mask_list', filter_ex)
        mod_mask_examples = self._construct_examples(
            mod_mask_data, 'mask_list')
        
        mask_ind_data = self._read_annotated_file(
            opt, js_list, 'mask_inds', filter_ex)
        mask_ind_examples = self._construct_examples(
            mask_ind_data, 'mask_inds')

        mod_label_data = self._read_annotated_file(opt, js_list, 'label_list', filter_ex)
        mod_label_examples = self._construct_examples(mod_label_data, 'label_list')
        
        label_num_data = self._read_annotated_file(opt, js_list, 'label_nums', filter_ex)
        label_num_examples = self._construct_examples(label_num_data, 'label_nums')

        # examples: one for each src line or (src, tgt) line pair.
        examples = [join_dicts(*it) for it in zip(src_examples, tbl_examples, tbl_split_examples, tbl_mask_examples,
                        sql_examples, mod_mask_examples, mask_ind_examples, mod_label_examples, label_num_examples)]
        # the examples should not contain None
        len_before_filter = len(examples)
        examples = list(filter(lambda x: all(
            (v is not None for k, v in x.items())), examples))
        len_after_filter = len(examples)
        num_filter = len_before_filter - len_after_filter
        if num_filter > 0:
            print('Filter #examples (with None): {} / {} = {:.2%}'.format(num_filter,
                                                                          len_before_filter, num_filter / len_before_filter))

        # Peek at the first to see which fields are used.
        ex = examples[0]
        keys = ex.keys()
        fields = [(k, fields[k])
                  for k in (list(keys) + ["indices"])]

        def construct_final(examples):
            for i, ex in enumerate(examples):
                yield torchtext.data.Example.fromlist(
                    [ex[k] for k in keys] + [i],
                    fields)

        def filter_pred(example):
            return True

        super(TableDataset, self).__init__(
            construct_final(examples), fields, filter_pred)

    def _read_annotated_file(self, opt, js_list, field, filter_ex):
        """
        path: location of a src or tgt file
        truncate: maximum sequence length (0 for unlimited)
        """
        if field in ('src', 'sql_history'):
            lines = (line[field] for line in js_list) # TODO: for question tokens and sql_history lower and tokenized
        elif field in ('tbl',):
            def _tbl(col_list):
                tk_list = [SPLIT_WORD]
                tk_split = '\t' + SPLIT_WORD + '\t'
                tk_list.extend(tk_split.join(
                    ['\t'.join(col) for col in col_list]).strip().split('\t'))
                tk_list.append(SPLIT_WORD)
                return tk_list
            lines = (_tbl(line["ts"]) for line in js_list) #TODO: a list of table schemas lower and tokenized
        elif field in ('tbl_split',):
            def _cum_length_for_split(col_list):
                len_list = [len(col) for col in col_list]
                r = [0]
                for i in range(len(len_list)):
                    r.append(r[-1] + len_list[i] + 1)
                return r
            lines = (_cum_length_for_split(line["ts"]) for line in js_list) #TODO: a list of table schemas
        elif field in ('tbl_mask',):
            lines = ([0 for col in line['ts']] for line in js_list) #TODO: a list of table schemas
        elif field in ('mask_list',):
            lines = (line[field] for line in js_list)
        elif field in ('mask_inds',):
            lines = (line[field] for line in js_list)
        elif field in ('label_list',):
            lines = (line[field] for line in js_list)
        elif field in ('label_nums',):
            lines = (line[field] for line in js_list)
        else:
            raise NotImplementedError
        for line in lines:
            yield line

    def _construct_examples(self, lines, side):
        for words in lines:
            example_dict = {side: words}
            yield example_dict

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)

    def __reduce_ex__(self, proto):
        "This is a hack. Something is broken with torch pickle."
        return super(TableDataset, self).__reduce_ex__()

    @staticmethod
    def load_fields(vocab):
        vocab = dict(vocab)
        fields = TableDataset.get_fields()
        for k, v in vocab.items():
            # Hack. Can't pickle defaultdict :(
            v.stoi = defaultdict(lambda: 0, v.stoi)
            fields[k].vocab = v
        return fields

    @staticmethod
    def save_vocab(fields):
        vocab = []
        for k, f in fields.items():
            if 'vocab' in f.__dict__:
                f.vocab.stoi = dict(f.vocab.stoi)
                vocab.append((k, f.vocab))
        return vocab

    @staticmethod
    def get_fields():
        fields = {}
        fields["src"] = torchtext.data.Field(
            pad_token=PAD_WORD, include_lengths=True)
        fields["tbl"] = torchtext.data.Field(
            pad_token=PAD_WORD, include_lengths=True)
        fields["tbl_split"] = torchtext.data.Field(
            use_vocab=False, pad_token=0)
        fields["tbl_mask"] = torchtext.data.Field(
            use_vocab=False, tensor_type=torch.ByteTensor, batch_first=True, pad_token=1)
        fields["sql_history"] = torchtext.data.Field(
            init_token=BOS_WORD, include_lengths=True, eos_token=EOS_WORD, pad_token=PAD_WORD)
        fields["mask_list"] = torchtext.data.Field(
            use_vocab=False,  pad_token=-1)
        fields["mask_inds"] = torchtext.data.Field(
            use_vocab=False, pad_token=-1)
        fields["label_list"] = torchtext.data.Field(
            use_vocab=False, pad_token=-1)
        fields["label_nums"] = torchtext.data.Field(
            use_vocab=False, pad_token=-1)
        fields["indices"] = torchtext.data.Field(
            use_vocab=False, sequential=False)
        return fields

    @staticmethod
    def build_vocab(train, dev, test, opt):
        fields = train.fields
        
        merge_list = []
        merge_name_list = ('src', 'tbl')
        for split in (dev, test, train,):
            for merge_name_it in merge_name_list:
                fields[merge_name_it].build_vocab(
                    split, max_size=opt.src_vocab_size, min_freq=0)
                merge_list.append(fields[merge_name_it].vocab)
                
        fields["sql_history"].build_vocab(
            train, max_size=opt.src_vocab_size, min_freq=0)

        # need to know all the words to filter the pretrained word embeddings
        merged_vocab = merge_vocabs(merge_list, vocab_size=opt.src_vocab_size)
        for merge_name_it in merge_name_list:
            fields[merge_name_it].vocab = merged_vocab
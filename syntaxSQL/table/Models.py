from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import torch.nn.functional as F

import table
from table.Utils import aeq, sort_for_pack
from table.modules.embed_regularize import embedded_dropout
from table.modules.cross_entropy_smooth import onehot
from table.Utils import argmax


def _build_rnn(rnn_type, input_size, hidden_size, num_layers, dropout, weight_dropout, bidirectional=False):
    rnn = getattr(nn, rnn_type)(input_size, hidden_size,
                                num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)
    if weight_dropout > 0:
        param_list = ['weight_hh_l' + str(i) for i in range(num_layers)]
        if bidirectional:
            param_list += [it + '_reverse' for it in param_list]
        rnn = table.modules.WeightDrop(rnn, param_list, dropout=weight_dropout)
    return rnn


# MODIFY: WIKISQL ENCODER USED HERE
class RNNEncoder(nn.Module):
    """ The standard RNN encoder. """

    def __init__(self, rnn_type, bidirectional, num_layers,
                 hidden_size, dropout, lock_dropout, weight_dropout, embeddings, ent_embedding):
        super(RNNEncoder, self).__init__()

        num_directions = 2 if bidirectional else 1
        self.hidden_size = hidden_size
        self.embeddings = embeddings
        self.ent_embedding = ent_embedding
        self.no_pack_padded_seq = False
        if lock_dropout:
            self.word_dropout = table.modules.LockedDropout(dropout)
        else:
            self.word_dropout = nn.Dropout(dropout)

        # Use pytorch version when available.
        input_size = embeddings.embedding_dim
        if ent_embedding is not None:
            input_size += ent_embedding.embedding_dim
        self.rnn = _build_rnn(rnn_type, input_size,
                              hidden_size // num_directions, num_layers, dropout, weight_dropout, bidirectional)

    def forward(self, input, lengths=None, hidden=None, ent=None):
        emb = self.embeddings(input)
        if self.ent_embedding is not None:
            emb_ent = self.ent_embedding(ent)
            emb = torch.cat((emb, emb_ent), 2)
        if self.word_dropout is not None:
            emb = self.word_dropout(emb)
        # s_len, batch, emb_dim = emb.size()

        packed_emb = emb
        need_pack = (lengths is not None) and (not self.no_pack_padded_seq)
        if need_pack:
            # Lengths data is wrapped inside a Variable.
            if not isinstance(lengths, list):
                lengths = lengths.view(-1).tolist()
            packed_emb = pack(emb, lengths)

        outputs, hidden_t = self.rnn(packed_emb, hidden)

        if need_pack:
            outputs = unpack(outputs)[0]

        return hidden_t, outputs


def encode_unsorted_batch(encoder, tbl, tbl_len):
    # sort for pack()
    idx_sorted, tbl_len_sorted, idx_map_back = sort_for_pack(tbl_len)
    tbl_sorted = tbl.index_select(1, Variable(
        torch.LongTensor(idx_sorted).cuda(), requires_grad=False))
    # tbl_context: (seq_len, batch, hidden_size * num_directions)
    __, tbl_context = encoder(tbl_sorted, tbl_len_sorted)
    # recover the sort for pack()
    v_idx_map_back = Variable(torch.LongTensor(
        idx_map_back).cuda(), requires_grad=False)
    tbl_context = tbl_context.index_select(1, v_idx_map_back)
    return tbl_context


class TableRNNEncoder(nn.Module):
    def __init__(self, encoder, split_type='incell', merge_type='cat'):
        super(TableRNNEncoder, self).__init__()
        self.split_type = split_type
        self.merge_type = merge_type
        self.hidden_size = encoder.hidden_size
        self.encoder = encoder
        if self.merge_type == 'mlp':
            self.merge = nn.Sequential(
                nn.Linear(2 * self.hidden_size, self.hidden_size),
                nn.Tanh())

    def forward(self, tbl, tbl_len, tbl_split):
        """
        Encode table headers.
            :param tbl: header token list
            :param tbl_len: length of token list (num_table_header, batch)
            :param tbl_split: table header boundary list
        """
        tbl_context = encode_unsorted_batch(self.encoder, tbl, tbl_len)
        # --> (num_table_header, batch, hidden_size * num_directions)
        if self.split_type == 'outcell':
            batch_index = torch.LongTensor(range(tbl_split.data.size(1))).unsqueeze_(
                0).cuda().expand_as(tbl_split.data)
            enc_split = tbl_context[tbl_split.data, batch_index, :]
            enc_left, enc_right = enc_split[:-1], enc_split[1:]
        elif self.split_type == 'incell':
            batch_index = torch.LongTensor(range(tbl_split.data.size(1))).unsqueeze_(
                0).cuda().expand(tbl_split.data.size(0) - 1, tbl_split.data.size(1))
            split_left = (tbl_split.data[:-1] +
                          1).clamp(0, tbl_context.size(0) - 1)
            enc_left = tbl_context[split_left, batch_index, :]
            split_right = (tbl_split.data[1:] -
                           1).clamp(0, tbl_context.size(0) - 1)
            enc_right = tbl_context[split_right, batch_index, :]

        if self.merge_type == 'sub':
            return (enc_right - enc_left)
        elif self.merge_type == 'cat':
            # take half vector for each direction
            half_hidden_size = self.hidden_size // 2
            return torch.cat([enc_right[:, :, :half_hidden_size], enc_left[:, :, half_hidden_size:]], 2)
        elif self.merge_type == 'mlp':
            return self.merge(torch.cat([enc_right, enc_left], 2))


class SeqDecoder(nn.Module):
    def __init__(self, rnn_type, bidirectional_encoder, num_layers, embeddings, input_size, hidden_size, attn_type, attn_hidden, dropout, dropout_i, lock_dropout, dropword, weight_dropout):
        super(SeqDecoder, self).__init__()

        # Basic attributes.
        self.decoder_type = 'rnn'
        self.bidirectional_encoder = bidirectional_encoder
        self.num_layers = num_layers
        self.embeddings = embeddings
        self.input_size = input_size
        self.hidden_size = hidden_size
        if lock_dropout:
            self.word_dropout = table.modules.LockedDropout(dropout_i)
        else:
            self.word_dropout = nn.Dropout(dropout_i)
        self.dropword = dropword

        # Build the RNN.
        self.rnn = _build_rnn(rnn_type, input_size,
                              hidden_size, num_layers, dropout, weight_dropout)

        # Set up the standard attention.
        self.attn = table.modules.GlobalAttention(
            hidden_size, True, attn_type=attn_type, attn_hidden=attn_hidden)

    def forward(self, inp, context, state, parent_index):
        """
        Forward through the decoder.
        Args:
            input (LongTensor): a sequence of input tokens tensors
                                of size (len x batch x nfeats).
            context (FloatTensor): output(tensor sequence) from the encoder
                        RNN of size (src_len x batch x hidden_size).
            state (FloatTensor): hidden state from the encoder RNN for
                                 initializing the decoder.
        Returns:
            outputs (FloatTensor): a Tensor sequence of output from the decoder
                                   of shape (len x batch x hidden_size).
            state (FloatTensor): final hidden state from the decoder.
            attns (dict of (str, FloatTensor)): a dictionary of different
                                type of attention Tensor from the decoder
                                of shape (src_len x batch).
        """
        # Args Check
        assert isinstance(state, RNNDecoderState)
        # END Args Check

        if self.embeddings is not None:
            if self.training and (self.dropword > 0):
                emb = embedded_dropout(
                    self.embeddings, inp, dropout=self.dropword)
            else:
                emb = self.embeddings(inp)
        else:
            emb = inp
        if self.word_dropout is not None:
            emb = self.word_dropout(emb)

        # Run the forward pass of the RNN.
        hidden, outputs, attns, rnn_output, concat_c = self._run_forward_pass(
            emb, context, state, parent_index)

        # Update the state with the result.
        state.update_state(hidden)

        # Concatenates sequence of tensors along a new dimension.
        outputs = torch.stack(outputs)
        attns = torch.stack(attns)

        return outputs, state, attns, rnn_output, concat_c

    def _fix_enc_hidden(self, h):
        """
        The encoder hidden is  (layers*directions) x batch x dim.
        We need to convert it to layers x batch x (directions*dim).
        """
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h

    def init_decoder_state(self, context, enc_hidden):
        return RNNDecoderState(context, self.hidden_size, tuple([self._fix_enc_hidden(enc_hidden[i]) for i in range(len(enc_hidden))]))

    def _run_forward_pass(self, emb, context, state, parent_index):
        """
        Private helper for running the specific RNN forward pass.
        Must be overriden by all subclasses.
        Args:
            input (LongTensor): a sequence of input tokens tensors
                                of size (len x batch x nfeats).
            context (FloatTensor): output(tensor sequence) from the encoder
                        RNN of size (src_len x batch x hidden_size).
            state (FloatTensor): hidden state from the encoder RNN for
                                 initializing the decoder.
        Returns:
            hidden (Variable): final hidden state from the decoder.
            outputs ([FloatTensor]): an array of output of every time
                                     step from the decoder.
            attns (dict of (str, [FloatTensor]): a dictionary of different
                            type of attention Tensor array of every time
                            step from the decoder.
        """

        # Initialize local and return variables.
        outputs = []

        if self.word_dropout is not None:
            emb = self.word_dropout(emb)

        # Run the forward pass of the RNN.
        rnn_output, hidden = self.rnn(emb, state.hidden)

        # Calculate the attention.
        attn_outputs, attn_scores, concat_c = self.attn(
            rnn_output.transpose(0, 1).contiguous(),  # (output_len, batch, d)
            context.transpose(0, 1)                   # (contxt_len, batch, d)
        )

        outputs = attn_outputs    # (input_len, batch, d)

        # Return result.
        return hidden, outputs, attn_scores, rnn_output, concat_c


class DecoderState(object):
    """
    DecoderState is a base class for models, used during translation
    for storing translation states.
    """

    def detach(self):
        """
        Detaches all Variables from the graph
        that created it, making it a leaf.
        """
        for h in self._all:
            if h is not None:
                h.detach_()

    def beam_update(self, idx, positions, beam_size):
        """ Update when beam advances. """
        for e in self._all:
            a, br, d = e.size()
            sentStates = e.view(a, beam_size, br // beam_size, d)[:, :, idx]
            sentStates.data.copy_(
                sentStates.data.index_select(1, positions))


class RNNDecoderState(DecoderState):
    def __init__(self, context, hidden_size, rnnstate):
        """
        Args:
            context (FloatTensor): output from the encoder of size
                                   len x batch x rnn_size.
            hidden_size (int): the size of hidden layer of the decoder.
            rnnstate (Variable): final hidden state from the encoder.
                transformed to shape: layers x batch x (directions*dim).
        """
        if not isinstance(rnnstate, tuple):
            self.hidden = (rnnstate,)
        else:
            self.hidden = rnnstate

    @property
    def _all(self):
        return self.hidden

    def update_state(self, rnnstate):
        if not isinstance(rnnstate, tuple):
            self.hidden = (rnnstate,)
        else:
            self.hidden = rnnstate

    def repeat_beam_size_times(self, beam_size):
        """ Repeat beam_size times along batch dimension. """
        v_list = [Variable(e.data.repeat(1, beam_size, 1), volatile=True)
                  for e in self._all]
        self.hidden = tuple(v_list)

        
class QtCoAttention(nn.Module):
    def __init__(self, rnn_type, bidirectional, num_layers, hidden_size, dropout, weight_dropout, attn_type, attn_hidden):
        super(CoAttention, self).__init__()

        num_directions = 2 if bidirectional else 1
        self.hidden_size = hidden_size
        self.no_pack_padded_seq = False

        self.rnn = _build_rnn(rnn_type, 2 * hidden_size, hidden_size //
                              num_directions, num_layers, dropout, weight_dropout, bidirectional)
        self.attn = table.modules.GlobalAttention(
            hidden_size, False, attn_type=attn_type, attn_hidden=attn_hidden)

    def forward(self, q_all, lengths, tbl_enc, tbl_mask):
        self.attn.applyMask(tbl_mask.data.unsqueeze(0))
        # attention
        emb, _ = self.attn(
            q_all.transpose(0, 1).contiguous(),  # (output_len, batch, d)
            tbl_enc.transpose(0, 1)              # (contxt_len, batch, d)
        )

        # feed to rnn
        if not isinstance(lengths, list):
            lengths = lengths.view(-1).tolist()
        packed_emb = pack(emb, lengths)

        outputs, hidden_t = self.rnn(packed_emb, None)

        outputs = unpack(outputs)[0]

        return hidden_t, outputs
    

class CoAttention(nn.Module):
    def __init__(self, rnn_type, bidirectional, num_layers, hidden_size, context_size, dropout, weight_dropout, attn_type, attn_hidden):
        super(CoAttention, self).__init__()

        if (hidden_size != context_size) and (attn_type != 'mlp'):
            self.linear_context = nn.Linear(
                context_size, hidden_size, bias=False)
            context_size = hidden_size
        else:
            self.linear_context = None

        num_directions = 2 if bidirectional else 1
        self.hidden_size = hidden_size
        self.context_size = context_size
        self.no_pack_padded_seq = False

        self.rnn = _build_rnn(rnn_type, hidden_size + context_size, hidden_size //
                              num_directions, num_layers, dropout, weight_dropout, bidirectional)
        self.attn = table.modules.GlobalAttention(
            hidden_size, False, attn_type=attn_type, attn_hidden=attn_hidden, context_size=context_size)


class QCoAttention(CoAttention):
    def forward(self, q_all, lengths, lay_all, lay):
        self.attn.applyMaskBySeqBatch(lay)
        if self.linear_context is not None:
            lay_all = self.linear_context(lay_all)
        # attention
        emb, _,_ = self.attn(
            q_all.transpose(0, 1).contiguous(),  # (output_len, batch, d)
            lay_all.transpose(0, 1)              # (contxt_len, batch, d)
        )

        # feed to rnn
        if not isinstance(lengths, list):
            lengths = lengths.view(-1).tolist()
        packed_emb = pack(emb, lengths)

        outputs, hidden_t = self.rnn(packed_emb, None)

        outputs = unpack(outputs)[0]

        return hidden_t, outputs


class LayCoAttention(CoAttention):
    def run_rnn_unsorted_batch(self, emb, lengths):
        # sort for pack()
        idx_sorted, tbl_len_sorted, idx_map_back = sort_for_pack(lengths)
        tbl_sorted = emb.index_select(1, Variable(
            torch.LongTensor(idx_sorted).cuda(), requires_grad=False))
        # tbl_context: (seq_len, batch, hidden_size * num_directions)
        packed_emb = pack(tbl_sorted, tbl_len_sorted)
        tbl_context, __ = self.rnn(packed_emb, None)
        tbl_context = unpack(tbl_context)[0]
        # recover the sort for pack()
        v_idx_map_back = Variable(torch.LongTensor(
            idx_map_back).cuda(), requires_grad=False)
        tbl_context = tbl_context.index_select(1, v_idx_map_back)
        return tbl_context

    def forward(self, lay_all, lengths, q_all, q):
        self.attn.applyMaskBySeqBatch(q)
        if self.linear_context is not None:
            q_all = self.linear_context(q_all)
        # attention
        emb, _,_ = self.attn(
            lay_all.transpose(0, 1).contiguous(),  # (output_len, batch, d)
            q_all.transpose(0, 1)              # (contxt_len, batch, d)
        )

        # feed to rnn
        outputs = self.run_rnn_unsorted_batch(emb, lengths)

        return outputs


class CopyGenerator(nn.Module):
    """Generator module that additionally considers copying
    words directly from the source. For each source sentence we have a `src_map` that maps each source word to an index in `tgt_dict` if it known, or else to an extra word. The copy generator is an extended version of the standard generator that computse three values.
    * :math:`p_{softmax}` the standard softmax over `tgt_dict`
    * :math:`p(z)` the probability of instead copying a word from the source, computed using a bernoulli
    * :math:`p_{copy}` the probility of copying a word instead. taken from the attention distribution directly.
    The model returns a distribution over the extend dictionary, computed as
    :math:`p(w) = p(z=1)  p_{copy}(w)  +  p(z=0)  p_{softmax}(w)`
    Args:
       hidden_size (int): size of input representation
       tgt_dict (Vocab): output target dictionary
    """

    def __init__(self, dropout, hidden_size, context_size, tgt_dict, ext_dict, copy_prb):
        super(CopyGenerator, self).__init__()
        self.copy_prb = copy_prb
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, len(tgt_dict))
        if copy_prb == 'hidden':
            self.linear_copy = nn.Linear(hidden_size, 1)
        elif copy_prb == 'hidden_context':
            self.linear_copy = nn.Linear(hidden_size + context_size, 1)
        else:
            raise NotImplementedError
        self.tgt_dict = tgt_dict
        self.ext_dict = ext_dict

    def forward(self, hidden, dec_rnn_output, concat_c, attn, copy_to_ext, copy_to_tgt):
        """
        Compute a distribution over the target dictionary extended by the dynamic dictionary implied by compying source words.
        Args:
           hidden (`FloatTensor`): hidden outputs `[tlen * batch, hidden_size]`
           attn (`FloatTensor`): attn for each `[tlen * batch, src_len]`
           copy_to_ext (`FloatTensor`): A sparse indicator matrix mapping each source word to its index in the "extended" vocab containing. `[src_len, batch]`
           copy_to_tgt (`FloatTensor`): A sparse indicator matrix mapping each source word to its index in the target vocab containing. `[src_len, batch]`
        """
        dec_seq_len = hidden.size(0)
        batch_size = hidden.size(1)
        # -> (targetL_ * batch_, rnn_size)
        hidden = hidden.view(dec_seq_len * batch_size, -1)
        dec_rnn_output = dec_rnn_output.view(dec_seq_len * batch_size, -1)
        concat_c = concat_c.view(dec_seq_len * batch_size, -1)
        # -> (targetL_ * batch_, sourceL_)
        attn = attn.view(dec_seq_len * batch_size, -1)

        # CHECKS
        batch_by_tlen, _ = hidden.size()
        batch_by_tlen_, slen = attn.size()
        slen_, batch = copy_to_ext.size()
        aeq(batch_by_tlen, batch_by_tlen_)
        aeq(slen, slen_)

        hidden = self.dropout(hidden)

        # Original probabilities.
        logits = self.linear(hidden)
        # logits[:, self.tgt_dict.stoi[table.IO.PAD_WORD]] = -float('inf')
        prob_log = F.log_softmax(logits)
        # return prob_log.view(dec_seq_len, batch_size, -1)

        # Probability of copying p(z=1) batch.
        # copy = F.sigmoid(self.linear_copy(hidden))
        if self.copy_prb == 'hidden':
            copy = F.sigmoid(self.linear_copy(dec_rnn_output))
        elif self.copy_prb == 'hidden_context':
            copy = F.sigmoid(self.linear_copy(concat_c))
        else:
            raise NotImplementedError

        def safe_log(v):
            return torch.log(v.clamp(1e-3, 1 - 1e-3))

        # Probibility of not copying: p_{word}(w) * (1 - p(z))
        out_prob_log = prob_log + safe_log(copy).expand_as(prob_log)
        mul_attn = torch.mul(attn, 1.0 - copy.expand_as(attn))
        # copy to extend vocabulary
        copy_to_ext_onehot = onehot(
            copy_to_ext, N=len(self.ext_dict), ignore_index=self.ext_dict.stoi[table.IO.UNK_WORD]).float()
        ext_copy_prob = torch.bmm(mul_attn.view(-1, batch, slen).transpose(0, 1),
                                  copy_to_ext_onehot.transpose(0, 1)).transpose(0, 1).contiguous().view(-1, len(self.ext_dict))
        ext_copy_prob_log = safe_log(ext_copy_prob)

        return torch.cat([prob_log, ext_copy_prob_log], 1).view(dec_seq_len, batch_size, -1)

        # copy to target vocabulary
        copy_to_tgt_onehot = onehot(
            copy_to_tgt, N=len(self.tgt_dict), ignore_index=self.tgt_dict.stoi[table.IO.UNK_WORD]).float()
        tgt_add_copy_prob = torch.bmm(mul_attn.view(-1, batch, slen).transpose(0, 1),
                                      copy_to_tgt_onehot.transpose(0, 1)).transpose(0, 1).contiguous().view(-1, len(self.tgt_dict))
        out_prob = torch.exp(out_prob_log) + tgt_add_copy_prob

        return torch.log(torch.cat([out_prob, ext_copy_prob], 1)).view(dec_seq_len, batch_size, -1)


class ParserModel(nn.Module):
    def __init__(self, q_encoder, tbl_encoder, qt_co_attention, lay_decoder, lay_classifier, q_co_attention, lay_co_attention, predictors, model_opt):
        super(ParserModel, self).__init__()
        
        self.q_encoder = q_encoder
        self.tbl_encoder = tbl_encoder
        self.qt_co_attention = qt_co_attention
        self.lay_decoder = lay_decoder
        self.lay_classifier = lay_classifier
        self.q_co_attention = q_co_attention
        self.lay_co_attention = lay_co_attention
        self.predictors = predictors
        self.opt = model_opt
        

    def enc(self, q, q_len, ent, tbl, tbl_len, tbl_split, tbl_mask):
        q_enc, q_all = self.q_encoder(q, lengths=q_len, ent=ent)
        tbl_enc = self.tbl_encoder(tbl, tbl_len, tbl_split)
        if self.co_attention is not None:
            q_enc, q_all = self.co_attention(q_all, q_len, tbl_enc, tbl_mask)
        # (num_layers * num_directions, batch, hidden_size)
        q_ht, q_ct = q_enc
        batch_size = q_ht.size(1)
        q_ht = q_ht[-1] if not self.opt.brnn else q_ht[-2:].transpose(
            0, 1).contiguous().view(batch_size, -1)

        return q_enc, q_all, tbl_enc, q_ht, batch_size
    
    
    def enc_to_ht(self, q_enc, batch_size):
        # (num_layers * num_directions, batch, hidden_size)
        q_ht, q_ct = q_enc
        q_ht = q_ht[-1] if not self.opt.brnn else q_ht[-2:].transpose(
            0, 1).contiguous().view(batch_size, -1).unsqueeze(0)
        return q_ht
    
    def run_predictors(self, predictors, feat_qhs, feat_qhstbl, tbl_mask): # TODO: lay_len -> lay_mask
        mulit_sql, keyword, col, op, agg, root_tem, des_asc, having, andor = predictors
        mulit_sql_score = mulit_sql(feat_qhs)
        keyword_score = keyword(feat_qhs)
        col_score = col(feat_qhs, feat_qhstbl, tbl_mask) # col_num_score: (B, sql_len, 3), col_score: (B, sql_len, max_col_len)
        op_score = op(feat_qhs)
        agg_score = agg(feat_qhs) # agg_num_score: (B, sql_len, 4), agg_score: (B, sql_len, 5)
        root_tem_score = root_tem(feat_qhs)
        des_asc_score = des_asc(feat_qhs)
        having_score = having(feat_qhs)
        andor_score = andor(feat_qhs)
        
        scores = [mulit_sql_score, keyword_score, col_score, op_score, agg_score,
                  root_tem_score, des_asc_score, having_score, andor_score]
        
        return scores

    def run_decoder(self, decoder, classifier, q, q_all, q_enc, inp, parent_index):
        batch_size = q.size(1)
        decoder.attn.applyMaskBySeqBatch(q)
        q_state = decoder.init_decoder_state(q_all, q_enc)
        dec_all, _, attn_scores, _, _ = decoder(
            inp, q_all, q_state, parent_index)
        dec_seq_len = dec_all.size(0)
        dec_all_c = dec_all.view(dec_seq_len * batch_size, -1)
        dec_out = classifier(dec_all_c)
        dec_out = dec_out.view(dec_seq_len, batch_size, -1)
        return dec_out, attn_scores, dec_all

    def run_copy_decoder(self, decoder, classifier, q, q_all, q_enc, inp, parent_index, copy_to_ext, copy_to_tgt):
        batch_size = q.size(1)
        decoder.attn.applyMaskBySeqBatch(q)
        q_state = decoder.init_decoder_state(q_all, q_enc)
        dec_all, _, attn_scores, dec_rnn_output, concat_c = decoder(
            inp, q_all, q_state, parent_index)
        dec_out = classifier(dec_all, dec_rnn_output,
                             concat_c, attn_scores, copy_to_ext, copy_to_tgt)
        return dec_out, attn_scores

    def forward(self, q, q_len, ent, tbl, tbl_len, tbl_split, tbl_mask, lay, lay_len):
        # encoding
        q_enc, q_all, tbl_enc, q_ht, batch_size = self.enc(
            q, q_len, ent, tbl, tbl_len, tbl_split, tbl_mask)
        
        # layout decoding
        dec_in_lay = lay[:-1]
        lay_out, lay_attn_scores, dec_all = self.run_decoder(
            self.lay_decoder, self.lay_classifier, q, q_all, q_enc, dec_in_lay, lay_parent_index)

        # q_enc: (batch, rnn_size)
        # tbl_enc: (num_table_header, batch, rnn_size)
        # dec_all: (sql_len, batch, rnn_size)
        # q_enc_expand: (sql_len, batch, rnn_size)
        q_enc_expand = q_enc.unsqueeze(0).expand(dec_all.size(0), dec_all.size(1), q_enc.size(1))
        # feat: (batch, sql_len, rnn_size*2)
        feat_qhs = torch.cat((q_enc_expand, dec_all), 2).transpose(0, 1)
        # feat_qhs_expand: (batch, sql_len, num_table_header, rnn_size*2)
        feat_qhs_expand = feat_qhs.unsqueenze(2).expand(
            feat_qhs.size(0), feat_qhs.size(1), tbl_enc.size(0), feat_qhs.size(2))
        # tbl_enc_expand: (batch, sql_len, num_table_header, rnn_size)
        tbl_enc_expand = tbl_enc.transpose(0, 1).unsqueenze(1).expand(
            feat_qhs.size(0), feat_qhs.size(1), tbl_enc.size(0), tbl_enc.size(2))
        # feat_qhstbl: (batch, sql_len, num_table_header, rnn_size*3)
        feat_qhstbl = torch.cat((feat_qhs_expand, tbl_enc), 3)
        
        scores = self.run_predictors(self.predictors, feat_qhs, feat_qhstbl, tbl_mask)
        
        return lay_out, scores, self.predictors

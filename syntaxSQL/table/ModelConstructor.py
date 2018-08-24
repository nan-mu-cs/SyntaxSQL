"""
This file is for models creation, which consults options
and creates each encoder and decoder accordingly.
"""
import torch.nn as nn
import torch.nn.functional as F

import table
import table.Models
import table.modules
from table.Models import ParserModel, RNNEncoder, TableRNNEncoder, SeqDecoder, LayCoAttention, QCoAttention, CopyGenerator, QtCoAttention
import torchtext.vocab
from table.modules.Embeddings import PartUpdateEmbedding
from table.Predictors import AggPredictor, AndOrPredictor, DesAscLimitPredictor, HavingPredictor, KeyWordPredictor, MultiSqlPredictor, OpPredictor, RootTeminalPredictor, ColPredictor


def make_word_embeddings(opt, word_dict, fields):
    word_padding_idx = word_dict.stoi[table.IO.PAD_WORD]
    num_word = len(word_dict)
    emb_word = nn.Embedding(num_word, opt.word_vec_size,
                            padding_idx=word_padding_idx)

    if len(opt.pre_word_vecs) > 0:
        if opt.word_vec_size == 150:
            dim_list = ['100', '50']
        elif opt.word_vec_size == 250:
            dim_list = ['200', '50']
        else:
            dim_list = [str(opt.word_vec_size), ]
        vectors = [torchtext.vocab.GloVe(
            name="6B", cache=opt.pre_word_vecs, dim=it) for it in dim_list]
        word_dict.load_vectors(vectors)
        emb_word.weight.data.copy_(word_dict.vectors)

    if opt.fix_word_vecs:
        # <unk> is 0
        num_special = len(table.IO.special_token_list)
        # zero vectors in the fixed embedding (emb_word)
        emb_word.weight.data[:num_special].zero_()
        emb_special = nn.Embedding(
            num_special, opt.word_vec_size, padding_idx=word_padding_idx)
        emb = PartUpdateEmbedding(num_special, emb_special, emb_word)
        return emb
    else:
        return emb_word
    
    
def make_word_embeddings_wikisql(opt, word_dict, fields):
    word_padding_idx = word_dict.stoi[table.IO.PAD_WORD]
    num_word = len(word_dict)
    emb_word = nn.Embedding(num_word, opt.word_vec_size,
                            padding_idx=word_padding_idx)

    if len(opt.pre_word_vecs) > 0:
        vectors = torchtext.vocab.GloVe(
            name="840B", cache=opt.pre_word_vecs, dim=str(opt.word_vec_size))
        fields["src"].vocab.load_vectors(vectors)
        emb_word.weight.data.copy_(fields["src"].vocab.vectors)

    if opt.fix_word_vecs:
        # <unk> is 0
        num_special = len(table.IO.special_token_list)
        # zero vectors in the fixed embedding (emb_word)
        emb_word.weight.data[:num_special].zero_()
        emb_special = nn.Embedding(
            num_special, opt.word_vec_size, padding_idx=word_padding_idx)
        emb = PartUpdateEmbedding(num_special, emb_special, emb_word)
        return emb
    else:
        return emb_word


def make_embeddings(word_dict, vec_size):
    word_padding_idx = word_dict.stoi[table.IO.PAD_WORD]
    num_word = len(word_dict)
    w_embeddings = nn.Embedding(
        num_word, vec_size, padding_idx=word_padding_idx)
    return w_embeddings

# MODIFY: WIKISQL ENCODER USED HERE
def make_encoder(opt, embeddings, ent_embedding=None):
    # "rnn" or "brnn"
    return RNNEncoder(opt.rnn_type, opt.brnn, opt.enc_layers, opt.rnn_size, opt.dropout, opt.lock_dropout, opt.weight_dropout, embeddings, ent_embedding)


def make_table_encoder(opt, embeddings, ent_embedding=None):
    # "rnn" or "brnn"
    return TableRNNEncoder(make_encoder(opt, embeddings, ent_embedding), opt.split_type, opt.merge_type)


def make_layout_encoder(opt, embeddings):
    return RNNEncoder(opt.rnn_type, opt.brnn, opt.enc_layers, opt.decoder_input_size, opt.dropout, opt.dropout_i, opt.lock_dropout, opt.dropword_enc, opt.weight_dropout, embeddings, None)


def make_predictors(opt):
    agg = AggPredictor(2 * opt.rnn_size, opt.score_size, opt.dropout)
    andor = AndOrPredictor(2 * opt.rnn_size, opt.score_size, opt.dropout)
    des_asc = DesAscLimitPredictor(2 * opt.rnn_size, opt.score_size, opt.dropout)
    having = HavingPredictor(2 * opt.rnn_size, opt.score_size, opt.dropout)
    keyword = KeyWordPredictor(2 * opt.rnn_size, opt.score_size, opt.dropout)
    mulit_sql = MultiSqlPredictor(2 * opt.rnn_size, opt.score_size, opt.dropout)
    op = OpPredictor(2 * opt.rnn_size, opt.score_size, opt.dropout)
    root_tem = RootTeminalPredictor(2 * opt.rnn_size, opt.score_size, opt.dropout)
    col = ColPredictor(3 * opt.rnn_size, opt.score_size, opt.dropout)
    value = None # TODO: to be added
    
    predictors = [mulit_sql, keyword, col, op, agg, root_tem, des_asc, having, andor, value]
    
    return predictors


def make_qt_co_attention(opt):
    if opt.qt_co_attention:
        return QtCoAttention(opt.rnn_type, opt.brnn, opt.enc_layers, opt.rnn_size, opt.dropout, opt.weight_dropout, opt.global_attention, opt.attn_hidden)
    return None


def make_q_co_attention(opt):
    if opt.q_co_attention:
        return QCoAttention(opt.rnn_type, opt.brnn, opt.enc_layers, opt.rnn_size, opt.decoder_input_size, opt.dropout, opt.weight_dropout, 'dot', opt.attn_hidden)
    return None


def make_sql_co_attention(opt):
    if opt.sql_co_attention:
        return LayCoAttention(opt.rnn_type, opt.brnn, opt.enc_layers, opt.decoder_input_size, opt.rnn_size, opt.dropout, opt.weight_dropout, 'mlp', opt.attn_hidden)
    return None


def make_decoder(opt, fields, field_name, embeddings, input_size):
    decoder = SeqDecoder(opt.rnn_type, opt.brnn, opt.dec_layers, embeddings, input_size, opt.rnn_size,
                         opt.global_attention, opt.attn_hidden, opt.dropout, opt.dropout_i, opt.lock_dropout, opt.dropword_dec, opt.weight_dropout)
    if field_name == 'tgt':
        classifier = CopyGenerator(opt.dropout, opt.rnn_size, opt.rnn_size, fields['tgt'].vocab, fields['copy_to_ext'].vocab, opt.copy_prb)
    else:
        classifier = nn.Sequential(
            nn.Dropout(opt.dropout),
            nn.Linear(opt.rnn_size, len(fields[field_name].vocab)),
            nn.LogSoftmax())
    return decoder, classifier


def make_base_model(model_opt, fields, checkpoint=None):
    """
    Args:
        model_opt: the option loaded from checkpoint.
        fields: `Field` objects for the model.
        gpu(bool): whether to use gpu.
        checkpoint: the model gnerated by train phase, or a resumed snapshot
                    model from a stopped training.
    Returns:
        the NMTModel.
    """

    # embedding
    w_embeddings = make_word_embeddings(model_opt, fields["src"].vocab, fields)

    if model_opt.src_type_vec_size > 0:
        src_type_embedding = make_embeddings(
            fields["src_type"].vocab, model_opt.src_type_vec_size) # TODO: add column types and double check ! = -> !=
    else:
        src_type_embedding = None

    # Make question encoder.
    q_encoder = make_encoder(model_opt, w_embeddings, src_type_embedding)
    
    if model_opt.col_type_vec_size > 0:
        col_type_embedding = make_embeddings(
            fields["col_type"].vocab, model_opt.col_type_vec_size) # TODO: add column types and double check ! = -> !=
    else:
        col_type_embedding = None
        
    # TODO: col_type encoder
        
    # Make table encoder.
    tbl_encoder = make_table_encoder(model_opt, w_embeddings, col_type_embedding)
    
    qt_co_attention = make_qt_co_attention(model_opt)

    # TODO: add value copy predictor and modify col/kw modules' loss based on michi
    predictors = make_predictors(model_opt)

    # Make sql history decoder models.
    sql_embeddings = make_embeddings(
        fields['sql_history'].vocab, model_opt.decoder_input_size)
    sql_decoder, sql_classifier = make_decoder(
        model_opt, fields, 'sql_history', sql_embeddings, model_opt.decoder_input_size)

    q_co_attention = make_q_co_attention(model_opt)
    sql_co_attention = make_sql_co_attention(model_opt)

    # Make ParserModel
    model = ParserModel(q_encoder, tbl_encoder, qt_co_attention, sql_decoder, sql_classifier,
                        q_co_attention, sql_co_attention, predictors, model_opt)

    if checkpoint is not None:
        print('Loading model')
        model.load_state_dict(checkpoint['model'])

    model.cuda()

    return model

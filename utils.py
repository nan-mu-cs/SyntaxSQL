import re
import io
import json
import numpy as np
import os
from nltk import word_tokenize
#from lib.dbengine import DBEngine
from word_embedding import gen_x_q_batch

def lower_keys(x):
    if isinstance(x, list):
        return [lower_keys(v) for v in x]
    elif isinstance(x, dict):
        return dict((k.lower(), lower_keys(v)) for k, v in x.iteritems())
    else:
        return x

def get_main_table_name(file_path):
    prefix_pattern = re.compile('(processed/.*/)(.*)(_table\.json)')
    if prefix_pattern.search(file_path):
        return prefix_pattern.search(file_path).group(2)
    return None

def load_data_new(sql_paths, table_paths, use_small=False):
    if not isinstance(sql_paths, list):
        sql_paths = (sql_paths, )
    if not isinstance(table_paths, list):
        table_paths = (table_paths, )
    sql_data = []
    table_data = {}
    for i, SQL_PATH in enumerate(sql_paths):
        if use_small and i >= 2:
            break
        print "Loading data from %s"%SQL_PATH
        with open(SQL_PATH) as inf:
            #for idx, line in enumerate(inf):
                #if use_small and idx >= 500:
                #    break

            sql = lower_keys(json.load(inf))
            sql_data += sql

    for i, TABLE_PATH in enumerate(table_paths):
        if use_small and i >= 2:
            break
        print "Loading data from %s"%TABLE_PATH
        with open(TABLE_PATH) as inf:
            table_data= json.load(inf)
            #for line in inf:
            #    tab = json.loads(line.strip())
            #    table_data[tab['db_id']] = tab

    sql_data_new, table_data_new = process(sql_data, table_data)  # comment out if not on full dataset
    if use_small:
        return sql_data_new[:500], table_data_new
    else:
        return sql_data_new, table_data_new

def load_data(sql_paths, table_paths, use_small=False):
    if not isinstance(sql_paths, list):
        sql_paths = (sql_paths, )
    if not isinstance(table_paths, list):
        table_paths = (table_paths, )
    sql_data = []
    table_data = {}

    max_col_num = 0
    for SQL_PATH in sql_paths:
        print "Loading data from %s"%SQL_PATH
        with open(SQL_PATH) as inf:
            for idx, line in enumerate(inf):
                if use_small and idx >= 1000:
                    break
                print line.strip()
                sql = json.loads(line.strip())
                sql_data.append(sql)

    for TABLE_PATH in table_paths:
        print "Loading data from %s"%TABLE_PATH
        with open(TABLE_PATH) as inf:
            for line in inf:
                tab = json.loads(line.strip())
                table_data[tab[u'id']] = tab

    for sql in sql_data:
        assert sql[u'table_id'] in table_data

    return sql_data, table_data


def load_train_dataset(component):
    return json.load(open("./generated_data/{}.json".format(component)))

def load_dataset(dataset_id, use_small=False):
    if dataset_id == 2:
        print "Loading from new dataset"
        sql_data, table_data = load_data_new(['/data/projects/nl2sql/label_git/nl2sql/data/train.json'],
                 ['/data/projects/nl2sql/label_git/nl2sql/data/tables.json'], use_small=use_small)
        val_sql_data, val_table_data = load_data_new(['/data/projects/nl2sql/label_git/nl2sql/data/dev.json'],
                 ['/data/projects/nl2sql/label_git/nl2sql/data/tables.json'], use_small=use_small)

        test_sql_data, test_table_data = load_data_new(['/data/projects/nl2sql/label_git/nl2sql/data/dev.json'],
                 ['/data/projects/nl2sql/label_git/nl2sql/data/tables.json'], use_small=use_small)

        TRAIN_DB = '../alt/data/train.db'
        DEV_DB = '../alt/data/dev.db'
        TEST_DB = '../alt/data/test.db'
    elif dataset_id == 0:
        print "Loading from original dataset"
        sql_data, table_data = load_data('../alt/data/train_tok.jsonl',
                 '../alt/data/train_tok.tables.jsonl', use_small=use_small)
        val_sql_data, val_table_data = load_data('../alt/ata/dev_tok.jsonl',
                 '../alt/data/dev_tok.tables.jsonl', use_small=use_small)

        test_sql_data, test_table_data = load_data('../alt/data/test_tok.jsonl',
                '../alt/data/test_tok.tables.jsonl', use_small=use_small)
        TRAIN_DB = '../alt/data/train.db'
        DEV_DB = '../alt/data/dev.db'
        TEST_DB = '../alt/data/test.db'
    else:
        print "Loading from re-split dataset"
        sql_data, table_data = load_data('data_resplit/train.jsonl',
                'data_resplit/tables.jsonl', use_small=use_small)
        val_sql_data, val_table_data = load_data('data_resplit/dev.jsonl',
                'data_resplit/tables.jsonl', use_small=use_small)
        test_sql_data, test_table_data = load_data('data_resplit/test.jsonl',
                'data_resplit/tables.jsonl', use_small=use_small)
        TRAIN_DB = 'data_resplit/table.db'
        DEV_DB = 'data_resplit/table.db'
        TEST_DB = 'data_resplit/table.db'

    return sql_data, table_data, val_sql_data, val_table_data,\
            test_sql_data, test_table_data, TRAIN_DB, DEV_DB, TEST_DB

def best_model_name(args, for_load=False):
    new_data = 'new' if args.dataset > 0 else 'old'
    mode = 'sqlnet'
    if for_load:
        use_emb = ''
    else:
        use_emb = '_train_emb' if args.train_emb else ''

    agg_model_name = args.sd + '/%s_%s%s.agg_model'%(new_data,
            mode, use_emb)
    sel_model_name = args.sd + '/%s_%s%s.sel_model'%(new_data,
            mode, use_emb)
    cond_model_name = args.sd + '/%s_%s%s.cond_model'%(new_data,
            mode, use_emb)

    return agg_model_name, sel_model_name, cond_model_name


def to_batch_seq(data, idxes, st, ed):
    q_seq = []
    history = []
    label = []
    for i in range(st, ed):
        q_seq.append(data[idxes[i]]['question_tokens'])
        history.append(data[idxes[i]]["history"])
        label.append(data[idxes[i]]["label"])
    return q_seq,history,label


def to_batch_tables(data,idxes,st,ed):
    col_lens = []
    tables = []
    for i in range(st,ed):
        ts = data[idxes[i]]
        column_names = []
        for col in ts["column_names"]:
            column_names.append(col[1])
        tables.append((column_names,ts["table_names"],ts["column_types"]))
        col_lens.append(len(column_names))
    return tables,col_lens

def to_batch_query(sql_data, idxes, st, ed):
    query_gt = []
    table_ids = []
    for i in range(st, ed):
        # query_gt.append(sql_data[idxes[i]]['sql1'])
        # query_gt.append(sql_data[idxes[i]]['sql'])
        query_gt.append(sql_data[idxes[i]])
        table_ids.append(sql_data[idxes[i]]['table_id'])
    return query_gt, table_ids


# def epoch_train(model, optimizer, batch_size, sql_data, table_data, pred_entry):
#     model.train()
#     perm=np.random.permutation(len(sql_data))
#     cum_loss = 0.0
#     st = 0
#     while st < len(sql_data):
#         ed = st+batch_size if st+batch_size < len(perm) else len(perm)
#
#         q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq = \
#                 to_batch_seq(sql_data, table_data, perm, st, ed)
#         gt_where_seq = None#model.generate_gt_where_seq(q_seq, col_seq, query_seq)
#         gt_sel_seq = [x[1] for x in ans_seq]
#         gt_agg_seq = [x[0] for x in ans_seq]
#         score = model.forward(q_seq, col_seq, col_num, pred_entry,
#                 gt_where=gt_where_seq, gt_cond=gt_cond_seq, gt_sel=gt_sel_seq)
#         loss = model.loss(score, ans_seq, pred_entry, gt_where_seq)
#         cum_loss += loss.data.cpu().numpy()[0]*(ed - st)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         st = ed
#
#     return cum_loss / len(sql_data)

def epoch_train(model, optimizer, batch_size, component,embed_layer,data):
    model.train()
    perm=np.random.permutation(len(data))
    cum_loss = 0.0
    st = 0

    while st < len(data):
        ed = st+batch_size if st+batch_size < len(perm) else len(perm)
        q_seq, history,label = to_batch_seq(data, perm, st, ed)
        q_emb_var, q_len = embed_layer.gen_x_q_batch(q_seq)
        hs_emb_var, hs_len = embed_layer.gen_x_history_batch(history)
        score = 0.0
        loss = 0.0
        if component == "multi_sql":
            #none, except, intersect,union
            #truth B*index(0,1,2,3)
            mkw_emb_var = embed_layer.gen_word_list_embedding(["none","except","intersect","union"])
            score = model.forward(q_emb_var, q_len, hs_emb_var, hs_len, mkw_emb_var=mkw_emb_var, mkw_len=4)
        elif component == "keyword":
            #where group by order by
            # [[0,1,2]]
            kw_emb_var = embed_layer.gen_word_list_embedding(["where", "group by", "order by"])
            score = model.forward(q_emb_var, q_len, hs_emb_var, hs_len, kw_emb_var=kw_emb_var, kw_len=3)
        elif component == "col":
            #col word embedding
            # [[0,1,3]]
            tables,col_lens = to_batch_tables(data,perm,st,ed)
            col_emb_var = embed_layer.gen_table_embedding(tables)
            score = model.forward(q_emb_var, q_len, hs_emb_var, hs_len, col_emb_var=col_emb_var, col_len=col_lens)
        elif component == "op":
            #B*index
            tables, col_lens = to_batch_tables(data, perm, st, ed,gen_gt_col=True)
            col_emb_var = embed_layer.gen_table_embedding(tables)
            gt_col = []
            for i in range(st,ed):
                gt_col.append(data["history"][-1][2])
            score = model.forward(q_emb_var, q_len, hs_emb_var, hs_len, col_emb_var=col_emb_var, col_len=col_lens, gt_col=gt_col)
        elif component == "agg":
            # [[0,1,3]]
            tables, col_lens = to_batch_tables(data, perm, st, ed)
            col_emb_var = embed_layer.gen_table_embedding(tables)
            gt_col = []
            for i in range(st, ed):
                gt_col.append(data["history"][-1][2])
            score = model.forward(q_emb_var, q_len, hs_emb_var, hs_len, col_emb_var=col_emb_var, col_len=col_lens, gt_col=gt_col)
        elif component == "root_tem":
            #B*0/1
            tables, col_lens = to_batch_tables(data, perm, st, ed)
            col_emb_var = embed_layer.gen_table_embedding(tables)
            gt_col = []
            for i in range(st, ed):
                gt_col.append(data["history"][-2][2])
            score = model.forward(q_emb_var, q_len, hs_emb_var, hs_len, col_emb_var=col_emb_var, col_len=col_lens, gt_col=None)
        elif component == "des_asc":
            # B*0/1
            tables, col_lens = to_batch_tables(data, perm, st, ed)
            col_emb_var = embed_layer.gen_table_embedding(tables)
            gt_col = []
            for i in range(st, ed):
                gt_col.append(data["history"][-2][2])
            score = model.forward(q_emb_var, q_len, hs_emb_var, hs_len, col_emb_var=col_emb_var, col_len=col_lens, gt_col=None)
        # score = model.forward(q_seq, col_seq, col_num, pred_entry,
        #         gt_where=gt_where_seq, gt_cond=gt_cond_seq, gt_sel=gt_sel_seq)
        loss = model.loss(score, label)
        cum_loss += loss.data.cpu().numpy()[0]*(ed - st)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        st = ed

    return cum_loss / len(data)


def epoch_acc(model, batch_size, sql_data, table_data, pred_entry, error_print=False, train_flag = False):
    model.eval()
    perm = list(range(len(sql_data)))
    st = 0
    one_acc_num = 0.0
    tot_acc_num = 0.0
    tot_prf = 0.0
    while st < len(sql_data):
        ed = st+batch_size if st+batch_size < len(perm) else len(perm)

        q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq,\
         raw_data = to_batch_seq(sql_data, table_data, perm, st, ed, ret_vis_data=True)
        raw_q_seq = [x[0] for x in raw_data]
        raw_col_seq = [x[1] for x in raw_data]
        query_gt, table_ids = to_batch_query(sql_data, perm, st, ed)
        gt_sel_seq = [x[1] for x in ans_seq]
        gt_ody_seq = [x[9] for x in ans_seq]
        if train_flag:
            score = model.forward(q_seq, col_seq, col_num, pred_entry, gt_sel=gt_sel_seq) #tmep for testing
        else:
            score = model.forward(q_seq, col_seq, col_num, pred_entry)
        pred_queries = model.gen_query(score, q_seq, col_seq,
                raw_q_seq, raw_col_seq, pred_entry, gt_sel = gt_sel_seq, gt_ody = gt_ody_seq)
        one_err, tot_err = model.check_acc(raw_data, pred_queries, query_gt, pred_entry, error_print)

        #tot_prf += prf
        one_acc_num += (ed-st-one_err)
        tot_acc_num += (ed-st-tot_err)

        st = ed
    return tot_acc_num / len(sql_data), one_acc_num / len(sql_data)


def load_para_wemb(file_name):
    f = io.open(file_name, 'r', encoding='utf-8')
    lines = f.readlines()
    ret = {}
    if len(lines[0].split()) == 2:
        lines.pop(0)
    for (n,line) in enumerate(lines):
        info = line.strip().split(' ')
        if info[0].lower() not in ret:
            ret[info[0]] = np.array(map(lambda x:float(x), info[1:]))

    return ret


def load_comb_wemb(fn1, fn2):
    wemb1 = load_word_emb(fn1)
    wemb2 = load_para_wemb(fn2)
    comb_emb = {k: wemb1.get(k, 0) + wemb2.get(k, 0) for k in set(wemb1) | set(wemb2)}

    return comb_emb


def load_concat_wemb(fn1, fn2):
    wemb1 = load_word_emb(fn1)
    wemb2 = load_para_wemb(fn2)
    backup = np.zeros(300, dtype=np.float32)
    comb_emb = {k: np.concatenate((wemb1.get(k, backup), wemb2.get(k, backup)), axis=0) for k in set(wemb1) | set(wemb2)}

    return comb_emb


def load_word_emb(file_name, load_used=False, use_small=False):
    if not load_used:
        print ('Loading word embedding from %s'%file_name)
        ret = {}
        with open(file_name) as inf:
            for idx, line in enumerate(inf):
                if (use_small and idx >= 5000):
                    break
                info = line.strip().split(' ')
                if info[0].lower() not in ret:
                    ret[info[0]] = np.array(map(lambda x:float(x), info[1:]))
        return ret
    else:
        print ('Load used word embedding')
        with open('../alt/glove/word2idx.json') as inf:
            w2i = json.load(inf)
        with open('../alt/glove/usedwordemb.npy') as inf:
            word_emb_val = np.load(inf)
        return w2i, word_emb_val


def process(sql_data, table_data):
    output_tab = {}
    for i in range(len(table_data)):
        table = table_data[i]
        temp = {}
        #temp['col_map'] = [col_name_str.split(" ") if i >= 0 else ['*', 'all'] for i, col_name_str in                    table['column_names']]
        temp['col_map'] = [(table['table_names'][i] + " " + col_name_str).split(" ") if i >= 0 else ['*', 'all'] for i, col_name_str in table['column_names']]

        db_name = table['db_id']
        # print table
        output_tab[db_name] = temp

    output_sql = []
    for i in range(len(sql_data)):
        sql = sql_data[i]
        sql_temp = {}

        # add query metadata
        sql_temp['question'] = sql['question']
        sql_temp['question_tok'] = [w.lower() for w in sql['question_toks']]
        sql_temp['query'] = sql['query']
        sql_temp['query_tok'] = sql['query_toks']
        sql_temp['table_id'] = sql['db_id']

        # process agg/sel
        sql_temp['agg'] = []
        sql_temp['sel'] = []
        gt_sel = sql['sql']['select'][1]
        if len(gt_sel) > 4:
            gt_sel = gt_sel[:4]
        for tup in gt_sel:
            sql_temp['agg'].append(tup[0])
            sql_temp['sel'].append(tup[1][1][1])

        # process where conditions and conjuctions
        sql_temp['cond'] = []
        gt_cond = sql['sql']['where']
        if len(gt_cond) > 0:
            conds = [gt_cond[x] for x in range(len(gt_cond)) if x % 2 == 0]
            for cond in conds:
                curr_cond = []
                curr_cond.append(cond[2][1][1])
                curr_cond.append(cond[1])
                if cond[4] is not None:
                    curr_cond.append([cond[3], cond[4]])
                else:
                    curr_cond.append(cond[3])
                sql_temp['cond'].append(curr_cond)

        sql_temp['conj'] = [gt_cond[x] for x in range(len(gt_cond)) if x % 2 == 1]

        # process group by / having
        sql_temp['group'] = [x[1] for x in sql['sql']['groupby']]
        having_cond = []
        if len(sql['sql']['having']) > 0:
            gt_having = sql['sql']['having'][0] # currently only do first having condition
            having_cond.append(gt_having[2][1][0]) # aggregator
            having_cond.append(gt_having[2][1][1]) # column
            having_cond.append(gt_having[1]) # operator
            if gt_having[4] is not None:
                having_cond.append([gt_having[3], gt_having[4]])
            else:
                having_cond.append(gt_having[3])
        sql_temp['group'].append(having_cond)

        # process order by / limit
        order_aggs = []
        order_cols = []
        order_par = -1
        gt_order = sql['sql']['orderby']
        if len(gt_order) > 0:
            order_aggs = [x[1][0] for x in gt_order[1][:1]] # limit to 1 order by
            order_cols = [x[1][1] for x in gt_order[1][:1]]
            order_par = 1 if gt_order[0] == 'asc' else 0
        sql_temp['order'] = [order_aggs, order_cols, order_par]

        # process intersect/except/union
        sql_temp['special'] = 0
        if sql['sql']['intersect'] is not None:
            sql_temp['special'] = 1
        elif sql['sql']['except'] is not None:
            sql_temp['special'] = 2
        elif sql['sql']['union'] is not None:
            sql_temp['special'] = 3

        output_sql.append(sql_temp)
    return output_sql, output_tab

################################
# val: number(float)/string(str)/sql(dict)
# col_unit: (agg_id, col_id, isDistinct(bool))
# val_unit: (unit_op, col_unit1, col_unit2)
# table_unit: (table_type, col_unit/sql)
# cond_unit: (not_op, op_id, val_unit, val1, val2)
# condition: [cond_unit1, 'and'/'or', cond_unit2, ...]
# sql {
#   'select': (isDistinct(bool), [(agg_id, val_unit), (agg_id, val_unit), ...])
#   'from': {'table_units': [table_unit1, table_unit2, ...], 'conds': condition}
#   'where': condition
#   'groupBy': [col_unit1, col_unit2, ...]
#   'orderBy': ('asc'/'desc', [val_unit1, val_unit2, ...])
#   'having': condition
#   'limit': None/limit value
#   'intersect': None/sql
#   'except': None/sql
#   'union': None/sql
# }
################################

import os, sys
import json
import sqlite3
import traceback

sys.path.append("/data/projects/nl2sql/datasets")
from process_sql import tokenize, get_schema, get_tables_with_alias, Schema, get_sql


ROOTPATH = "/data/projects/nl2sql/database/"

CLAUSE_KEYWORDS = ('select', 'from', 'where', 'group', 'order', 'limit', 'intersect', 'union', 'except')
JOIN_KEYWORDS = ('join', 'on', 'as')

WHERE_OPS = ('not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists')
UNIT_OPS = ('none', '-', '+', "*", '/')
AGG_OPS = ('none', 'max', 'min', 'count', 'sum', 'avg')
TABLE_TYPE = {
    'sql': "sql",
    'table_unit': "table_unit",
}

COND_OPS = ('and', 'or')
SQL_OPS = ('intersect', 'union', 'except')
ORDER_OPS = ('desc', 'asc')


HARDNESS = {
    "component1": ('where', 'group', 'order', 'limit', 'join', 'or', 'like'),
    "component2": ('except', 'union', 'intersect')
}


def condition_has_or(conds):
    return 'or' in conds[1::2]


def condition_has_like(conds):
    return WHERE_OPS.index('like') in [cond_unit[1] for cond_unit in conds[::2]]


def condition_has_sql(conds):
    for cond_unit in conds[::2]:
        val1, val2 = cond_unit[3], cond_unit[4]
        if val1 is not None and type(val1) is dict:
            return True
        if val2 is not None and type(val2) is dict:
            return True
    return False


def val_has_op(val_unit):
    return val_unit[0] != UNIT_OPS.index('none')


def has_agg(unit):
    return unit[0] != AGG_OPS.index('none')


def accuracy(count, total):
    if count == total:
        return 1
    # return float(count) / total
    return 0

def recall(count, total):
    if count == total:
        return 1
    return 0
    # return float(count) / total


def F1(acc, rec):
    if (acc + rec) == 0:
        return 0
    return (2. * acc * rec) / (acc + rec)


def get_scores(count, pred_total, label_total):
    # acc = accuracy(count, pred_total)
    # rec = recall(count, label_total)
    if pred_total != label_total:
        return 0,0,0
    elif count == pred_total:
        return 1,1,1
    return 0,0,0
        # else:
            # acc = 0
            # rec = 0
    # f1 = F1(acc, rec)
    # return acc, rec, f1


def eval_sel(pred, label):
    pred_sel = pred['select'][1]
    label_sel = label['select'][1]
    label_wo_agg = [unit[1] for unit in label_sel]
    pred_total = len(pred_sel)
    label_total = len(label_sel)
    cnt = 0
    cnt_wo_agg = 0

    for unit in pred_sel:
        if unit in label_sel:
            cnt += 1
        if unit[1] in label_wo_agg:
            cnt_wo_agg += 1

    return label_total, pred_total, cnt, cnt_wo_agg


def eval_where(pred, label):
    pred_conds = [unit[:3] for unit in pred['where'][::2]]  # ignore value
    label_conds = [unit[:3] for unit in label['where'][::2]]  # ignore value
    label_wo_agg = [unit[2] for unit in label_conds]
    # print(pred_conds)
    # print(label_conds)
    pred_total = len(pred_conds)
    label_total = len(label_conds)
    cnt = 0
    cnt_wo_agg = 0

    for unit in pred_conds:
        if unit in label_conds:
            cnt += 1
        if unit[2] in label_wo_agg:
            cnt_wo_agg += 1

    return label_total, pred_total, cnt, cnt_wo_agg


def eval_group(pred, label):
    pred_cols = [unit[1] for unit in pred['groupBy']]
    label_cols = [unit[1] for unit in label['groupBy']]
    pred_total = len(pred_cols)
    label_total = len(label_cols)
    cnt = 0
    pred_cols = [pred.split(".")[1] if "." in pred else pred for pred in pred_cols]
    label_cols = [label.split(".")[1] if "." in label else label for label in label_cols]
    # print(pred_cols)
    # print(label_cols)
    for col in pred_cols:
        if col in label_cols:
            cnt += 1
    return label_total, pred_total, cnt


def eval_having(pred, label):
    pred_total = label_total = cnt = 0
    if len(pred['groupBy']) > 0:
        pred_total = 1
    if len(label['groupBy']) > 0:
        label_total = 1

    pred_cols = [unit[1] for unit in pred['groupBy']]
    label_cols = [unit[1] for unit in label['groupBy']]
    if pred_total == label_total == 1 and \
        pred_cols == label_cols and pred['having'] == label['having']:
        cnt = 1

    return label_total, pred_total, cnt


def eval_order(pred, label):
    pred_total = label_total = cnt = 0
    if len(pred['orderBy']) > 0:
        pred_total = 1
    if len(label['orderBy']) > 0:
        label_total = 1
    # print("pred:{}".format(pred))
    # print("label:{}\n".format(label))
    if len(label['orderBy']) > 0 and pred['orderBy'] == label['orderBy'] and \
            ((pred['limit'] is None and label['limit'] is None) or (pred['limit'] is not None and label['limit'] is not None)):
        cnt = 1
    return label_total, pred_total, cnt


def eval_and_or(pred, label):
    # pred_ao = pred['from']['conds'][1::2] + pred['where'][1::2] + pred['having'][1::2]
    # label_ao = label['from']['conds'][1::2] + label['where'][1::2] + label['having'][1::2]
    pred_ao = pred['where'][1::2]
    label_ao = label['where'][1::2]
    pred_total = len(pred_ao)
    label_total = len(label_ao)
    # print("pred:{}".format(pred_ao))
    # print("gold:{}".format(label_ao))
    cnt = 0
    pred_ao = set(pred_ao)
    label_ao = set(label_ao)
    # print("pred:{}".format(pred_ao))
    # print("gold:{}".format(label_ao))
    if pred_ao == label_ao:
        return 1,1,1
    return len(pred_ao),len(label_ao),0
    # num_pred_a = len([token for token in pred_ao if token == 'and'])
    # num_label_a = len([token for token in label_ao if token == 'and'])
    # cnt += min(num_pred_a, num_label_a)
    #
    # num_pred_o = len([token for token in pred_ao if token=='or'])
    # num_label_o = len([token for token in label_ao if token=='or'])
    # cnt += min(num_pred_o, num_label_o)

    # return label_total, pred_total, cnt


def get_nestedSQL(sql):
    nested = []
    for cond_unit in sql['from']['conds'][::2] + sql['where'][::2] + sql['having'][::2]:
        if type(cond_unit[3]) is dict:
            nested.append(cond_unit[3])
        if type(cond_unit[4]) is dict:
            nested.append(cond_unit[4])
    if sql['intersect'] is not None:
        nested.append(sql['intersect'])
    if sql['except'] is not None:
        nested.append(sql['except'])
    if sql['union'] is not None:
        nested.append(sql['union'])
    return nested


def eval_IUEN(pred, label):
    pred_nested = get_nestedSQL(pred)
    label_nested = get_nestedSQL(label)
    pred_total = len(pred_nested)
    label_total = len(label_nested)
    cnt = 0

    for sql in pred_nested:
        if sql in label_nested:
            cnt += 1
    return label_total, pred_total, cnt


def get_keywords(sql):
    res = set()
    if len(sql['where']) > 0:
        res.add('where')
    if len(sql['groupBy']) > 0:
        res.add('group')
    if len(sql['having']) > 0:
        res.add('having')
    if len(sql['orderBy']) > 0:
        res.add(sql['orderBy'][0])
        res.add('order')
    if sql['limit'] is not None:
        res.add('limit')
    if sql['except'] is not None:
        res.add('except')
    if sql['union'] is not None:
        res.add('union')
    if sql['intersect'] is not None:
        res.add('intersect')

    # or keyword
    ao = sql['from']['conds'][1::2] + sql['where'][1::2] + sql['having'][1::2]
    if len([token for token in ao if token == 'or']) > 0:
        res.add('or')

    cond_units = sql['from']['conds'][::2] + sql['where'][::2] + sql['having'][::2]
    # not keyword
    if len([cond_unit for cond_unit in cond_units if cond_unit[0]]) > 0:
        res.add('not')

    # in keyword
    if len([cond_unit for cond_unit in cond_units if cond_unit[1] == WHERE_OPS.index('in')]) > 0:
        res.add('in')

    # like keyword
    if len([cond_unit for cond_unit in cond_units if cond_unit[1] == WHERE_OPS.index('like')]) > 0:
        res.add('like')

    return res


def eval_keywords(pred, label):
    pred_keywords = get_keywords(pred)
    label_keywords = get_keywords(label)
    pred_total = len(pred_keywords)
    label_total = len(label_keywords)
    cnt = 0

    for k in pred_keywords:
        if k in label_keywords:
            cnt += 1
    return label_total, pred_total, cnt


def count_agg(units):
    return len([unit for unit in units if has_agg(unit)])


def count_component1(sql):
    count = 0
    if len(sql['where']) > 0:
        count += 1
    if len(sql['groupBy']) > 0:
        count += 1
    if len(sql['orderBy']) > 0:
        count += 1
    if sql['limit'] is not None:
        count += 1
    if len(sql['from']['table_units']) > 0:  # JOIN
        count += len(sql['from']['table_units']) - 1

    ao = sql['from']['conds'][1::2] + sql['where'][1::2] + sql['having'][1::2]
    count += len([token for token in ao if token == 'or'])
    cond_units = sql['from']['conds'][::2] + sql['where'][::2] + sql['having'][::2]
    count += len([cond_unit for cond_unit in cond_units if cond_unit[1] == WHERE_OPS.index('like')])

    return count


def count_component2(sql):
    nested = get_nestedSQL(sql)
    return len(nested)


def count_others(sql):
    count = 0
    # number of aggregation
    agg_count = count_agg(sql['select'][1])
    agg_count += count_agg(sql['where'][::2])
    agg_count += count_agg(sql['groupBy'])
    if len(sql['orderBy']) > 0:
        agg_count += count_agg([unit[1] for unit in sql['orderBy'][1] if unit[1]] +
                            [unit[2] for unit in sql['orderBy'][1] if unit[2]])
    agg_count += count_agg(sql['having'])
    if agg_count > 1:
        count += 1

    # number of select columns
    if len(sql['select'][1]) > 1:
        count += 1

    # number of where conditions
    if len(sql['where']) > 1:
        count += 1

    # number of group by clauses
    if len(sql['groupBy']) > 1:
        count += 1

    return count


class Evaluator:
    """A simple evaluator"""
    def __init__(self):
        self.partial_scores = None

    def eval_hardness(self, sql):
        count_comp1_ = count_component1(sql)
        count_comp2_ = count_component2(sql)
        count_others_ = count_others(sql)

        if count_comp1_ <= 1 and count_others_ == 0 and count_comp2_ == 0:
            return "easy"
        elif (count_others_ <= 2 and count_comp1_ <= 1 and count_comp2_ == 0) or \
                (count_comp1_ <= 2 and count_others_ < 2 and count_comp2_ == 0):
            return "medium"
        elif (count_others_ > 2 and count_comp1_ <= 2 and count_comp2_ == 0) or \
                (2 < count_comp1_ <= 3 and count_others_ <= 2 and count_comp2_ == 0) or \
                (count_comp1_ <= 1 and count_others_ == 0 and count_comp2_ <= 1):
            return "hard"
        else:
            return "extra"

    def eval_exact_match(self, pred, label):
        partial_scores = self.eval_partial_match(pred, label)
        self.partial_scores = partial_scores
        # for _, score in partial_scores.items():
        #     if score['acc'] != 1:
        #         return 0
        # print("pred:{}".format(pred))
        # print("label:{}".format(label))
        # print("\n")
        for _, score in partial_scores.items():
            if score['f1'] != 1:
                return 0
        return 1

    def eval_partial_match(self, pred, label):
        res = {}

        label_total, pred_total, cnt, cnt_wo_agg = eval_sel(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['select'] = {'acc': acc, 'rec': rec, 'f1': f1,'label_total':label_total,'pred_total':pred_total}
        acc, rec, f1 = get_scores(cnt_wo_agg, pred_total, label_total)
        res['select(no AGG)'] = {'acc': acc, 'rec': rec, 'f1': f1,'label_total':label_total,'pred_total':pred_total}

        label_total, pred_total, cnt, cnt_wo_agg = eval_where(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['where'] = {'acc': acc, 'rec': rec, 'f1': f1,'label_total':label_total,'pred_total':pred_total}
        acc, rec, f1 = get_scores(cnt_wo_agg, pred_total, label_total)
        res['where(no OP)'] = {'acc': acc, 'rec': rec, 'f1': f1,'label_total':label_total,'pred_total':pred_total}

        label_total, pred_total, cnt = eval_group(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['group(no Having)'] = {'acc': acc, 'rec': rec, 'f1': f1,'label_total':label_total,'pred_total':pred_total}

        label_total, pred_total, cnt = eval_having(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['group'] = {'acc': acc, 'rec': rec, 'f1': f1,'label_total':label_total,'pred_total':pred_total}

        label_total, pred_total, cnt = eval_order(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['order'] = {'acc': acc, 'rec': rec, 'f1': f1,'label_total':label_total,'pred_total':pred_total}

        label_total, pred_total, cnt = eval_and_or(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['and/or'] = {'acc': acc, 'rec': rec, 'f1': f1,'label_total':label_total,'pred_total':pred_total}

        label_total, pred_total, cnt = eval_IUEN(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['IUEN'] = {'acc': acc, 'rec': rec, 'f1': f1,'label_total':label_total,'pred_total':pred_total}

        label_total, pred_total, cnt = eval_keywords(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['keywords'] = {'acc': acc, 'rec': rec, 'f1': f1,'label_total':label_total,'pred_total':pred_total}

        return res


def isValidSQL(sql, db):
    conn = sqlite3.connect(db)
    cursor = conn.cursor()
    try:
        cursor.execute(sql)
    except:
        return False
    return True


def print_scores(scores):
    levels = ['easy', 'medium', 'hard', 'extra', 'all']
    partial_types = ['select', 'select(no AGG)', 'where', 'where(no OP)', 'group(no Having)',
                     'group', 'order', 'and/or', 'IUEN', 'keywords']

    print "{:20} {:20} {:20} {:20} {:20} {:20}".format("", *levels)
    counts = [scores[level]['count'] for level in levels]
    print "{:20} {:<20d} {:<20d} {:<20d} {:<20d} {:<20d}".format("count", *counts)
    exact_scores = [scores[level]['exact'] for level in levels]
    print "{:20} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f}".format("exact", *exact_scores)

    print '-------------------EXEC ACCURACY-----------------------'
    this_scores = [scores[level]['exec'] for level in levels]
    print "{:20} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f}".format("exec", *this_scores)
    
    print '---------------------ACCURACY--------------------------'
    for type_ in partial_types:
        this_scores = [scores[level]['partial'][type_]['acc'] for level in levels]
        print "{:20} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f}".format(type_, *this_scores)

    print '---------------------Recall--------------------------'
    for type_ in partial_types:
        this_scores = [scores[level]['partial'][type_]['rec'] for level in levels]
        print "{:20} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f}".format(type_, *this_scores)

    print '---------------------F1--------------------------'
    for type_ in partial_types:
        this_scores = [scores[level]['partial'][type_]['f1'] for level in levels]
        print "{:20} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f}".format(type_, *this_scores)


def evaluate(gold, predict):
    with open(gold) as f:
        glist = [l.strip().split('\t') for l in f.readlines() if len(l.strip()) > 0]

    with open(predict) as f:
        plist = [l.strip().split('\t') for l in f.readlines() if len(l.strip()) > 0]

    evaluator = Evaluator()

    levels = ['easy', 'medium', 'hard', 'extra', 'all']
    partial_types = ['select', 'select(no AGG)', 'where', 'where(no OP)', 'group(no Having)',
                     'group', 'order', 'and/or', 'IUEN', 'keywords']
    entries = []
    scores = {}

    for level in levels:
        scores[level] = {'count': 0, 'partial': {}, 'exact': 0.}
        scores[level]['exec'] = 0
        for type_ in partial_types:
            scores[level]['partial'][type_] = {'acc': 0., 'rec': 0., 'f1': 0.,'acc_count':0,'rec_count':0}

    eval_err_num = 0
    for p, g in zip(plist, glist):
        # print(p)
        # print(g)
        # p_str, _ = p
        p_str = p[0]
        #print(g)
        g_str, db = g
        db = os.path.join(ROOTPATH, db, db + ".sqlite")
        schema = Schema(get_schema(db))
        g_sql = get_sql(schema, g_str)
        hardness = evaluator.eval_hardness(g_sql)
        # if hardness == "medium":
        #     print("medium:{}\n".format(g_str))
        scores[hardness]['count'] += 1
        scores['all']['count'] += 1

        # print("p:{}".format(p_str))

        # if not isValidSQL(p_str, db):
        #     entries.append({
        #         'predictSQL': p_str,
        #         'goldSQL': g_str,
        #         'hardness': hardness,
        #         'exact': None,
        #         'partial': None
        #     })
        #     continue
        # print("p:{}".format(p_str))
        try:
            p_sql = get_sql(schema, p_str)
            #print(p_str)
        except:
            traceback.print_exc()
            # p_sql = {}
            # print("p:{}".format(p_str))
            # print("gold: {}".format(g_str))
            # print("")

            p_sql = {
            "except": None,
            "from": {
                "conds": [],
                "table_units": [
                    [
                        "table_unit",
                        1
                    ]
                ]
            },
            "groupBy": [],
            "having": [],
            "intersect": None,
            "limit": None,
            "orderBy": [],
            "select": [
                False,
                []
            ],
            "union": None,
            "where": []
            }
            eval_err_num += 1
            print("eval_err_num:{}".format(eval_err_num))

        #scores[hardness]['count'] += 1
        #scores['all']['count'] += 1
        exec_score = eval_exec_match(db, p_str, g_str, p_sql, g_sql)
        if exec_score:
            scores[hardness]['exec'] += 1
        exact_score = evaluator.eval_exact_match(p_sql, g_sql)
        partial_scores = evaluator.partial_scores
        if exact_score == 0:
            print("{} pred: {}".format(hardness,p_str))
            print("{} gold: {}".format(hardness,g_str))
            print("")
        scores[hardness]['exact'] += exact_score
        scores['all']['exact'] += exact_score
        for type_ in partial_types:
            # if type_ == "IUEN" and partial_scores[type_]['acc'] == 1:
            #     print("pred: {}".format(p_str))
            #     print("gold: {}".format(g_str))
            #     print("")
            # print(scores[hardness]['partial'][type_])
            # print(partial_scores[type_]['acc'])
            if partial_scores[type_]['pred_total'] > 0:
                # if type_ == "group" and partial_scores[type_]['acc'] == 0:
                #     print("pred:{}".format(p_str))
                #     print("gold:{}\n".format(g_str))
                scores[hardness]['partial'][type_]['acc'] += partial_scores[type_]['acc']
                scores[hardness]['partial'][type_]['acc_count'] += 1
            if partial_scores[type_]['label_total'] > 0:
                scores[hardness]['partial'][type_]['rec'] += partial_scores[type_]['rec']
                scores[hardness]['partial'][type_]['rec_count'] += 1
            scores[hardness]['partial'][type_]['f1'] += partial_scores[type_]['f1']
            if partial_scores[type_]['pred_total'] > 0:
                scores['all']['partial'][type_]['acc'] += partial_scores[type_]['acc']
                scores['all']['partial'][type_]['acc_count'] += 1
            if partial_scores[type_]['label_total'] > 0:
                scores['all']['partial'][type_]['rec'] += partial_scores[type_]['rec']
                scores['all']['partial'][type_]['rec_count'] += 1
            scores['all']['partial'][type_]['f1'] += partial_scores[type_]['f1']

        entries.append({
            'predictSQL': p_str,
            'goldSQL': g_str,
            'hardness': hardness,
            'exact': exact_score,
            'partial': partial_scores
        })

    for level in levels:
        scores[level]['exact'] /= scores[level]['count']
        scores[level]['exec'] /= scores[level]['count']
        for type_ in partial_types:
            # print("part:{} level:{} acc:{} acc_count:{} rec:{} rec_count:{}".format(type_,level,scores[level]['partial'][type_]['acc'],scores[level]['partial'][type_]['acc_count'],scores[level]['partial'][type_]['rec'],scores[level]['partial'][type_]['rec_count']))
            if scores[level]['partial'][type_]['acc_count'] == 0:
                scores[level]['partial'][type_]['acc'] = 0
            else:
                scores[level]['partial'][type_]['acc'] = scores[level]['partial'][type_]['acc']/scores[level]['partial'][type_]['acc_count']*1.0
            if scores[level]['partial'][type_]['rec_count'] == 0:
                scores[level]['partial'][type_]['rec'] = 0
            else:
                scores[level]['partial'][type_]['rec'] = scores[level]['partial'][type_]['rec']/scores[level]['partial'][type_]['rec_count']*1.0
            if scores[level]['partial'][type_]['acc'] == 0 and scores[level]['partial'][type_]['rec'] == 0:
                scores[level]['partial'][type_]['f1'] = 1
            else:
                scores[level]['partial'][type_]['f1'] = \
                    2.0*scores[level]['partial'][type_]['acc']*scores[level]['partial'][type_]['rec']/( scores[level]['partial'][type_]['rec']+ scores[level]['partial'][type_]['acc'])
        # scores['all']['partial'][type_]['acc'] = scores['all']['partial'][type_]['acc']/scores['all']['partial'][type_]['acc_count']*1.0
        # scores['all']['partial'][type_]['rec'] = scores['all']['partial'][type_]['rec']/scores['all']['partial'][type_]['rec_count']*1.0
        # scores['all']['partial'][type_]['f1'] /= \
        #     2.0 * scores['all']['partial'][type_]['acc_count'] * scores['all']['partial'][type_]['rec_count'] / (
        #     scores['all']['partial'][type_]['rec_count'] + scores['all']['partial'][type_]['acc_count'])

    print_scores(scores)

    # with open('scores.json', 'wb') as f:
    #     json.dump(obj=entries, fp=f, indent=4)


def eval_exec_match(db, p_str, g_str, pred, gold):
    """
    return 1 if the values between prediction and gold are matching
    in the corresponding index. Currently not support multiple col_unit(pairs).
    """
    db = os.path.join(ROOTPATH, db, db + ".sqlite")
    conn = sqlite3.connect(db)
    cursor = conn.cursor()
    try:
        cursor.execute(p_str)
        p_res = cursor.fetchall()
    except:
        return False

    cursor.execute(g_str)
    q_res = cursor.fetchall()

    def res_map(res, val_units):
        rmap = {}
        for idx, val_unit in enumerate(val_units):
            key = tuple(val_unit[1]) if not val_unit[2] else (val_unit[0], tuple(val_unit[1]), tuple(val_unit[2]))
            rmap[key] = [r[idx] for r in res]
        return rmap

    p_val_units = [unit[1] for unit in pred['select'][1]]
    q_val_units = [unit[1] for unit in gold['select'][1]]
    return res_map(p_res, p_val_units) == res_map(q_res, q_val_units)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_path',type=str)
    args = parser.parse_args()

    gold = "/data/projects/nl2sql/datasets/data/gold.sql"
    # pred = "./results/ours_fullhs_result.txt"
    pred = args.pred_path

    evaluate(gold, pred)

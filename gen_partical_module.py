import json

train_data_path = "./data/train.json"
table_data_path = "./data/tables.json"
train_data = json.load(open(train_data_path))


WHERE_OPS = ('not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists')
# SQL_OPS = ('none','intersect', 'union', 'except')
SQL_OPS = {
    'none':0,
    'intersect':1,
    'union':2,
    'except':3
}
KW_DICT = {
    'where':0,
    'groupBy':1,
    'orderBy':2
}
ORDER_OPS = ('desc', 'asc')
AGG_OPS = ('none', 'max', 'min', 'count', 'sum', 'avg')
def index_to_column_name(index,table):
    column_name = table["column_names"][index][1]
    table_index = table["column_names"][index][0]
    table_name = table["table_names"][table_index]
    return table_name,column_name,index

class MultiSqlPredictor:
    def __init__(self, question, sql, history):
        self.sql = sql
        self.question = question
        self.history = history
        self.keywords = ('intersect', 'except', 'union')

    def generate_output(self):
        for key in self.sql:
            if key in self.keywords and self.sql[key]:
                return self.history + ['ROOT'], key, self.sql[key]
        return self.history + ['ROOT'], 'none', self.sql


class KeyWordPredictor:
    def __init__(self, question, sql, history):
        self.sql = sql
        self.question = question
        self.history = history
        self.keywords = ('select', 'where', 'groupBy', 'orderBy', 'limit', 'having')

    def generate_output(self):
        sql_keywords = []
        for key in self.sql:
            if key in self.keywords and self.sql[key]:
                sql_keywords.append(key)
        return self.history, [len(sql_keywords), sql_keywords], self.sql


class ColPredictor:
    def __init__(self, question, sql, table,history):
        self.sql = sql
        self.question = question
        self.history = history
        self.table = table
        self.keywords = ('select', 'where', 'groupBy', 'orderBy', 'having')

    def generate_output(self):
        ret = []
        for key in self.sql:
            if key in self.keywords and self.sql[key]:
                cols = []
                sqls = []
                if key == 'groupBy':
                    sql_cols = self.sql[key]
                    for col in sql_cols:
                        cols.append((index_to_column_name(col[1],self.table), col[2]))
                        sqls.append(col)
                elif key == 'orderBy':
                    sql_cols = self.sql[key][1]
                    for col in sql_cols:
                        cols.append((index_to_column_name(col[1][1],self.table), col[1][2]))
                        sqls.append(col)
                elif key == 'select':
                    sql_cols = self.sql[key][1]
                    for col in sql_cols:
                        cols.append((index_to_column_name(col[1][1][1],self.table), col[1][1][2]))
                        sqls.append(col)
                elif key == 'where' or key == 'having':
                    sql_cols = self.sql[key]
                    for col in sql_cols:
                        if not isinstance(col,list):
                            continue
                        try:
                            cols.append((index_to_column_name(col[2][1][1],self.table), col[2][1][2]))
                        except:
                            print("Key:{} Col:{} Question:{}".format(key,col,self.question))
                        sqls.append(col)
                ret.append((
                    self.history + [key], (len(cols), cols), sqls
                ))
        return ret
        # ret.append(history+[key],)


class OpPredictor:
    def __init__(self, question, sql, history):
        self.sql = sql
        self.question = question
        self.history = history
        # self.keywords = ('select', 'where', 'groupBy', 'orderBy', 'having')

    def generate_output(self):
        return self.history, self.sql[1], (self.sql[3], self.sql[4])


class AggPredictor:
    def __init__(self, question, sql, history):
        self.sql = sql
        self.question = question
        self.history = history

    def generate_output(self):
        label = 0
        key = self.history[-2]
        if key == 'select':
            label = self.sql[1][1][0]
        elif key == 'orderBy':
            label = self.sql[1][0]
        elif key == 'having':
            label = self.sql[2][1][0]
        return self.history, label


# class RootTemPredictor:
#     def __init__(self, question, sql):
#         self.sql = sql
#         self.question = question
#         self.keywords = ('intersect', 'except', 'union')
#
#     def generate_output(self):
#         for key in self.sql:
#             if key in self.keywords:
#                 return ['ROOT'], key, self.sql[key]
#         return ['ROOT'], 'none', self.sql


class DesAscPredictor:
    def __init__(self, question, sql, table,history):
        self.sql = sql
        self.question = question
        self.history = history
        self.table = table

    def generate_output(self):
        for key in self.sql:
            if key == "orderBy" and self.sql[key]:
                self.history.append(key)
                try:
                    col = self.sql[key][1][0][1][1]
                except:
                    print("question:{} sql:{}".format(self.question,self.sql))
                self.history.append(index_to_column_name(col,self.table))
                self.history.append(self.sql[key][1][0][1][0])
                if self.sql[key][0] == "asc" and self.sql["limit"]:
                    label = 0
                elif self.sql[key][0] == "asc" and not self.sql["limit"]:
                    label = 1
                elif self.sql[key][0] == "desc" and self.sql["limit"]:
                    label = 2
                else:
                    label = 3
                return self.history, self.sql[key][0]


def parser_item(question_tokens,sql,table,history, dataset):
    # try:
    #     question_tokens = item['question_toks']
    # except:
    #     print(item)
    # sql = item['sql']
    table_schema = [
        table["table_names"],
        table["column_names"],
        table["column_types"]
    ]
    history, label, sql = MultiSqlPredictor(question_tokens, sql, history).generate_output()
    dataset['multi_sql_dataset'].append({
        "question_tokens": question_tokens,
        "ts":table_schema,
        "history": history[:],
        "label": SQL_OPS[label]
    })
    history.append(label)
    history, label, sql = KeyWordPredictor(question_tokens, sql, history).generate_output()
    label_idxs = []
    for item in label[1]:
        if item in KW_DICT:
            label_idxs.append(KW_DICT[item])
    label_idxs.sort()
    dataset['keyword_dataset'].append({
        "question_tokens": question_tokens,
        "ts":table_schema,
        "history": history[:],
        "keywords_num": label[0],
        "label": label_idxs
    })
    orderby_ret = DesAscPredictor(question_tokens, sql, table,history).generate_output()
    if orderby_ret:
        dataset['des_asc_dataset'].append({
            "question_tokens": question_tokens,
            "ts": table_schema,
            "history": orderby_ret[0][:],
            "label": orderby_ret[1]
        })
    col_ret = ColPredictor(question_tokens, sql, table,history).generate_output()
    agg_candidates = []
    op_candidates = []
    for h, l, s in col_ret:
        if l[0] == 0:
            print("Warning: predicted 0 columns!")
            continue
        dataset['col_dataset'].append({
            "question_tokens": question_tokens,
            "ts": table_schema,
            "history": h[:],
            "cols_num": l[0],
            "label": [val[0][2] for val in l[1]]
        })
        for col, sql_item in zip(l[1], s):
            if h[-1] in ('where', 'having'):
                op_candidates.append((h + [col[0]], sql_item))
            if h[-1] in ('select', 'orderBy', 'having'):
                agg_candidates.append((h + [col[0]], sql_item))
    for h, sql_item in op_candidates:
        _, label, s = OpPredictor(question_tokens, sql_item, h).generate_output()
        dataset['op_dataset'].append({
            "question_tokens": question_tokens,
            "ts": table_schema,
            "history": h[:],
            "label": label
        })
        if isinstance(s[0], dict):
            dataset['root_tem_dataset'].append({
                "question_tokens": question_tokens,
                "ts": table_schema,
                "history": h[:] + [WHERE_OPS[label]],
                "label": 0
            })
            parser_item(question_tokens,s[0],table,h[:] + [label], dataset)
        else:
            dataset['root_tem_dataset'].append({
                "question_tokens": question_tokens,
                "ts": table_schema,
                "history": h[:] + [WHERE_OPS[label]],
                "label": 1
            })
    for h, sql_item in agg_candidates:
        _, label = AggPredictor(question_tokens,sql_item,h).generate_output()
        dataset['agg_dataset'].append({
            "question_tokens": question_tokens,
            "ts": table_schema,
            "history": h[:],
            "label": label
        })


def get_table_dict(table_data_path):
    data = json.load(open(table_data_path))
    table = dict()
    for item in data:
        table[item["db_id"]] = item
    return table
def parse_data(data):
    dataset = {
        "multi_sql_dataset": [],
        "keyword_dataset": [],
        "col_dataset": [],
        "op_dataset": [],
        "agg_dataset": [],
        "root_tem_dataset": [],
        "des_asc_dataset": []
    }
    table_dict = get_table_dict(table_data_path)
    for item in data:
        parser_item(item["question_toks"],item["sql"],table_dict[item["db_id"]], [], dataset)
    print("finished preprocess")
    for key in dataset:
        print("dataset:{} size:{}".format(key,len(dataset[key])))
        json.dump(dataset[key],open("./generated_data/{}.json".format(key),"w"),indent=2)

parse_data(train_data)

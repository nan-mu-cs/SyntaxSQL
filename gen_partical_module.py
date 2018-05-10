import json

train_data_path = "./data/dev.json"

train_data = json.load(open(train_data_path))


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
    def __init__(self, question, sql, history):
        self.sql = sql
        self.question = question
        self.history = history
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
                        cols.append((col[1], col[2]))
                        sqls.append(col)
                elif key == 'orderBy':
                    sql_cols = self.sql[key][1]
                    for col in sql_cols:
                        cols.append((col[1][1], col[1][2]))
                        sqls.append(col)
                elif key == 'select':
                    sql_cols = self.sql[key][1]
                    for col in sql_cols:
                        cols.append((col[1][1][1], col[1][1][2]))
                        sqls.append(col)
                elif key == 'where' or key == 'having':
                    sql_cols = self.sql[key]
                    for col in sql_cols:
                        if not isinstance(col,list):
                            continue
                        try:
                            cols.append((col[2][1][1], col[2][1][2]))
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
    def __init__(self, question, sql, history):
        self.sql = sql
        self.question = question
        self.history = history

    def generate_output(self):
        for key in self.sql:
            if key == "orderBy" and self.sql[key]:
                return self.history + [key], self.sql[key][0]


def parser_item(question_tokens,sql,history, dataset):
    # try:
    #     question_tokens = item['question_toks']
    # except:
    #     print(item)
    # sql = item['sql']
    history, label, sql = MultiSqlPredictor(question_tokens, sql, history).generate_output()
    dataset['multi_sql_dataset'].append({
        "question_tokens": question_tokens,
        "history": history[:],
        "label": label
    })
    history.append(label)
    history, label, sql = KeyWordPredictor(question_tokens, sql, history).generate_output()
    dataset['keyword_dataset'].append({
        "question_tokens": question_tokens,
        "history": history[:],
        "keywords_num": label[0],
        "keywords": label[1]
    })
    orderby_ret = DesAscPredictor(question_tokens, sql, history).generate_output()
    if orderby_ret:
        dataset['des_asc_dataset'].append({
            "question_tokens": question_tokens,
            "history": orderby_ret[0][:],
            "label": orderby_ret[1]
        })
    col_ret = ColPredictor(question_tokens, sql, history).generate_output()
    agg_candidates = []
    op_candidates = []
    for h, l, s in col_ret:
        dataset['col_dataset'].append({
            "question_tokens": question_tokens,
            "history": h[:],
            "cols_num": l[0],
            "cols": l[1]
        })
        for col, sql_item in zip(l[1], s):
            if h[-1] in ('where', 'having'):
                op_candidates.append((h + [col], sql_item))
            if h[-1] in ('select', 'orderBy', 'having'):
                agg_candidates.append((h + [col], sql_item))
    for h, sql_item in op_candidates:
        _, label, s = OpPredictor(question_tokens, sql_item, h).generate_output()
        dataset['op_dataset'].append({
            "question_tokens": question_tokens,
            "history": h[:],
            "op": label
        })
        if isinstance(s[0], dict):
            dataset['root_tem_dataset'].append({
                "question_tokens": question_tokens,
                "history": h[:] + [label],
                "label": "ROOT"
            })
            parser_item(question_tokens,s[0],h[:] + [label], dataset)
        else:
            dataset['root_tem_dataset'].append({
                "question_tokens": question_tokens,
                "history": h[:] + [label],
                "label": "TEM"
            })
    for h, sql_item in agg_candidates:
        _, label = AggPredictor(question_tokens,sql_item,h).generate_output()
        dataset['agg_dataset'].append({
            "question_tokens": question_tokens,
            "history": h[:],
            "agg": label
        })


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

    for item in data:
        parser_item(item["question_toks"],item["sql"], [], dataset)
    print("finished preprocess")
    for key in dataset:
        print("dataset:{} size:{}".format(key,len(dataset[key])))
        json.dump(dataset[key],open("./generated_data/{}.json".format(key),"w"),indent=2)

parse_data(train_data)

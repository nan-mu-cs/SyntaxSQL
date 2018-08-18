import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import spline

stat = {
    "min_fk":1,
    "max_fk":14,
    "min_table":2,
    "max_table":15,
    "min_col":6,
    "max_col":70
}

def plot_based(key,gap,paths,model_names,line_style):
    max_num = stat["max_{}".format(key)]
    min_num = stat["min_{}".format(key)]
    # gap = (max_num - min_num) // num_points
    num_points = (max_num - min_num) // gap
    count_list = [[[0,0] for _ in range(num_points)] for _ in range(len(paths))]
    db_datas = [json.load(open(path)) for path in paths]

    for model in range(len(db_datas)):
        for db_id in db_datas[model]:
            item = db_datas[model][db_id]
            val = item[key]
            val -= min_num
            if val == 0:
                idx = 0
            else:
                val = val // gap
                if val >= num_points:
                    idx = -1
                else:
                    idx = val
            count_list[model][idx][0] += item["correct"]
            count_list[model][idx][1] += item["total"]
    labels = []
    # for i in range(num_points):
        # labels.append((min_num + i * gap, min_num + (i + 1) * gap))
    # print(count_list)
    # print(labels)
    # acc = [item[0]*1.0/item[1] for item in count_list]
    acc = [[] for _ in range(len(paths))]
    prev = min_num
    for i in range(len(count_list[0])):
        if count_list[0][i][1] == 0:
            continue
        for j in range(len(paths)):
            acc[j].append(count_list[j][i][0]*100 / count_list[j][i][1])
        labels.append([prev, min_num + (i + 1) * gap])
        prev = min_num + (i + 1) * gap

    # for i,item in enumerate(count_list):
    #     if item[1] == 0:
    #         continue
    #     labels.append([prev,min_num+(i+1)*gap])
    #     prev = min_num + (i+1)*gap
    #     acc.append(item[0]/item[1])
    labels = ["{}-{}".format(val[0],val[1]) if i != len(labels)-1 else " > {}".format(val[0]) for i,val in enumerate(labels)]
    # print(acc)
    # acc = np.array(acc)
    # T = np.array(range(len(labels)))
    # xnew = np.linspace(T.min(), T.max(), 50)
    # power_smooth = spline(T, acc, xnew)
    plt.figure()
    model_lines = []
    for item,name,line in zip(acc,model_names,line_style):
        plt.plot(item,line,label=name)
    plt.legend()
    # plt.legend(model_lines,model_names)
    # plt.plot(acc)
    # plt.title("This is the title")
    plt.xticks(range(len(labels)),labels)
    plt.xlabel('Number of Foreign Keys in Each Database')
    plt.ylabel('Exact Matching Accuracy')
    plt.show()

plot_based("fk",3,["./typesql_db_data.json","./seq2seq_atten_db_data.json"],["TypeSQL","Seq2Seq with Attention"],["-","--"])

#seq2seq atten typesql
#type fullhistyory wikisql
# ours_wiki_augment_result.txt
# ours_fullhs_result.txt
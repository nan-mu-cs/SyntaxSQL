import os
import json
import sqlite3
from gen_partical_module import get_table_dict
ROOTPATH = "/data/projects/nl2sql/database/"


# conn = sqlite3.connect(db)
# cursor = conn.cursor()
# cursor.execute(sql)

# for f in os.listdir(ROOTPATH):
#     if not os.path.isdir(f):
#         continue
#     db = os.path.join(ROOTPATH, f, f + ".sqlite")
#     conn = sqlite3.connect(db)
#     cursor = conn.cursor()
#     cursor.execute(sql)

def dump_single_db(db_id,db_path,schema):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    ret = {
        "db_id":db_id,
        "data":{}
    }
    try:
        for table in schema["table_names_original"]:
            sql = "select * from {} limit 30".format(table)
            cursor.execute(sql)
            data = cursor.fetchall()
            ret["data"][table] = data
    except:
        print("error happens when dump {}".format(db_id))
        # print(ret)
        return None
    finally:
        conn.close()
    return ret

def dump_db(db_dict):
    data = []
    for f in os.listdir(ROOTPATH):
        if not os.path.isdir(os.path.join(ROOTPATH, f)):
            # print(f)
            continue
        if f not in db_dict:
            continue
        schema = db_dict[f]
        db_path = os.path.join(ROOTPATH, f, f + ".sqlite")
        # print("start dump db {}".format(f))
        ret = dump_single_db(f,db_path,schema)
        if ret:
            # print("successful dump db {}".format(f))
            data.append(ret)
    json.dump(data,open("./data/db_data.json","w"))
if __name__ == "__main__":
    db_dict = get_table_dict("./data/tables.json")
    # f = "flight_1"
    # dump_single_db(os.path.join(ROOTPATH, f, f + ".sqlite"),db_dict[f])
    dump_db(db_dict)
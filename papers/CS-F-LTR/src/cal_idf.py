"""[summary]
"""
import pickle
from math import log
from collections import defaultdict

common_dics = pickle.load(open("/data/icde_data/common_dics.pkl", "rb"))
doc_num = len(common_dics)

avg_idf = 0
word_idf = dict()
cnt = 0
for doc_dic in common_dics:
    cnt += 1
    if cnt % 100 == 0:
        print(cnt)
    for word in doc_dic:
        if word not in word_idf:
            include = 0
            for doc in common_dics:
                if doc[word] != 0:
                    include += 1
            word_idf[word] = log(doc_num / (include + 0.5))
            avg_idf += word_idf[word]

avg_idf /= len(word_idf)
print(avg_idf)
word_idf_default = defaultdict()
for word in word_idf:
    word_idf_default[word] = word_idf[word]

pickle.dump(word_idf_default, open("/data/icde_data/idf_dict.pkl", "wb"))

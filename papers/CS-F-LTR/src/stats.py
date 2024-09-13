"""[summary]
"""
import pickle
import random


dic = pickle.load(open("/data/icde_data/common_dics.pkl", "rb"))
random.seed(0)
words = random.sample([i for i in range(20000)], 100)

all_res = [[each[word] for each in dic] for word in words]

for each in all_res:
    each.sort(reverse=True)

pickle.dump(all_res, open("./all_res", "wb"))

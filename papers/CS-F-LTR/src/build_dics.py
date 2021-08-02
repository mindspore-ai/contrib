"""[summary]
"""
import pickle
from collections import defaultdict

doc_sketch = pickle.load(
    open("/data/icde_data/common_docs_sketches.pkl", "rb"))
docs = doc_sketch[0]

dics = []

i = 0
for d in docs:
    i += 1
    title, body, score = d
    one_dict = defaultdict(int)
    for w in body:
        one_dict[w] += 1
    dics.append(one_dict)
    if i % 100 == 0:
        print(i)

pickle.dump(dics, open("/data/icde_data/common_dics.pkl", "wb"))

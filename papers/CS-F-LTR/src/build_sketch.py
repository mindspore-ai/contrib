"""[summary]
"""
import pickle
import sys

from countminsketch import CountMinSketch


common_dics = pickle.load(open("/data/icde_data/common_dics.pkl", "rb"))

d = int(sys.argv[1])
w = int(sys.argv[2])
print("d=%d,w=%d" % (d, w))

sketches = []
for i, common_dic in enumerate(common_dics):
    if i % 100 == 0:
        print(i)
    sketch = CountMinSketch(d=d, m=w)
    for key in common_dic:
        sketch.add(key, common_dic[key])
        sketches.append(sketch)

pickle.dump(
    sketches, open(
        "/data/icde_data/common_sketches_(%d_%d).pkl" %
        (d, w), "wb"))

"""[summary]
"""
import pickle
import sys
from src.sketch_heap import SketchHeap

doc_sketch = pickle.load(
    open("/data/icde_data/common_docs_sketches.pkl", "rb"))
docs = doc_sketch[0]  # [:2]
# a = int(sys.argv[1])
# print(a)
a = 5
d = 30
w = 200
# k = 100
k = int(sys.argv[1])
print(k)
sketch_heap = SketchHeap(a, k, d, w)
for i in range(len(docs)):
    title, body, score = docs[i]
    for word in body:
        # print(word)
        sketch_heap.push_in_dict(word)
    sketch_heap.build_sketch_heap(i)
    sketch_heap.clear_dict()
    if i % 100 == 0:
        print(i)
# sketch_heap.contract()

pickle.dump(
    sketch_heap, open(
        "/data/icde_data/sketch_heap_(%d_%d_%d_%d).pkl" %
        (a, d, w, k), "wb"))

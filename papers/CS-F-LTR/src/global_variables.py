"""[summary]
"""
# use it: dictionary
DOC_NUM = 4000000
# use it: dictionary
QUERY_NUM = 10000000
# use it: mapper
TOP_NUM = 80000  # 100000  # 1000000

FED_NUM = 4  # 8
D = 10  # 10
M = 60  # 120
with_sketch = 1
EPS = 0.5
file_path = "/data/icde_data/xfed_normal_w%d_d%d_m%d_e%.5f_%d/" % (
    with_sketch, D, M, EPS, TOP_NUM)

TRAIN_MODE = 6
a = 5
d = 30
w = 200


avg_idf = 7.162413702369469

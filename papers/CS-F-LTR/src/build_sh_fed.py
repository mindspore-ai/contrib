"""[summary]
"""
from src.global_variables import FED_NUM, a, d, w
from src.federation import Federation

FED_PATH = "/data/icde_data/"
FED_NAME = "fed_80000"
for i in range(3, FED_NUM):
    print(i)
    fed_i = Federation()
    fed_i = fed_i.load_fed(FED_PATH, FED_NAME + str(i))
    fed_i.build_sketch_heap(a, d, w, k=150)
    fed_i.save_fed(FED_PATH, "fed_with_sh_(%d_%d_%d)" % (a, d, w) + str(i))

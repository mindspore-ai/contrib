"""[summary]
"""
import pickle
from src.global_variables import FED_NUM
from src.federation import Federation

common_docs = []
common_sketches = []
FED_NUM = 0
for i in range(FED_NUM):
    fed_i = Federation()
    fed_i = fed_i.load_fed(
        "/data/icde_data/",
        "fed_with_sd_d10_m6080000" + str(i))
    sub_docs, sub_sketches = fed_i.contribute()
    common_docs.extend(sub_docs)
    common_sketches.extend(sub_sketches)

print(len(common_docs), len(common_sketches))
# pickle.dump((common_docs, common_sketches), open("/data/icde_data/common_docs_sketches.pkl", "wb"))
f = open("/data/icde_data/common_docs_sketches.pkl", "rb")
common_docs, common_sketches = pickle.load(f)
f.close()
common_fed = Federation(docs=common_docs)
common_fed.sketches = common_sketches
#pickle.dump(common_fed, open("/data/icde_data/common_fed", "wb"))
#common_fed = pickle.load(open("/data/icde_data/common_fed", "rb"))
common_fed.build_cnt_dics()
common_fed.build_invert_table()
with open("/data/icde_data/common_fed", "wb") as f:
    pickle.dump(common_fed, f)
# pickle.dump(common_fed2, open("/data/icde_data/common_fed2", "wb"))

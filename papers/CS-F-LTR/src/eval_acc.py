"""[summary]
"""
import pickle
import random
import time

is_sh = True
is_sk = False
dics = pickle.load(open("/data/icde_data/common_dics.pkl", "rb"))
# sketches = pickle.load(open("/data/icde_data/common_docs_sketches.pkl", "rb"))[1]
# dics = dics[:2]
a = 5
d = 30
w = 200
k = 250
ratio = 0.1
sketch_heap = pickle.load(
    open(
        "/data/icde_data/sketch_heap_(%d_%d_%d_%d).pkl" %
        (a, d, w, k), "rb"))
sketches = pickle.load(
    open(
        "/data/icde_data/common_sketches_(%d_%d).pkl" %
        (d, w), "rb"))
random.seed(100)
words = random.sample([i for i in range(40000)], 10)
# words = [21,54,1465,41,54,438,1448,1251,175,2127,9,18,1450,5,242,1,112,17,5804,58,5488,1066,2886,296,164,13241,7600,1367,23547,15552,2174,1502,97,2904,5,174032,68557,5826,4099,2651,4099,63,5343,4099,12270,5895,5839,4272,5895,4446,12271,12272,5895,5897,5759,5895,5884,5895,5885,5895,12275,5895,4446,18]
STEP = False
accu_cover_rate0 = 0.
accu_cover_rate1 = 0.
accu_time_acc = 0.
accu_time_sk = 0.
accu_time_sh = 0.
for word_n, word in enumerate(words):
    # print(word_n)
    # accurcy freq in all docs
    a = time.clock()
    acc = []
    for doc_id in range(len(dics)):
        dic = dics[doc_id]
        if dic[word] > 0:
            acc.append((dic[word], doc_id))
    acc.sort(reverse=True)
    acc = acc[:150]
    b = time.clock()
    acc_ids = {each[1] for each in acc}
    if STEP:
        print(acc)
        print("############")

    # naive sketches freq
    c = time.clock()
    # [0] body title issue
    # sk = [(sketches[doc_id][0].query_hash(sketches[doc_id][0].hash2(word)), doc_id) for doc_id in range(len(sketches))]
    if is_sk:
        sk = []
        for doc_id in range(len(sketches)):
            sk.append((sketches[doc_id].query_hash(
                sketches[doc_id].hash2(word)), doc_id))
    else:
        sk = []
    sk.sort(reverse=True)
    d = time.clock()
    sk_ids = {each[1] for each in sk}
    cover_ids = acc_ids & sk_ids
    if bool(acc_ids):
        cover_rate0 = len(cover_ids) / len(acc_ids)
    else:
        cover_rate0 = 1.

    # sketch heap solution
    e = time.clock()
    if is_sh:
        sh = sketch_heap.query_hash(sketch_heap.hash(word), ratio)
    else:
        sh = []
    sh.sort(reverse=True)
    if STEP:
        print(sh)
    f = time.clock()
    sh_ids = {each[1] for each in sh}
    cover_ids = acc_ids & sh_ids
    if bool(acc_ids):
        cover_rate1 = len(cover_ids) / len(acc_ids)
    else:
        cover_rate1 = 1.
    if STEP:
        print("id", "acc", "sketch")
        acc_dict = {each[1]: each[0] for each in acc}
        sh_dict = {each[1]: each[0] for each in sh}
        for each in cover_ids:
            print(each, acc_dict[each], sh_dict[each])
        print(cover_rate0, cover_rate1, b - a, d - c, f - e)
        input()
    accu_cover_rate0 += cover_rate0
    accu_cover_rate1 += cover_rate1
    accu_time_acc += (b - a)
    accu_time_sh += (f - e)
    accu_time_sk += (d - c)

print(accu_cover_rate0 / len(words), accu_cover_rate1 / len(words),
      accu_time_acc / len(words), accu_time_sk / len(words), accu_time_sh / len(words))

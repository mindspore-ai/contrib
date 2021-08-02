"""[summary]

Returns:
    [type]: [description]
"""
import pickle
import sys
import numpy as np


def gen_label(x):
    """[summary]

    Args:
        x ([type]): [description]

    Returns:
        [type]: [description]
    """
    if 1 <= x <= 10:
        return 2
    if 11 <= x <= 100:
        return 1
    return 0


def load_only(src_path):
    """[summary]

    Args:
        src_path ([type]): [description]

    Returns:
        [type]: [description]
    """
    f = open(src_path)
    line = f.readline()
    ext_label_id_features = []
    real_label_id_features = []
    cnt = 0
    while line:
        cnt += 1
        if cnt % 10000 == 0:
            print(cnt)
        pure_line, comm = line.split('#')  # 去除行尾注释
        pure_line = pure_line.strip()
        comm = comm.strip()
        relevance_id_features = pure_line.split(' ')
        label_id_feature = [
            gen_label(int(relevance_id_features[0])) \
            if comm in ("rel", "non") \
            else int(relevance_id_features[0])
        ]
        for i in range(1, len(relevance_id_features)):
            label_id_feature.append(float(relevance_id_features[i].split(':')[1]))
        if comm in ("non", "rel"):
            real_label_id_features.append(label_id_feature)
        else:
            ext_label_id_features.append(label_id_feature)
        line = f.readline()
    f.close()
    return real_label_id_features, ext_label_id_features

def main():
    """[summary]
    """
    try:
        fed_no = int(sys.argv[1])
    except ValueError:
        fed_no = 0
    print(fed_no)
    real_label_id_features, ext_label_id_features = load_only(
        "/data/icde_data/fed_with_sh_(5_30_200)ext%d.txt" % fed_no)
    pickle.dump([np.array(real_label_id_features), np.array(ext_label_id_features)], open(
        "/data/icde_data/fed_with_sh_(5_30_200)ext%d.pkl" % fed_no, "wb"))

if __name__ == "__main__":
    main()

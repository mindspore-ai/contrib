"""[summary]
"""
import copy
import pickle
from src.global_variables import FED_NUM, a, w, EPS, D, M
from src.global_variables import d as d_global
from src.federation import Federation


class Server:
    """[summary]
    """
    def __init__(self, fed_num):
        """[summary]

        Args:
            fed_num ([type]): [description]
        """
        self.fed_num = fed_num

    def exchange_most_rel(self, dst_path, fed_path, fed_name,
                          eps=-1, min_median='min', d=10, m=120, with_sketch=True):
        """[summary]

        Args:
            dst_path ([type]): [description]
            fed_path ([type]): [description]
            fed_name ([type]): [description]
            eps (int, optional): [description]. Defaults to -1.
            min_median (str, optional): [description]. Defaults to 'min'.
            d (int, optional): [description]. Defaults to 10.
            m (int, optional): [description]. Defaults to 120.
            with_sketch (bool, optional): [description]. Defaults to True.
        """
        _, _, _, _, _ = eps, min_median, d, m, with_sketch
        idf_dict = pickle.load(open("/data/icde_data/idf_dict.pkl", "rb"))
        for i in range(self.fed_num):
            fed_i = Federation()
            fed_i = fed_i.load_fed(fed_path, fed_name + str(i))
            # quereis
            queries = copy.deepcopy(fed_i.queries)
            query_num = len(queries)
            cases = [[] for i in range(query_num)]
            for k in range(self.fed_num):
                if i == k:
                    continue
                fed_k = Federation()
                fed_k = fed_k.load_fed(fed_path, fed_name + str(k))
                for v in range(query_num):
                    query = queries[v]
                    # how to get cases
                    cases[v].extend(
                        fed_k.get_most_rel_sh(
                            query, idf_dict, 7.162413702369469))

            for v in range(query_num):
                cases[v] = cases[v][:150]
                for x in range(len(cases[v])):
                    if x < 5:
                        cases[v][x].append(2)
                    else:
                        cases[v][x].append(1)
                fed_i.extend_cases(v, cases[v])
            length = sum(len(cases[x]) for x in range(query_num))
            print("fed %d get %d cases from others" % (i, length))
            fed_i.gen(dst_path + str(i) + '.txt')

def main():
    """[summary]
    """
    server = Server(FED_NUM)
    fed_path = "/data/icde_data/"
    fed_name = "fed_with_sh_(%d_%d_%d)" % (a, d_global, w)
    # for i in range(3, FED_NUM):
    #    fed = pickle.load(open(FED_PATH + FED_NAME + str(i), "rb"))
    #    fed.build_cnt_dics()
    #    pickle.dump(fed, open(FED_PATH + FED_NAME + str(i), "wb"))
    #    print(i)
    server.exchange_most_rel(
        fed_path +
        fed_name +
        "ext",
        fed_path,
        fed_name,
        eps=EPS,
        d=D,
        m=M)

if __name__ == "__main__":
    main()

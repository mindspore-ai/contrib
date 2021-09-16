"""[summary]
"""
import copy
from src.federation import Federation
from src.global_variables import FED_NUM, D, M, TOP_NUM, EPS


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
        if not with_sketch:
            for i in range(self.fed_num):
                fed_i = Federation()
                fed_i = fed_i.load_fed(fed_path, fed_name + str(i))
                fed_i.build_cnt_dics()
                fed_i.save_fed(fed_path, fed_name + str(i))
        fed_k = Federation()
        fed_k = fed_k.load_fed(fed_path, "common_fed")
        for i in range(self.fed_num):
            fed_i = Federation()
            fed_i = fed_i.load_fed(fed_path, fed_name + str(i))
            if with_sketch:
                queries_hashed = fed_i.get_hashed_query(d, m)
            else:
                queries_hashed = copy.deepcopy(fed_i.queries)
            query_num = len(queries_hashed)
            cases = [[] for i in range(query_num)]
            for v in range(query_num):
                query_hashed = queries_hashed[v]
                cases[v].extend(
                    fed_k.get_most_rel(
                        query_hashed,
                        eps=eps,
                        min_median=min_median,
                        with_sketch=with_sketch))

            for v in range(query_num):
                cases[v].sort(reverse=True, key=lambda each: each[8])
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
    fed_name = 'fed_with_sd_d' + str(D) + '_m' + str(M) + str(TOP_NUM)
    server.exchange_most_rel(
        fed_path +
        "x" +
        fed_name +
        'e' +
        str(EPS) +
        "ext",
        fed_path,
        fed_name,
        eps=EPS,
        d=D,
        m=M)

if __name__ == "__main__":
    main()

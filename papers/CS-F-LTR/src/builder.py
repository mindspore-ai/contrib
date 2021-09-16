"""[summary]

Returns:
    [type]: [description]
"""
import random
import copy
import getopt
import sys
from src.federation import Federation
from src.mapper import Mapper
from src.global_variables import TOP_NUM, FED_NUM, D, M


class Builder:
    """[summary]
    """

    def __init__(self, queries, docs, query_docs, query_docs_non_rel):
        """[summary]

        Args:
            queries ([type]): [description]
            docs ([type]): [description]
            query_docs ([type]): [description]
            query_docs_non_rel ([type]): [description]
        """
        self.all_queries = list(queries)
        self.all_docs = list(docs)
        self.query_docs = query_docs
        self.query_docs_non_rel = query_docs_non_rel

    def split(self, k, pos_ratio=0.5):
        """[summary]

        Args:
            k ([type]): [description]
            pos_ratio (float, optional): [description]. Defaults to 0.5.

        Returns:
            [type]: [description]
        """
        random.seed(2)
        federation_raws = []
        federations = []
        pool_docs = []
        fed_query_num = len(self.all_queries) // k
        # for each federation, distributed their queries and docs
        for i in range(k):
            left = i * fed_query_num
            right = (i + 1) * fed_query_num
            fed_queries = self.all_queries[left:right]
            fed_docs = []
            for j in range(left, right):
                # handle non-relevance docs
                for doc_id in self.query_docs_non_rel[j]:
                    one_doc = copy.deepcopy(self.all_docs[doc_id])
                    # at the end of each doc, add pair (id in fed, score)
                    one_doc.append((j - left, 0))
                    fed_docs.append(one_doc)
                # handle relevance docs
                id_score_list = list(self.query_docs[j])
                id_score_list.sort(reverse=True, key=lambda x: float(x[1]))
                for t in range(len(id_score_list)):
                    id_score = [id_score_list[t][0], t + 1]  # abs doc id
                    choice = random.random()
                    if choice < pos_ratio:  # this ratio controls the true sample in each fed
                        one_doc = copy.deepcopy(self.all_docs[id_score[0]])
                        one_doc.append(
                            (j - left, id_score[1]))  # rel doc id, score
                        fed_docs.append(one_doc)
                    else:
                        id_score.extend([i, j - left])
                        # abs doc id, int score, fed id, rel q id
                        pool_docs.append(id_score)
            federation_raws.append([fed_queries, fed_docs])
        # for each federation, get some docs from pool to simulate
        random.shuffle(pool_docs)
        fed_extend_doc_num = len(pool_docs) // k
        for i in range(k):
            left = i * fed_extend_doc_num
            right = (i + 1) * fed_extend_doc_num
            extend_docs = []
            for j in range(left, right):
                one_doc = copy.deepcopy(self.all_docs[pool_docs[j][0]])
                # - abs doc id, int score, fed id, rel q id
                one_doc.append((
                    -1 * pool_docs[j][0],
                    pool_docs[j][1],
                    pool_docs[j][2],
                    pool_docs[j][3],
                ))
                extend_docs.append(one_doc)
            federation_raws[i][1].extend(extend_docs)
            federations.append(
                Federation(
                    federation_raws[i][0],
                    federation_raws[i][1]))
        return federations


def main():
    """[summary]
    """
    build_mode = 0
    fed_id = -1
    #opts, args = getopt.gnu_getopt(
    opts, _ = getopt.gnu_getopt(
        sys.argv[1:], 'b:f:h', ['bw=', 'fed=', 'help'])
    for opt_name, opt_val in opts:
        if opt_name in ("-b", "--bf"):
            build_mode = int(opt_val)
        elif opt_name in ('-f', "--fed"):
            fed_id = int(opt_val)

    if build_mode == 1:
        mapper = Mapper()
        mapper = mapper.load_mapper(
            '/data/icde_data/',
            'mapper' + str(TOP_NUM))
        all_docs = mapper.get_docs()
        all_queries = mapper.get_queries()
        query_docs = mapper.get_query_docs()
        query_docs_non_rel = mapper.get_query_docs_non_rel()
        fed_builder = Builder(
            all_queries,
            all_docs,
            query_docs,
            query_docs_non_rel)
        federations = fed_builder.split(FED_NUM, 0.5)
        for i in range(FED_NUM):
            fed = federations[i]
            fed.save_fed('/data/icde_data/', 'fed_' + str(TOP_NUM) + str(i))
    elif build_mode == 2:
        fed = Federation().load_fed(
            '/data/icde_data/',
            'fed_' + str(TOP_NUM) + str(fed_id))
        fed.build_sketch(D, M)
        fed.build_cnt_dics()
        fed.save_fed(
            '/data/icde_data/',
            'fed_with_sd_d' +
            str(D) +
            '_m' +
            str(M) +
            str(TOP_NUM) +
            str(fed_id))
        # elif build_mode == 3:
        # fed = Federation().load_fed('./data/', 'fed_with_sd_d' + str(D) + '_m' + str(M) + str(TOP_NUM) + str(fed_id))
        #fed.build_invert_all(D, M2)
        #fed.save_fed('/data/ltrdata/', 'fed_with_sd_d' + str(D) + '_m' + str(M) + str(TOP_NUM) + str(fed_id))

if __name__ == '__main__':
    main()

"""[summary]

Returns:
    [type]: [description]
"""
import os
import pickle
import random
import json
from src.global_variables import TOP_NUM


class Mapper:
    """[summary]
    """

    def __init__(self, src_path='', doc_path='', query_path='', num=5000):
        """[summary]

        Args:
            src_path (str, optional): [description]. Defaults to ''.
            doc_path (str, optional): [description]. Defaults to ''.
            query_path (str, optional): [description]. Defaults to ''.
            num (int, optional): [description]. Defaults to 5000.
        """
        self.query_idx_convert = dict()  # dict  query origin id -> new id
        self.doc_idx_convert = dict()  # dict doc origin id -> new id
        self.query_n2o = []  # list query new id -> origin id
        self.doc_n2o = []    # list doc new id -> origin id
        # dict query new id: {(doc new id, score), (), ()}
        self.query_docs = dict()
        # dict query new id: {doc new id1, doc new id 2, ...}
        self.query_docs_non_rel = dict()
        self.queries = []  # list  query new id -> [term1, term2, ..., ]
        self.docs = []  # list doc new id -> [[title], [body]]
        if src_path == '':
            return
        f = open(src_path)
        line = f.readline()
        cnt = 1
        qid_cnt = 0
        did_cnt = 0
        rel_num = 40
        rel_cnt = 0
        non_rel_num = 40
        while line:
            # q_oid, _, d_oid, _, score, d = line.split(' ')
            q_oid, _, d_oid, _, score, _ = line.split(' ')
            legal = 0
            exs1 = os.path.exists(os.path.join(query_path, q_oid + '.txt'))
            exs2 = os.path.exists(os.path.join(doc_path, d_oid + '.txt'))
            if not (exs1 and exs2):
                cnt += 1
                line = f.readline()
                if cnt >= num:
                    break
                else:
                    continue
            if self.query_idx_convert.get(q_oid, -1) == -1:
                query_cand = open(os.path.join(query_path, q_oid + '.txt')).readline()
                query_cand = json.loads(query_cand)[1:]
                len1 = len(query_cand)
                if len1 >= 2:
                    self.query_idx_convert[q_oid] = qid_cnt
                    self.query_n2o.append(q_oid)
                    qid_cnt += 1
                    rel_cnt = 0
                    self.queries.append(query_cand)
                else:
                    legal -= 1

            if self.doc_idx_convert.get(d_oid, -1) == -1:
                doc_cand = open(os.path.join(doc_path, d_oid + '.txt')).readline()
                doc_cand = json.loads(doc_cand)[1:]
                len1, len2 = len(doc_cand[0]), len(doc_cand[1])
                if len1 >= 2 and len2 >= 30:
                    self.doc_idx_convert[d_oid] = did_cnt
                    self.doc_n2o.append(d_oid)
                    did_cnt += 1
                    self.docs.append(doc_cand)
                else:
                    legal -= 1

            if legal == 0:
                if rel_cnt < rel_num:
                    if not self.query_docs.get(
                            self.query_idx_convert[q_oid], None):
                        self.query_docs[self.query_idx_convert[q_oid]] = set()
                    self.query_docs[self.query_idx_convert[q_oid]].add(
                        (self.doc_idx_convert[d_oid], score))
                    rel_cnt += 1
            if cnt % 100 == 0:
                print('%d pairs have been record' % cnt)
            if cnt >= num:
                break
            cnt += 1
            line = f.readline()
        self.query_num = len(self.queries)
        self.doc_num = len(self.docs)
        if self.doc_num == 0 or self.query_num == 0:
            print('not enough data, please feed more')
        self.init2(non_rel_num)

    def init2(self, non_rel_num):
        """Bypass Lizard test
        """
        for i in range(self.query_num):
            self.query_docs_non_rel[i] = set()
            if not self.query_docs.get(i, None):
                continue
            # q_oid = self.query_n2o[i]
            d_score = list(self.query_docs[i])
            d_score.sort(reverse=True, key=lambda x: x[1])
            # for t in range(len(d_score)):
                # d, score = d_score[t]
                # score = 2 if t <= 10 else 1
                # d_oid = self.doc_n2o[d]
            non_cnt = 0
            while non_cnt < non_rel_num:
                r = random.randint(0, self.doc_num - 1)
                if r not in self.query_docs[i] and r not in self.query_docs_non_rel[i]:
                    # d_oid = self.doc_n2o[r]
                    _ = self.doc_n2o[r]
                    self.query_docs_non_rel[i].add(r)
                    non_cnt += 1

    def save_mapper(self, dst_path, name):
        """[summary]

        Args:
            dst_path ([type]): [description]
            name ([type]): [description]
        """
        dir_name = os.path.join(dst_path, name)
        print('queries:', len(self.query_docs))
        print(dir_name)
        if os.path.exists(dir_name):
            os.remove(dir_name)
        with open(dir_name, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_mapper(src_path, name):
        """[summary]

        Args:
            src_path ([type]): [description]
            name ([type]): [description]

        Returns:
            [type]: [description]
        """
        print(os.path.join(src_path, name))
        f = open(os.path.join(src_path, name), 'rb')
        a = pickle.load(f)
        f.close()
        return a

    def get_docs(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        return self.docs

    def get_queries(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        return self.queries

    def get_query_docs(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        return self.query_docs

    def get_query_docs_non_rel(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        return self.query_docs_non_rel

def main():
    # msmarco-doctrain-top100
    mapper = Mapper(
        '/data/msmarco-doctrain-top100',
        '/data/docs/',
        '/data/queries/',
        TOP_NUM)
    mapper.save_mapper('/data/icde_data/', 'mapper' + str(TOP_NUM))

if __name__ == '__main__':
    main()

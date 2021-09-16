"""[summary]

Returns:
    [type]: [description]
"""
import os
import pickle
from math import log


class FeatureGenerator:
    """[summary]

    Returns:
        [type]: [description]
    """

    # init: get all docs and queries and some independent statistics
    def __init__(self, all_docs=None, all_queries=None,
                 k1=1.2, b=0.75, lamb=0.1, mu=1000, delta=0.7):
        """
        :param all_docs: all the documents of a federation
        :param all_queries: all the queries of a federation
        :param k1: parameter for calculation of bm 25
        :param b: parameter for calculation of bm 25
        :param lamb: parameter for calculation of jm value
        :param mu: parameter for calculation of Dirichlet value
        :param delta: parameter for calculation of absolute distance
        model attributions:
            parameters: k1, b, lamb, mu, delta
            data: all_docs, all_queries
            document info: doc_num, doc_lens, doc_avg_len, doc_term_freq, all_doc_term_freq, word_num
            query info: query_num, query_idf, query_LMIR
        """

        self.k1 = k1
        self.b = b
        self.lamb = lamb
        self.mu = mu
        self.delta = delta
        self.all_docs = all_docs
        self.all_queries = all_queries

        # Fetch all of the necessary quantities for the document language
        # models.
        # doc_term_counts[i] indicates terms-counts pairs of the i-th document
        doc_term_counts = []
        doc_term_freq = []  # doc_term_feq[i] indicates term-frequency pairs of the i-th document
        doc_lens = []  # doc_lens[i] indicates the length of the i-th document

        all_doc_term_counts = {}  # all of the documents' terms and their counts
        flag = 0
        if not all_docs or not all_queries:
            return
        for doc in self.all_docs:
            if flag % 100 == 0:
                print(flag)
            flag += 1
            # doc_len = [len of body, len of title]
            doc_len = [len(doc[0]), len(doc[1])]
            doc_lens.append(doc_len)
            # print("len", doc_len)
            term_counts = {}
            # calculate document term count
            for i in range(2):
                for term in doc[i]:
                    if term_counts.get(term, [0, 0]) == [0, 0]:
                        term_counts[term] = [0, 0]
                    term_counts[term][i] += 1
                    if all_doc_term_counts.get(term, [0, 0]) == [0, 0]:
                        all_doc_term_counts[term] = [0, 0]
                    all_doc_term_counts[term][i] += 1
            doc_term_counts.append(term_counts)
            # calculate document term frequent
            term_freq = {}
            for term in term_counts:
                term_freq[term] = [0, 0]
                for i in range(2):
                    term_freq[term][i] = term_counts[term][i] / doc_len[i]
                    # print(term_freq[term][i])
            doc_term_freq.append(term_freq)
            # print('@', term_freq)
        # all numbers of document term counts
        cnt1 = 0
        cnt2 = 0
        for each in all_doc_term_counts.values():
            cnt1 += each[0]
            cnt2 += each[1]
        all_doc_term_freq = {
            term: [term_count[0] / cnt1, term_count[1] / cnt2]
            for (term, term_count) in all_doc_term_counts.items()
        }

        self.doc_num = len(self.all_docs)
        self.query_num = len(self.all_queries)
        # self.word_num[i]  the i-th doc: body word num, title word num
        self.word_num = doc_term_counts
        # self.doc_lens[i] the i-th doc: len of body, len of title
        self.doc_lens = doc_lens
        body_len = 0
        title_len = 0
        for each in doc_lens:
            body_len += each[0]
            title_len += each[1]
        self.doc_avg_len = [body_len / self.doc_num, title_len / self.doc_num]
        # doc_term_freq[i][t]  the i-th doc, term t: freq of body, freq of
        # title
        self.doc_term_freq = doc_term_freq
        # all_doc_term_freq[t]  term t: freq of all doc body, freq of all doc
        # title
        self.all_doc_term_freq = all_doc_term_freq

        query_idf = []
        query_idf_pos = []
        q_flag = 0
        for q_idx in range(self.query_num):
            if q_flag % 10 == 0:
                print(q_flag)
            q_flag += 1
            idf, idf_pos = self._cal_idf(q_idx)
            query_idf.append(idf)
            query_idf_pos.append(idf_pos)

        self.query_idf = query_idf

    def gen_feature(self, q_idx, d_idx):
        """[summary]

        Args:
            q_idx ([type]): [description]
            d_idx ([type]): [description]

        Returns:
            [type]: [description]
        """
        # generate feature vector for all pairs of (query, doc)
        # the features contain:
        # [document length, tf, idf, tf-idf, bm25, jm, dir, abs]
        lens = self.doc_lens[d_idx]
        tfs = self._cal_tf(q_idx, d_idx)
        q_idfs = self.query_idf[q_idx]
        tf_idfs = [tfs[0] * q_idfs[0], tfs[1] * q_idfs[1]]
        bm25s = self._cal_bm25(q_idx, d_idx)
        jms = self._cal_jm(q_idx, d_idx)
        dirichlet = self._cal_dirichlet(q_idx, d_idx)
        abs_dis = self._cal_abs_dis(q_idx, d_idx)

        #vector = lens + tfs + q_idfs + tf_idfs + bm25s + jms + dir + abs_dis
        # print(vector)
        vector = [
            lens[1],
            tfs[0],
            q_idfs[0],
            tf_idfs[0],
            bm25s[0],
            jms[0],
            dirichlet[0],
            abs_dis[0]]
        # print(vector)
        # print(lens)
        # input()

        line = ''
        for i in range(len(vector)):
            line += (str(i) + ':' + str(vector[i]) + ' ')
        # print(line)
        return line

    def _cal_tf(self, q_idx, d_idx):
        """[summary]

        Args:
            q_idx ([type]): [description]
            d_idx ([type]): [description]

        Returns:
            [type]: [description]
        """
        query = self.all_queries[q_idx]
        tf = [0, 0]
        for term in query:
            for i in range(2):
                tf[i] += self.doc_term_freq[d_idx].get(term, [0, 0])[i]
        return tf

    def _cal_idf(self, q_idx):
        """[summary]

        Args:
            q_idx ([type]): [description]

        Returns:
            [type]: [description]
        """
        include = [0, 0]
        query = self.all_queries[q_idx]
        for term in query:
            for d_idx in range(self.doc_num):
                for i in range(2):
                    if self.doc_term_freq[d_idx].get(term, [0, 0])[i] > 0:
                        include[i] += 1
        for i in range(2):
            if bool(query):
                include[i] /= len(query)
            else:
                print("WARNING: query %d has no term!" % q_idx)
        idf = [log((self.doc_num - include[0] + 0.5) / (include[0] + 0.5), 2),
               log((self.doc_num - include[1] + 0.5) / (include[1] + 0.5), 2)]
        idf_pos = [log(1 + (self.doc_num - include[1] + 0.5) / (include[1] + 0.5), 2),
                   log(1 + (self.doc_num - include[1] + 0.5) / (include[1] + 0.5), 2)]

        return idf, idf_pos

    def _cal_bm25(self, q_idx, d_idx):
        """[summary]

        Args:
            q_idx ([type]): [description]
            d_idx ([type]): [description]

        Returns:
            [type]: [description]
        """
        score = [0, 0]
        query = self.all_queries[q_idx]
        dl = self.doc_lens[d_idx]
        for term in query:
            tf = self.doc_term_freq[d_idx].get(term, [0, 0])
            for i in range(2):
                score[i] += self.query_idf[q_idx][i] * tf[i] * (self.k1 + 1) / \
                    (tf[i] + self.k1 * (1 - self.b + self.b * dl[i] / self.doc_avg_len[i]))
        return score

    def _cal_jm(self, q_idx, d_idx):
        """[summary]

        Args:
            q_idx ([type]): [description]
            d_idx ([type]): [description]

        Returns:
            [type]: [description]
        """
        lamb = self.lamb
        all_tfs = self.all_doc_term_freq
        query = self.all_queries[q_idx]
        tfs = self.doc_term_freq[d_idx]
        score = [0, 0]
        for term in query:
            if term not in all_tfs:
                continue
            for i in range(2):
                # print(all_tfs[term])
                if (1 - lamb) * tfs.get(term, [0, 0])[i] + lamb * all_tfs[term][i]:
                    score[i] -= log((1 - lamb) * tfs.get(term, [0, 0])[i] + lamb * all_tfs[term][i])
        return score

    def _cal_dirichlet(self, q_idx, d_idx):
        """[summary]

        Args:
            q_idx ([type]): [description]
            d_idx ([type]): [description]

        Returns:
            [type]: [description]
        """
        mu = self.mu
        all_tfs = self.all_doc_term_freq
        query = self.all_queries[q_idx]
        c = self.word_num[d_idx]
        doc_len = self.doc_lens[d_idx]
        score = [0, 0]
        for term in query:
            if term not in all_tfs:
                continue
            for i in range(2):
                # print(all_tfs[term])
                denom = (c.get(term, [0, 0])[i] + mu *
                         all_tfs[term][i]) / (doc_len[i] + mu)
                if denom > 0:
                    score[i] -= log(denom)
        return score

    def _cal_abs_dis(self, q_idx, d_idx):
        """[summary]

        Args:
            q_idx ([type]): [description]
            d_idx ([type]): [description]

        Returns:
            [type]: [description]
        """
        query = self.all_queries[q_idx]
        delta = self.delta
        all_tfs = self.all_doc_term_freq
        c = self.word_num[d_idx]
        doc_len = self.doc_lens[d_idx]
        d_u = len(c)
        score = [0, 0]
        for term in query:
            for i in range(2):
                if all_tfs.get(term, [0, 0])[i] > 0:
                    # print(doc_len[i], i)
                    score[i] -= log(max(c.get(term, [0, 0])[i] -
                                        delta, 0) /
                                    doc_len[i] +
                                    delta *
                                    d_u /
                                    doc_len[i] *
                                    all_tfs[term][i])
        return score

    def save_gen(self, dst_path, fed_name):
        """[summary]

        Args:
            dst_path ([type]): [description]
            fed_name ([type]): [description]
        """
        dir_name = os.path.join(dst_path, fed_name)
        print(dir_name)
        if os.path.exists(dir_name):
            os.remove(dir_name)
        with open(dir_name, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_gen(src_path, fed_name):
        """[summary]

        Args:
            src_path ([type]): [description]
            fed_name ([type]): [description]

        Returns:
            [type]: [description]
        """
        print(os.path.join(src_path, fed_name))
        f = open(os.path.join(src_path, fed_name), 'rb')
        a = pickle.load(f)
        f.close()
        return a

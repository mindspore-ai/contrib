"""[summary]

Returns:
    [type]: [description]
"""
import os
import pickle
import random
from math import log
from collections import defaultdict
from src.countminsketch import CountMinSketch
from src.feature_generator import FeatureGenerator
from src.utils import add_lap
from src.sketch_heap import SketchHeap


class Federation:
    """[summary]
    """

    def __init__(self, queries=None, docs=None):
        """[summary]

        Args:
            queries ([type], optional): [description]. Defaults to None.
            docs ([type], optional): [description]. Defaults to None.
        """
        self.queries = queries
        self.docs = docs
        if docs:
            self.doc_num = len(self.docs)
        else:
            self.doc_num = 0
        if queries:
            self.query_num = len(self.queries)
        else:
            self.query_num = 0
        self.sketches = []
        self.all_sketches = None
        self.invert_table = None
        if self.docs:
            self.dl = [[len(each[0]), len(each[1])] for each in self.docs]
        else:
            self.dl = [[0, 0]]
        if self.docs:
            len0 = 0
            len1 = 1
            for each in self.docs:
                len0 += len(each[0])
                len1 += len(each[1])
            self.doc_avg_len = [len0 / self.doc_num, len1 / self.doc_num]
        else:
            self.doc_avg_len = [0, 0]
        # caution!! len[0] is title len[1] is body
        self.extended_cases = dict()

    def save_fed(self, dst_path, fed_name):
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
    def load_fed(src_path, fed_name):
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

    def build_sketch(self, d=10, m=120):
        """[summary]

        Args:
            d (int, optional): [description]. Defaults to 10.
            m (int, optional): [description]. Defaults to 120.
        """
        self.d = d
        self.m = m
        all_body_sketch = CountMinSketch(d=d, m=m)
        all_title_sketch = CountMinSketch(d=d, m=m)
        self.invert_table = [
            CountMinSketch(
                d=d,
                m=int(
                    1.7 * m)),
            CountMinSketch(
                d=d,
                m=m)]
        cnt = 0
        term_inc = dict()
        # print(self.doc_num, len(self.docs))
        for doc in self.docs:
            if cnt % 100 == 0:
                print(cnt)
            cnt += 1
            body_sketch = CountMinSketch(d=d, m=m)
            title_sketch = CountMinSketch(d=d, m=m)
            # title, body, a = doc
            title, body, _ = doc
            for term in body:
                body_sketch.add(term)
                all_body_sketch.add(term)
            for term in title:
                title_sketch.add(term)
                all_title_sketch.add(term)
            self.sketches.append([body_sketch, title_sketch])
            # build invert table for idf
            body = set(body)
            title = set(title)
            for each in body:
                if term_inc.get(each, 0) == 0:
                    term_inc[each] = 0
                term_inc[each] += 1
                if term_inc[each] > cnt:
                    print(term_inc[each], cnt)
                # self.invert_table[0].add(each)
            for each in title:
                self.invert_table[1].add(each)
        self.all_sketches = [all_body_sketch, all_title_sketch]
        for each in term_inc:
            self.invert_table[0].add(each, value=term_inc[each])
            # if self.invert_table[0].query(each) > term_inc[each]:
            #    print(self.invert_table[0].query(each), term_inc[each], self.doc_num)

    def build_sketch_heap(self, a=1, d=50, w=200, k=150):
        """[summary]

        Args:
            a (int, optional): [description]. Defaults to 1.
            d (int, optional): [description]. Defaults to 50.
            w (int, optional): [description]. Defaults to 200.
            k (int, optional): [description]. Defaults to 150.
        """
        self.sketch_heap = SketchHeap(a, k, d, w)
        for i in range(len(self.docs)):
            # title, body, score = self.docs[i]
            _, body, _ = self.docs[i]
            for word in body:
                # print(word)
                self.sketch_heap.push_in_dict(word)
            self.sketch_heap.build_sketch_heap(i)
            self.sketch_heap.clear_dict()
            if i % 100 == 0:
                print(i)

    def build_invert_table(self):
        """[summary]
        """
        # build invert table for idf
        d = 10
        m = 60
        self.d = d
        self.m = m
        all_body_sketch = CountMinSketch(d=d, m=m)
        all_title_sketch = CountMinSketch(d=d, m=m)
        self.invert_table = [
            CountMinSketch(
                d=d,
                m=int(
                    1.7 * m)),
            CountMinSketch(
                d=d,
                m=m)]
        cnt = 0
        term_inc = dict()
        # print(self.doc_num, len(self.docs))
        for doc in self.docs:
            if cnt % 100 == 0:
                print(cnt)
            cnt += 1
            # title, body, a = doc
            title, body, _ = doc
            for term in body:
                all_body_sketch.add(term)
            for term in title:
                all_title_sketch.add(term)
            # build invert table for idf
            body = set(body)
            title = set(title)
            for each in body:
                if term_inc.get(each, 0) == 0:
                    term_inc[each] = 0
                term_inc[each] += 1
            for each in title:
                self.invert_table[1].add(each)
        self.all_sketches = [all_body_sketch, all_title_sketch]
        for each in term_inc:
            self.invert_table[0].add(each, value=term_inc[each])

    def get_hashed_query(self, d=10, m=120):
        """[summary]

        Args:
            d (int, optional): [description]. Defaults to 10.
            m (int, optional): [description]. Defaults to 120.

        Returns:
            [type]: [description]
        """
        hashed_queries = []
        # hasher = CountMinSketch(d=self.d, m=self.m)
        hasher = CountMinSketch(d=d, m=m)
        for each in self.queries:
            query_hashed = []
            for term in each:
                query_hashed.append(hasher.hash2(term))
            hashed_queries.append(query_hashed)
        return hashed_queries

    def get_hashed_queries_sh(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        return [[self.sketch_heap.hash(term) for term in each]
                for each in self.queries]

    def extend_cases(self, q_id, cases):
        """[summary]

        Args:
            q_id ([type]): [description]
            cases ([type]): [description]
        """
        self.extended_cases[q_id] = cases

    def build_cnt_dics(self):
        """[summary]
        """
        dics = []
        all_body_dic = dict()
        all_title_dic = dict()
        invert_body_dic = dict()
        invert_title_dic = dict()
        for one_doc in self.docs:
            # title, body, a = one_doc
            title, body, _ = one_doc
            body_dic = dict()
            for term in body:
                if body_dic.get(term, 0) == 0:
                    body_dic[term] = 0
                    if invert_body_dic.get(term, 0) == 0:
                        invert_body_dic[term] = 0
                    invert_body_dic[term] += 1
                body_dic[term] += 1
                if all_body_dic.get(term, 0) == 0:
                    all_body_dic[term] = 0
                all_body_dic[term] += 1
            title_dic = dict()
            for term in title:
                if title_dic.get(term, 0) == 0:
                    title_dic[term] = 0
                    if invert_title_dic.get(term, 0) == 0:
                        invert_title_dic[term] = 0
                    invert_title_dic[term] += 1
                title_dic[term] += 1
                if all_title_dic.get(term, 0) == 0:
                    all_title_dic[term] = 0
                all_title_dic[term] += 1
            dics.append([body_dic, title_dic])
        self.dics = dics
        self.all_dics = [all_body_dic, all_title_dic]
        self.invert_dic = [invert_body_dic, invert_title_dic]

    def get_most_rel(self, query_hashed, eps=-1, min_median='min',
                     k1=1.2, b=0.75, lamb=0.1, mu=1000, delta=0.7, with_sketch=True):
        """[summary]

        Args:
            query_hashed ([type]): [description]
            eps (int, optional): [description]. Defaults to -1.
            min_median (str, optional): [description]. Defaults to 'min'.
            k1 (float, optional): [description]. Defaults to 1.2.
            b (float, optional): [description]. Defaults to 0.75.
            lamb (float, optional): [description]. Defaults to 0.1.
            mu (int, optional): [description]. Defaults to 1000.
            delta (float, optional): [description]. Defaults to 0.7.
            with_sketch (bool, optional): [description]. Defaults to True.

        Returns:
            [type]: [description]
        """
        # cal idf
        include = [0, 0]
        for term_hashed in query_hashed:
            if with_sketch:
                if min_median == 'min':
                    include[0] += self.invert_table[0].query_hash(term_hashed)
                    include[1] += self.invert_table[1].query_hash(term_hashed)
                else:
                    include[0] += self.invert_table[0].query_hash_median(
                        term_hashed)
                    include[1] += self.invert_table[1].query_hash_median(
                        term_hashed)
            else:
                include[0] += self.invert_dic[0].get(term_hashed, 0)
                include[1] += self.invert_dic[1].get(term_hashed, 0)
        if eps > -0.1:
            include[0] = add_lap(include[0], eps)
            include[1] = add_lap(include[1], eps)
        if bool(query_hashed):
            include = [include[0] / len(query_hashed), include[1] / len(query_hashed)]
        # print(self.doc_num, include[0], include[1])
        idf = [log((max(0, self.doc_num - include[0]) + 0.5) / (include[0] + 0.5), 2),
               log((max(0, self.doc_num - include[1]) + 0.5) / (include[1] + 0.5), 2)]
        # cal tf, tf_idf and bm25s
        tfs = []
        tf_idfs = []
        bm25s = []
        for d_id in range(self.doc_num):
            tf = [0, 0]
            bm25 = [0, 0, d_id]
            for term_hashed in query_hashed:
                if with_sketch:
                    if min_median == 'min':
                        local_a = self.sketches[d_id][0].query_hash(
                            term_hashed)
                        local_b = self.sketches[d_id][1].query_hash(
                            term_hashed)
                    else:
                        local_a = self.sketches[d_id][0].query_hash_median(
                            term_hashed)
                        local_b = self.sketches[d_id][1].query_hash_median(
                            term_hashed)
                else:
                    local_a = self.dics[d_id][0].get(term_hashed, 0)
                    local_b = self.dics[d_id][1].get(term_hashed, 0)
                local_c = k1 * \
                    (1 - b + b * self.dl[d_id][0] / self.doc_avg_len[0])
                local_d = k1 * \
                    (1 - b + b * self.dl[d_id][1] / self.doc_avg_len[1])
                if eps > -0.1:
                    local_a = add_lap(local_a, eps)
                    local_b = add_lap(local_b, eps)
                local_a /= self.dl[d_id][0]
                local_b /= self.dl[d_id][1]
                tf[0] += local_a
                tf[1] += local_b
                bm25[0] += idf[0] * local_a * (k1 + 1) / (local_a + local_c)
                bm25[1] += idf[1] * local_b * (k1 + 1) / (local_b + local_d)
            tfs.append(tf)
            tf_idfs.append([tfs[d_id][0] * idf[0], tfs[d_id][1] * idf[1]])
            bm25s.append(bm25)
        # get most relevant based on bm25
        bm25s.sort(reverse=True)
        cands = [bm25s[x][2] for x in range(min(100, len(bm25s)))]
        features = []
        for k in range(len(cands)):
            d_id = cands[k]
            jm = [0, 0]
            dirs = [0, 0]
            ab = [0, 0]
            for term_hashed in query_hashed:
                for i in range(2):
                    if with_sketch:
                        if min_median == 'min':
                            local_a = self.sketches[d_id][i].query_hash(
                                term_hashed)
                            local_b = self.all_sketches[i].query_hash(
                                term_hashed)
                        else:
                            local_a = self.sketches[d_id][i].query_hash_median(
                                term_hashed)
                            local_b = self.all_sketches[i].query_hash_median(
                                term_hashed)
                    else:
                        local_a = self.dics[d_id][i].get(term_hashed, 0)
                        local_b = self.all_dics[i].get(term_hashed, 0)
                    if eps > -0.1:
                        local_a = add_lap(local_a, eps)
                        local_b = add_lap(local_b, eps)
                    local_a /= self.dl[d_id][i]
                    local_b /= (self.doc_avg_len[i] * self.doc_num)
                    denom = (1 - lamb) * local_a + lamb * local_b
                    if denom > 0:
                        jm[i] -= log(denom)
                    denom = (local_a * self.dl[d_id][i] +
                             mu * local_b) / (self.dl[d_id][i] + mu)
                    if denom > 0:
                        dirs[i] -= log(denom)
                    if local_b > 0:
                        ab[i] -= log(max(local_a *
                                         self.dl[d_id][i] -
                                         delta, 0) /
                                     self.dl[d_id][i] +
                                     delta *
                                     0.8 *
                                     local_b)
            # exit(0)
            vector = self.dl[d_id] + tfs[d_id] + idf + \
                tf_idfs[d_id] + bm25s[k][:2] + jm + dirs + ab
            features.append(vector)
        return features

    def get_most_rel_sh(self, query, idf_dict, avg_idf,
                        k1=1.2, b=0.75, lamb=0.1, mu=1000, delta=0.7):
        """[summary]

        Args:
            query ([type]): [description]
            idf_dict ([type]): [description]
            avg_idf ([type]): [description]
            k1 (float, optional): [description]. Defaults to 1.2.
            b (float, optional): [description]. Defaults to 0.75.
            lamb (float, optional): [description]. Defaults to 0.1.
            mu (int, optional): [description]. Defaults to 1000.
            delta (float, optional): [description]. Defaults to 0.7.

        Returns:
            [type]: [description]
        """
        # cal idf
        idf = sum(
            [idf_dict[word] if word in idf_dict else avg_idf for word in query]) / len(query)
        # cal tf, tf_idf and bm25s
        # top three words of the query according to idf
        idf_word = [(idf_dict[term] if term in idf_dict else avg_idf, term)
                    for term in query]
        idf_word.sort(reverse=True)
        cand_terms = [idf_word[i][1] for i in range(min(len(query), 3))]
        doc_cnts = defaultdict(lambda: 0)
        for cand_term in cand_terms:
            cnt_terms = self.sketch_heap.query(cand_term)
            for cnt_term in cnt_terms:
                doc_cnts[cnt_term[1]] += cnt_term[0]
        avg_cnt_docs = [(doc_cnts[key] / len(cand_terms), key)
                        for key in doc_cnts]
        avg_cnt_docs.sort(reverse=True)
        cand_docs = [avg_cnt_docs[i][1]
                     for i in range(min(len(avg_cnt_docs), 150))]
        tfs = {cand_docs[i]: avg_cnt_docs[i][0] / self.dl[cand_docs[i]][1] for i in range(len(cand_docs))}
        features = []
        for cand_doc in cand_docs:
            tf = tfs[cand_doc]
            tf_idf = tf * idf
            local_c = k1 * \
                (1 - b + b * self.dl[cand_doc][0] / self.doc_avg_len[0])
            bm25 = tf_idf * (k1 + 1) / (tf + local_c)
            jm = 0
            dirs = 0
            ab = 0
            for cand_term in cand_terms:
                try:
                    local_b = self.all_dics[0][cand_term]
                except IndexError:
                    local_b = 0
                local_b /= (self.doc_avg_len[0] * self.doc_num)
                denom = (1 - lamb) * tf + lamb * local_b
                if denom > 0:
                    jm -= log(denom)
                denom = (tf * self.dl[cand_doc][0] + mu *
                         local_b) / (self.dl[cand_doc][0] + mu)
                if denom > 0:
                    dirs -= log(denom)
                if local_b > 0:
                    ab -= log(max(tf *
                                  self.dl[cand_doc][0] -
                                  delta, 0) /
                              self.dl[cand_doc][0] +
                              delta *
                              0.8 *
                              local_b)
            # exit(0)
            features.append([self.dl[cand_doc][0], tf, idf, tf_idf, bm25, jm, dirs, ab])
        return features

    def gen(self, dst_path):
        """[summary]

        Args:
            dst_path ([type]): [description]
        """
        print("gen feature...")
        local_docs = []
        query_docs = dict()
        query_docs_non_rel = dict()
        for i in range(self.doc_num):
            doc = self.docs[i]
            if len(doc[-1]) > 2:
                continue
            q_id, score = doc[-1]
            doc = doc[:-1]
            if q_id != -1:
                local_docs.append(doc)
                if score > 0:
                    if not query_docs.get(q_id, None):
                        query_docs[q_id] = set()
                    query_docs[q_id].add((i, score))
                elif score == 0:
                    if not query_docs_non_rel.get(q_id, None):
                        query_docs_non_rel[q_id] = set()
                    query_docs_non_rel[q_id].add(i)
        fg = FeatureGenerator(local_docs, self.queries)
        buffer_strs = []
        for i in range(self.query_num):
            if not query_docs.get(i, None):
                continue
            for d, score in query_docs[i]:
                # print("query_docs:", self.query_docs[i], "len", self.docs[d])
                line = str(score) + " qid:" + str(i) + ' ' + \
                    fg.gen_feature(i, d) + ' # rel\n'
                buffer_strs.append(line)
            for d in query_docs_non_rel[i]:
                line = "0 qid:" + str(i) + ' ' + \
                    fg.gen_feature(i, d) + ' # non\n'
                buffer_strs.append(line)
            if self.extended_cases.get(i, None):
                for vector_score in self.extended_cases[i]:
                    vector = vector_score[:-1]
                    score = vector_score[-1]
                    line = str(score) + " qid:" + str(i) + ' '
                    for k in range(len(vector)):
                        line += (str(k) + ':' + str(vector[k]) + ' ')
                    line += " # ext\n"
                    buffer_strs.append(line)
            if len(buffer_strs) > 100:
                print('push to file')
                f = open(dst_path, 'a')
                for each in buffer_strs:
                    f.write(each)
                f.close()
                buffer_strs.clear()
        if bool(buffer_strs):
            print('push to file')
            f = open(dst_path, 'a')
            for each in buffer_strs:
                f.write(each)
            f.close()
            buffer_strs.clear()

    def contribute(self, number=1000):
        """[summary]

        Args:
            number (int, optional): [description]. Defaults to 1000.

        Returns:
            [type]: [description]
        """
        random.seed(0)
        indices = random.sample([i for i in range(len(self.docs))], number)
        sub_docs = []
        sub_sketches = []
        print(indices)
        for i in indices:
            sub_docs.append(self.docs[i])
            sub_sketches.append(self.sketches[i])
        return sub_docs, sub_sketches

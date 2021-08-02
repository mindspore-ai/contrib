"""[summary]

Returns:
    [type]: [description]
"""
import pickle
import os
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from src.global_variables import DOC_NUM, QUERY_NUM


# CC 连词 and, or,but, if, while,although
# CD 数词 twenty-four, fourth, 1991,14:24
# DT 限定词 the, a, some, most,every, no
# EX 存在量词 there, there's
# FW 外来词 ersatz, esprit, quo
# IN 介词连词 on, of,at, with,by,into, under
# JJ 形容词 new,good, high, special, big, local
# JJR 比较级词语 bleaker braver breezier briefer brighter brisker
# JJS 最高级词语 calmest cheapest choicest classiest cleanest clearest
# LS 标记 A A. B B. C C. D E F First G H I J K
# MD 情态动词 can cannot could couldn't
# NN 名词 year,home, costs, time, education
# NNS 名词复数 undergraduates scotches
# NNP 专有名词 Alison,Africa,April,Washington
# NNP S 专有名词复数 Americans Americas
# PDT 前限定词 all both half many
# POS 所有格标记 ' 's
# PRP 人称代词 hers herself him himself hisself
# PRP$ 所有格 her his mine my our ours
# RB 副词 occasionally maddeningly
# RBR 副词比较级 further gloomier grander
# RBS 副词最高级 best biggest bluntest earliest
# RP 虚词 aboard about across along apart
# SYM 符号 % & ' '' ''. ) )
# TO 词to to
# UH 感叹词 Goodbye Goody Gosh Wow
# VB 动词 ask assemble assess
# VBD 动词过去式 dipped pleaded swiped
# VBG 动词现在分词 telegraphing stirring focusing
# VBN 动词过去分词  dilapidated
# VBP 动词现在式非第三人称时态 predominate wrap resort sue
# VBZ 动词现在式第三人称时态 bases reconstructs marks
# WDT Wh限定词 who,which,when,what,where,how
# WP WH代词 that what whatever
# WP$ WH代词所有格 whose
# WRB WH副词


class Dictionary:
    """[summary]
    """

    def __init__(self):
        """[summary]
        """
        self.i2w = []           # word index to word
        self.w2i = {}           # word to word index
        self.lm = WordNetLemmatizer()
        self.check_set = {'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNPS', 'PDT', 'RB',
                          'RBR', 'RBS', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBZ', 'VBP'}

    @staticmethod
    def _legal(token):
        """[summary]

        Args:
            token ([type]): [description]

        Returns:
            [type]: [description]
        """
        for char in token:
            if ord(char) > ord('z') or ord(char) < ord('a'):
                return False
        return True

    def _purify(self, text):
        """[summary]

        Args:
            text ([type]): [description]

        Returns:
            [type]: [description]
        """
        text.lower()
        tokens = nltk.word_tokenize(text)
        terms = []
        for token in tokens:
            token = token.lower()
            if self._legal(token):
                terms.append(self.lm.lemmatize(token))
        # pos_tags = nltk.pos_tag(new_tokens)
        # terms = []
        # for word, pos in pos_tags:
        #     if pos in self.check_set and len(word) < 30 and word not in self.check_set:
        #         terms.append(word)
        # print(terms)
        return terms

    def convert_doc(self, dst_path, src_path, from_idx=0, max_num=1000):
        """[summary]

        Args:
            dst_path ([type]): [description]
            src_path ([type]): [description]
            from_idx (int, optional): [description]. Defaults to 0.
            max_num (int, optional): [description]. Defaults to 1000.
        """
        if not os.path.exists(dst_path):
            os.mkdir(dst_path)
        f = open(src_path)
        cnt = 0
        line = f.readline()
        term_cnt = 1
        while line:
            if cnt < from_idx:
                if cnt % 500000 == 0 and cnt > 0:
                    print(cnt, 'documents have been skipped')
                cnt += 1
                line = f.readline()
                continue
            if cnt % 1000 == 0 and cnt > 0:
                print(cnt, 'documents have been converted')
            # idx, url, body, title = line.split('\t')
            idx, _, body, title = line.split('\t')
            terms = self._purify(body)
            title = self._purify(title)
            doc_cvt = [idx]
            body_cvt = []
            title_cvt = []
            for term in terms:
                if self.w2i.get(term, 0) == 0:
                    self.w2i[term] = term_cnt
                    self.i2w.append(term)
                    term_cnt += 1
                body_cvt.append(self.w2i[term])
            for term in title:
                if self.w2i.get(term, 0) == 0:
                    self.w2i[term] = term_cnt
                    self.i2w.append(term)
                    term_cnt += 1
                title_cvt.append(self.w2i[term])
            doc_cvt.append(body_cvt)
            doc_cvt.append(title_cvt)
            fw = open(
                os.path.join(
                    dst_path,
                    str(idx) +
                    '.txt'),
                mode="w",
                encoding="utf-8")
            fw.writelines(str(doc_cvt))
            fw.close()
            cnt += 1
            if cnt >= max_num != -1:
                break
            line = f.readline()
        f.close()

    def convert_query(self, dst_path, src_path, from_idx=0, max_num=1000):
        """[summary]

        Args:
            dst_path ([type]): [description]
            src_path ([type]): [description]
            from_idx (int, optional): [description]. Defaults to 0.
            max_num (int, optional): [description]. Defaults to 1000.
        """
        if not os.path.exists(dst_path):
            os.mkdir(dst_path)
        f = open(src_path)
        cnt = 0
        line = f.readline()
        term_cnt = 1
        while line:
            if cnt < from_idx:
                cnt += 1
                line = f.readline()
            if cnt % 1000 == 0 and cnt > 0:
                print(cnt, 'queries have been converted')
            idx, content = line.split('\t')
            terms = self._purify(content)
            query_cvt = [idx]
            for term in terms:
                if self.w2i.get(term, 0) == 0:
                    self.w2i[term] = term_cnt
                    self.i2w.append(term)
                    term_cnt += 1
                query_cvt.append(self.w2i[term])

            fw = open(
                os.path.join(
                    dst_path,
                    str(idx) +
                    '.txt'),
                mode="w",
                encoding="utf-8")
            fw.writelines(str(query_cvt))
            fw.close()
            cnt += 1
            if cnt >= max_num != -1:
                break
            line = f.readline()
        f.close()

    def save_dic(self, dic_path, dic_name):
        """[summary]

        Args:
            dic_path ([type]): [description]
            dic_name ([type]): [description]
        """
        print('dictionary with', len(self.i2w), 'terms')
        dir_name = os.path.join(dic_path, dic_name)
        print(dir_name)
        if os.path.exists(dir_name):
            os.remove(dir_name)
        with open(dir_name, 'wb') as f:
            pickle.dump(self, f)

    def load_dic(self, dic_path, dic_name):
        """[summary]

        Args:
            dic_path ([type]): [description]
            dic_name ([type]): [description]

        Returns:
            [type]: [description]
        """
        print(os.path.join(dic_path, dic_name))
        with open(os.path.join(dic_path, dic_name), 'rb') as f:
            a = pickle.load(f)
            # self = a
            # return self
            return a

    def get_raw_word(self, idx):
        """[summary]

        Args:
            idx ([type]): [description]

        Returns:
            [type]: [description]
        """
        return self.i2w[idx]

    def get_word_idx(self, word):
        """[summary]

        Args:
            word ([type]): [description]

        Returns:
            [type]: [description]
        """
        return self.w2i.get(word, -1)

def main():
    """[summary]
    """
    # this script can read the raw docs and queries and convert them into numbers
    # meanwhile, it splits all the docs and queries into docs/ and queries/,
    # respectively
    dic = Dictionary()
    # dic = dic.load_dic('./data/', 'dic_d4000000_q10000000')
    dic.convert_doc('./data/docs/', './data/msmarco-docs.tsv', 0, DOC_NUM)
    dic.convert_query(
        './data/queries/',
        './data/msmarco-doctrain-queries.tsv',
        0,
        QUERY_NUM)
    dic.save_dic('./data/', 'dic_d' + str(DOC_NUM) + '_q' + str(QUERY_NUM))
    exit(0)

if __name__ == '__main__':
    main()

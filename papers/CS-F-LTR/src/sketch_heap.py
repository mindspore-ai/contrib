"""[summary]

Returns:
    [type]: [description]

Yields:
    [type]: [description]
"""
import heapq
import hashlib
import random
from collections import defaultdict, Counter


class SketchHeap:
    """[summary]
    """
    def __init__(self, a=5, k=150, d=10, w=60):
        """
            a: int, times of k
            k: int, number of top
            d: int, height of sketch
            w: int, width of sketch
        """
        self._a = a
        self._k = k
        self._d = d
        self._w = w
        self._sketch = [[[] for j in range(w)] for i in range(d)]
        self._aid_dict = defaultdict(int)

    def clear_dict(self):
        """[summary]
        """
        self._aid_dict.clear()

    def push_in_dict(self, word, value=1):
        """[summary]

        Args:
            word ([type]): [description]
            value (int, optional): [description]. Defaults to 1.
        """
        self._aid_dict[word] += value

    def build_sketch_heap(self, doc_id=""):
        """[summary]

        Args:
            doc_id (str, optional): [description]. Defaults to "".
        """
        for word in self._aid_dict:
            self.add(word, doc_id, self._aid_dict[word])

    def contract(self):
        """[summary]
        """
        for i in range(self._d):
            for j in range(self._w):
                if len(self._sketch[i][j]) > self._a * self._k:
                    tmp = heapq.nlargest(
                        self._a * self._k, [(self._sketch[i][j][key], key) for key in self._sketch[i][j]])
                    self._sketch[i][j].clear()
                    for each in tmp:
                        self._sketch[i][j][each[1]] = each[0]

    def _hash(self, x):
        """[summary]

        Args:
            x ([type]): [description]

        Yields:
            [type]: [description]
        """
        md5 = hashlib.md5(str(hash(x)).encode("utf8"))
        for i in range(self._d):
            md5.update(str(i).encode("utf8"))
            yield int(md5.hexdigest(), 16) % self._w

    def hash(self, x):
        """[summary]

        Args:
            x ([type]): [description]

        Returns:
            [type]: [description]
        """
        gen = self._hash(x)
        return [i for i in gen]

    # def add(self, word, doc_id="", value=1):
    #     for table, i in zip(self._sketch, self._hash(word)):
    #         table[i][doc_id] += value

    def add(self, word, doc_id="", value=1):
        """[summary]

        Args:
            word ([type]): [description]
            doc_id (str, optional): [description]. Defaults to "".
            value (int, optional): [description]. Defaults to 1.
        """
        for table, i in zip(self._sketch, self._hash(word)):
            heapq.heappush(table[i], (value, doc_id))
            if len(table[i]) > self._a * self._k:
                heapq.heappop(table[i])

    # def query_hash(self, x_hashed, ratio=0.8):
    #    keys = []
    #    for table, i in zip(self._sketch, x_hashed):
    #        keys.extend(table[i].keys())
    #    counter = Counter(keys)
    #    min_cnt = dict()
    #    for table, i in zip(self._sketch, x_hashed):
    #        for key in table[i]:
    #            if counter[key] > ratio * self._d:
    #                min_cnt[key] = min(min_cnt[key], table[i][key]) if key in min_cnt else table[i][key]
    #    return [(min_cnt[each], each) for each in min_cnt]
    def query_hash(self, x_hashed, ratio=0.8):
        """[summary]

        Args:
            x_hashed ([type]): [description]
            ratio (float, optional): [description]. Defaults to 0.8.

        Returns:
            [type]: [description]
        """
        keys = []
        for table, i in zip(self._sketch, x_hashed):
            keys.extend([each[1] for each in table[i]])
        counter = Counter(keys)
        min_cnt = dict()
        for table, i in zip(self._sketch, x_hashed):
            for each in table[i]:
                key = each[1]
                if counter[key] > ratio * self._d:
                    min_cnt[key] = min(
                        min_cnt[key], each[0]) if key in min_cnt else each[0]
        return [(min_cnt[each], each) for each in min_cnt]

    def query(self, x, ratio=0.8):
        """[summary]

        Args:
            x ([type]): [description]
            ratio (float, optional): [description]. Defaults to 0.8.

        Returns:
            [type]: [description]
        """
        return self.query_hash(self.hash(x), ratio=ratio)

    def __str__(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        res = ""
        for each in self._sketch:
            res += (str(each) + "\n")
        return res

def main():
    """[summary]
    """
    # local test
    sketch_heap = SketchHeap(2, 150, 10, 60)
    doc1 = [1 for i in range(7)]
    doc1.extend([2 for i in range(5)])
    doc2 = [4 for i in range(10)]
    doc2.extend([5 for i in range(15)])
    doc2.extend([1 for i in range(3)])
    random.shuffle(doc1)
    random.shuffle(doc2)
    print(doc1)
    print(doc2)
    docs = [doc1, doc2]
    for i in range(len(docs)):
        d = docs[i]
        for w in d:
            sketch_heap.push_in_dict(w)
        sketch_heap.build_sketch_heap("doc" + str(i + 1))
        sketch_heap.clear_dict()
    sketch_heap.contract()
    print(sketch_heap)
    input()
    print(sketch_heap.hash(1))
    print(sketch_heap.query_hash(sketch_heap.hash(1)))

if __name__ == "__main__":
    main()

"""[summary]

Raises:
    ValueError: [description]

Returns:
    [type]: [description]

Yields:
    [type]: [description]
"""
# -*- coding: utf-8 -*-
import hashlib
import array
import numpy as np


class CountMinSketch():
    """
    A class for counting hashable items using the Count-min Sketch strategy.
    It fulfills a similar purpose than `itertools.Counter`.

    The Count-min Sketch is a randomized data structure that uses a constant
    amount of memory and has constant insertion and lookup times at the cost
    of an arbitrarily small overestimation of the counts.

    It has two parameters:
     - `m` the size of the hash tables, larger implies smaller overestimation
     - `d` the amount of hash tables, larger implies lower probability of
           overestimation.

    An example usage:

        from countminsketch import CountMinSketch
        sketch = CountMinSketch(1000, 10)  # m=1000, d=10
        sketch.add("oh yeah")
        sketch.add(tuple())
        sketch.add(1, value=123)
        print sketch["oh yeah"]       # prints 1
        print sketch[tuple()]         # prints 1
        print sketch[1]               # prints 123
        print sketch["non-existent"]  # prints 0

    Note that this class can be used to count *any* hashable type, so it's
    possible to "count apples" and then "ask for oranges". Validation is up to
    the user.
    """

    def __init__(self, m, d):
        """ `m` is the size of the hash tables, larger implies smaller
        overestimation. `d` the amount of hash tables, larger implies lower
        probability of overestimation.
        """
        if not m or not d:
            raise ValueError("Table size (m) and amount of hash functions (d)"
                             " must be non-zero")
        self.m = m
        self.d = d
        self.item_list = [i for i in range(self.d)]
        self.n = 0
        self.tables = []
        # self.pool = ThreadPool()
        # self.executor = concurrent.futures.ProcessPoolExecutor()
        for _ in range(d):
            table = array.array("l", (0 for _ in range(m)))
            self.tables.append(table)

    def hash(self, x):
        """[summary]

        Args:
            x ([type]): [description]

        Yields:
            [type]: [description]
        """
        # print("should not appear")
        md5 = hashlib.md5(str(hash(x)).encode("utf8"))
        for i in range(self.d):
            md5.update(str(i).encode("utf8"))
            yield int(md5.hexdigest(), 16) % self.m

    def hash2(self, x):
        """[summary]

        Args:
            x ([type]): [description]

        Returns:
            [type]: [description]
        """
        gen = self.hash(x)
        return [i for i in gen]

    # def hash2(self, x):
        # md5 = hashlib.md5(str(hash(x)).encode("utf8"))
    #    def f(i):
        # md5.update(str(i).encode("utf8"))
    #        md5 = hashlib.md5(str(hash(x*100 + i)).encode("utf8"))
    #        return int(md5.hexdigest(), 16) % self.m
    #   # results = self.pool.map(f, self.item_list)
    #    results = self.executor.map(f, self.item_list)
    #    return results

    def add(self, x, value=1):
        """
        Count element `x` as if had appeared `value` times.
        By default `value=1` so:

            sketch.add(x)

        Effectively counts `x` as occurring once.
        """
        self.n += value
        #hashes = self.hash2(x)
        # for table, i in zip(self.tables, hashes):
        for table, i in zip(self.tables, self.hash(x)):
            table[i] += value

    def query(self, x):
        """
        Return an estimation of the amount of times `x` has occurred.
        The returned value always overestimates the real value.
        """
        return min((table[i] for table, i in zip(self.tables, self.hash(x))))

    def query_hash(self, x_hashed):
        """[summary]

        Args:
            x_hashed ([type]): [description]

        Returns:
            [type]: [description]
        """
        return min((table[i] for table, i in zip(self.tables, x_hashed)))

    def query_median(self, x):
        """[summary]

        Args:
            x ([type]): [description]

        Returns:
            [type]: [description]
        """
        return abs(np.median(
            [table[i] - table[i + 1 if i % 2 == 0 else i - 1] for table, i in zip(self.tables, self.hash(x))]
        ))

    def query_hash_median(self, x_hashed):
        """[summary]

        Args:
            x_hashed ([type]): [description]

        Returns:
            [type]: [description]
        """
        return abs(np.median(
            [table[i] - table[i + 1 if i % 2 == 0 else i - 1] for table, i in zip(self.tables, x_hashed)]
        ))

    def __getitem__(self, x):
        """
        A convenience method to call `query`.
        """
        return self.query(x)

    def __len__(self):
        """
        The amount of things counted. Takes into account that the `value`
        argument of `add` might be different from 1.
        """
        return self.n

import numpy as np

class MyIterable:
    def __init__(self, dataset):
        self._index = 0
        self._data = dataset
        self.len = len(self._data)
        self.meta = []

    def get_meta_list(self):
        self.meta = []
        for i in range(self.len):
            self.meta.append(self._data[i][4])

    def get_meta_item(self,index):
        return self._data[index][4]

    def __next__(self):
        if self._index >= len(self._data):
            raise StopIteration
        else:
            item = (self._data[self._index][0], self._data[self._index][1],
                    self._data[self._index][2], self._data[self._index][3])
            self._index += 1
            return item

    def __iter__(self):
        self._index = 0
        return self

    def __len__(self):
        return len(self._data)

class MyIterable_meta:
    def __init__(self, dataset):
        super().__init__()
        self._index = 0
        self._meta = dataset
        self.len = len(self._meta)
        self.meta = []

    def get_item(self):
        self.meta = []
        for i in range(self.len):
            self.meta.append(self._meta[i][4])
        

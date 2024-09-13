"""
创建词典和数据集划分
"""
import random
import json
import os
from PIL import Image
import numpy as np

text0 = 'dataset/multimodel_data/text0.txt'
text1 = 'dataset/multimodel_data/text1.txt'
text5 = 'dataset/multimodel_data/text5.txt'

wordset = set()

data = []

with open(text0, encoding='utf8') as f:
    for line in f:
        # print(line)
        line = line.strip()
        parts = line.split(' ')
        word = parts[:-2]
        img = parts[-2]
        label = parts[-1]
        # print(word,'dataset/multimodel_data/0/'+img,label)
        img_path = 'dataset/multimodel_data/0/' + img
        if not os.path.exists(img_path):
            continue
        image = Image.open(img_path)
        image = np.copy(image)
        if len(image.shape) != 3 or image.shape[2] != 3:
            continue

        words = ' '.join(word)
        data.append((words, 'dataset/multimodel_data/0/' + img, label))
        wordset = wordset.union(set(word))

with open(text1, encoding='utf8') as f:
    for line in f:
        # print(line)
        line = line.strip()
        parts = line.split(' ')
        word = parts[:-2]
        img = parts[-2]
        label = parts[-1]

        img_path = 'dataset/multimodel_data/1/' + img
        if not os.path.exists(img_path):
            continue
        image = Image.open(img_path)
        image = np.copy(image)
        if len(image.shape) != 3 or image.shape[2] != 3:
            continue

        # print(word,'dataset/multimodel_data/1/'+img,label)
        words = ' '.join(word)
        data.append((words, img_path, label))
        wordset = wordset.union(set(word))

with open(text5, encoding='utf8') as f:
    for line in f:
        # print(line)
        line = line.strip()
        parts = line.split(' ')
        word = parts[:-2]
        img = parts[-2]
        label = parts[-1]

        img_path = 'dataset/multimodel_data/5/' + img
        if not os.path.exists(img_path):
            continue
        image = Image.open(img_path)
        image = np.copy(image)

        if len(image.shape) != 3 or image.shape[2] != 3:
            continue

        # print(word,'dataset/multimodel_data/3/'+img,label)
        words = ' '.join(word)
        data.append((words, img_path, label))
        wordset = wordset.union(set(word))

print(len(wordset))

word2id = {}
for i, word in enumerate(list(wordset)):
    word2id[word] = i + 1
word2id['<pad>'] = 0

with open('dataset/word2id.json', 'w', encoding='utf8') as f:
    json.dump(word2id, f, ensure_ascii=False, indent=4)

random.shuffle(data)
print(len(data))

train_data = data[:int(len(data) * 0.6)]
val_data = data[int(len(data) * 0.6):int(len(data) * 0.8)]
test_data = data[int(len(data) * 0.8):]

print(len(train_data), len(val_data), len(test_data))

with open('dataset/train/train.txt', 'w', encoding='utf8') as f:
    for t in train_data:
        f.write(t[0] + '\t' + t[1] + '\t' + str(t[2]) + '\n')

with open('dataset/val/val.txt', 'w', encoding='utf8') as f:
    for t in val_data:
        f.write(t[0] + '\t' + t[1] + '\t' + str(t[2]) + '\n')

with open('dataset/test/test.txt', 'w', encoding='utf8') as f:
    for t in test_data:
        f.write(t[0] + '\t' + t[1] + '\t' + str(t[2]) + '\n')

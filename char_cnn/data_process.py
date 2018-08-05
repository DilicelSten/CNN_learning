"""
created on:2018/8/4
author:DilicelSten
target: preprocess data
finished on:2018/8/4
"""
import numpy as np
from char_cnn.config import Config
import pandas as pd
import os
from sklearn.metrics import accuracy_score

dir_path = "/media/iiip/文档/user_profiling/20_newsgroups/"


def label_transform():
    """
    turn texual label to 1, 0
    :return:
    """
    label_dict = {}
    label_lst = os.listdir(dir_path)
    for i in range(len(label_lst)):
        label_dict[label_lst[i]] = int(i + 1)
    return label_dict


def score(pred, label):
    """
    compute the pre
    :param pred:
    :param label:
    :return:
    """
    label = np.argmax(label, axis=1).tolist()
    acc = accuracy_score(pred, label)
    return acc


def loadData():
    dataset = pd.read_csv(Config.data_source)
    content = []
    category = []
    for ids, class_name in enumerate(dataset.classname):
        category.append(class_name)
    for ids, con in enumerate(dataset.content):
        content.append((label_transform()[category[ids]], con))

    data = np.asarray(content)

    dev_sample_index = -1 * int(Config.dev_sample_percentage * float(len(data)))
    train, dev = data[:dev_sample_index], data[dev_sample_index:]
    return train, dev


class Data(object):

    def __init__(self,
                 data,
                 alphabet = Config.alphabet,
                 l0 = Config.l0,
                 batch_size = Config.batch_size,
                 num_class = Config.num_class):

        self.alphabet = alphabet
        self.alphabet_size = len(self.alphabet)
        self.dict = {}
        self.num_class = num_class

        for i, c in enumerate(self.alphabet):
            self.dict[c] = i + 1

        self.length = l0
        self.batch_size = batch_size
        self.data = data
        self.shuffled_data = self.data

    def shuffleData(self):
        np.random.seed(256)
        data_size = len(self.data)

        shuffle_indices = np.random.permutation(np.arange(data_size))
        self.shuffled_data = self.data[shuffle_indices]

    def getBatch(self, batch_num=0):
        data_size = len(self.data)
        start_index = batch_num * self.batch_size
        end_index = min((batch_num + 1) * self.batch_size, data_size)
        return self.shuffled_data[start_index:end_index]

    def GetBatchToIndices(self, batch_num=0):
        data_size = len(self.data)
        start_index = batch_num * self.batch_size
        end_index = data_size if self.batch_size == 0 else min((batch_num + 1) * self.batch_size, data_size)
        batch_texts = self.shuffled_data[start_index:end_index]
        batch_indices = []
        one_hot = np.eye(self.num_class, dtype='int64')
        classes = []
        for c, s in batch_texts:
            batch_indices.append(self.strToIndexs(s))
            c = int(c) - 1
            classes.append(one_hot[c])
        return np.asarray(batch_indices, dtype='int64'), classes

    def strToIndexs(self, s):
        s = s.lower()
        m = len(s)
        n = min(m, self.length)
        str2idx = np.zeros(self.length, dtype='int64')
        k = 0
        for i in range(1, n+1):
            c = s[-i]
            if c in self.dict:
                str2idx[i-1] = self.dict[c]
        return str2idx

    def getLength(self):
        return len(self.data)


if __name__ == '__main__':
    train_set, test_set = loadData()
    train_data = Data(data=train_set)
    test_data = Data(data=test_set)
    print(train_data.getLength())
    print(test_data.getLength())






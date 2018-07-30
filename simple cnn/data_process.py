"""
created on:2018/7/28
author: DilicelSten
target:preprocess data
finished on:2018/7/28
"""
import pandas as pd
import os
import numpy as np

data_path = '../data/20newsgroup.csv'
dir_path = "/media/iiip/文档/user_profiling/20_newsgroups/"


def label_transform():
    """
    turn texual label to 1, 0
    :return:
    """
    label_dict = {}
    label_lst = os.listdir(dir_path)
    for i in range(len(label_lst)):
        lst = [0] * 20
        lst[i] = 1
        label_dict[label_lst[i]] = lst
    return label_dict


def load_data():
    """
    extract data
    :return: content, label
    """
    label_dic = label_transform()
    dataset = pd.read_csv(data_path)
    content = []
    category = []
    for ids, con in enumerate(dataset.content):
        content.append(con)
    for ids, class_name in enumerate(dataset.classname):
        category.append(label_dic[class_name])
    return content, category


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    a batch iterator for a dataset
    :param data:
    :param batch_size:
    :param num_epochs:
    :param shuffle:
    :return:
    """
    data = np.array(data)
    data_size = len(data)
    num_batch_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # shuffle the data for each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batch_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index: end_index]

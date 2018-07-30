"""
created on:2018/7/26
author:DilicelSten
target:data preprocess
finished on:2018/7/27
"""
import os
import re
import csv
import pandas as pd


def read_wrong_document():
    """
    remove some wrong documents
    :return: doc_list
    """
    with open("/home/iiip/PycharmProjects/Experiment/data/problem document", 'r') as f:
        raw_list = re.findall('convert(.*?)from', f.read())
        doc_list = [x.replace(" `", '').replace("' ", '') for x in raw_list]
    return doc_list


def clean_data(string):
    """
    data preprocess
    :param string: document
    :return: cleaned document
    """
    """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"` ,", "", string)
    return string.strip().lower()


def load_data():
    """
    loading data and preprocess
    :return: csv file
    """
    content = {}
    wrong_doc = read_wrong_document()  # remove 66 files
    dir_path = "/media/iiip/文档/user_profiling/20_newsgroups/"
    for each in os.listdir(dir_path):
        print(each)
        f_path = dir_path + each
        for doc in os.listdir(f_path):
            result = []
            if doc in wrong_doc:
                # print(doc + 'is wrong')
                continue
            else:
                # print(doc)
                with open(f_path + '/' + doc, 'r') as fr:
                    if fr.read() == '':
                        print(doc + " is null")
                    else:
                        result.append(each, )
                        result.append(clean_data(fr.read()))
                        content[doc] = result
    header = ["classname", "content"]
    file = open('20newsgroup.csv', 'w')
    writer = csv.DictWriter(file, fieldnames=header)
    writer.writeheader()
    data = []
    for key in content:
        dic = dict(map(lambda x, y: [x, y], header, content[key]))
        data.append(dic)
    writer.writerows(data)


def read_csv():
    file = pd.read_csv('20newsgroup.csv', sep=',')
    print(file.classname)


if __name__ == '__main__':
    load_data()
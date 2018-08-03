"""
created on:2018/7/31
author: DilicelSten
target: turn data into vector using google vectors
finished on:2018/8/2
"""
from collections import defaultdict
import pickle
import numpy as np
from simple_cnn import data_process
import pandas as pd
import datetime
import gensim


def get_max_len():
    """
    get max length
    :return:
    """
    # max_len = 0
    length = []
    contents = data_process.load_data()[0]
    for d_id, content in enumerate(contents):
        words = content.split()
        length.append(len(words))
    return int(np.max(np.array(length)))


def create_word_cab(cv=10):
    """
    Using dataset to create its own vocabulary
    :return:
    """
    contents, labels = data_process.load_data()
    # final data
    word_cab = defaultdict(float)  # word frequency
    revs = []
    for i in range(len(labels)):
        words = set(contents[i].split(" "))
        for word in words:
            word_cab[word] += 1
        datum = {
            "y": labels[i],  # like[0,0,0,..,1]
            "text": contents[i],  # sentence
            "num_words": len(words),
            "split": np.random.randint(0, cv)}  # cross-validation
        revs.append(datum)
    return word_cab, revs


def get_vocab_and_W(word_cab, w2v_file, vocab_save_path=None, embedding_save_path=None):
    """
    Loading the pretrained Google vector and deal with the unexist word
    :param word_cab:
    :param vocab_save_path:
    :param embedding_save_path:
    :return:
    """
    vocab = {}
    i = 1
    min_freq = 1
    vocab['UNK'] = 0
    for word, freq in word_cab.items():
        if freq >= min_freq:
            vocab[word] = i
            i += 1
    print("vocab size: ", len(vocab))

    if vocab_save_path is not None:
        with open(vocab_save_path, 'wb') as g:
            pickle.dump(vocab, g)

    embed_W = [np.random.uniform(-0.25, 0.25, 300) for j in range(len(vocab))]
    w2v = gensim.models.KeyedVectors.load_word2vec_format(w2v_file, binary=True)

    num = 0
    for word in vocab:
        index = vocab[word]
        if word in w2v:
            print(word)
            num += 1
            embed_W[index] = np.array(w2v[word])
    embed_W = np.array(embed_W)
    print(embed_W)
    print("valid num: ", num)

    if embedding_save_path is not None:
        with open(embedding_save_path, 'wb') as f:
            pickle.dump(embed_W, f)


def get_index_by_sentence_list(word_ids, sentence_list, maxlen):
    indexs = []
    # words_length = len(word_ids)
    for sentence in sentence_list:
        # get a sentence
        sentence_indexs = []
        words = sentence.split()
        for word in words:
            sentence_indexs.append(word_ids[word])
        # padding sentence to maxlen
        length = len(sentence_indexs)
        if length < maxlen:
            for i in range(maxlen - length):
                sentence_indexs.append(0)
        # add a sentence vector
        indexs.append(sentence_indexs)

    return np.array(indexs)


def get_train_test(cv):
    vocab = pickle.load(open("/media/iiip/文档/user_profiling/result/vocab/vocab.pickle", 'rb'))
    revs = create_word_cab(cv=10)[1]
    data_set_df = pd.DataFrame(revs)
    max_len = get_max_len()
    # DataFrame
    # y text num_words spilt
    # 1 'I like this movie' 4 3
    # data_set_df = data_set_df.sample(frac=1)  # 打乱顺序

    data_set_cv_train = data_set_df[data_set_df['split'] != cv]  # 训练集
    data_set_cv_test = data_set_df[data_set_df['split'] == cv]  # 测试集

    # train
    train_y = np.array(data_set_cv_train['y'].tolist(), dtype='int')
    test_y = np.array(data_set_cv_test['y'].tolist(), dtype='int')

    train_sentence_list = data_set_cv_train['text'].tolist()
    test_sentence_list = data_set_cv_test['text'].tolist()

    train_x = get_index_by_sentence_list(vocab, train_sentence_list, max_len)
    test_x = get_index_by_sentence_list(vocab, test_sentence_list, max_len)

    return train_x, train_y, test_x, test_y


if __name__ == '__main__':
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    w2v_file = '/media/iiip/文档/user_profiling/word2vec/GoogleNews-vectors-negative300.bin'
    print("Loading data....")
    word_cab, revs = create_word_cab()
    print("Data loaded!")
    print("Number of documents:" + str(len(revs)))
    print("Size of vocab: " + str(len(word_cab)))
    senten_max_len = np.max(pd.DataFrame(revs)["num_words"])
    print("The max length of all the document is {}".format(senten_max_len))
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print('load word2vec vectors...')
    vocab_save_path = '/media/iiip/文档/user_profiling/result/vocab/vocab.pickle'
    embedding_save_path = '/media/iiip/文档/user_profiling/result/vocab/embedding.pickle'
    get_vocab_and_W(word_cab, w2v_file, vocab_save_path=vocab_save_path, embedding_save_path=embedding_save_path)
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    pass
    # print(get_max_len())
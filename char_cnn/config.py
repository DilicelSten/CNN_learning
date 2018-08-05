"""
created on:2018/8/3
author:DilicelSten
target: config file
finished on:2018/8/4
"""


class TrainingConfig(object):
    """
    used in training process
    """
    decay_step = 15000
    decay_rate = 0.95
    epochs = 30
    evaluate_every = 100
    checkpoint_every = 100


class ModelConfig(object):
    """
    used in char cnn model
    """
    conv_layer = [[256, 7, 3],
                  [256, 7, 3],
                  [256, 3, None],
                  [256, 3, None],
                  [256, 3, None],
                  [256, 3, 3]]
    fc_layer = [1024, 1024]
    dropout_keep_prob = 0.9
    learning_rate = 0.0005


class Config(object):
    """
    all the configuration
    """
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
    alphabet_size = len(alphabet)
    l0 = 3000
    batch_size = 64
    num_class = 20

    # data
    data_source = '../data/20newsgroup.csv'
    dev_sample_percentage = 0.1

    training = TrainingConfig()
    model = ModelConfig()


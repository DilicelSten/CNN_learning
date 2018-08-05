"""
created on:2019/8/4
author:DilicelSten
target: create char cnn model
finished on:2018/8/4
"""
import tensorflow as tf
from math import sqrt


class CharCNN(object):
    """
    A CNN for text classification.
    Use an embedding layer, followed by a convolutional layer, max-pooling and softmax layer.
    """
    def __init__(
            self, l0, num_class, alphabet_size, conv_layers, fc_layers, l2_reg_lambda=0.0):

        # placeholder for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, l0], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_class], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # keeping track of l2 regularization loss
        l2_loss = tf.constant(0.0)

        # Embedding layer( can use another method)
        with tf.device('/cpu:0'), tf.name_scope("embedding"):

            # Quantization (It's important!)
            Q = tf.concat(
                [
                    tf.zeros([1, alphabet_size]),  # Zero padding vector for out of alphabet characters
                    tf.one_hot(list(range(alphabet_size)), alphabet_size, 1.0, 0.0)
                    # one-hot vector representation for alphabets
                ],
                0,
                name='Q')
            self.x_image = tf.nn.embedding_lookup(Q, self.input_x)
            print(self.x_image)
            self.x_flat = tf.expand_dims(self.x_image, -1)  # Add the channel dim, thus the shape of x is [batch_size, l0, alphabet_size, 1]

        for i, cl in enumerate(conv_layers):
            with tf.name_scope("conv_layer-%s" % (i + 1)):
                filter_width = self.x_flat.get_shape()[2].value
                filter_shape = [cl[1], filter_width, 1, cl[0]]

                stdv = 1 / sqrt(cl[0] * cl[1])
                w_conv = tf.Variable(tf.random_uniform(filter_shape, minval=-stdv, maxval=stdv),
                                     dtype='float32', name='w')

                b_conv = tf.Variable(tf.random_uniform(shape=[cl[0]], minval=-stdv, maxval=stdv), name='b')

                conv = tf.nn.conv2d(
                    self.x_flat,
                    w_conv,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv"
                )
                h_conv = tf.nn.bias_add(conv, b_conv)

                if not cl[-1] is None:
                    ksize_shape = [1, cl[2], 1, 1]
                    h_pool = tf.nn.max_pool(
                        h_conv,
                        ksize=ksize_shape,
                        strides=ksize_shape,
                        padding='VALID',
                        name='pool')
                else:
                    h_pool = h_conv

                self.x_flat = tf.transpose(h_pool, [0, 1, 3, 2], name='transpose')

        with tf.name_scope('reshape'):
            fc_dim = self.x_flat.get_shape()[1].value * self.x_flat.get_shape()[2].value
            self.x_flat = tf.reshape(self.x_flat, [-1, fc_dim])

        weights = [fc_dim] + fc_layers
        for i, fl in enumerate(fc_layers):
            with tf.name_scope('fc_layer-%s' % (i + 1)):
                stdv = 1 / sqrt(weights[i])
                w_fc = tf.Variable(tf.random_uniform([weights[i], fl], minval=-stdv, maxval=stdv),
                                   dtype='float32', name='w')
                b_fc = tf.Variable(tf.random_uniform(shape=[fl], minval=-stdv, maxval=stdv), dtype='float32', name='b')
                self.x_flat = tf.nn.relu(tf.matmul(self.x_flat, w_fc) + b_fc)

                with tf.name_scope('drop_out'):
                    self.x_flat = tf.nn.dropout(self.x_flat, self.dropout_keep_prob)

        with tf.name_scope('output_layer'):
            stdv = 1 / sqrt(weights[-1])
            w_out = tf.Variable(tf.random_uniform([fc_layers[-1], num_class], minval=-stdv, maxval=stdv),
                                dtype='float32', name='W')
            b_out = tf.Variable(tf.random_uniform(shape=[num_class], minval=-stdv, maxval=stdv), name='b')
            self.y_pred = tf.nn.xw_plus_b(self.x_flat, w_out, b_out, name="y_pred")
            self.predictions = tf.argmax(self.y_pred, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.y_pred, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")





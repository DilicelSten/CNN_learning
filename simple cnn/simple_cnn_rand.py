"""
created on:2018/7/29
author: DilicelSten
target: train process(rand cnn)
finished on:2018/7/29
"""
from tensorflow.contrib import learn
import numpy as np
from simple_cnn import data_process
import tensorflow as tf
from simple_cnn.text_cnn import TextCNN
import os
import datetime


# params
# ==================================================================
# data loading param
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation.")

# model hyperparameters
# embedding (dimension: 256; conv kernel: 4; num: 128; dropout: 0.5)
tf.flags.DEFINE_integer("embedding_dim", 256, "Dimensionality of character embedding(default: 128)")
tf.flags.DEFINE_string("filter_sizes", "2,3,4,5", "Comma-separated filter sizes (default: '2,3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# training params
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 10, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

# misc params
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()


def preprocess():
    """
    turn data into inputs and split sets(rand cnn)
    :return:
    """
    # load data
    print("loading data...")
    x_test, y = data_process.load_data()  # 16259

    # build vocabulary
    max_document_length = max([len(x.split(" ")) for x in x_test])  # attention: null data influence  (20705)
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    x = np.array(list(vocab_processor.fit_transform(x_test)))
    y = np.array(y)

    # randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # split train/test set (cross-validation ?)
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

    del x, y, x_shuffled, y_shuffled

    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    return x_train, y_train, vocab_processor, x_dev, y_dev


def train(x_train, y_train, vocab_processor, x_dev, y_dev):
    """
    training process
    :param x_train:
    :param y_train:
    :param vocab_processor:
    :param x_dev:
    :param y_dev:
    :return:
    """
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement
        )
        # session_conf.gpu_options.per_process_gpu_memory_fraction = 0.5 # maximun alloc gpu50% of MEM
        # session_conf.gpu_options.allow_growth = True #allocate dynamically
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda
            )

            # define training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # output directory
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs"))

            # Checkpoint directory.
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # write vocaulary
            vocab_processor.save(os.path.join(out_dir, "vocab"))

            # initial all variables
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch):
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, loss, accuracy = sess.run(
                    [train_op, global_step, cnn.loss, cnn.accuracy],
                    feed_dict=feed_dict
                )
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

            def dev_step(x_batch, y_batch):
                """
                Something wrong happened so we should use batch
                """
                num = 20
                x_batch = x_batch.tolist()
                y_batch = y_batch.tolist()
                l = len(y_batch)
                l_20 = int(l / num)
                x_set = []
                y_set = []
                for i in range(num - 1):
                    x_temp = x_batch[i * l_20:(i + 1) * l_20]
                    x_set.append(x_temp)
                    y_temp = y_batch[i * l_20:(i + 1) * l_20]
                    y_set.append(y_temp)
                x_temp = x_batch[(num - 1) * l_20:]
                x_set.append(x_temp)
                y_temp = y_batch[(num - 1) * l_20:]
                y_set.append(y_temp)

                # each batch computation and then add up
                lis_loss = []
                lis_acc = []
                for i in range(num):
                    feed_dict = {
                        cnn.input_x: np.array(x_set[i]),
                        cnn.input_y: np.array(y_set[i]),
                        cnn.dropout_keep_prob: 1.0
                    }
                    step, loss, accuracy = sess.run(
                        [global_step, cnn.loss, cnn.accuracy],
                        feed_dict=feed_dict
                    )
                    lis_loss.append(loss)
                    lis_acc.append(accuracy)
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                print("test_loss and test_acc" + "\t\t" + str(sum(lis_loss) / num) + "\t\t" + str(sum(lis_acc) / num))

            # generate batches
            batches = data_process.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs
            )

            # training loop
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_step(x_dev, y_dev)
                    print("")
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))


def main(argv=None):
    x_train, y_train, vocab_processor, x_dev, y_dev = preprocess()
    train(x_train, y_train, vocab_processor, x_dev, y_dev)


if __name__ == '__main__':
    tf.app.run()
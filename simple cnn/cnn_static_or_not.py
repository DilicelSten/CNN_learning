"""
created on:2018/8/2
author: DilicelSten
target: train textcnn(2 versions: static and non static)
finished on:2018/8/2
"""
import numpy as np
from simple_cnn import data_process, data_2_vec
import tensorflow as tf
from simple_cnn.text_cnn_static import TextCNN
import os
import datetime


# params
# ==================================================================
# data loading param
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation.")

# model hyperparameters
# embedding (dimension: 256; conv kernel: 4; num: 128; dropout: 0.5)
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding(default: 128)")  # Attention!
tf.flags.DEFINE_string("filter_sizes", "2,3,4,5", "Comma-separated filter sizes (default: '2,3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# training params
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 20, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

# misc params
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS


def train(x_train, y_train, x_dev, y_dev, static_flag):
    """
    training process
    :param x_train:
    :param y_train:
    :param x_dev:
    :param y_dev:
    :return:
    """
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement
        )

        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                embedding_size=FLAGS.embedding_dim,
                static_flag=static_flag,
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
            out_dir = os.path.abspath(os.path.join('/media/iiip/文档/user_profiling/', "runs"))

            # Checkpoint directory.
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

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
                batch_num = 20
                x_batch = x_batch.tolist()
                y_batch = y_batch.tolist()
                l = len(y_batch)
                l_20 = int(l / batch_num)
                x_set = []
                y_set = []
                for i in range(batch_num - 1):
                    x_temp = x_batch[i * l_20:(i + 1) * l_20]
                    x_set.append(x_temp)
                    y_temp = y_batch[i * l_20:(i + 1) * l_20]
                    y_set.append(y_temp)
                x_temp = x_batch[(batch_num - 1) * l_20:]
                x_set.append(x_temp)
                y_temp = y_batch[(batch_num - 1) * l_20:]
                y_set.append(y_temp)

                # get the whole predictions and then compute accuracy
                lis_predictions = []
                lis_labels = []
                for i in range(batch_num):
                    feed_dict = {
                        cnn.input_x: np.array(x_set[i]),
                        cnn.dropout_keep_prob: 1.0
                    }
                    step, predictions = sess.run(
                        [global_step, cnn.predictions],
                        feed_dict=feed_dict
                    )
                    lis_predictions.extend(predictions)  # Attention! Don't use append
                    lis_labels.extend(y_set[i])
                print("test_acc" + "\t\t" + str(data_process.score(lis_predictions, lis_labels)))

            # generate batches
            batches = data_process.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs
            )

            # training loop
            for batch in batches:
                x_batch, y_batch = zip(*batch)   # data ok!
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
    x_train, y_train, x_dev, y_dev = data_2_vec.get_train_test(1)
    train(x_train, y_train, x_dev, y_dev, static_flag=True)  # True = static  False = nonstatic


if __name__ == '__main__':
    tf.app.run()
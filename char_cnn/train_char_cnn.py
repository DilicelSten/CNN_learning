"""
created on:2018/8/4
author:DilicelSten
target: train char cnn model
finished on:2018/8/4
"""

from char_cnn import config, data_process, char_cnn_model
import tensorflow as tf
import numpy as np
import os
import time
import datetime


if __name__ == '__main__':
    train_set, test_set = data_process.loadData()
    train_data = data_process.Data(data=train_set)
    dev_data = data_process.Data(data=test_set)

    num_batches_per_epoch = int(train_data.getLength() / config.Config.batch_size) + 1
    num_batch_dev = dev_data.getLength()

    print("Loading")

    print("Training ======>")

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True,
                                      log_device_placement=False)
        sess = tf.Session(config=session_conf)

        with sess.as_default():

            char_cnn = char_cnn_model.CharCNN(l0=config.Config.l0,
                                              num_class=config.Config.num_class,
                                              alphabet_size=config.Config.alphabet_size,
                                              conv_layers=config.ModelConfig.conv_layer,
                                              fc_layers=config.ModelConfig.fc_layer)

            global_step = tf.Variable(0, trainable=False)

            learning_rate = config.ModelConfig.learning_rate

            optimizer = tf.train.AdamOptimizer(learning_rate)
            grads_and_vars = optimizer.compute_gradients(char_cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # output directory
            time_stamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join('/media/iiip/文档/user_profiling/', "runs", time_stamp))
            print("Writing to {}\n".format(out_dir))

            # checkpoint
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables())

            # initialize
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch):
                feed_dict = {
                    char_cnn.input_x: x_batch,
                    char_cnn.input_y: y_batch,
                    char_cnn.dropout_keep_prob: config.ModelConfig.dropout_keep_prob
                }
                _, step, loss, accuracy = sess.run(
                    [train_op, global_step, char_cnn.loss, char_cnn.accuracy],
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
                        char_cnn.input_x: np.array(x_set[i]),
                        char_cnn.dropout_keep_prob: 1.0
                    }
                    step, predictions = sess.run(
                        [global_step, char_cnn.predictions],
                        feed_dict=feed_dict
                    )
                    lis_predictions.extend(predictions)  # Attention! Don't use append
                    lis_labels.extend(y_set[i])
                print("test_acc" + "\t\t" + str(data_process.score(lis_predictions, lis_labels)))


            for e in range(config.TrainingConfig.epochs):
                print("Epoch: ", e)
                train_data.shuffleData()
                for k in range(num_batches_per_epoch):
                    print("batch: ", k)

                    batch_x, batch_y = train_data.GetBatchToIndices(k)

                    train_step(batch_x, batch_y)
                    current_step = tf.train.global_step(sess, global_step)

                    if current_step % config.TrainingConfig.evaluate_every == 0:
                        dex_x, dev_y = dev_data.GetBatchToIndices()
                        print("\nEvaluation: ")
                        dev_step(dex_x, dev_y)
                        print("")

                    if current_step % config.TrainingConfig.checkpoint_every == 0:
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        print("Saved model checkpoint to {}\n".format(path))




# @Author:      HgS_1217_
# @Create Date: 2017/10/9

import random
import tensorflow as tf
from config import CKPT_PATH


def variable_with_weight_decay(name, shape, initializer, wd=None):
    var = tf.get_variable(name, shape, initializer=initializer)
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def add_loss_summaries(total_loss):
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    for l in losses + [total_loss]:
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op


def batch_norm_layer(x, is_training):
    x_shape = x.get_shape()
    params_shape = x_shape[-1:]

    axis = list(range(len(x_shape) - 1))

    beta = variable_with_weight_decay('beta', params_shape, initializer=tf.truncated_normal_initializer())
    gamma = variable_with_weight_decay('gamma', params_shape, initializer=tf.truncated_normal_initializer())

    batch_mean, batch_var = tf.nn.moments(x, axis, name='moments')
    ema = tf.train.ExponentialMovingAverage(decay=0.5)

    def mean_var_with_update():
        ema_apply_op = ema.apply([batch_mean, batch_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)

    mean, var = tf.cond(is_training, mean_var_with_update,
                        lambda: (ema.average(batch_mean), ema.average(batch_var)))
    normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed


def fc_layer(x, feature_num, is_training, name=None, relu_flag=True):
    with tf.variable_scope(name) as scope:
        w = variable_with_weight_decay('w', shape=[x.get_shape()[-1], feature_num],
                                       initializer=tf.orthogonal_initializer(), wd=None)
        b = tf.get_variable("b", [feature_num], initializer=tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(tf.matmul(x, w), b)
        norm = batch_norm_layer(bias, is_training)
        return tf.nn.relu(norm) if relu_flag else norm


class FCNet:
    def __init__(self, raws, labels, test_raws, test_labels, keep_pb=0.5, batch_size=5000, epoch_size=20000,
                 learning_rate=0.001, start_step=0):
        self.raws = raws
        self.labels = labels
        self.test_raws = test_raws
        self.test_labels = test_labels
        self.keep_pb = keep_pb
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.start_step = start_step
        self.learning_rate = learning_rate

        self.x = tf.placeholder(tf.float32, shape=[None, 75], name="input_x")
        self.y = tf.placeholder(tf.float32, shape=[None, 1], name="input_y")
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        self.is_training = tf.placeholder(tf.bool, name="is_training")
        self.global_step = tf.Variable(0, trainable=False)

    def build_network(self, x, y, is_training):
        flat = tf.reshape(x, [-1, 75])

        fc1 = fc_layer(flat, 256, is_training, "fc1")
        fc1_drop = tf.nn.dropout(fc1, self.keep_prob)
        fc2 = fc_layer(fc1_drop, 256, is_training, "fc2")
        fc2_drop = tf.nn.dropout(fc2, self.keep_prob)
        fc3 = fc_layer(fc2_drop, 1, is_training, "fc3", relu_flag=False)

        out = fc3
        loss = tf.reduce_mean(tf.reduce_sum(tf.abs((y - out) / y), reduction_indices=[1]))
        return loss, out

    def train_set(self, total_loss, global_step):
        loss_averages_op = add_loss_summaries(total_loss)

        with tf.control_dependencies([loss_averages_op]):
            opt = tf.train.AdamOptimizer(self.learning_rate)
            grads = opt.compute_gradients(total_loss)

        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)

        variable_averages = tf.train.ExponentialMovingAverage(0.9999, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
            train_op = tf.no_op(name='train')

        return train_op

    def train(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            loss, prediction = self.build_network(self.x, self.y, self.is_training)
            train_op = self.train_set(loss, self.global_step)

            prediction_in_5 = tf.equal(tf.round((prediction - self.y) / self.y * 10), tf.zeros(self.test_raws.shape))
            prediction_in_20 = tf.equal(tf.round((prediction - self.y) / self.y * 2.5), tf.zeros(self.test_raws.shape))
            accuracy_in_5 = tf.reduce_mean(tf.cast(prediction_in_5, tf.float32))
            accuracy_in_20 = tf.reduce_mean(tf.cast(prediction_in_20, tf.float32))

            saver = tf.train.Saver()
            tf.add_to_collection('prediction', prediction)

            if self.start_step > 0:
                saver.restore(sess, CKPT_PATH)
            else:
                sess.run(tf.global_variables_initializer())

            min_loss = 9999
            for i in range(self.start_step, self.start_step + self.epoch_size):
                rand_num = random.sample(range(self.raws.shape[0]), self.batch_size)
                batch_xs, batch_ys = [self.raws[i] for i in rand_num], [self.labels[i] for i in rand_num]
                _, batch_loss = sess.run([train_op, loss], feed_dict={self.x: batch_xs, self.y: batch_ys,
                                                                      self.is_training: True,
                                                                      self.keep_prob: self.keep_pb})
                if i % 100 == 0:
                    x_test, y_test = self.test_raws, self.test_labels
                    accu5, accu20, los = sess.run([accuracy_in_5, accuracy_in_20, loss],
                                                  feed_dict={self.x: x_test, self.y: y_test,
                                                             self.is_training: False,
                                                             self.keep_prob: 1.0})
                    print("train %d, batch loss %g, test accu5 %g, test accu20 %g, test loss %g"
                          % (i, batch_loss, accu5, accu20, los))

                    if i % 1000 == 0 and los < min_loss:
                        min_loss = los
                        print("saving model.....")
                        saver.save(sess, CKPT_PATH)
                        print("end saving....\n")

            x_test, y_test = self.test_raws, self.test_labels
            accu5, accu20, los = sess.run([accuracy_in_5, accuracy_in_20, loss],
                                          feed_dict={self.x: x_test, self.y: y_test,
                                                     self.is_training: False,
                                                     self.keep_prob: 1.0})
            print("train total, test accu5 %g, test accu20 %g, test loss %g"
                  % (accu5, accu20, los))

            print("\nsaving model.....")
            saver.save(sess, CKPT_PATH)
            print("end saving....\n")

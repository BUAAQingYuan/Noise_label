__author__ = 'PC-LiNing'

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
from tensorflow.python.framework import dtypes
import numpy as np
import logging
import data_uitl


tf.app.flags.DEFINE_string("data_dir", "mnist", "data dir.")


IMAGE_SIZE_MNIST = 28
NUM_CLASS = 10
Batch_size = 128
NUM_EPOCHS = 5

validation_size = 0
train_size = 60000
test_size = 10000

FLAGS = tf.app.flags.FLAGS


def main(_):
    # load data
    # mnist = NameTuple(train,validation,test)
    # one_hot encodes labels to vector
    mnist = input_data.read_data_sets("./" + FLAGS.data_dir, validation_size=validation_size, one_hot=True)
    # add noise to train
    noise_labels, prior_phi = data_uitl.add_uniform_noise(mnist.train.labels, q=0.2, num_class=NUM_CLASS)

    is_same = (noise_labels == mnist.train.labels).all()
    print("is same: " + str(is_same))

    dim_img = IMAGE_SIZE_MNIST**2
    # x_input = [batch_size,28x28]
    x_input = tf.placeholder(tf.float32, shape=[None, dim_img], name='input_img')
    labels = tf.placeholder(tf.float32, shape=[None, NUM_CLASS], name='targets')
    keep_prob = tf.placeholder(tf.float32, name='dropout')
    bern_q = tf.placeholder(tf.float32, name='bern_q')

    # phi
    phi = tf.get_variable(name="phi", dtype=tf.float32, initializer=tf.constant(prior_phi))
    # phi_bias = tf.Variable(tf.constant(0.1, shape=[NUM_CLASS]))

    # x_image = [batch_size,28,28,1]
    x_image = tf.reshape(x_input, [-1, IMAGE_SIZE_MNIST, IMAGE_SIZE_MNIST, 1])
    #1st conv layer
    conv1_weight = tf.get_variable("conv1_weight", shape=[5, 5, 1, 32], initializer=tf.contrib.layers.xavier_initializer())
    conv1_bias = tf.Variable(tf.constant(0.1, shape=[32]))
    conv1_output = tf.nn.conv2d(x_image, conv1_weight, strides=[1, 1, 1, 1], padding='SAME')
    conv1_output = tf.nn.relu(conv1_output + conv1_bias)
    pooling1_output = tf.nn.max_pool(conv1_output, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    #2nd conv layer
    conv2_weight = tf.get_variable("conv2_weight", shape=[5, 5, 32, 64], initializer=tf.contrib.layers.xavier_initializer())
    conv2_bias = tf.Variable(tf.constant(0.1, shape=[64]))
    conv2_output = tf.nn.conv2d(pooling1_output, conv2_weight, strides=[1, 1, 1, 1], padding='SAME')
    conv2_output = tf.nn.relu(conv2_output + conv2_bias)
    pooling2_output = tf.nn.max_pool(conv2_output, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # fc layer
    # Densely Connected Layer
    W_fc1 = tf.get_variable("W_fc1", shape=[7 * 7 * 64, 1024], initializer=tf.contrib.layers.xavier_initializer())
    b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))
    h_pool2_flat = tf.reshape(pooling2_output, [-1, 7*7*64])
    fc1_output = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    # Dropout
    h_fc1_drop = tf.nn.dropout(fc1_output, keep_prob)
    # softmax
    W_fc2 = tf.get_variable("W_fc2", shape=[1024, 10], initializer=tf.contrib.layers.xavier_initializer())
    b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))
    # logits = [batch_size, num_classes]
    logits = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    # add dropout regularization
    logits_dropout = tf.nn.dropout(logits, bern_q)
    # add noise softmax
    logits_noise = tf.nn.softmax(tf.matmul(logits_dropout, phi))

    test_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=tf.clip_by_value(logits, 1e-10, 1.0), labels=labels))
    train_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=tf.clip_by_value(logits_noise, 1e-10, 1.0), labels=labels))
    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(1e-3)
    grads_and_vars = optimizer.compute_gradients(train_loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
    # prediction
    test_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    test_accuracy = tf.reduce_mean(tf.cast(test_prediction, tf.float32))

    train_prediction = tf.equal(tf.argmax(logits_noise, 1), tf.argmax(labels, 1))
    train_accuracy = tf.reduce_mean(tf.cast(train_prediction, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # Generate batches
        batches = data_uitl.batch_iter(list(zip(mnist.train.images, noise_labels)), Batch_size, NUM_EPOCHS)
        # batch count
        batch_count = 0
        for batch in batches:
            batch_count += 1
            batch_data, batch_label = zip(*batch)
            _, step, losses, acc = sess.run([train_op, global_step, train_loss, train_accuracy],
                                            feed_dict={x_input: batch_data, labels: batch_label, keep_prob: 0.5, bern_q:1.0})
            print("step %d, loss %g, train accuracy %g" % (step, losses, acc))

            if batch_count % 200 == 0:
                step, test_acc, losses = sess.run([global_step, test_accuracy, test_loss],
                                                  feed_dict={x_input: mnist.test.images, labels: mnist.test.labels, keep_prob: 1.0})
                print("\nEvaluation:\n")
                print("step %d, loss %g, test accuracy %g" % (step, losses, test_acc))
                print("\n")

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s",
                        datefmt='%b %d %H:%M')
    tf.app.run()
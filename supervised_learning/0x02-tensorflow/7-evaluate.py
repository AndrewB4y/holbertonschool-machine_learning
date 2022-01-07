#!/usr/bin/env python3

"""
7-evaluate module
"""

import tensorflow.compat.v1 as tf


def evaluate(X, Y, save_path):
    """
    evaluate(X, Y, save_path) - Evaluates the output of a neural network.

    @X: is a numpy.ndarray containing the input data to evaluate.
    @Y: is a numpy.ndarray containing the one-hot labels for X.
    @save_path: is the location to load the model from.

    Returns: the network's prediction, accuracy, and loss, respectively.
    """

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph("{}.meta".format(save_path))
        saver.restore(sess, save_path)

        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        t_y_pred = tf.get_collection("y_pred")[0]
        pred_accuracy = tf.get_collection("accuracy")[0]
        t_loss = tf.get_collection("loss")[0]

        t_prediction = sess.run(t_y_pred, feed_dict={x: X, y: Y})
        t_accuracy = sess.run(pred_accuracy, feed_dict={x: X, y: Y})
        cost = sess.run(t_loss, feed_dict={x: X, y: Y})
        return t_prediction, t_accuracy, cost

#!/usr/bin/env python3

"""
7-evaluate module
"""

import tensorflow as tf


def evaluate(X, Y, save_path):
    """
    evaluate(X, Y, save_path) - Evaluates the output of a neural network.

    @X: is a numpy.ndarray containing the input data to evaluate.
    @Y: is a numpy.ndarray containing the one-hot labels for X.
    @save_path: is the location to load the model from.

    Returns: the network’s prediction, accuracy, and loss, respectively.
    """

    with tf.Session() as session:
        saver = tf.train.import_meta_graph("{}.meta".format(save_path))
        saver.restore(session, save_path)

        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        y_pred = tf.get_collection("y_pred")[0]
        accuracy = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]

        prediction = session.run(y_pred, feed_dict={x: X, y: Y})
        accuracy = session.run(accuracy, feed_dict={x: X, y: Y})
        net_loss = session.run(loss, feed_dict={x: X, y: Y})
        return prediction, accuracy, net_loss

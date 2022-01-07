#!/usr/bin/env python3

"""
3-calculate_accuracy module
"""

import tensorflow.compat.v1 as tf


def calculate_accuracy(y, y_pred):
    """
    calculate_accuracy(y, y_pred) - calculates the accuracy of a prediction.

    @y: is a placeholder for the labels of the input data.
    @y_pred: is a tensor containing the network's predictions

    Returns: a tensor containing the decimal accuracy of the prediction.
             accuracy = correct_predictions / all_predictions
    """

    correct_preds = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))

    return accuracy

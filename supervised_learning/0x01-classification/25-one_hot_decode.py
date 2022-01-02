#!/usr/bin/env python3

"""
One hot encode module
"""
import numpy as np


def one_hot_decode(one_hot):
    """
    one_hot_decode(one_hot): - Converts a one-hot matrix into a vector
                                of labels.

    @one_hot: a one-hot encoded numpy.ndarray with shape (classes, m)
        *classes: is the maximum number of classes.
        *m: is the number of examples.

    Returns: a numpy.ndarray with shape (m, ) containing the numeric labels
             for each example. None on failure.
    """

    if one_hot is None or type(one_hot) is not np.ndarray:
        return None

    oh_decoded = np.argmax(one_hot, axis=0)

    return oh_decoded.T

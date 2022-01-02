#!/usr/bin/env python3

"""
One hot encode module
"""
import numpy as np


def one_hot_encode(Y, classes):
    """
    one_hot_encode(Y, classes) - Converts a numeric label vector into
                                 a one-hot matrix.

    @Y: a numpy.ndarray with shape (m,) containing numeric class labels.
        *m is the number of examples.

    @classes: is the maximum number of classes found in Y.

    Returns: a one-hot encoding of Y with shape (classes, m).
             None on failure.
    """

    oh_encoded = np.zeros((len(Y), classes))
    for idx, cat in enumerate(Y):
        if Y[idx] >= classes or Y[idx] < 0:
            return None
        oh_encoded[idx, Y[idx]] = 1

    return oh_encoded.T

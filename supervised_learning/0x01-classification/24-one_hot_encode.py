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

    if Y is None or type(Y) is not np.ndarray or type(classes) is not int \
            or not all([x >= 0 and x < classes for x in Y]):
        return None

    oh_encoded = np.eye(classes)[Y]

    return oh_encoded.T

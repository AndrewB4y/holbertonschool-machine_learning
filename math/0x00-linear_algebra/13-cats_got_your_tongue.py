#!/usr/bin/env python3

"""
cats_got_your_tongue module
"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    np_cat(mat1, mat2, axis=0) - concatenates two matrices along a specific
                                 axis.

    @mat1: a numpy array which is never empty.
    @mat2: a numpy array which is never empty.

    Returns: a new numpy.ndarray as result of concatenating @mat1 and @mat2
    """

    return np.concatenate((mat1, mat2), axis=axis)

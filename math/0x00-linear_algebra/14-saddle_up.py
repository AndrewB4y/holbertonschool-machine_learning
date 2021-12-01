#!/usr/bin/env python3

"""
saddle_up module
"""
import numpy as np


def np_matmul(mat1, mat2):
    """
    np_matmul(mat1, mat2) - performs matrix multiplication.

    @mat1: a numpy array which is never empty.
    @mat2: a numpy array which is never empty.

    Returns: a new numpy array as result of multiplicating @mat1 with @mat2
    """

    # return mat1.dot(mat2)
    return mat1 @ mat2

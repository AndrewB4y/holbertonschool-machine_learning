#!/usr/bin/env python3

"""
size_me_please module
"""


def matrix_shape(matrix):
    """
    matrix_shape(matrix) - calculates the shape of a matrix

    @matrix: nested object (often nested list).
             This object is asssumed to never be empty.
             all elements in the same dimension are of the same type/shape.

    Returns: the dimensions of @matrix in a list.
    """
    result = []
    a = matrix
    m_class = type(matrix)
    while type(a) == m_class:
        result.append(len(a))
        a = a[0]
    return result

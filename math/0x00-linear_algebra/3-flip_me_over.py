#!/usr/bin/env python3

"""
flip_me_over module
"""


def matrix_transpose(matrix):
    """
    matrix_transpose(matrix) - returns the transpose of a 2D matrix.

    @matrix: nested object (often nested list).
             This object is asssumed to never be empty.
             all elements in the same dimension are of the same type/shape.

    Returns: A transposed matrix of @matrix
    """
    transpose = []
    cols = len(matrix[0])
    for col in range(cols):
        aux = []
        for row in matrix:
            aux.append(row[col])
        transpose.append(aux)
    # transpose = [list(row) for row in list(zip(*matrix))]
    return transpose

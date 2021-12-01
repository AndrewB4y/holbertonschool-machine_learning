#!/usr/bin/env python3

"""
across_the_planes module
"""

def add_matrices2D(mat1, mat2):
    """
    add_matrices2D(mat1, mat2) - adds two matrices element-wise.

    @mat1: 2D matrix containing ints/floats.
           All elements in the same dimension are of the same type/shape.
    @mat2: 2D matrix containing ints/floats.
           All elements in the same dimension are of the same type/shape.

    Returns: A new matrix as result of adding element-wise @mat1 with @mat2.
             If @mat1 and @mat2 are not the same shape, return None.
    """

    rows = len(mat1)
    cols = len(mat1[0])
    if rows != len(mat2) or cols != len(mat2[0]):
        return None

    result = []
    for i in range(rows):
        r = []
        for j in range(cols):
            r.append(mat1[i][j] + mat2[i][j])
        result.append(r)

    return result

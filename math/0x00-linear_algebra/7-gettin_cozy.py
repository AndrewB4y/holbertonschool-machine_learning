#!/usr/bin/env python3

"""
gettin_cozy module
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    cat_matrices2D(mat1, mat2, axis=0) - concatenates two matrices along a
                                         specific axis.

    @mat1: 2D matrix containing ints/floats.
           All elements in the same dimension are of the same type/shape.
    @mat2: 2D matrix containing ints/floats.
           All elements in the same dimension are of the same type/shape.

    Returns: A new matrix as result of adding element-wise @mat1 with @mat2.
             If @mat1 and @mat2 are not the same shape, return None.
    """
    if mat1 is None or mat2 is None:
        return None

    result = []
    if axis == 0:
        cols = len(mat1[0])
        if cols != len(mat2[0]):
            return None
        result = [[col for col in row] for row in mat1]
        result.extend(mat2)

    elif axis == 1:
        rows = len(mat1)
        if rows != len(mat2):
            return None
        result = [[col for col in row] for row in mat1]
        for i in range(rows):
            result[i].extend(mat2[i])
    else:
        return None

    return result

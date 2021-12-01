#!/usr/bin/env python3

"""
ridin_bareback module
"""

def mat_mul(mat1, mat2):
    """
    mat_mul(mat1, mat2) - performs matrix multiplication

    @mat1: 2D matrix containing ints/floats.
           All elements in the same dimension are of the same type/shape.
    @mat2: 2D matrix containing ints/floats.
           All elements in the same dimension are of the same type/shape.

    Returns: A new matrix as result of multiplying @mat1 with @mat2.
             If @mat1 and @mat2 cannot be multiplied returns None.
    """
    if mat1 is None or mat2 is None:
        return None

    rows1 = len(mat1)
    cols1 = len(mat1[0])
    rows2 = len(mat2)
    cols2 = len(mat2[0])

    if cols1 != rows2:
        return None

    result = [[0, ]*cols2 for _ in range(rows1)]
    for i in range(rows1):
        # holdin on row from mat1
        r1 = mat1[i]
        for h in range(cols2):
            # holding on col from mat2
            v_sum = 0
            for j in range(cols1):
                # sum of products "elemen-wise"
                prod = r1[j] * mat2[j][h]
                v_sum += prod
            result[i][h] = v_sum

    return result

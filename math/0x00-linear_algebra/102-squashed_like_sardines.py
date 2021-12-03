#!/usr/bin/env python3

"""
squashed_like_sardines module
"""


def cat_matrices(mat1, mat2, axis=0):
    """
    cat_matrices(mat1, mat2, axis=0) - concatenates two matrices along
                                       a specific axis.

    @mat1: matrix containing ints/floats.
           All elements in the same dimension are of the same type/shape.
    @mat2: matrix containing ints/floats.
           All elements in the same dimension are of the same type/shape.

    Return: A new matrix as result of adding @mat1 and @mat2.
            If @mat1 and @mat2 can't be concatenated, returns None.
    """

    # getting sizes of mat1 and mat2.
    a = mat1
    a_class = type(mat1)
    a_shape = []
    while type(a) == a_class:
        l_a = len(a)
        a = a[0]
        a_shape.append(l_a)

    b = mat2
    b_class = type(mat2)
    b_shape = []
    while type(b) == b_class:
        l_b = len(b)
        b = b[0]
        b_shape.append(l_b)

    a_s = a_shape.copy()
    a_s[axis] = 0
    b_s = b_shape.copy()
    b_s[axis] = 0
    if a_s != b_s:
        return None

    c_mat1 = eval(str(mat1))
    c_mat2 = eval(str(mat2))

    result = c_mat1
    if axis == 0:
        result.extend(c_mat2)
        return result

    current = [0, ]*len(a_s[:axis])
    while current[0] < a_shape[0]:
        #  Dinamically accesing indices
        indices = "[" + "][".join([str(x) for x in current])+"]"
        #  Dinamically extending matrix using indices
        exec("result"+indices+".extend(c_mat2"+indices+")")
        #  Updating indices in list current
        dim = axis
        while dim:
            dim = dim - 1
            if current[dim] == a_shape[dim] - 1:
                if dim == 0:
                    current[dim] += 1
                    # reaching full dimensions: breaking out of loop
                    break
                current[dim] = 0
                continue
            current[dim] += 1
            dim = 0

    return result

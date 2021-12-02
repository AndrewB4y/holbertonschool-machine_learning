#!/usr/bin/env python3

"""
the_whole_barn module
"""


def add_matrices(mat1, mat2):
    """
    add_matrices(mat1, mat2) - adds two matrices.

    @mat1: matrix containing ints/floats.
           All elements in the same dimension are of the same type/shape.
    @mat2: matrix containing ints/floats.
           All elements in the same dimension are of the same type/shape.

    Return: A new matrix as result of adding @mat1 and @mat2.
            If @mat1 and @mat2 are not the same shape, returns None.
    """

    # validating sizes of mat1 and mat2 to be the same.
    a = mat1
    b = mat2
    a_class = type(mat1)
    shape = []
    while type(a) == a_class:
        l_a = len(a)
        if l_a != len(b):
            return None
        b = b[0]
        a = a[0]
        if type(a) != type(b):
            return None
        shape.append(l_a)

    ndim = len(shape)
    current = [0]*ndim
    str_ndim = '0,'
    for d in shape[:0:-1]:
        str_ndim = '[' + str_ndim*d + '],'
    str_ndim = '[' + str_ndim*shape[0] + ']'
    # str_ndim = '[0]*{}'.format(shape[-1])
    # for dim in shape[-2::-1]:
    #     str_ndim = '[' + str_ndim + ']*{}'.format(dim)
    result = eval(str_ndim)
    while current[0] < shape[0]:
        #  Dinamically accesing indices
        indices = "[" + "][".join([str(x) for x in current])+"]"
        n_mat1 = eval("mat1"+indices)
        n_mat2 = eval("mat2"+indices)
        exec("result"+indices+" = {}".format(n_mat1+n_mat2))
        #  Updating indices in list current
        dim = ndim
        while dim:
            dim = dim - 1
            if current[dim] == shape[dim] - 1:
                if dim == 0:
                    current[dim] += 1
                    break
                current[dim] = 0
                continue
            current[dim] += 1
            dim = 0

    return result

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

    a_s = [x for x in a_shape]
    del a_s[axis]
    b_s = [x for x in b_shape]
    del b_s[axis]
    if a_s != b_s:
        return None

    ndim = len(a_shape)
    current = [0]*ndim
    str_ndim = '0,'
    for d in a_shape[:0:-1]:
        str_ndim = '[' + str_ndim*d + '],'
    str_ndim = '[' + str_ndim*a_shape[0] + ']'
    # str_ndim = '[0]*{}'.format(a_shape[-1])
    # for dim in a_shape[-2::-1]:
    #     str_ndim = '[' + str_ndim + ']*{}'.format(dim)
    result = eval(str_ndim)
    # while current[0] < a_shape[0]:
    #     #  Dinamically accesing indices
    #     indices = "[" + "][".join([str(x) for x in current])+"]"
    #     n_mat1 = eval("mat1"+indices)
    #     n_mat2 = eval("mat2"+indices)
    #     exec("result"+indices+" = {}".format(n_mat1+n_mat2))
    #     #  Updating indices in list current
    #     dim = ndim
    #     while dim:
    #         dim = dim - 1
    #         if current[dim] == a_shape[dim] - 1:
    #             if dim == 0:
    #                 current[dim] += 1
    #                 break
    #             current[dim] = 0
    #             continue
    #         current[dim] += 1
    #         dim = 0

    return result

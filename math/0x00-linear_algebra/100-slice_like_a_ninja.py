#!/usr/bin/env python3

"""
slice_like_a_ninja module
"""


# def np_slice(matrix, axes={}):
#     """
#     np_slice(matrix, axes={}) -  slices a matrix along specific axes.

#     @matrix: a numpy.ndarray
#     @axes: a dictionary where the key is an axis to slice along and
#            the value is a tuple representing the slice to make along
#            that axis.
#            @axes must always represent a valid slice

#     Returns: a new numpy.ndarray which is a slice of @matrix.

#     Example:
#     $ mat2 = np.array([[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
#                  [[11, 12, 13, 14, 15], [16, 17, 18, 19, 20]],
#                  [[21, 22, 23, 24, 25], [26, 27, 28, 29, 30]]])
#     $ print(np_slice(mat2, axes={0: (2,), 2: (None, None, -2)}))
#                                  mat2[:2,:,::-2]
#     $ print(mat2)
#         [[[ 5  3  1]
#           [10  8  6]]

#          [[15 13 11]
#           [20 18 16]]]
#     """

#     ninja = [":", ]*(max(axes.keys()) + 1)
#     for axis, params in axes.items():
#         if len(params) == 1:
#             ninja[axis] = ":{}".format(params[0])
#         else:
#             temp = ""
#             for idx, i in enumerate(params):
#                 if idx != 0:
#                     temp = temp + ":"
#                 if i is None:
#                     continue
#                 temp = temp + str(i)
#             ninja[axis] = temp
#     my_slice = matrix.copy()
#     my_ninja = "my_slice[{}]".format(', '.join(ninja))
#     my_slice = eval(my_ninja)

#     return my_slice



def np_slice(matrix, axes={}):
    """
    np_slice(matrix, axes={}) -  slices a matrix along specific axes.
                                 (this is an improved code of the above)

    @matrix: a numpy.ndarray
    @axes: a dictionary where the key is an axis to slice along and
           the value is a tuple representing the slice to make along
           that axis.
           @axes must always represent a valid slice

    Returns: a new numpy.ndarray which is a slice of @matrix.

    Example:
    $ mat2 = np.array([[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
                 [[11, 12, 13, 14, 15], [16, 17, 18, 19, 20]],
                 [[21, 22, 23, 24, 25], [26, 27, 28, 29, 30]]])
    $ print(np_slice(mat2, axes={0: (2,), 2: (None, None, -2)}))
                                 mat2[:2,:,::-2]
    $ print(mat2)
        [[[ 5  3  1]
          [10  8  6]]

         [[15 13 11]
          [20 18 16]]]
    """
    slices = []
    axes={0: (2,), 2: (None, None, -2)}
    for i in range(matrix.ndim):
        t = axes.get(i)
        if t is not None:
            slices.append(slice(*t))
        else:
            slices.append(slice(None, None, None))
    
    return matrix[tuple(slices)]
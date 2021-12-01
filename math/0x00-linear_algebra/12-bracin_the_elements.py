#!/usr/bin/env python3

"""
bracin_the_elements module
"""


def np_elementwise(mat1, mat2):
    """
    np_elementwise(mat1, mat2) - performs element-wise addition, subtraction,
                                 multiplication, and division.

    @mat1: a numpy array which is never empty.
    @mat2: a numpy array which is never empty.

    Returns: a tuple containing the element-wise sum, difference, product,
             and quotient, respectively.
    """

    return (mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2)

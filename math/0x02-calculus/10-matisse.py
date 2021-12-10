#!/usr/bin/env python3

"""
matisse module
"""


def poly_derivative(poly):
    """
    poly_derivative(poly) - calculates the derivative of a polynomial.

    @poly: is a list of coefficients representing a polynomial.
           The index of the list represents the power of x that the
           coefficient belongs to.

    Returns: a new list of coefficients representing the derivative of
             the polynomial.
             If poly is not valid, return None.
             If the derivative is 0, return [0]
    """

    if type(poly) is not list:
        return None

    res = [x * idx for idx, x in enumerate(poly)]
    del res[0]

    if len(res) == 0:
        return [0,]

    return res

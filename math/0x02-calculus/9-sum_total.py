#!/usr/bin/env python3

"""
sum_total module
"""


def summation_i_squared(n):
    """
    summation_i_squared(n) - calculates sum squared i, from 1 to n

    @n: is the stopping condition.

    Returns: the integer value of the sum.
             If n is not a valid number, returns None.
    """

    if n == 1:
        return 1

    return n**2 + summation_i_squared(n - 1)

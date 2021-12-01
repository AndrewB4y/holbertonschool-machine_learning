#!/usr/bin/env python3

"""
howdy_partner module
"""


def cat_arrays(arr1, arr2):
    """
    cat_arrays(arr1, arr2) - concatenates two arrays.

    @arr1: list of ints/floats
    @arr2: list of ints/floats

    Returns: A new list as result of the concatenation.
    """

    if arr1 is None:
        return arr2
    elif arr2 is None:
        return arr1
    else:
        res = [x for x in arr1]
        res.extend(arr2)
        return res

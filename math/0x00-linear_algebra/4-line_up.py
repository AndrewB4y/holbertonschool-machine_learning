#!/usr/bin/env python3


def add_arrays(arr1, arr2):
    """
    add_arrays(arr1, arr2) - adds two arrays element-wise

    @arr1: list of ints/floats
    @arr2: list of ints/floats

    Returns: A new list as result of adding element-wise @arr1 with @arr2
    """
    lenght = len(arr1)
    if lenght != len(arr2):
        return None
    result = []
    for i in range(lenght):
        result.append(arr1[i] + arr2[i])
    return result

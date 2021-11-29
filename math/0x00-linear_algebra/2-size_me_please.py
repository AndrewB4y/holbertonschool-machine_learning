#!/usr/bin/env python3

def matrix_shape(matrix):
    result = []
    a = matrix
    m_class = type(matrix)
    while type(a) == m_class:
        result.append(len(a))
        a = a[0]
    return result
#!/usr/bin/env python3
"""Calculates the shape of a matrix"""


def add_arrays(arr1, arr2):


    if len(arr1) != len(arr2):
        return None
    result = []
    for i in range(len(arr1)):
        result.append(arr1[i] + arr2[i])
    return result
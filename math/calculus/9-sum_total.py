#!/usr/bin/env python3
""" A scripts that calculates the sum of squares """


def summation_i_squared(n):
"""function that calculate the sum of squares"""
    if type(n) is not int:
       return None
    sum = (n*(n+1)*(2*n+1)) / 6
    return sum

#!/usr/bin/env python3
"""
A script that calculates the derivative of a polynomial
"""


def poly_derivative(poly):
    '''
    A function that calculates the derivative of a polynomial
    '''
    if type(poly) is not list or len(poly) < 1:
        return None
    for coeff in poly:
        if type(coeff) is not int:
            return None

    for power, coefficient in enumerate(poly):
        if power == 0:
            derivative = [0]
        if power == 1:
            derivative = []
        derivative.append(power * coefficient)
    while derivative[-1] == 0 and len(derivative) > 1:
        derivative = derivative[:-1]
    return derivative

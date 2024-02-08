#!/usr/bin/env python3
""" ... """


def poly_integral(poly, C=0):
    """ ... """
    if not isinstance(poly, list) or not all(isinstance(coeff, (int, float)) for coeff in poly) or not isinstance(C, (int, float)):
        return None
    
    integral = [C]
    for i in range(len(poly)):
        if not isinstance(poly[i], (int, float)):
            return None
        if i == 0:
            integral.append(poly[i] / (i + 1))
        else:
            integral.append(poly[i] / (i + 1))
    return integral

# Example usage:
poly = [5, 3, 0, 1]
integral = poly_integral(poly, C=0)
print(integral)  # Output should be [0, 5.0, 1.5, 0.0, 0.25]

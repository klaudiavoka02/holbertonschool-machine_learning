#!/usr/bin/env python3
"""
The function calculates the likelihood 
"""
import numpy as np


def likelihood(x, n, P):
    """
    Calculate the hypothetical likelihood of severe side effects
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be an integer that is greater than "
                         "or equal to 0")
    if type(P) is not np.ndarray or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if np.any(P < 0) or np.any(P > 1):
        raise ValueError("All values in P must be in the range [0, 1]")
    n_factorials = np.math.factorial(n)
    x_factorial = np.math.factorial(x)
    likelihoods = ((n_factorials / (x_factorial * np.math.factorial(n - x)))
                   * (P ** x) * ((1 - P) ** (n - x)))
    return likelihoods

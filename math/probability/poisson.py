#!/usr/bin/env python3
"""
represent a poisson distribution
"""


class Poisson:
    """
    the poisson distribution class
    """
    def __init__(self, data=None, lambtha=1):
        """
        represents a poisson distribution
        """
        self.lambtha = float(lambtha)
        if self.lambtha <= 0:
            raise ValueError("lambtha must be a positive value")
        if data is not None:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """
        ...
        """
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0
        e = 2.7182818285
        lambtha = self.lambtha
        factorial = 1
        for i in range(1, k + 1):
            factorial *= i
        pmf = ((e ** -lambtha) * lambtha ** k) / factorial
        return pmf

    def cdf(self, k):
        """
        Calculates the cumulative distribution for a given number 
        """
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0
        cdf = 0
        for i in range(k + 1):
            cdf += self.pmf(i)
        return cdf

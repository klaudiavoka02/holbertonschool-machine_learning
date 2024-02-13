#!/usr/bin/env python3
"""
Creates a class named Binomial which represents a binomial distribution
"""


class Binomial:
    """
    class which represents the binomial distribution
    """
    def __init__(self, data=None, n=1, p=0.5):
        """
        the init method
        """
        self.n = n
        self.p = float(p)
        if data is None:
            if self.n <= 0:
                raise ValueError("n must be a positive value")
            else:
                self.n = n
            if 0 <= self.p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            else:
                self.p = p
        if data is not None:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                mean = float(sum(data) / len(data))
                summation = 0
                for x in data:
                    summation += (x - mean) ** 2
                variance = summation / len(data)
                q = variance / mean
                p = (1 - q)
                n = round(mean / p)
                p = float(mean / n)
                self.n = n
                self.p = p

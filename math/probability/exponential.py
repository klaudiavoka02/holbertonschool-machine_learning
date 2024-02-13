#!/usr/bin/env python3
""" ... """


def __init__(self, data=None, lambtha=1.):
    """ ... """
if data is None:
    if lambtha ValueError("lamtha must be a positive value")


#4
def pdf(self, x):
"""Calculates the value of the PDF for a given time period"""
    if x < 0 :
        return 0

    e = 2.7182818285
    lambtha = self.lambtha

    result = lambtha * (e ** (-lambtha *x))
    return result

#5
def cdf(self, x):
"""Calculates the value of the CDF for a given time period"""
    if x < 0 :
        return 0

    e = 2.7182818285
    lambtha = self.lambtha

    result = 1 - (e ** (-lambtha *x))
    return result

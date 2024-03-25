#!/usr/bin/env python3
"""
A  class Neuron that defines a single neuron performing binary classification (Based on 0-neuron.py)
"""



import numpy as np

class Neuron:
    """
A single neuron performing binary classification
    """

def __init__(self, nx):
    """ class constructor """
    if type(nx) is not int:
        raise TypeError("nx must be an integer")
    if nx < 1 :
        raise ValueError("nx must be a positive integer")
    self.__W = np.random.randn(1, nx)
    self.__b = 0
    self.__a = 0

    @property
    def W(self):
        """ get method for property weights"""
        return self.__W
    
    @property
    def b(self):
        """ get method for property bias"""
        return self.__b

    @property
    def a(self):
        """ get method for property activation function"""
        return self.__A

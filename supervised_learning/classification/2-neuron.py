#!/usr/bin/env python3
"""
A  class Neuron that defines a single neuron performing binary classification (Based on 0-neuron.py)
"""



import numpy as np

class Neuron:
    """
A single neuron performing binary classification
    """

def forward_prop(self, X):
    """ Calculates the forward propagation of the neuron """
    z = np.matmul(self.W, X) + self.b
    self.__A = 1 / (1 + np.exp(-z))
    return self.__A

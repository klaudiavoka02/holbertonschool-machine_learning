#!/usr/bin/env python3
"""
class Neuron that defines a single neuron
"""



import numpy as np

class Neuron :
    """
A single neuron performing binary classification
    """

def cost(self, Y, A) :
    """ Calculates the cost of the model using logistic regression """
    m = Y.shape[1]
    m_loss = np.sum((Y * np.log(A)) + (1 - Y) * np.log(1.0000001 - A))
    cost = (1/m) * (-(m_loss))
    return cost

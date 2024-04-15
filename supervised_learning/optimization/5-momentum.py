#!/usr/bin/env python3
"""
   Momentum
"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
        method that updates a variable using the gradient descent
        with momentum optimization algorithm
    """
    dW = beta1 * v + (1 - beta1) * grad

    var_new = var - dW * alpha

    return var_new, dW

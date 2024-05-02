#!/usr/bin/env python3
"""
    Gradient descent with L2 regularization
"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
        function that updates weights of NN with Dropout reg
        using gradient descent
    """

    m = Y.shape[1]
    A = cache['A' + str(L)]
    dZ = A - Y

    A_prev = cache['A' + str(L - 1)]
    W = weights['W' + str(L)]
    dW = np.matmul(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.matmul(W.T, dZ)

    weights['W' + str(L)] -= alpha * dW
    weights['b' + str(L)] -= alpha * db

    for layer in range(L - 1, 0, -1):
        D = cache['D' + str(layer)]
        dA = dA_prev * (D / keep_prob)

        A = cache['A' + str(layer)]
        A_prev = cache['A' + str(layer - 1)]
        dZ = dA * (1 - A ** 2)
        dW = np.matmul(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        W = weights['W' + str(layer)]
        dA_prev = np.matmul(W.T, dZ)

        weights['W' + str(layer)] -= alpha * dW
        weights['b' + str(layer)] -= alpha * db

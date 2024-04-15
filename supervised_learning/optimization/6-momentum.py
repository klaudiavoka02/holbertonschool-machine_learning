#!/usr/bin/env python3
"""
   Momentum upgraded
"""

import tensorflow.compat.v1 as tf


def create_momentum_op(loss, alpha, beta1):
    """
        Method to creates the training operation for a NN
        in tf using gradient descent with momentum opt algo
    """

    optimizer = tf.train.MomentumOptimizer(learning_rate=alpha,
                                           momentum=beta1)

    train_op = optimizer.minimize(loss)

    return train_op
c
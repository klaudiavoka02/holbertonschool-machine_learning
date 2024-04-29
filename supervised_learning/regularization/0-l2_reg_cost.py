#!/usr/bin/env python3
"""
    Cost of NN with L2 regularization
"""

import numpy as np


def l2_reg_cost(cost):
    """
        Function that calculate the cost of NN with L2 regularization
    """
    regularization_L2 = tf.compat.v1.losses.get_regularization_losses()

    cost_L2reg = cost + regularization_L2

    return cost_L2reg

#!/usr/bin/env python3
"""
    Create layer with L2 regularization
"""

import tensorflow.compat.v1 as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
        Function that creates a tensorflow layer includes L2 regularization
    """
    initializer = tf.keras.initializers.VarianceScaling(scale=2.0,
                                                        mode='fan_avg')

    new_layer = (
        tf.layers.Dense(n,
                        activation=activation,
                        kernel_initializer=initializer,
                        kernel_regularizer=tf.keras.regularizers.l2(lambtha),
                        name="layer"))

    output = new_layer(prev)

    return output

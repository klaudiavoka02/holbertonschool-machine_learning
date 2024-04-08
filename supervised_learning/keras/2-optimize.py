#!/usr/bin/env python3
"""
    Optimize
"""

import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """
        function that sets up Adam optimization for a keras model
        with categorical crossentropy loss and accuracy metrics
    """
    Adam_optimizer = K.optimizers.Adam(learning_rate=alpha,
                                       beta_1=beta1,
                                       beta_2=beta2)

    model.compile(optimizer='Adam',loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

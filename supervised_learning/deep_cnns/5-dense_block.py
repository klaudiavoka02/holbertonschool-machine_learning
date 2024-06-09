#!/usr/bin/env python3
"""Dense Block Module"""

from tensorflow import keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """Builds a Dense Block.
    """

    concatenated_layers = [X]
    current_filters = nb_filters
    init = K.initializers.he_normal

    for _ in range(layers):
        X = K.layers.BatchNormalization()(X)
        X = K.layers.Activation("relu")(X)
        X = K.layers.Conv2D(4 * growth_rate, 1,
                            padding="same", kernel_initializer=init)(X)

        X = K.layers.BatchNormalization()(X)
        X = K.layers.Activation("relu")(X)
        X = K.layers.Conv2D(growth_rate, 3, padding="same",
                            kernel_initializer=init)(X)

        concatenated_layers.append(X)
        X = K.layers.Concatenate()(concatenated_layers)
        concatenated_layers.pop(0)
        concatenated_layers.pop(0)
        concatenated_layers.append(X)

        current_filters += growth_rate

    return X, current_filters

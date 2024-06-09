#!/usr/bin/env python3
"""Dense Block Module"""

from tensorflow import keras as K


def transition_layer(X, nb_filters, compression):
    """Builds a Transition Layer.
    """

    init = K.initializers.he_normal
    compressed_filters = int(nb_filters * compression)

    X = K.layers.BatchNormalization()(X)
    X = K.layers.ReLU()(X)
    X = K.layers.Conv2D(compressed_filters, 1,
                        padding="same", kernel_initializer=init)(X)
    X = K.layers.AveragePooling2D(2, 2)(X)

    return X, compressed_filters

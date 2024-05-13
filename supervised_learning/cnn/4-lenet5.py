#!/usr/bin/env python3
"""
    LeNet-5 (Tensorflow 1 implementation)
"""

import tensorflow.compat.v1 as tf


def lenet5(x, y):
    """
        function that builds a modified version of the LeNet-5
        architecture using TensorFlow version 1
    """
    initializer = tf.keras.initializers.VarianceScaling(scale=2.0)
    conv1 = tf.layers.Conv2D(filters=6,
                             kernel_size=5,
                             padding='same',
                             kernel_initializer=initializer,
                             activation='relu')(x)
                              pool1 = tf.layers.MaxPooling2D(pool_size=2,
                                   strides=2)(conv1)
    conv2 = tf.layers.Conv2D(filters=16,
                             kernel_size=5,
                             padding='valid',
                             kernel_initializer=initializer,
                             activation='relu')(pool1)
    pool2 = tf.layers.MaxPooling2D(pool_size=2,
                                   strides=2)(conv2)
    flat = tf.layers.Flatten()(pool2)
    full1 = tf.layers.Dense(120,
                            activation='relu',
                            kernel_initializer=initializer)(flat)
    full2 = tf.layers.Dense(84,
                            activation='relu',
                            kernel_initializer=initializer)(full1)
    output = tf.layers.Dense(10,
                             activation=None,
                             kernel_initializer=initializer)(full2)
    softmax = tf.nn.softmax(output)

    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=y,
        logits=output)
    train_Adam = tf.train.AdamOptimizer().minimize(loss)

    y_pred = tf.argmax(output, axis=1)
    y_true = tf.argmax(y, axis=1)
    correct_prediction = tf.equal(y_pred, y_true)
    correct_prediction = tf.cast(correct_prediction, dtype=tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    return softmax, train_Adam, loss, accuracy

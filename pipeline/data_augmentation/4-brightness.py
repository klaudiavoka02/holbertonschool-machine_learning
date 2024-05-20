#!/usr/bin/env python3
"""
Defines function that randomly changes the brightness of an image
"""


import tensorflow as tf


def change_brightness(image, max_delta):
    """
    Randomly changes the brightness of an image
    """
    return (tf.image.random_brightness(image, max_delta))

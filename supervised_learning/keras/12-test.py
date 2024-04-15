#!/usr/bin/env python3
"""
    Test neural network
"""

import tensorflow.keras as keras


def test_model(network, data, labels, verbose=True):
    """
  function that tests a neural network
    """
    return network.evaluate(data, labels, verbose=verbose)
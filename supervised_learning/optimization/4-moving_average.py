#!/usr/bin/env python3
"""
   Moving Average
"""

import numpy as np


def moving_average(data, beta):
    """
        Method that calculates the weighted moving average of
        a data set
    """
    m_av = []

    w = 0

    for i, d in enumerate(data):
        w = beta * w + (1 - beta) * d
        w_new = w / (1 - beta ** (i + 1))
        m_av.append(w_new)
    return m_av

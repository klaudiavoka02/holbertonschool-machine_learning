#!/usr/bin/env python3
"""
    Convolution Back Propagation
"""

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
        function that performs back propagation over a convolutional
        layer of NN
    """

    _, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, _ = W.shape
    sh, sw = stride


    if padding == 'valid':

        ph, pw = 0, 0
    elif padding == 'same':
        ph = int((((h_prev - 1) * sh + kh - h_prev) / 2 + 0.5))
        pw = int((((w_prev - 1) * sw + kw - w_prev) / 2 + 0.5))


    A_prev_pad = np.pad(A_prev,
                        [(0, 0), (ph, ph), (pw, pw), (0, 0)],
                        mode='constant')


    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    dA_pad = np.zeros(shape=A_prev_pad.shape)
    dW = np.zeros(shape=W.shape)


    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):

                for f in range(c_new):

                    v_start = h * sh
                    v_end = v_start + kh
                    h_start = w * sw
                    h_end = h_start + kw

                    dA_pad[i, v_start:v_end, h_start:h_end, :]\
                        += W[:, :, :, f] * dZ[i, h, w, f]
                    dW[:, :, :, f] += (A_prev_pad[i, v_start:v_end,
                                                  h_start:h_end, :]
                                       * dZ[i, h, w, f])

    if padding == "same":
        dA = dA_pad[:, ph:-ph, pw:-pw, :]
    else:
        dA = dA_pad

    return dA, dW, db

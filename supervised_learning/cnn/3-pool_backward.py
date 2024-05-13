#!/usr/bin/env python3
"""
    Pooling Back Propagation
"""

import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
        function that performs back propagation over a pooling layer of NN
    """
    m, h_new, w_new, c = dA.shape
    _, h_prev, w_prev, _ = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros(A_prev.shape)

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for f in range(c):
                    v_start = h * sh
                    v_end = v_start + kh
                    h_start = w * sw
                    h_end = h_start + kw

                    if mode == 'avg':
                        avg_dA = dA[i, h, w, f] / kh / kw
                        dA_prev[i, v_start:v_end, h_start:h_end, f] += (
                                np.ones((kh, kw)) * avg_dA)
                    elif mode == 'max':
                        a_prev_slice \
                            = A_prev[i, v_start:v_end, h_start:h_end, f]
                        mask = (a_prev_slice == np.max(a_prev_slice))
                        dA_prev[i, v_start:v_end, h_start:h_end, f] +=\
                            mask * dA[i, h, w, f]
                            
    return dA_prev

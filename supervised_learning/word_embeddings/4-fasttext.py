#!/usr/bin/env python3
"""
Creates and trains a simple gensim fasttext model
"""


import gensim


def fasttext_model(
        sentences, size=100, min_count=5, negative=5, window=5, cbow=True,
        iterations=5, seed=0, workers=1
):
    """
    Creates and trains a simple gensim fasttext model
    """
    fast_model = gensim.models.FastText

    model = fast_model(
        sentences,
        size,
        min_count,
        window,
        negative,
        iter=iterations,
        seed=seed,
        workers=workers,
        sg =not cbow
    )
    return model

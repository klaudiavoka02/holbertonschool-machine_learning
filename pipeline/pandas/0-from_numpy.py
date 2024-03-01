import pandas as pd

def from_numpy(array):

    if array.shape == (5, 8):
        df = pd.DataFrame(array, columns=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])
    else:
        df = pd.DataFrame(array, columns=['A', 'B', 'C'])
    return df
# Main cell

import numpy as np

np.random.seed(0)
A = np.random.randn(5, 8)
print(from_numpy(A))
B = np.random.randn(9, 3)
print(from_numpy(B))

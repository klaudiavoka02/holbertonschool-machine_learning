#!/usr/bin/env python3
""" returns the transpose of a 2D matrix """


def matrix_transpose(matrix):
    rows = len(matrix)
    cols = len(matrix[0])

    transpose = [[0] * rows for _ in range(cols)]

for i in range(rows):
    for j in range(cols):
        transpose[j][i] = matrix[i][j]

return transpose

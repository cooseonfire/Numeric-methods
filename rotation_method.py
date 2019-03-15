import numpy as np
from math import sin, cos, atan, isclose


def symmetrize_matrix(A):
    return np.transpose(A) @ A


def max_el_above(A):
    max_el = 0
    i0 = 0
    j0 = 0
    for i in range(len(A)):
        for j in range(i+1, len(A)):
            if abs(A[i][j]) > abs(max_el):
                max_el = A[i][j]
                i0, j0 = i, j
    return max_el, i0, j0


def rotate_matrix(A):
    A = symmetrize_matrix(A)
    U_k = np.diag([1.0] * len(A))
    while True:
        U = np.diag([1.0] * len(A))
        max_el, i, j = max_el_above(A)
        if isclose(abs(max_el), 0.0, abs_tol=1e-10):
            break

        angle = 1/2 * atan(2*A[i][j] / (A[i][i] - A[j][j]))
        U[i][i] = cos(angle)
        U[j][j] = cos(angle)
        U[i][j] = -sin(angle)
        U[j][i] = sin(angle)

        A = np.transpose(U) @ A @ U
        U_k = U_k @ U

    return np.diag(A), U_k


if __name__ == "__main__":
    n = int(input('Enter dimension n: '))
    print('Enter A matrix: ')
    input_matrix = [[float(i) for i in input().split()] for j in range(n)]
    values, vectors = rotate_matrix(input_matrix[:])
    print('\nEigen values:\n')
    print(values, end='\n')
    print('\nEigen vectors:\n')
    print(vectors, end='\n')

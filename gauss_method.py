import numpy as np


def check_compatability(matrix: list):
    matr = matrix
    for k in range(n):
        if matr[k] == [0] * n:
            raise ValueError
        for i in range(n - 1):
            if matr[i][i] == 0:
                matr[i], matr[i + 1] = matr[i + 1], matr[k]
    return matr


n = int(input('Enter dimension n: '))
print('Enter A matrix: ')
input_matrix = [[float(i) for i in input().split()] for j in range(n)]
try:
    a = check_compatability(input_matrix[:])
except ValueError:
    print('Rank(A) < Rank(A|B)')
    exit()
b = [float(i) for i in input('Enter free coefficients: ').split()]
norm_b = max(b)
k = 1  # number of step
row_mod = []  # [(k-1) step]
diagonal = []
b_mod = []  # [(k-1) step]

for k in range(n):
    row_mod.append(a[k])
    diagonal.append(a[k][k])
    b_mod.append(b[k])
    for i in range(k + 1, n):
        factor = -a[i][k]/a[k][k]
        eq_modified = [factor * a_past for a_past in a[k]]
        a[i] = [r + rk for (r, rk) in zip(a[i], eq_modified)]
        b[i] = b[i] + factor*b[k]

x = [0 for i in range(n)]
for k in reversed(range(n)):
    submitted = [i * j for (i, j) in zip(x.__reversed__(),
                                         row_mod[k].__reversed__())]
    root = (b_mod[k] - sum(submitted)) / diagonal[k]
    x[k] = root
print('The solutions are {0}'.format(x))

try:
    inv_a = np.linalg.inv(input_matrix)
except np.linalg.LinAlgError:
    print("A can't be inverted")
else:
    print('Inversed A: {}'.format(inv_a))
    norm_a = max(sum(input_matrix[i]) for i in range(n))
    norm_inv_a = max(sum(inv_a[i]) for i in range(n))
    abs_b = 0.001
    rel_b = abs_b / norm_b
    abs_x = norm_inv_a * abs_b
    rel_x = norm_a * norm_inv_a * rel_b
    print('delta x = {0}, relative delta x <= {1}'.format(abs_x, rel_x))

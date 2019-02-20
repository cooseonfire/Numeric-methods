import numpy as np
from math import sqrt, log, ceil


def column(A, j):
    return [row[j] for row in A]


def check_convergence(B: list):
    norm1_cond = max(sum(map(abs, B[i])) for i in range(len(B))) < 1
    sum_B_sq = 0
    for i in range(len(B)):
        sum_B_sq += sum([ij*ij for ij in B[i]])
    norm2_cond = sqrt(sum_B_sq) < 1
    norm3_cond = max(sum(map(abs, j)) for j in zip(*B)) < 1  # *B for rows

    eig_abs_vals = map(abs, np.linalg.eigvals(B))
    eigv_cond = all(x < 1 for x in eig_abs_vals)

    if norm1_cond or norm2_cond or norm3_cond or eigv_cond:
        return True
    return False


def iteration_method(A, b, eps):
    x0 = b[:]   # x 0
    k0 = 0
    E = np.diag([1] * len(A))
    B = [[1.0 for i in range(len(A))] for j in range(len(A))]
    for i in range(len(A)):
        B[i] = [e - a for (e, a) in zip(E[i], A[i])]

    if check_convergence(B):
        x = [m + n for (m, n) in zip(np.dot(B, x0), b)]
        k0 = ceil(log((eps*(1 - np.linalg.norm(B)) /
                       np.linalg.norm(np.subtract(x, x0)))) /
                  log(np.linalg.norm(B)))
        x0 = x

        for _ in range(k0 - 1):
            x = [m + n for (m, n) in zip(np.dot(B, x0), b)]
            x0 = x
        return x0, k0
    else:
        raise ValueError


def diagonal_priority(A: list):
    if np.allclose(np.array(A), np.array(A).transpose(), 1e-8) and \
            np.all(np.linalg.eigvals(A) > 0):
        return True
    for i in range(len(A)):
        if not (2 * abs(A[i][i]) - sum(map(abs, A[i])) > 0 or 2 * abs(A[i][i])
                - sum(map(abs, column(A, i))) > 0):
            return False
    return True


def seidel(A, b, eps):
    n = len(A)
    x = [.0 for i in range(n)]

    converge = False
    while not converge:
        x_new = np.copy(x)
        for i in range(n):
            s1 = sum(A[i][j] * x_new[j] for j in range(i))
            s2 = sum(A[i][j] * x[j] for j in range(i + 1, n))
            x_new[i] = (b[i] - s1 - s2) / A[i][i]

        converge = sqrt(sum((x_new[i] - x[i]) ** 2 for i in range(n))) <= eps
        x = x_new

    return x


n = int(input('Enter dimension n: '))
print('Enter A matrix: ')
input_matrix = [[float(i) for i in input().split()] for j in range(n)]
b = [float(i) for i in input('Enter free coefficients: ').split()]
try:
    x, k = iteration_method(input_matrix[:], b, 0.01)
    x_f = [f'{i:.2f}' for i in x]
    print(x_f)
    print('k0 = ' + str(k))
except ValueError:
    print('There is no convergence')
if diagonal_priority(input_matrix[:]):
    x = seidel(input_matrix[:], b, 0.01)
    x_f = [f'{i:.2f}' for i in x]
    print(x_f)
else:
    print('There\'s no diagonal priority')

import math
import numpy as np


def cof_input(i, state):
    while True:
        try:
            show = 'x' if state == 0 else 'b'
            var = float(input(f"Коэффициент при {show}{i+1}: "))
        except ValueError:
            print("Неверное значение, попробуйте снова")
        else:
            return var


n = int(input("Введите размерность матрицы: "))
try:
    if n == 0:
        raise ValueError
except ValueError:
    exit()

print("\nВведите коэффициенты при \'x\'", end='\n')
a_input = np.array([[cof_input(i, 0) for i in range(n)] for j in range(n)])

print("\nВведите свободные коэффициенты", end='\n\n')
b_input = np.array([cof_input(i, 1) for i in range(n)])

u = np.array([[0.0 for i in range(n)] for j in range(n)])

print("\nМатрица A:\n")
print(a_input, end='\n\n')
print("Вектор свободных членов b:\n")
print(b_input, end='\n\n')

#симметризация системы 
a_transporent = np.transpose(a_input)
a = a_transporent @ a_input
b = a_transporent @ b_input
print("Симметризированная система:\n")
print(a, end='\n\n')
print(b, end='\n\n')

#решение методом квадратного корня
count = 0
for i in range(count, n):
    for j in range(count, n):
        if not i: #подсчет коэффициентов для первой строки
            try:
                u[i][i] = math.sqrt(a[i][i])
            except ValueError:
                raise Exception("Матрица не является положительно опредленной")
            else:    
                for k in range(1, n):
                    u[i][k] = a[i][k] / u[i][i]
                break
        else:
            if i == j:
                try:
                    u[i][i] = math.sqrt(a[i][i] - sum([u[k][i]**2 for k in range(i)]))
                except ValueError:
                    raise Exception("Матрица не является положительно опредленной")
            else:
                u[i][j] = (a[i][j] - sum([u[k][i]*u[k][j] for k in range(i)])) / u[i][i]
    count += 1

u_transporent = np.transpose(u)
u_mul = u_transporent @ u
u_mul_inv = np.linalg.inv(u_mul)
x = u_mul_inv @ b ###########
format_roots = [f"x{i + 1}: {x[i]}; " for i in range(len(x))]
print("Корни СЛАУ: ", "".join(format_roots))

#определитель матрицы А
det_a = 1
for i in (u[i][i]**2 for i in range(n)):
    det_a *= i
print(f"\nОпределитель матрицы A:\n\n{det_a}", end='\n\n')

#обратная матрица 
inv_a = np.array([[0.0 for i in range(n)] for j in range(n)])
for i in range(n):
    e_i = np.array([[0.0]] * n)
    e_i[i] = 1.0
    y = np.linalg.inv(u_transporent) @ e_i
    x_new = np.linalg.inv(u) @ y
    for j in range(n):
        inv_a[j][i] = x_new[j]
print(f"Обратная матрица:\n\n{inv_a}", end='\n\n')
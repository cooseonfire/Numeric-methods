import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, sqrt, asin


def tangent_method(f, df, ddf, a, b, eps):
    x0 = a if f(a)*ddf(a) > 0 else b  # start approximation
    x = x0 - f(x0)/df(x0)
    while abs(x - x0) > eps:
        x0 = x
        x = x0 - f(x0)/df(x0)
    return x


def chord_method(f, a, b, eps):
    x0 = a
    x1 = b
    while abs(x1 - x0) > eps:
        x = x1 - f(x1)*(x1 - x0)/(f(x1) - f(x0))
        x0, x1 = x1, x
    return x


def sim_method(fi, eps):
    k = 0
    x0 = [0.7, 0.7]
    x = fi(x0)
    while abs(max(x) - max(x0)) > eps:
        x0 = x
        x = fi(x0)
        k += 1
    return x, k


def newton_method(F, J, eps):
    k = 0  # iteration counter
    x0 = [0.4, 0.4]
    x = [m - n for (m, n) in zip(x0, np.linalg.inv(J(x0)) @ F(x0))]
    while abs(max(x) - max(x0)) > eps:
        x0 = x
        x = [m - n for (m, n) in zip(x0, np.linalg.inv(J(x0)) @ F(x0))]
        k += 1
    return x, k


def mod_newton_method(F, J, eps):
    k = 0
    x0 = [0.4, 0.4]
    inv_J = np.linalg.inv(J(x0))
    x = [m - n for (m, n) in zip(x0, inv_J @ F(x0))]
    while abs(max(x) - max(x0)) > eps:
        x0 = x
        x = [m - n for (m, n) in zip(x0, inv_J @ F(x0))]
        k += 1
    return x, k


def f1(x):
    return np.cos(1 + 0.2*x**2) - x


def df1(x):
    return -0.4 * x * np.sin(1 - 0.2*x**2) - 1


def ddf1(x):
    return -0.16 * np.cos(1 + 0.2*x**2)*x**2 - 0.4*np.sin(1 + 0.2*x**2)


def F(x):
    return [sin(x[0] + 2*x[1]) - 1.2*x[0], x[0]**2 + x[1]**2 - 1]


def J(x):
    return [[cos(x[0] + 2*x[1]) - 1.2, 2*cos(x[0] + 2*x[1])],
            [2*x[0], 2*x[1]]]


def fi(x):
    return [-0.5*x[0] + asin(1.2*x[0]/2), sqrt(1 - x[1]**2)]


# graphical root separation
x = np.linspace(-2*np.pi, 2*np.pi)
a = np.cos(1 + 0.2*x**2)
b = x
plt.plot(x, a, x, b)
plt.show()

# x = np.linspace(1e-5, 5)
# y = np.linspace(1e-5, 5)
# a = np.sin(x + 2*y) - 1.2*x
# b = x**2 + y**2 - 1
# plt.plot(x, a, x, b)
# plt.show()

x0 = tangent_method(f1, df1, ddf1, -2*np.pi, 2*np.pi, 1e-5)
x00 = chord_method(f1, -2*np.pi, 2*np.pi, 1e-5)
print(f'Tangent method:\nx = {x0:.5}\nChord method:\nx = {x00:.5}', end='\n')
x0, k = newton_method(F, J, 1e-5)
print(f'System:\nNewton\'s Method\nroot = {x0}, iterations = {k}', end='\n')
x0, k = mod_newton_method(F, J, 1e-5)
print(f'Modified Newton\'s method\nroot = {x0}, iterations = {k}', end='\n')
#  мпи не сходится, так как не выполнено условие сходимости dfi > 1

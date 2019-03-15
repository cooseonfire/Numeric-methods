from prettytable import PrettyTable
from math import exp, sqrt
import matplotlib.pyplot as plt


def f(x):
    return exp(-sqrt(x))


def F(x):
    return -2*exp(-sqrt(x))*(sqrt(x) + 1)


def d2f(x):
    return 1/4*exp(-sqrt(x))*(1/x**(3/2) + 1/x)


def d4f(x):
    return (15/16*exp(-sqrt(x))*(1/x**(7/2) + 1/x**3) +
            3/8*(exp(-sqrt(x))/x**(5/2)) + 1/16*(exp(-sqrt(x))/x**2))


def fk(x, y):
    return 0.5 * x * y**2 - y


def find_step(f, a, b, eps, order, method):
    h0 = eps**(1/order)
    amount = int(round((b - a)/h0))
    if amount:
        h0 /= 2
    while 1/15*abs(method(f, a, b, 2*h0) - method(f, a, b, h0)) >= eps:
        h0 /= 2
    return h0


def find_step_k(f, a, b, eps, order, method):
    h0 = eps**(1/order)
    amount = int(round((b - a)/h0))
    if amount % 2:
        h0 /= 2
    delta = [abs(i - j) for (i, j) in zip(method(f, a, b, 2*h0)[1], method(f, a, b, h0)[1])]
    while 1/15*max(delta) >= eps:
        h0 /= 2
        delta = [i - j for (i, j) in zip(method(f, a, b, 2 * h0)[1],
                                         method(f, a, b, h0)[1])]
    return h0


def trapeze(f, a, b, h, d2f=None):
    amount = int((b - a)/h)
    x_vals = [a + h*i for i in range(1, amount)]
    integral = h/2 *(f(a) + 2 * sum([f(i) for i in x_vals]) + f(b))
    if d2f:
        m2 = max([d2f(i) for i in x_vals])
        cond_R = m2 * (b - a) * h**2 / 12
    else:
        return integral
    return integral, cond_R


def simpson(f, a, b, h, d4f=None):
    h /= 2
    amount = int((b - a) / h)
    x_vals = [a + h*i for i in range(1, amount)]
    integral = h/3 * (f(a) + f(b) +
                      2*sum(f(k) for i, k in
                            enumerate(x_vals) if i % 2) +
                      4*sum(f(k) for i, k in
                            enumerate(x_vals) if not i % 2))
    if d4f is not None:
        m4 = max([d4f(i) for i in x_vals])
        cond_R = m4 * (b - a) * h**4 / 180
    else:
        return integral
    return integral, cond_R


def runge_cutte(f, a, b, h):
    amount = int((b - a) / h) + 1
    x = [a + h*i for i in range(amount)]
    y = [0.0 for i in range(amount)]
    y[0] = 2.0
    for i in range(1, amount):
        k1 = h * f(x[i - 1], y[i - 1])
        k2 = h * f(x[i - 1] + h/2, y[i - 1] + k1/2)
        k3 = h * f(x[i - 1] + h/2, y[i - 1] + k2/2)
        k4 = h * f(x[i - 1] + h, y[i - 1] + k3)
        y[i] = y[i - 1] + 1/6*(k1 + 2*(k2 + k3) + k4)
    return x, y


def adams(f, a, b, h):
    amount = int((b - a) / h) + 1
    x = [a + h * i for i in range(amount)]
    y = [0.0 for i in range(amount)]
    y[0] = 2.0
    y[1] = y[0] + h * f(x[0], y[0])
    for i in range(2, amount):
        y[i] = y[i - 1] + h * (3/2*f(x[i - 1], y[i - 1]) -
                               1/2*f(x[i - 2], y[i - 2]))
    return x, y


def eueler(f, a, b, h):
    amount = int((b - a) / h) + 1
    x = [a + h*i for i in range(amount)]
    y = [0.0 for i in range(amount)]
    y[0] = 2.0
    for i in range(1, amount):
        y[i] = y[i - 1] + h * f(x[i - 1], y[i - 1])
    return x, y


def print_table(x, yi, y2i, title):
    t = PrettyTable()
    column_names = ['xi', 'yi', '~yi', '|yi - ~yi|']
    t.add_column(column_names[0], [f'{i:.6}' for i in x])
    t.add_column(column_names[1], [f'{i:.6}' for i in yi])
    t.add_column(column_names[2], [f'{i:.6}' for i in y2i])
    t.add_column(column_names[3],
                 [f'{abs(i - j):.6}' for i, j in zip(yi, y2i)])
    print(t.get_string(title=title))


if __name__ == '__main__':
    h = find_step(f, 1, 4, 1e-3, 2, trapeze)
    print(f'Step is {h:.5}')
    it1 = trapeze(f, 1, 4, h, d2f)
    print(f'Trapeze: \nH:\n I = {it1[0]:.5}'
          f'\n R <= {it1[1]:.5}')
    it2 = trapeze(f, 1, 4, 2*h, d2f)
    print(f'\n2H:\n I = {it2[0]:.5}'
          f'\n R <= {it2[1]:.5}')

    is1 = simpson(f, 1, 4, h, d4f)
    print(f'\nSimpson:\nH:\n I = {is1[0]:.5}'
          f'\n R <= {is1[1]:.5}')
    is2 = simpson(f, 1, 4, 2*h, d4f)
    print(f'\nSimpson:\n2H:\n I = {is2[0]:.5}'
          f'\n R <= {is2[1]:.5}')

    print(f'\nNewton-L.:\n{F(4) - F(1):.5}')

    h = find_step_k(fk, 0, 2, 1e-4, 4, runge_cutte)
    t_res = PrettyTable()
    col_names = ['xi', 'Kosha', 'Runge-Kutta', 'Delta1',
                 'Adams', 'Delta2']
    xn, yn = runge_cutte(fk, 0, 2, h)
    plt.plot(xn, yn, 'ro', xn, yn, markersize=3)
    xn2, yn2 = runge_cutte(fk, 0, 2, 2*h)
    yn_even = [k for i, k in enumerate(yn) if not i % 2]
    kosha = [2/(j + 1) for j in xn2]

    t_res.add_column(col_names[0], [f'{i:.6}' for i in xn2])
    t_res.add_column(col_names[1], [f'{i:.6}' for i in kosha])
    t_res.add_column(col_names[2], [f'{i:.6}' for i in yn_even])
    delt1 = [abs(i - j) for i, j in zip(kosha, yn_even)]
    t_res.add_column(col_names[3], [f'{i:.6}' for i in delt1])
    max1 = max(delt1)

    print_table(xn2, yn_even, yn2, 'Runge-Kutta')

    xn, yn = eueler(fk, 0, 2, h)
    plt.plot(xn, yn, 'ro', xn, yn, markersize=3)
    xn2, yn2 = eueler(fk, 0, 2, 2*h)
    print_table(xn2, [k for i, k in enumerate(yn) if not i % 2],
                yn2, 'Eueler')

    xn, yn = adams(fk, 0, 2, h)
    plt.plot(xn, yn, 'ro', xn, yn, markersize=3)
    xn2, yn2 = adams(fk, 0, 2, 2*h)
    print_table(xn2, [k for i, k in enumerate(yn) if not i % 2],
                yn2, 'Adams')

    yn_even = [k for i, k in enumerate(yn) if not i % 2]
    delt2 = [abs(i - j) for i, j in zip(kosha, yn_even)]
    t_res.add_column(col_names[4], [f'{i:.6}' for i in yn_even])
    t_res.add_column(col_names[5], [f'{i:.6}' for i in delt2])
    max2 = max(delt2)
    print('Kosha solve: y = 2/(x + 1)')
    print(t_res.get_string())
    print(f'\nMax(d1) = {max1:.4}\nMax(d2) = {max2:.4}')


    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(('_', 'Runge-Kutta', '_', 'Eueler', '_', 'Adams'))
    plt.grid()
    plt.show()


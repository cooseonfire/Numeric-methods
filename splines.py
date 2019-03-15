import numpy as np
import sympy as sy
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from functools import reduce


def lagrange(points, order):
    if order > len(points):
        raise ValueError
    x_val, y_val = zip(*points)
    x = sy.Symbol('x')

    def basis(j):
        p = [(x - x_val[i])/(x_val[j] - x_val[i])
             for i in range(len(x_val)) if i != j]
        return reduce(lambda m, n: m * n, p)

    return sy.simplify(sum(basis(j) * y_val[j] for j in range(order)))


def newton(points, order):
    n = 0
    x_n = [p[0] for p in points]
    x = sy.Symbol('x')
    d_x = 1
    diff = divided_diffs(points)
    for i in range(order):
        n += diff[i][0] * d_x
        d_x *= (x - x_n[i])
    return sy.simplify(n)



def divided_diffs(points):
    x, _ = zip(*points)
    d = [['' for i in range(len(points))] for j in range(len(points))]
    d[0] = [p[1] for p in points]
    for i in range(1, len(points)):
        for j in range(0, len(points) - i):
            d[i][j] = (d[i-1][j+1] - d[i-1][j]) / (x[j + i] - x[j])
    return d


def finite_diffs(points):
    d = [['' for i in range(len(points))] for j in range(len(points))]
    d[0] = [p[1] for p in points]
    for i in range(1, len(points)):
        for j in range(len(points) - i):
            d[i][j] = d[i - 1][j + 1] - d[i - 1][j]
    return d


def linear_spline(points, xi=None):
    x = sy.Symbol('x')
    x_n, f = zip(*points)
    l_splines = []
    for i in range(1, len(points)):
        l_splines.append(sy.simplify(
            f[i-1] + (f[i] - f[i-1]) * (x - x_n[i-1]) / (x_n[i] - x_n[i-1])))

    if xi is None:
        return l_splines
    else:
        for i in range(1, len(points)):
            if xi <= x_n[i]:
                return l_splines[i-1].subs({'x': xi})



def quadratic_spline(points, xi=None):
    if (len(points) + 1) % 2:
        print('Can\'t build valid quadratic spline')
    else:
        x = sy.Symbol('x')
        x_n, f = zip(*points)
        q_splines = []
        for i in range(2, len(points), 2):
            a2 = (f[i] - f[i - 2])/((x_n[i] - x_n[i - 2]) *
                                    (x_n[i] - x_n[i - 1])) - \
                 (f[i - 1] - f[i - 2])/((x_n[i - 1] - x_n[i - 2]) *
                                        (x_n[i] - x_n[i - 1]))

            a1 = (f[i - 1] - f[i - 2])/(x_n[i - 1] - x_n[i - 2]) - \
                a2 * (x_n[i - 1] + x_n[i - 2])

            a0 = f[i - 2] - a1*x_n[i - 2] - a2*x_n[i - 2]**2

            q_splines.append(a0 + a1*x + a2*x**2)

        if xi is None:
            return q_splines
        else:
            for i in range(2, len(points), 2):
                if xi <= x_n[i]:
                    return q_splines[i//2 - 1].subs({'x': xi})


def cubic_spline(points, xi=None):
    x = sy.Symbol('x')
    c_splines =[]
    np1 = len(points)
    n = np1 - 1  # between 2 points
    x_n, y_n = zip(*points)
    a = y_n[:]
    b = [.0] * n
    d = [.0] * n
    h = [x_n[i + 1] - x_n[i] for i in range(n)]
    alpha = [.0] * n

    for i in range(1, n):
        alpha[i] = 3/h[i]*(a[i+1]-a[i]) - 3/h[i-1]*(a[i]-a[i-1])
    c = [.0] * np1
    L = [.0] * np1
    u = [.0] * np1
    z = [.0] * np1
    L[0] = 1.0
    u[0] = z[0] = 0.0
    for i in range(1, n):
        L[i] = 2*(x_n[i+1]-x_n[i-1]) - h[i-1]*u[i-1]
        u[i] = h[i]/L[i]
        z[i] = (alpha[i]-h[i-1]*z[i-1])/L[i]
    L[n] = 1.0
    z[n] = c[n] = 0.0
    for j in reversed(range(n)):
        c[j] = z[j] - u[j]*c[j+1]
        b[j] = (a[j+1]-a[j])/h[j] - (h[j]*(c[j+1]+2*c[j]))/3
        d[j] = (c[j+1]-c[j])/(3*h[j])
    for i in range(n):
        c_splines.append(a[i] + b[i]*x + c[i]*x**2 + d[i]*x**3)

    if xi is None:
        return c_splines
    else:
        return [(a[i], b[i], c[i], d[i], x_n[i]) for i in range(n)], x_n[n]


def splines_to_plot(splines, xn, res):
    n = len(splines)
    perSpline = int(res / n)
    if perSpline < 3:
        perSpline = 3
    X = []
    Y = []
    for i in range(n - 1):
        S = splines[i]
        x0 = S[4]
        x1 = splines[i + 1][4]
        x = np.linspace(x0, x1, perSpline)
        for xi in x:
            X.append(xi)
            h = (xi - S[4])
            Y.append(S[0] + S[1] * h + S[2] * h ** 2 + S[3] * h ** 3)
    S = splines[n - 1]
    x = np.linspace(S[4], xn, perSpline)
    for xi in x:
        X.append(xi)
        h = (xi - S[4])
        Y.append(S[0] + S[1] * h + S[2] * h ** 2 + S[3] * h ** 3)

    return X, Y


if __name__ == "__main__":
    points = [(0.135, -2.132), (0.876, -2.113), (1.336, -1.613),
            (2.301, -0.842), (2.642, 1.204)]

    # Lagrange polynomial
    x, y = zip(*points)
    x_n = np.linspace(min(x), max(x), 30)
    y_n1 = [lagrange(points, len(points)).subs({'x': i})
        for i in x_n]
    plt.title('Lagrange')
    plt.plot(x, y, 'o', x_n, y_n1)
    plt.grid(True)
    plt.show()
    print(f'L(x) = {lagrange(points, len(points))}')
    print(f'L4(x1 + x2) = '
        f'{lagrange(points, 4).subs( {"x": points[0][0] + points[1][0]})}')


    # Finite diffs table
    f_diffs = finite_diffs(points)
    t1 = PrettyTable([])
    column_names = ['xk', 'yk', 'd1', 'd2', 'd3', 'd4']
    t1.add_column(column_names[0], x)
    for i in range(len(f_diffs)):
        t1.add_column(column_names[i + 1], f_diffs[i])
    print(t1.get_string(title='Finite diffs'))


    # Divided diffs
    d_diffs = divided_diffs(points)
    t1 = PrettyTable([])
    column_names = ['xk', 'yk', 'f1', 'f2', 'f3', 'f4']
    t1.add_column(column_names[0], x)
    for i in range(len(f_diffs)):
        t1.add_column(column_names[i + 1], d_diffs[i])
    print(t1.get_string(title='Divided diffs'))


    # Newton polynomial
    print(f'N(x) = {newton(points, len(points))}')
    print(f'N4(x1 + x2) = '
        f'{newton(points, 4).subs( {"x": points[0][0] + points[1][0]})}')
    y_n2 = [newton(points, len(points)).subs({'x': i})
        for i in x_n]
    plt.plot(x, y, 'o', x_n, y_n2)
    plt.title('Newton')
    plt.grid(True)
    plt.show()


    # Linear spline
    print('Linear spline: ')
    print(f'F(x) = {linear_spline(points)}')
    y_n3 = [linear_spline(points, i) for i in x_n]
    plt.plot(x, y, 'o', x_n, y_n3)
    plt.title('Linear')
    plt.grid(True)
    plt.show()


    # Quadratic spline
    print('Quadratic spline: ')
    print(f'F(x) = {quadratic_spline(points)}')
    y_n4 = [quadratic_spline(points, i) for i in x_n]
    plt.plot(x, y, 'o', x_n, y_n4)
    plt.title('Quadratic')
    plt.grid(True)
    plt.show()


    print('Cubic spline: ')
    print(f'F(x) = {cubic_spline(points)}')
    splines, xn = cubic_spline(points, True)
    x_n2, y_n5 = splines_to_plot(splines, xn, 40)
    plt.plot(x, y, 'o', x_n2, y_n5)
    plt.title('Cubic')
    plt.grid(True)
    plt.show()


    plt.plot(x, y, 'o', x_n, y_n1, x_n, y_n2, x_n, y_n3, x_n, y_n4, x_n2, y_n5)
    plt.legend(('Points', 'Lagrange', 'Newton', 'Linear', 'Quadratic', 'Cubic'),
            loc='upper left')
    plt.grid(True)
    plt.show()

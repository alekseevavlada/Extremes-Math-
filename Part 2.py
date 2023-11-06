# Часть 2. Численный метод для функции z(x,y) = (2 * (x ** 2) + y ** 2) * np.exp(-x ** 2 - y ** 2)

# Возьмем точку экстремума для функции (1, 0)

import numpy as np

# Нахождение экстремума методом градиентного спуска


def f(x):
    return (2 * (x[0] ** 2) + x[1] ** 2) * np.exp(-x[0] ** 2 - x[1] ** 2)


def dfdx(x):
    return np.array([-2 * x[0] * (2 * x[0] ** 2 + x[1] ** 2) * np.exp(-x[0] ** 2 - x[1] ** 2) + 4 * x[0] * np.exp(-x[0] ** 2 - x[1] ** 2), -2 * x[1] * (2 * x[0] ** 2 + x[1] ** 2) * np.exp(-x[0] ** 2 - x[1] ** 2) + 2 * x[1] * np.exp(-x[0] ** 2 - x[1] ** 2)])


def gradsteps(f, r, epsg=1e-6, alpha=0.4, maxiter=100):
    # xlist = [r]
    for iteration in range(maxiter):
        arr_r = dfdx(r)
        summa = sum(arr_r ** 2)
        r = r + alpha * arr_r
        if np.sqrt(summa) < epsg:
            break
    r = np.round(r, 6)
    return r.tolist(), f(r), iteration


r = np.array([1, 1])

gradsteps_f_x = gradsteps(f, r)

print('Точка максимума функции, значение функции в этой точке и количество итераций:', gradsteps_f_x)

print('Точный результат, полученный аналитическим методом (точка максимума и значение функции): ([1.0, 0], 2e ** 2)')

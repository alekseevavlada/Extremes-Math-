import numpy as np
import matplotlib.pyplot as plt

# Метод Эйлера


def Euler(x0, y0, h, n):
    x = [x0]
    y = [y0]

    for i in range(n):
        for i in range(n):
            dx = (2 * (x[-1] ** 2) + y[-1] ** 2) * np.exp(-x[-1] ** 2 - y[-1] ** 2)
            dy = (2 * x[-1] * y[-1]) * np.exp(-x[-1] ** 2 - y[-1] ** 2)

        x.append(x[-1] + h * dx)
        y.append(y[-1] + h * dy)

    return x, y


x0 = 0.05
y0 = 0.05
h = 0.1
n = 100

x_values, y_values = Euler(x0, y0, h, n)


for i in range(n+1):
    print('Метод Эйлера:', f"x[{i}] = {x_values[i]}, y[{i}] = {y_values[i]}")


# Метод Кунге-Кутты


def z(x, y):
    return (2*x**2 + y**2) * np.exp(-x**2 - y**2)


def Runge(x0, y0, h, n):
    x = [x0]
    y = [y0]

    for i in range(n):
        k1x = (2 * (x[-1] ** 2) + y[-1] ** 2) * np.exp(-x[-1] ** 2 - y[-1] ** 2)
        k1y = (2 * x[-1] * y[-1]) * np.exp(-x[-1] ** 2 - y[-1] ** 2)

        x_mid = x[-1] + (h / 2) * k1x
        y_mid = y[-1] + (h / 2) * k1y

        k2x = (2 * (x_mid ** 2) + y_mid ** 2) * np.exp(-x_mid ** 2 - y_mid ** 2)
        k2y = (2 * x_mid * y_mid) * np.exp(-x_mid ** 2 - y_mid ** 2)

        xn = x[-1] + h * k2x
        yn = y[-1] + h * k2y

        x.append(xn)
        y.append(yn)

    return x, y


x, y = Runge(x0, y0, h, n)

# Значения функции z(x, y) для каждого значения x и y

z = (2 * np.array(x) ** 2 + np.array(y) ** 2) * np.exp(-np.array(x) ** 2 - np.array(y) ** 2)


max_index = np.argmax(z)
x_max = x[max_index]
y_max = y[max_index]
z_max = z[max_index]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c='r', marker='o')
ax.scatter(x_max, y_max, z_max, c='b', marker='o', label='Максимум')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.legend()
plt.show()

for i in range(n+1):
    print('Метод Кунге-Кутты:', f"x[{i}] = {x[i]}, y[{i}] = {y[i]}")
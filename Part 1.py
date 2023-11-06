import numpy as np
import matplotlib.pyplot as plt

# Построение графика функции z(x,y) = (2 * (x ** 2) + y ** 2) * np.exp(-x ** 2 - y ** 2)


def z(x, y):
    return (2*x**2 + y**2) * np.exp(-x**2 - y**2)


x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)

X, Y = np.meshgrid(x, y)
Z = z(X, Y)

Z_flat = Z.flatten()
max_index = np.argmax(Z_flat)
x_max = X.flatten()[max_index]
y_max = Y.flatten()[max_index]
z_max = Z_flat[max_index]

# График функции z(x, y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')
ax.scatter(x_max, y_max, z_max, color='r', label='Maximum')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.legend()
plt.show()

print('Получен график функции z(x, y)')
print('В численных методах максимум достигается в значениях: ', x_max, y_max, z_max)
print('В аналитическом методе это значение: (-1, 0, 0.74)')
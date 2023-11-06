# Задание 6. Лабораторная работа "Экстремумы ФНП"

# Построение графика функции z(x,y) = (2 * (x ** 2) + y ** 2) * np.exp(-x ** 2 - y ** 2)

import matplotlib.pyplot as plt
import numpy as np

# x = 0
# y = -1

x, y = np.meshgrid([i for i in range(-5, 5)], [i for i in range(-5, 5)])
z = (2 * (x ** 2) + y ** 2) * np.exp(-x ** 2 - y ** 2)

# print(z)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_title('Extremes?')
ax.plot_surface(x, y, z, cmap='inferno')

plt.show()


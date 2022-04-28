import matplotlib.pyplot as plt
import numpy as np


def h_theta(t, x1, x2):
    z = t[0] + t[1] * x1 + t[2] * x2
    return 1 / (1 + np.exp(-z))


def logistic_regression(d, t, lr, ep):
    for _ in range(ep):
        for s in d:
            x1 = s[0]
            x2 = s[1]
            y = s[2]
            t[0] += lr * (y - h_theta(t, x1, x2))
            t[1] += lr * (y - h_theta(t, x1, x2)) * x1
            t[2] += lr * (y - h_theta(t, x1, x2)) * x2
    return t


def x_2(t, x):
    return (t[0] + t[1] * x) * (-1 / t[2])


if __name__ == '__main__':
    data = np.loadtxt('data.txt', delimiter=' ')
    plt.scatter(data[:, 0], data[:, 1], c=data[:, 2], cmap='coolwarm')
    alpha = 1e-2
    epochs = 100
    thetas = np.random.uniform(-.01, .01, (3, 1))
    plt.plot(data[:, 0], [x_2(thetas, x) for x in data[:, 0]], 'g', label='Initial Boundary')
    thetas = logistic_regression(data, thetas, alpha, epochs)
    plt.plot(data[:, 0], [x_2(thetas, x) for x in data[:, 0]], 'k', label='Final Boundary')
    plt.legend()
    plt.show()

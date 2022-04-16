import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def f(x):
    return np.sin(2 * np.pi * x) + np.random.uniform(-.3, .3)


def h_theta_x(x, a):
    h = 0
    for d in range(len(a)):
        h += a[d] * x ** d
    return h


def sgd(x, y, tj, a, e):
    errs = {}
    for e in tqdm(range(e), file=sys.stdout):
        err = 0
        for i in range(len(x)):
            h = h_theta_x(x[i], tj)
            err += (h - y[i]) ** 2
            for j in range(len(tj)):
                tj[j] += a * (y[i] - h_theta_x(x[i], tj)) * x[i] ** j
        errs[e] = np.sqrt(err / len(x))
    return tj, errs


if __name__ == '__main__':
    x = np.random.uniform(0, 1, 100)
    y = np.array(list(map(f, x)))
    theta_j = np.random.uniform(-.5, .5, (8, 1))
    alpha = .05
    epochs = 5000
    theta_j, errors = sgd(x, y, theta_j, alpha, epochs)
    plt.plot(x, y, 'go')
    plt.plot(x, np.sin(2 * np.pi * x), 'bo')
    plt.plot(x, h_theta_x(x, theta_j), 'ro')
    plt.legend(['Data', 'Sine', 'Poly'])
    plt.show()
    plt.plot(list(errors.keys()), list(errors.values()), 'r')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.show()

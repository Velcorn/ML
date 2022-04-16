import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def f(x):
    """
    Calculate y_i = sin(2pix_i) + Epsilon
    :param x: x value
    :return: y value
    """
    return np.sin(2 * np.pi * x) + np.random.uniform(-.3, .3)


def h_theta_x(x, d):
    """
    Calculate h_theta(x)
    :param x: x value
    :param d: np.array of polynomial coefficients
    :return: h_theta(x)
    """
    h_theta = 0
    for i in range(len(d)):
        h_theta += d[i] * x ** i
    return h_theta


def sgd(x, y, d, a, e):
    """
    Stochastic Gradient Descent
    :param x: np.array x values
    :param y: np.array y values
    :param d: np.array polynomial coefficients
    :param a: float learning rate
    :param e: int epochs
    :return: np.array polynomial coefficients and dict of errors per epoch
    """
    errs = {}
    for e in tqdm(range(e), file=sys.stdout):
        err = 0
        for i in range(len(x)):
            h = h_theta_x(x[i], d)
            err += (h - y[i]) ** 2
            for j in range(len(d)):
                d[j] += a * (y[i] - h_theta_x(x[i], d)) * x[i] ** j
        err = 1/2 * err
        errs[e] = np.sqrt(2 * err / len(x))
    return d, errs


if __name__ == '__main__':
    # Initialize data points, polynomial coefficients and set learning rate and epochs
    x = np.random.uniform(0, 1, 100)
    y = np.array(list(map(f, x)))
    dim = np.random.uniform(-.5, .5, (5, 1))
    alpha = 1e-2
    epochs = 10000

    # Run SGD
    theta_j, errors = sgd(x, y, dim, alpha, epochs)

    # Plot cloud of data points, sine function, and polynomial
    plt.plot(x, y, 'go')
    plt.plot(x, np.sin(2 * np.pi * x), 'bo')
    plt.plot(x, h_theta_x(x, theta_j), 'ro')
    plt.legend(['Data', 'Sine', 'Poly'])
    plt.show()

    # Plot error vs epoch
    plt.plot(list(errors.keys()), list(errors.values()), 'r')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.show()

    print('All done!')

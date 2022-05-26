import matplotlib.pyplot as plt
import numpy as np
import sys
from tqdm import tqdm


def adaboost(X, y, n_clfs):
    # Initialize weights
    w = np.ones(len(X)) / len(X)
    # Initialize dict of classifiers
    clfs = {}
    # Iterate over classifiers
    for i in tqdm(range(n_clfs), file=sys.stdout):
        # Create classifier as random thresholds
        clf = np.random.uniform(np.min(X), np.max(X), len(X))
        # Randomly select x- or y-axis
        axis = np.random.choice([0, 1])
        data = X[:, axis]
        # Calculate error
        min_error = np.inf
        for t in clf:
            p = 1
            preds = np.ones(len(data))
            preds[data < t] = -1
            error = np.sum(w[preds != y])

            # Swap parity if error higher than 50%
            if error > .5:
                error = 1 - error
                p = -1

            # Update minimum error
            if error < min_error:
                min_error = error
                parity = p
                threshold = t

        # Calculate alpha
        alpha = .5 * np.log((1 - min_error) / min_error)

        # Predict with fitted weak classifier and update weights
        preds = np.ones(len(data))
        if parity == 1:
            preds[data < threshold] = -1
        else:
            preds[data > threshold] = -1
        w *= np.exp(-alpha * y * preds)
        w /= np.sum(w)

        # Add classifier and params to dict
        clfs[i] = (axis, parity, threshold, alpha, preds)

    return clfs


if __name__ == '__main__':
    # Set number of classifiers
    n_clfs = 20

    # Get matrix from textfile as numpy array and split into X and y
    mat = np.loadtxt('dataCircle.txt')
    X = mat[:, 0:2]
    y = mat[:, 2]

    # Run AdaBoost
    print('Running AdaBoost...')
    clfs = adaboost(X, y, n_clfs)
    H = np.sign(np.sum([clf[-1] * clf[-2] for clf in clfs.values()], axis=0))

    # Plot ground truth
    print('Plotting...')
    colors = ['lightgreen' if x == 1 else 'red' for x in y]
    plt.scatter(X[:, 0], X[:, 1], c=colors)
    plt.title('Ground Truth')
    plt.show()

    # Plot AdaBoost result
    colors = ['lightgreen' if x == 1 else 'red' for x in H]
    plt.scatter(mat[:, 0], mat[:, 1], c=colors)
    plt.title('AdaBoost')

    # Plot weak classifier lines and show plot
    for clf in clfs.values():
        if clf[0] == 0:
            plt.hlines(clf[2], np.min(X), np.max(X), colors='k', linestyles='dashed')
        else:
            plt.vlines(clf[2], np.min(X), np.max(X), colors='k', linestyles='dashed')
    plt.show()

    # Print some results
    print(f'Prediction: {list(H)}')
    print(f'Ground truth: {list(y)}')
    print(f'The params are (axis, parity, threshold, alpha):')
    for val in clfs.values():
        print(val[:-1])

    # All done
    print('All Done!')

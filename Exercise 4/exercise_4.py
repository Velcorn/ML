import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score


def adaboost(X, y, n_clfs):
    # Initialize weights and dict of classifiers
    w = np.ones(len(X)) / len(X)
    clfs = {}

    # Iterate over classifiers
    for i in range(n_clfs):
        # Create weak classifier as len(X) random thresholds between min and max value of X
        clf = np.random.uniform(np.min(X), np.max(X), len(X))

        # Randomly select x- or y-axis
        axis = np.random.choice([0, 1])
        data = X[:, axis]

        # Calculate minimum error by iterating over thresholds and comparing predictions with y
        min_error = np.inf
        for t in clf:
            # Predict with parity 1 and calculate error
            p = 1
            preds = np.ones(len(data))
            preds[data < t] = -1
            error = np.sum(w[preds != y])

            # Update error and parity if error higher than 50%
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

        # Predict with fitted weak classifier
        preds = np.ones(len(data))
        if parity == 1:
            preds[data < threshold] = -1
        else:
            preds[data > threshold] = -1

        # Update weights
        w *= np.exp(-alpha * y * preds)
        w /= np.sum(w)

        # Add weak classifier params to dict
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
    print(f'Accuracy: {round(accuracy_score(H, y) * 100, 2)}%')
    print(f'The params are (axis, parity, threshold, alpha):')
    for val in clfs.values():
        print(val[:-1])

    # All done
    print('All Done!')

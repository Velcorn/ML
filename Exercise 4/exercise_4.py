import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score


def adaboost(X, y, n_clfs):
    # Initialize distribution and dict of classifiers
    D_t = np.ones(len(X)) / len(X)
    h_ts = {}

    # Iterate over classifiers
    for i in range(n_clfs):
        # Create weak classifier as len(X) random thresholds between min and max value of X
        h_t = np.random.uniform(np.min(X), np.max(X), len(X))

        # Randomly select x- or y-axis
        axis = np.random.choice([0, 1])
        data = X[:, axis]

        # Calculate minimum error by iterating over thresholds and comparing predictions with y
        min_e_t = np.inf
        for t in h_t:
            # Predict with parity 1 and calculate error
            p_t = 1
            preds = np.ones(len(data))
            preds[data < t] = -1
            e_t = np.sum(D_t[preds != y])

            # Update error and parity if error higher than 50%
            if e_t > .5:
                e_t = 1 - e_t
                p_t = -1

            # If error lower than minimum error, update minimum error, parity and threshold
            if e_t < min_e_t:
                min_e_t = e_t
                parity = p_t
                threshold = t

        # Calculate alpha
        alpha_t = .5 * np.log((1 - min_e_t) / min_e_t)

        # Predict with fitted weak classifier
        preds = np.ones(len(data))
        if parity == 1:
            preds[data < threshold] = -1
        else:
            preds[data > threshold] = -1

        # Update distribution
        D_t *= np.exp(-alpha_t * y * preds)
        D_t /= np.sum(D_t)

        # Add weak classifier params to dict
        h_ts[i] = (axis, parity, threshold, alpha_t, preds)

    return h_ts


if __name__ == '__main__':
    # Set number of classifiers
    n_clfs = 20

    # Get matrix from textfile as numpy array and split into X and y
    mat = np.loadtxt('dataCircle.txt')
    X = mat[:, 0:2]
    y = mat[:, 2]

    # Run AdaBoost
    print('Running AdaBoost...')
    h_ts = adaboost(X, y, n_clfs)

    # Calculate prediction of strong classifier
    H = np.sign(np.sum([clf[-1] * clf[-2] for clf in h_ts.values()], axis=0))

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
    for clf in h_ts.values():
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
    for val in h_ts.values():
        print(val[:-1])

    # All done
    print('All Done!')

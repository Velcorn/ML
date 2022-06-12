import glob
import numpy as np
from PIL import Image


# Load images into list as numpy arrays, create ground truth list and create feature vectors
images = [np.asarray(Image.open(x)) for x in glob.glob('../Data/Exercise 3&5/*/*.png')]
ground_truth = np.asarray([0] * 30 + [1] * 30)
features = [[img[:, :, 0].min(), img[:, :, 1].min(), img[:, :, 2].min(), round(img[:, :, 0].mean()),
             round(img[:, :, 1].mean()), round(img[:, :, 2].mean())] for img in images]
features = np.asarray(features).astype(np.float64)

# Fit the model
X, y = features, ground_truth
# Number of classes
c = len(np.unique(y))
phi = y.mean()
mu_0 = X[y == 0].mean(axis=0)
mu_1 = X[y == 1].mean(axis=0)
# Combine the two mus for easier indexing
mu = np.array([mu_0, mu_1])
# Subtract the two mus from the data
# Create a copy to avoid changing the original data
X_mu = X.copy()
X_mu[y == 0] -= mu_0
X_mu[y == 1] -= mu_1
# Compute the covariance matrix and its inverse
sigma = X_mu.T @ X_mu / len(y)
sigma_inv = np.linalg.inv(sigma)

# Predict with the fitted model
probabilities = []
for i in range(len(mu)):
    # Get corresponding mu and calculate phi
    _mu = mu[i]
    _phi = phi ** i * (1 - phi) ** (1 - i)
    # Calculate the probability according to the formula
    probabilities.append(np.exp(-1 * np.sum((X - _mu).dot(sigma_inv) * (X - _mu), axis=1)) * _phi)

# Finally predict the classes based on the probabilities
y_pred = np.argmax(probabilities, axis=0)

# Print prediction, ground truth and accuracy
print(f'Prediction:   {", ".join(y_pred.astype(str))}')
print(f'Ground truth: {", ".join(np.char.mod("%d", y))}')
print(f'The model accuracy is: {round(np.mean(y_pred == y) * 100, 2)}%\n')

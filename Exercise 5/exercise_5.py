import glob
import numpy as np
from PIL import Image


# Load images into list as numpy arrays, create ground truth list and create feature vectors
images = [np.asarray(Image.open(x)) for x in glob.glob('../Data/Exercise 3&5/*/*.png')]
ground_truth = [0] * 30 + [1] * 30
features = []
for img in images:
    features.append([img[:, :, 0].min(), img[:, :, 1].min(), img[:, :, 2].min(), round(img[:, :, 0].mean()),
                     round(img[:, :, 1].mean()), round(img[:, :, 2].mean())])


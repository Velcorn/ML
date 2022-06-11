import glob
import numpy as np
from PIL import Image
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


for k in ['linear', 'poly', 'sigmoid', 'rbf']:
    # Initialize the classifier, give multiple params to test, pass kernel
    params = {
        'C': [.01, .1, 1, 10, 100], 'gamma': [.00001, .001, .01, .1, 1], 'kernel': [k]
    }
    svc = svm.SVC()
    model = GridSearchCV(svc, params)

    # Load images into list as numpy arrays, create ground truth list and create feature vectors
    images = [np.asarray(Image.open(x)) for x in glob.glob('*/*.png')]
    ground_truth = [0] * 30 + [1] * 30
    features = []
    for img in images:
        features.append([img[:, :, 0].min(), img[:, :, 1].min(), img[:, :, 2].min(), round(img[:, :, 0].mean()),
                         round(img[:, :, 1].mean()), round(img[:, :, 2].mean())])

    # Split the data into training and testing sets, fit the model and print the results
    x_train, x_test, y_train, y_test = train_test_split(features, ground_truth, test_size=0.5, random_state=42)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    # Print some results
    print(f'Prediction:   {list(y_pred)}')
    print(f'Ground truth: {y_test}')
    print(f'The best params are: {model.best_params_}')
    print(f'The model accuracy is: {round(accuracy_score(y_pred, y_test) * 100, 2)}%\n')

import numpy as np


def mean_variance(y):

    return np.sum(np.square(y - np.mean(y)))


def fit_variance(y, y_pred):

    return np.sum(np.square(y - y_pred))


def r_squared(y, y_pred):

    print(fit_variance(y, y_pred))
    print(mean_variance(y))

    return 1 - fit_variance(y, y_pred)/mean_variance(y)


def normal_equations(X, y):

    X_transpose = X.T
    return np.linalg.inv(X_transpose.dot(X)).dot(X_transpose).dot(y)

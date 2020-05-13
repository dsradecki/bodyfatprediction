import numpy as np
from scipy.stats import t


def ss_error(y, y_pred):
    return np.sum(np.square(y - y_pred))


def ss_total(y):
    return np.sum(np.square(y - np.mean(y)))


def ss_regression(y, y_pred):
    return np.sum(np.square(y_pred - np.mean(y)))


def r_squared(y, y_pred):
    return 1 - ss_error(y, y_pred)/ss_total(y)


def adjusted_r_2(y, y_pred, n, k):
    return r_squared(y, y_pred) * (n-1)/(n-k-1)


def normal_equations(X, y):
    X_transpose = X.T
    return np.linalg.inv(X_transpose.dot(X)).dot(X_transpose).dot(y)


def hat_matrix(X):

    return X.dot(np.linalg.inv(X.T.dot(X)).dot(X.T))


def predict(X, params):
    return X.dot(params)


def ms_error(y, y_pred, n, k):
    return ss_error(y, y_pred)/(n - k - 1)


def ms_res(y, y_pred, k):
    return ss_regression(y, y_pred) / k


def var_cor_matrix(X, y, y_pred, n, k):
    return ms_error(y, y_pred, n, k) * np.linalg.inv(X.T.dot(X))


def t_values(X, y, y_pred, n, k, params):
    return [param / np.sqrt(st_error) for param, st_error in zip(params, var_cor_matrix(X, y, y_pred, n, k).diagonal())]


def p_values(X, y, y_pred, n, k, params):
    return [(1 - t.cdf(abs(t_value), n - k) - 1) * 2 for t_value in t_values(X, y, y_pred, n, k, params)]


def dependent_corr(X, y):

    params = normal_equations(X, y)
    y_pred = predict(X, params)
    sst = ss_total(y)
    ssr = ss_regression(y, y_pred)

    return ssr/sst


def variance_inflation_factor(X):

    vifs = []

    for param_name in X.columns:

        new_y = X.loc[:, param_name]
        new_X = X.drop(param_name, axis=1)
        r = dependent_corr(new_X, new_y)
        vif = 1/(1 - r)
        vifs.append(vif)

    return vifs


def calc_residuals(y, y_pred):

    return y - y_pred


def residual_analysis(X, k, n, y, y_pred):

    hat_diagonal = np.array(hat_matrix(X)).diagonal()
    outliers = [idx for idx, element in enumerate(hat_diagonal) if element > (2*k+1)/n]

    residuals = calc_residuals(y, y_pred)


    return outliers

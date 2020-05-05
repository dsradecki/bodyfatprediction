from numpy.linalg import inv


class MultipleRegression:

    def __init__(self):
        self.params = []

    def fit(self, X, y):
        X_transpose = X.T
        self.params = inv(X_transpose.dot(X)).dot(X_transpose).dot(y)

    def predict(self, X):
        if len(self.params) == 0:
            return None

        return X.dot(self.params)

    def get_params(self):
        return self.params



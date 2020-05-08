from numpy.linalg import inv
import src.modelling.helper as helper
import importlib

importlib.reload(helper)


class MultipleRegression:

    class Prediction:

        def __init__(self, params, X, Y):
            self.params = params
            self.X = X
            self.Y = Y
            self.r_squared = helper.r_squared(self.Y, self.predict(self.X))

        def predict(self, X):
            return X.dot(self.params)

        def analyse(self):
            return self.r_squared

    def fit(self, X, y):

        params = helper.normal_equations(X, y)

        return self.Prediction(params, X, y)

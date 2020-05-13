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
            self.obs_num = X.shape[0]
            self.params_num = X.shape[1]
            self.predictions = self.predict(self.X)
            self.r_squared = helper.r_squared(self.Y, self.predictions)
            self.adjusted_r = helper.adjusted_r_2(Y, self.predictions, self.obs_num, self.params_num)
            self.p_values = helper.p_values(self.X, self.Y, self.predictions, self.obs_num, self.params_num, self.params)

        def predict(self, X):
            return X.dot(self.params)

        def analyse(self):

            return helper.variance_inflation_factor(self.X)

    def fit(self, X, y):

        params = helper.normal_equations(X, y)

        return self.Prediction(params, X, y)

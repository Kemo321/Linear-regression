import numpy


class LinearRegression:
    def __init__(self):
        self.coef = None
        self.intercept = None

    def fit(self, X, y):
        top = 0
        bottom = 0
        for i in range(len(X)):
            top += (X[i] - numpy.mean(X)) * (y[i] - numpy.mean(y))
            bottom += (X[i] - numpy.mean(X)) ** 2

        self.coef = top / bottom
        self.intercept = numpy.mean(y) - self.coef * numpy.mean(X)

    def predict(self, X):
        return self.coef * X + self.intercept

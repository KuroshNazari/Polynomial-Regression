from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class PolynomialRegression:

    """
    A class to perform polynomial regression on a single-feature dataset.

    - Works only with datasets having a single feature (1D input X).
    - The user must specify the desired polynomial degree during initialization.

    Methods:
    - fit(X, y): Fits a polynomial to the data (X, y).
    - predict(X): Predicts values for input X using the fitted polynomial.
    - plot(data_x, data_y): Plots the data points and the fitted polynomial curve.

    Example usage:

        >>> X = [1, 2, 3], y = [1, 4, 9]
        >>> model = PolynomialRegression(degree=3)
        >>> model.fit(X, y)
        >>> model.plot(X, y)
        >>> predictions = model.predict([1.5, 2.5])
            [2.25, 6.25]

    """

    def __init__(self, degree:int):
        self.degree = degree
        self.n_samples = None
        self.polynomial = None


    def _build_summation_matrix(self, X, y):
        x_summation_matrix = []
        x_summation_matrix.append(len(X))
        for i in range(1, self.degree*2 + 1):
            list = [x**i for x in X]
            x_summation_matrix.append(sum(list))

        xy_summation_matrix = []
        for i in range(0, self.degree + 1):
            list = [(X[ind]**i)*(y[ind]) for ind in range(self.n_samples)]
            xy_summation_matrix.append(sum(list))

        return x_summation_matrix, xy_summation_matrix


    def _buid_coefficient_matrix(self, X, y):
        x_summation_matrix, xy_summation_matrix = self._build_summation_matrix(X, y)
        coefficient_matrix = []
        y_matrix = xy_summation_matrix[::-1]
       
        for i in range(self.degree + 1):
            list = [x_summation_matrix[-ind-i] for ind in range(1, self.degree + 2)]
            coefficient_matrix.append(list)

        return coefficient_matrix, y_matrix


    def _solve_linear_system(self, coefficient_matrix, y_matrix):
        ans = np.linalg.solve(coefficient_matrix, y_matrix)
        return ans


    def fit(self, X: list, y: list) -> None:
        self.n_samples = len(X)

        x1 = (self._buid_coefficient_matrix(X, y)[0])
        y1 = (self._buid_coefficient_matrix(X, y)[1])
        coefficients = self._solve_linear_system(x1, y1)

        self.polynomial = np.poly1d(coefficients)


    def predict(self, X: list) -> list:
        return self.polynomial(X)
    
    
    def plot(self, data_x: list, data_y: list) -> None:
        polinomial_X = np.arange(min(data_x)-1, max(data_x)+1, 0.01)
        polinomial_y = self.polynomial(polinomial_X)

        plt.title(self.polynomial)
        plt.scatter(data_x, data_y, label='Data Points', color='blue')
        plt.plot(polinomial_X, polinomial_y, label='Fit', color='red')
        plt.grid()
        plt.legend()
        plt.show()




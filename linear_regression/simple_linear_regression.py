import numpy as np

# sklearn like interface
class SimpleLinearRegression:
    def __init__(self, is_analytical=True, lr=None, eps=None):
        self.is_analytical = is_analytical
        if not self.is_analytical and (lr is None or eps is None):
            raise ValueError("lr and eps must be set if you want to use iterative appraoch")

        self.lr = lr
        self.eps = eps
        self.coef_ = None

    def fit(self, x, y):
        if self.is_analytical:
            self.analytical_linear_regression(x, y)
        else:
            self.gradient_descent_linear_regression(x, y)

    def predict(self, x):
        return self.coef_[0]*x + self.coef_[1]

    def analytical_linear_regression(self, x, y):
        n = len(x)
        sum_x = np.sum(x)
        sum_y = np.sum(y)

        num = -x @ y + sum_x * sum_y / n
        denom = -np.sum(x ** 2) + (sum_x ** 2) / n
        a = num / denom

        b = (sum_y - a * sum_x) / n

        self.coef_ = (a, b)

    def gradient_descent_linear_regression(self, x, y):
        # initialize a and b
        a, b = 0, 0

        while True:
            dLda, dLdb = self.get_dLda(a, b, x, y), self.get_dLdb(a, b, x, y)

            a = a - self.lr * dLda
            b = b - self.lr * dLdb

            if self.magnitude(dLda, dLdb) < self.eps:
                break

        self.coef_ = (a, b)

    def get_dLda(self, a, b, x, y):
        sum_ = np.sum(x * y) - a * np.sum(x ** 2) - b * np.sum(x)

        return -2 * sum_ / len(x)

    def get_dLdb(self, a, b, x, y):
        sum_ = np.sum(y) - a * np.sum(x) - b * len(x)

        return -2 * sum_ / len(x)

    def magnitude(self, dLda, dLdb):
        return np.sqrt(dLda ** 2 + dLdb ** 2)
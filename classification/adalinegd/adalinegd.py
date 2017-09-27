from typing import TypeVar, Generic
import numpy as np

T = TypeVar('AdalineGD')


class AdalineGD(Generic[T]):
    """ADAptive LInear NEuron classifier.

    Attributes:
        w_: 1d-array
            Weights after fitting.
        cost_: list
            Value of cost function
    """
    w_ = None
    cost_ = None

    def __init__(self, eta: float = 0.01, n_iter: int = 10):
        """
        :param eta: float Learning rate (between 0.0 and 1.0)
        :param n_iter: int Passes over the training dataset (epochs)
        """
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X: np.ndarray, y: np.ndarray) -> T:
        """Fit training data.

        :param X:  np.ndarray shape = [n_samples, n_features] Training vectors, where n_samples is
            the number of samples and n_features is the number of features.
        :param y:  np.ndarray shape = [n_samples] Target values.

        :return: AdalineGD self object
        """

        # create n (features) + 1 long zeros array of initial weights
        # self.w_[0] is going to be our bias
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()

            cost = (errors ** 2).sum() / 2.0
            self.cost_.append(cost)

        return self

    def net_input(self, X: np.ndarray) -> np.ndarray:
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X) -> np.ndarray:
        return self.net_input(X)

    def predict(self, X: np.ndarray) -> int:
        """Return class label after unit step"""
        return np.where(self.activation(X) >= 0.0, 1, -1)

from typing import TypeVar, Generic
import numpy as np
from numpy.random import seed, permutation

T = TypeVar('AdalineSGD')


class AdalineSGD(Generic[T]):
    """ADAptive LInear NEuron classifier.

    Attributes:
        w_: 1d-array
            Weights after fitting.
        cost_: list
            Value of cost function
    """
    w_ = None
    cost_ = None

    def __init__(self, eta: float = 0.01, n_iter: int = 10, shuffle=True, random_state=None):
        """
        :param eta: float Learning rate (between 0.0 and 1.0)
        :param n_iter: int Passes over the training dataset (epochs)
        :param shuffle: bool (default: True) Shuffles training data every epoch if True to prevent cycles
        :param random_state: int (default: None) Set random state for shuffling and initializing the weights
        """
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle

        if random_state:
            seed(random_state)

    def fit(self, X: np.ndarray, y: np.ndarray) -> T:
        """Fit training data.

        :param X:  np.ndarray shape = [n_samples, n_features] Training vectors, where n_samples is
            the number of samples and n_features is the number of features.
        :param y:  np.ndarray shape = [n_samples] Target values.

        :return: AdalineSGD self object
        """

        # create n (features) + 1 long zeros array of initial weights
        # self.w_[0] is going to be our bias
        self._initialize_weights(X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)

            cost = []

            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))

            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)

        return self

    def partial_fit(self, X, y):
        """Fit training data without reinitializing the weights"""
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])

        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)

        return self

    def _initialize_weights(self, m):
        self.w_ = np.zeros(1 + m)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        output = self.net_input(xi)
        error = (target - output)
        self.w_[1:] += self.eta * xi.T.dot(error)
        self.w_[0] += self.eta * error

        cost = 0.5 * error ** 2
        return cost

    def _shuffle(self, X: np.ndarray, y: np.ndarray):
        r = np.random.permutation(len(y))
        return X[r], y[r]

    def net_input(self, X: np.ndarray) -> np.ndarray:
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X) -> np.ndarray:
        return self.net_input(X)

    def predict(self, X: np.ndarray) -> int:
        """Return class label after unit step"""
        return np.where(self.activation(X) >= 0.0, 1, -1)

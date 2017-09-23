from typing import TypeVar, Generic
import numpy as np

T = TypeVar('Perceptron')


class Perceptron(Generic[T]):
    """Percepton classifier
    
    Attributes:
        w_: 1d-array
            Weights after fitting.
        errors_: list
            Number of misclassifications in every epoch.
    """
    w_ = None
    errors_ = None

    def __init__(self, eta: float = 0.01, n_iter: int = 10):
        """
        :param eta: float Learning rate (between 0.0 and 1.0)
        :param n_iter: int Passes over the training dataset (epochs)
        """
        self.eta = eta
        self.n__iter = n_iter

    def fit(self, X: np.ndarray, y: np.ndarray) -> T:
        """Fit training data.
        
        :param X:  np.ndarray shape = [n_samples, n_features] Training vectors, where n_samples is
            the number of samples and n_features is the number of features.
        :param y:  np.ndarray shape = [n_samples] Target values.
        
        :return: Perceptron self object
        """

        # create n (features) + 1 long zeros array of initial weights
        # self.w_[0] is going to be our bias
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        # for number of iteration to teach Perceptron
        for _ in range(self.n__iter):
            errors = 0

            # for each sample data take data and its desired output value
            for xi, target in zip(X, y):
                # scalar update weight value
                update = self.eta * (target - self.predict(xi))

                # update all of the weight
                self.w_[1:] += update * xi

                # update bias
                self.w_[0] += update

                errors += int(update != 0.0)

            self.errors_.append(errors)

        return self

    def net_input(self, X: np.ndarray) -> float:
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X: np.ndarray) -> int:
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)

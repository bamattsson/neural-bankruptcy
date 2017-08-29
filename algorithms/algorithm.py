from abc import ABCMeta, abstractmethod
import numpy as np


class Algorithm(metaclass=ABCMeta):
    """Abstract algorithm class."""

    @abstractmethod
    def fit(self, samples, labels):
        """
        Args:
            samples (np.ndarray): X data, shape (n_samples, n_features)
            labels (np.ndarray): y data, shape (n_samples)
        Returns:
            fit_info (dict): information from the fit for later analysis
        """
        return None

    @abstractmethod
    def predict_proba(self, samples):
        """
        Args:
            samples (np.ndarray): X data, shape (n_samples, n_features)
        Returns:
            proba (np.ndarray): Probability of belonging to a particular class,
                shape (n_samples, n_classes)
        """
        proba = np.zeros((samples.shape[0], 2))
        return proba

    def predict(self, samples):
        """
        Args:
            samples (np.ndarray): X data, shape (n_samples, n_features)
        Returns:
            predict (np.ndarray): Predicted class, shape (n_samples)
        """
        predict_proba = self.predict_proba(samples)
        predict = np.argmax(predict_proba, axis=1)
        return predict

import numpy as np
from .algorithm import Algorithm


class RandomGuessAlgorithm(Algorithm):

    def __init__(self):
        pass

    def fit(self, samples, labels):
        """
        Args:
            samples (np.ndarray): X data, shape (n_samples, n_features)
            labels (np.ndarray): y data, shape (n_samples)
        Returns:
            fit_info (dict): information from the fit for later analysis
        """
        return None

    def predict_proba(self, samples):
        """
        Args:
            samples (np.ndarray): X data, shape (n_samples, n_features)
        Returns:
            proba (np.ndarray): Probability of belonging to a particular class,
                shape (n_samples,n_classes)
        """

        proba = np.zeros((samples.shape[0], 2))
        proba[:, 0] = np.random.rand(samples.shape[0])
        proba[:, 1] = np.ones(samples.shape[0]) - proba[:, 0]
        return proba

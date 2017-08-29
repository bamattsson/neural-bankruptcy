from .algorithm import Algorithm
from sklearn.ensemble import GradientBoostingClassifier


class GradientBoostingAlgorithm(Algorithm):

    def __init__(self, loss='deviance', learning_rate=0.1, n_estimators=100,
            subsample=1.0, criterion='friedman_mse', min_samples_split=2,
            min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3,
            min_impurity_split=1e-07, init=None, random_state=None,
            max_features=None, verbose=0, max_leaf_nodes=None,
            warm_start=False, presort='auto'):

        self.clf = GradientBoostingClassifier(loss=loss,
                learning_rate=learning_rate, n_estimators=n_estimators,
                subsample=subsample, criterion=criterion,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                min_weight_fraction_leaf=min_weight_fraction_leaf,
                max_depth=max_depth, min_impurity_split=min_impurity_split,
                init=init, random_state=random_state, max_features=max_features,
                verbose=verbose, max_leaf_nodes=max_leaf_nodes,
                warm_start=warm_start, presort=presort)

    def fit(self, samples, labels):
        """
        Args:
            samples (np.ndarray): X data, shape (n_samples, n_features)
            labels (np.ndarray): y data, shape (n_samples)
        Returns:
            fit_info (dict): information from the fit for later analysis
        """
        self.clf.fit(samples, labels)

    def predict_proba(self, samples):
        """
        Args:
            samples (np.ndarray): X data, shape (n_samples, n_features)
        Returns:
            proba (np.ndarray): Probability of belonging to a particular class,
                shape (n_samples,n_classes)
        """
        return self.clf.predict_proba(samples)

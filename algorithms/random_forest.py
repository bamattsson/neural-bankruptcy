from .algorithm import Algorithm
from sklearn.ensemble import RandomForestClassifier


class RandomForestAlgorithm(Algorithm):

    def __init__(self, n_estimators=10, criterion='gini', min_samples_split=2,
            min_samples_leaf=1, min_weight_fraction_leaf=0.0, warm_start=False,
            bootstrap=True, oob_score=False, max_features='auto',
            max_depth=None, max_leaf_nodes=None, min_impurity_split=1e-07,
            class_weight=None):

        self.clf = RandomForestClassifier(n_estimators=n_estimators,
                criterion=criterion, min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                min_weight_fraction_leaf=min_weight_fraction_leaf,
                warm_start=warm_start, bootstrap=bootstrap,
                oob_score=oob_score, max_features=max_features,
                max_depth=max_depth, max_leaf_nodes=max_leaf_nodes,
                min_impurity_split=min_impurity_split,
                class_weight=class_weight)

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

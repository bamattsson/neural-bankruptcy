import numpy as np
from algorithm import Algorithm
from sklearn.ensemble import RandomForestClassifier

class RandomForestAlgorithm(Algorithm, RandomForestClassifier):
    
    def __init__(self, n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, bootstrap=True, oob_score=False, n_jobs=1, verbose=0, warm_start=False):
    
        super(RandomForestAlgorithm, self).__init__(n_estimators=n_estimators, min_samples_split=min_samples_split,  min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, n_jobs=n_jobs, verbose=verbose, warm_start=warm_start, criterion=criterion, bootstrap=bootstrap, oob_score=oob_score, max_features=max_features, max_depth=max_depth, max_leaf_nodes=max_leaf_nodes, min_impurity_split=min_impurity_split)

    #are fit and predict proba used? just leave?
    def fit(self, samples, labels):
        """
        Args:
        samples (np.ndarray): X data, shape (n_samples, n_features)
        labels (np.ndarray): y data, shape (n_samples)
        Returns:
        fit_info (dict): information from the fit for later analysis
        """
        super(RandomForestClassifier,self).fit(samples, labels)
        return None
    
    def predict_proba(self, samples):
        """
        Args:
        samples (np.ndarray): X data, shape (n_samples, n_features)
        Returns:
        proba (np.ndarray): Probability of belonging to a particular class, shape (n_samples,n_classes)
        """
        #random_state=None, class_weight=None ,random_state=random_state, class_weight=class_weight max_depth=max_depth , max_leaf_nodes min_impurity_split=min_impurity_split, max_leaf_nodes=max_leaf_nodes
        return RandomForestClassifier.predict_proba(self, samples)

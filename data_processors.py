from abc import ABCMeta, abstractmethod
import numpy as np


class DataProcessor(metaclass=ABCMeta):
    """Abstract data processor class."""

    @abstractmethod
    def fit(self, data):
        """Fits the internal DataProcessor values to the data."""
        raise NotImplementedError

    @abstractmethod
    def transform(self, data):
        """Transforms the data with the DataProcessor."""
        raise NotImplementedError

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)


class Imputer(DataProcessor):
    def __init__(self, strategy, new_features=False):
        """Initialize object.

        Args:
            strategy (str): strategy to follow when imputing. Available: 'mean', 'min'
            new_features (str): whether we should create new features depending from the information from missing
                values. False creates no new features, 'sum' creates one new and '1-hot' creates multiple new features.
        """
        self.strategy = strategy
        self.new_features = new_features

    def fit(self, data):
        if self.strategy == 'mean':
            self.imputing_values = np.nanmean(data, axis=0)
        elif self.strategy == 'min':
            self.imputing_values = np.nanmin(data, axis=0)
        else:
            raise ValueError('{} is not a valid value for `strategy`'.format(self.strategy))
        if self.new_features == '1-hot':
            self.contains_nan = np.any(np.isnan(data), axis=0)

    def transform(self, data):
        # Add new features from nan values
        if not self.new_features:
            extra_features = np.zeros([len(data), 0])
        elif self.new_features == 'sum':
            extra_features = np.isnan(data).sum(axis=1)[:, None]
        elif self.new_features == '1-hot':
            extra_features = np.isnan(data)[:, self.contains_nan]
            extra_features = np.atleast_2d(extra_features)
        else:
            raise ValueError('{} is not a valid value for `new_features`'.format(self.new_features))
        # Imputes nan values
        data = np.copy(data)
        for i in range(len(data)):
            isnan = np.isnan(data[i])
            data[i, isnan] = self.imputing_values[isnan]

        out_data = np.concatenate((data, extra_features), axis=1)
        return out_data


class Processor(DataProcessor):

    def __init__(self, normalize):
        self.normalize = normalize

    def fit(self, data):
        if self.normalize:
            self.mean = np.nanmean(data, axis=0)
            self.std = np.nanstd(data, axis=0)

    def transform(self, data):
        data = np.copy(data)
        if self.normalize:
            data = (data - self.mean) / self.std
        return data

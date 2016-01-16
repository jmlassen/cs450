import numpy as np


class KnnClassifier:
    def __init__(self, n_neighbors=1):
        self.n_neighbors = n_neighbors
        self.mean = None
        self.std = None
        self.train_z_scores = None
        self.train_target = None

    def fit(self, train_data, train_target):
        self.train_z_scores = self._find_z_scores(train_data)
        self.train_target = train_target

    def predict(self, test_data):
        # Make sure we have trained before we start predicting
        if self.train_z_scores is None:
            return
        test_z_scores = self._find_z_scores(test_data)
        # Calculate distance
        dist = np.linalg.norm(self.train_z_scores - test_z_scores)
        return np.zeros(len(test_data))

    def _find_z_scores(self, data):
        np_data = np.asarray(data)
        # Check to see if we need to set the population mean and std
        if self.mean is None:
            self.mean = np_data.mean()
        if self.std is None:
            self.std = np_data.std()
        return (np_data - self.mean) / self.std

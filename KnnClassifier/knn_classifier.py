import numpy as np
from collections import Counter


class KnnClassifier:
    def __init__(self, n_neighbors=1):
        self.n_neighbors = n_neighbors
        self.mean = None
        self.std = None
        self.train_z_scores = None
        self.train_target = None

    def fit(self, train_data, train_target):
        self._set_population_values(train_data)
        self.train_z_scores = self._find_z_scores(train_data)
        self.train_target = train_target

    def predict(self, test_data):
        # Make sure we have trained before we start predicting
        if self.train_z_scores is None:
            return
        test_z_scores = self._find_z_scores(test_data)
        # Create an empty array for results
        results = []
        # Make prediction for each test_data point
        for point in test_z_scores:
            results.append(self._predict_point(point))
        return results

    def _set_population_values(self, train_data):
        np_data = np.asarray(train_data)
        self.mean = np_data.mean()
        self.std = np_data.std()

    def _find_z_scores(self, data):
        return (data - self.mean) / self.std

    def _predict_point(self, point):
        distances = []
        # Loop through each trained z score
        for train_point in self.train_z_scores:
            distances.append(np.linalg.norm(point - train_point))
        # Sort the distances along side the target list
        d_sorted,t_sorted = zip(*sorted(zip(distances, self.train_target)))
        return self._return_majority_value(list(t_sorted)[:self.n_neighbors])

    def _return_majority_value(self, list):
        # Put list into a counter
        c = Counter(list)
        value, count = c.most_common()[0]
        return value

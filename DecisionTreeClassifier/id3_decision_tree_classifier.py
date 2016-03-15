from collections import Counter

import numpy as np
from math import log

from utilities import bin_data


class Id3DecisionTreeClassifier:
    def __init__(self, categories=3):
        self.tree = None
        self.categories = categories

    def fit(self, data, target):
        # Check if we need to bin data here?
        binned_data = bin_data(data, self.categories)
        # Make a copy of data and target here, then pass to create tree
        self._create_tree(binned_data, target)

    def predict(self, data):
        results = []
        for point in data:
            results.append(0)
        return results

    def _create_tree(self, data, target):
        features = []
        for i in range(len(data[0])):
            features.append(i)
        self._make_node(0, features, data, target)

    def _make_node(self, node_label, features, data, target):
        labels = self._find_unique(target)
        # If all examples have the same label
        if len(labels) < 2:
            # return a leaf with that label
            return {node_label: labels[0]}
        # Else if there are no features left to test
        elif len(features) < 1:
            # return a leaf with the most common label
            return {node_label: self._find_most_common(target)}
        else:
            # choose the feature F that maximises the information gain of S to be the next node
            feature_gain = self._calculate_gain(labels, features, data, target)
            # add a branch from the node for each possible value f in F
            # for each branch:

    def _find_unique(self, a):
        return np.unique(a)

    def _find_most_common(self, target):
        return Counter(target).most_common(1)

    def _calculate_gain(self, labels, features, data, target):
        entropy = self._calculate_entropy(labels, target)
        gain = []
        for feature in features:
            gain.append(self._calculate_feature_gain(entropy, feature, data, target))
        return gain

    def _calculate_feature_gain(self, entropy, feature, data, target):
        gain = entropy
        feature_data = data[:, feature]
        unique_values = self._find_unique(feature_data)
        for value in unique_values:
            # stopping here 03-14-2016
            gain -= Counter(feature_data)[value] / feature_data.size
        return gain

    def _calculate_entropy(self, labels, target):
        entropy = 0
        for label in labels:
            probability = Counter(target)[label] / len(target)
            entropy -= probability * log(probability, 2)
        return entropy

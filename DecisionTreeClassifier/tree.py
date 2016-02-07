import logging
from collections import Counter
import numpy as np


def calc_entropy(p):
    if p != 0:
        return -p * np.log2(p)
    else:
        return 0


class Tree:
    def __init__(self):
        self.logger = logging.getLogger(Tree.__name__)
        self.logger.debug('Initialized new class object.')
        self.total_entropy = None

    def make_tree(self, data, target, features):
        self.logger.debug('Started make_tree function.')
        self.logger.debug('Passing off to private function.')
        self._make_tree(data, target, features)

    def _make_tree(self, data, target, features):
        self.logger.debug('Started _make_tree function.')
        default = Counter(target).most_common(1)[0]
        self.logger.debug("For this branch, default value found to be {}.".format(default[0]))
        if len(target) == 0 or len(features) == 0:
            self.logger.debug("Empty branch found.")
            return default
        elif default[1] == len(target):
            self.logger.debug("Only one class remains.")
            return target[0]
        else:
            gain = np.zeros(len(features))
            for i in range(len(features)):
                gain[i] = self._calc_gain(data, target, i)
            best_feature = np.argmax(gain)
            tree = {features[best_feature]: {}}

    def _calc_gain(self, data, target, feature):
        gain = 0
        n_data = len(data)
        values = []
        for point in data:
            if point[feature] not in values:
                values.append(point[feature])
        feature_counts = np.zeros(len(values))
        entropy = np.zeros(len(values))
        value_index = 0
        for value in values:
            data_index = 0
            new_target = []
            for point in data:
                if point[feature] == value:
                    feature_counts[value_index] += 1
                    new_target.append(target[data_index])
                data_index += 1
            target_values = []
            for a_target in new_target:
                if target_values.count(a_target) == 0:
                    target_values.append(a_target)
            target_counts = np.zeros(len(target_values))
            target_index = 0
            for target_value in target_values:
                for a_target in new_target:
                    if a_target == target_value:
                        target_counts[target_index] += 1
                target_index += 1
            for target_index in range(len(target_values)):
                entropy[value_index] += calc_entropy(float(target_counts[target_index]) / sum(target_counts))
            gain += float(feature_counts[value_index]) / n_data * entropy[value_index]
            value_index += 1
        return gain

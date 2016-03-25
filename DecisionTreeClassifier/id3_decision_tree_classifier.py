from collections import Counter

import numpy as np

from utilities import bin_data, calculate_entropy


class Id3DecisionTreeClassifier:
    def __init__(self, categories=6):
        self.tree = None
        self.categories = categories

    def fit(self, data, target):
        # Check if we need to bin data here?
        binned_data = bin_data(data, self.categories)
        # Make a copy of data and target here, then pass to create tree
        self._create_tree(binned_data, target)
        print("Tree: {}".format(self.tree))

    def predict(self, data):
        results = []
        binned_data = bin_data(data, self.categories)
        for point in binned_data:
            results.append(self._traverse_tree(self.tree, point))
        return results

    def _traverse_tree(self, tree, point):
        if isinstance(tree, np.int32) or isinstance(tree, np.int64) or isinstance(tree, np.str_):
            return tree
        else:
            feature = list(tree.keys())[0]
            if point[feature] not in tree[feature]:
                return 0
            subtree = tree[feature][point[feature]]
            return self._traverse_tree(subtree, point)

    def _create_tree(self, data, target):
        features = []
        for i in range(len(data[0])):
            features.append(i)
        self.tree = self._make_tree(data, target, features)

    def _make_tree(self, data, target, features):
        target_labels = np.unique(target)
        default = target.argmax()
        # If all data points have the same label
        if len(data) == 0 or len(features) == 0:
            return default
        # Else if there are no features left to test
        elif len(target_labels) == 1:
            return target[0]
        else:
            # choose the feature F that maximises the information gain of S to be the next node
            best_feature_index = self._get_feature_with_highest_gain(features, data, target)
            tree = {features[best_feature_index]: {}}
            best_feature_values = np.unique(data[:, best_feature_index])
            # add a branch from the node for each possible value f in F
            for value in best_feature_values:
                data_subset = []
                target_subset = []
                feature_subset = []
                for point_index, point in enumerate(data):
                    if point[best_feature_index] == value:
                        if best_feature_index == 0:
                            point = point[1:]
                            # feature_subset = features[1:]
                        elif best_feature_index == len(features):
                            point = point[:-1]
                            # feature_subset = features[:-1]
                        else:
                            point = np.concatenate((point[:best_feature_index], point[best_feature_index + 1:]), axis=0)
                            # feature_subset = features[:-1]
                        data_subset.append(point)
                        target_subset.append(target[point_index])
                for i in range(len(features) - 1):
                    feature_subset.append(i)
                subtree = self._make_tree(np.array(data_subset), np.array(target_subset), feature_subset)
                tree[features[best_feature_index]][value] = subtree
            return tree

    def _get_feature_with_highest_gain(self, features, data, target):
        feature_gains = np.zeros(len(features))
        for feature in features:
            gain = self._calculate_info_gain(data, target, feature)
            feature_gains[feature] = gain
        return np.argmin(feature_gains)

    def _calculate_info_gain(self, data, target, feature):
        gain = 0
        feature_values = np.unique(data[:, feature])
        feature_counts = np.zeros(len(feature_values))
        entropy = np.zeros(len(feature_values))
        for value_index, value in enumerate(feature_values):
            target_subset = []
            for point_index, point in enumerate(data):
                if point[feature] == value:
                    feature_counts[value_index] += 1
                    target_subset.append(target[point_index])
            target_subset_values = np.unique(target_subset)
            target_subset_counts = np.zeros(len(target_subset_values))
            for target_subset_index, target_subset_value in enumerate(target_subset_values):
                for s_target in target_subset:
                    if s_target == target_subset_value:
                        target_subset_counts[target_subset_index] += 1
            for target_subset_index in range(len(target_subset_values)):
                entropy[value_index] += calculate_entropy(float(target_subset_counts[target_subset_index] /
                                                                sum(target_subset_counts)))
            gain += float(feature_counts[value_index] / len(data) * entropy[value_index])
        return gain

import copy
import numpy as np
import logging
import pandas as pd
from DecisionTreeClassifier.tree import Tree


class Id3DecisionTreeClassifier:
    def __init__(self):
        self.logger = logging.getLogger(Id3DecisionTreeClassifier.__name__)
        self.logger.debug('Initialized new class object.')
        self.tree = None

    def fit(self, data, target):
        self.logger.debug('Started fit function.')
        feature_count = len(data[0])
        self.logger.debug("Number of features for data found to be {}.".format(feature_count))
        feature_tracker = list(range(feature_count))
        self.logger.debug("Created array to track features in tree: {}".format(feature_tracker))
        binned_data = self._convert_numeric_data_to_nominal(data)
        self.logger.debug("Sending binned_data to build decision tree.")
        self.tree = self._build_tree(binned_data, target, feature_tracker)

    def predict(self, data):
        if self.tree is None:
            self.logger.critical('Tree has not been initialized yet! You must train data before you can predict!')
            raise Exception('No data has been fit before calling predict function.')
        return np.empty_like(data)

    def _convert_numeric_data_to_nominal(self, data, bin_count: int = 5):
        """Converts numeric data into nominal data.

        Searches each column looking for numeric values. If any around found, it converts the whole column to nominal
        data. If no columns are found with numeric values, it returns the same array back.

        TODO: Turn bin_count into something that can be set for each column.

        :param bin_count: number of bins desired to be created for each numeric column.
        :param data: data set containing 0 to n columns with numeric data.
        :return: processed 2d array with nominal data replaced the numeric data.
        """
        self.logger.debug('Started _convert_numeric_data_to_nominal function.')
        converted_data = np.array(copy.deepcopy(data))
        for i in range(len(converted_data[0])):
            self.logger.debug("Inspecting column {}.".format(i))
            if isinstance(converted_data[0][i], str) or isinstance(converted_data[0][i], bool):
                self.logger.debug("Data in column {} found to be a str or bool, no processing needed.".format(i))
            else:
                self.logger.debug("Data in column {} is not a str or bool, converting to nominal value.".format(i))
                converted_data[:, i] = pd.cut(converted_data[:, i], bin_count, labels=list(range(bin_count)))
                self.logger.debug("Converted column {} into data contained in {} bins.".format(i, bin_count))
        return converted_data

    def _build_tree(self, binned_data, target, feature_tracker):
        """Builds a new decision tree to be used when classifying data.

        :param binned_data: data assumed to be binned
        :param target: labels of binned data
        :param feature_tracker: array containing a value for each column in data
        :return:
        """
        self.logger.debug('Started _build_tree function.')
        tree = Tree()
        self.logger.debug('Calling make_tree on Tree object.')
        tree.make_tree(binned_data, target, feature_tracker)
        return tree

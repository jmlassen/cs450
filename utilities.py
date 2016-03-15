import copy

from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import numpy as np
import pandas as pd


def cross_val_score(classifier, data, target, cv):
    """Runs a cross validation test with a provided number of folds.

    :param classifier: The classifier we want to test, must have 'fit' and 'predict' methods implemented
    :param data:
    :param target:
    :param cv:
    :return:
    """
    data, target = shuffle(data, target)
    fold_len = int(len(data) / cv)
    results = []
    for i in range(cv):
        start_index = fold_len * i
        end_index = fold_len * (i + 1)
        train_data = np.concatenate((data[:start_index], data[end_index:]), axis=0)
        train_target = np.concatenate((target[:start_index], target[end_index:]), axis=0)
        classifier.fit(train_data, train_target)
        prediction = classifier.predict(data[start_index:end_index])
        accuracy = accuracy_score(target[start_index:end_index], prediction)
        results.append(accuracy)
    return np.array(results)


def bin_data(data, bin_count):
    binned_data = np.array(copy.deepcopy(data))
    for i in range(binned_data[0].size):
        binned_data[:, i] = pd.cut(binned_data[:, i], bin_count, labels=list(range(bin_count)))
    return binned_data

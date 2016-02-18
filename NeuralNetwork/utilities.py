import math
import numpy as np
from sklearn.utils import shuffle

DEFAULT_WEIGHT_FLOOR = -1
DEFAULT_WEIGHT_CEILING = 1


def get_starting_weights(n_inputs):
    return np.random.uniform(DEFAULT_WEIGHT_FLOOR, DEFAULT_WEIGHT_CEILING, n_inputs)


def sigmoid(x):
    return 1 / (1 + math.exp((-1 * x)))


def cross_val_score(classifier, data, target, cv):
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
        if len(prediction) != len(target[start_index:end_index]):
            raise Exception('Classifier predict function did not return correct number of results!')
        accuracy = np.sum(prediction == target[start_index:end_index]) / len(prediction)
        results.append(accuracy)
    return np.array(results)

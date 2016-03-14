import math
import random

import numpy as np
from numpy.ma import around
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle


DEFAULT_WEIGHT_FLOOR = -1
DEFAULT_WEIGHT_CEILING = 1


def get_starting_weights(n_inputs):
    """

    :param n_inputs:
    :return:
    """
    weights = []
    for _ in range(n_inputs):
        weights.append(random.uniform(DEFAULT_WEIGHT_FLOOR, DEFAULT_WEIGHT_CEILING))
    return weights


def sigmoid(x):
    """

    :param x:
    :return:
    """
    return 1 / (1 + math.exp((-1 * float(x))))


def cross_val_score(classifier, data, target, cv):
    """

    :param classifier:
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
        if len(prediction) != len(target[start_index:end_index]):
            raise Exception('Classifier predict function did not return correct number of results!')
        accuracy = accuracy_score(target[start_index:end_index], prediction)
        results.append(accuracy)
    return np.array(results)


def calc_output_delta(a, t):
    """

    :param a: Activation
    :param t: Target
    :return: The delta for the given data.
    """
    return a * (1 - a) * (a - t)


def calc_hidden_delta(a, w, d):
    """

    :param a:
    :param w:
    :param d:
    :return:
    """
    weight_sum = 0
    for i in range(len(w)):
        weight_sum += w[i] * d[i]
    return a * (1 - a) * weight_sum


def calc_updated_weight(w, l, d, a):
    """

    :param w: Weight
    :param l: Learning rate
    :param d: Delta
    :param a: Activation
    :return: New weight
    """
    return w - l * d * a


def print_network(network):
    print("[")
    max_layer_size = 0
    for i in range(len(network)):
        if len(network[i]) > max_layer_size:
            max_layer_size = len(network[i])
    for i in range(max_layer_size):
        print("\t", end="")
        for layer in network:
            if len(layer) > i:
                    print("{}   ".format(around(layer[i], 2)), end="")
            else:
                print("{}   ".format("    "), end="")
        print()
    print("]")

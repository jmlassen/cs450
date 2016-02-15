import math
import numpy as np

DEFAULT_WEIGHT_FLOOR = -1
DEFAULT_WEIGHT_CEILING = 1


def get_starting_weights(n_inputs):
    return np.random.uniform(DEFAULT_WEIGHT_FLOOR, DEFAULT_WEIGHT_CEILING, n_inputs)


def sigmoid(x):
    return 1 / (1 + math.exp((-1 * x)))

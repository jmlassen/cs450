import numpy as np

DEFAULT_WEIGHT_FLOOR = -1
DEFAULT_WEIGHT_CEILING = 1


def get_starting_weights(n_inputs):
    return np.random.uniform(DEFAULT_WEIGHT_FLOOR, DEFAULT_WEIGHT_CEILING, n_inputs)